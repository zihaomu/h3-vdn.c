#include "h3_audio_vae.h"
#include "h3_safetensors.h"
#include "h3_video_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VIDEO_CHANNELS = 24,
    VIDEO_T = 17,
    VIDEO_H = 2,
    VIDEO_W = 4,
    OUTPUT_FRAMES = 56,
    OUTPUT_H = 32,
    OUTPUT_W = 64,
    RGB_CHANNELS = 3,
    AUDIO_CHANNELS = 32,
    AUDIO_BATCH = 2,
    AUDIO_T = 3,
    AUDIO_SAMPLES = 2400
};

typedef struct {
    double max_abs;
    double relative_rmse;
    double cosine;
    size_t changed;
    size_t nonfinite;
} metrics;

static void *read_tensor(const h3_st_header *header, const char *name,
                         h3_dtype dtype, int ndim, const uint64_t *shape,
                         size_t *bytes, char *error, size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor || tensor->dtype != dtype || tensor->ndim != ndim) {
        snprintf(error, error_size, "missing or incompatible tensor %s", name);
        return NULL;
    }
    for (int axis = 0; axis < ndim; axis++) {
        if (tensor->shape[axis] != shape[axis]) {
            snprintf(error, error_size, "shape mismatch for %s", name);
            return NULL;
        }
    }
    uint64_t elements = h3_st_tensor_elements(tensor);
    size_t item_size = h3_dtype_size(dtype);
    if (!item_size || elements > SIZE_MAX / item_size) {
        snprintf(error, error_size, "size overflow for %s", name);
        return NULL;
    }
    *bytes = (size_t)elements * item_size;
    void *data = malloc(*bytes ? *bytes : 1);
    if (!data || !h3_st_read_data(header, tensor, data, *bytes,
                                  error, error_size)) {
        free(data);
        return NULL;
    }
    return data;
}

static metrics calculate_metrics(const float *actual, const float *expected,
                                 size_t elements) {
    metrics result = {0};
    double squared_error = 0.0, squared_reference = 0.0;
    double squared_actual = 0.0, dot = 0.0;
    for (size_t index = 0; index < elements; index++) {
        double a = actual[index];
        double b = expected[index];
        if (!isfinite(a) || !isfinite(b)) {
            result.nonfinite++;
            continue;
        }
        double difference = a - b;
        if (fabs(difference) > result.max_abs)
            result.max_abs = fabs(difference);
        squared_error += difference * difference;
        squared_reference += b * b;
        squared_actual += a * a;
        dot += a * b;
        result.changed += actual[index] != expected[index];
    }
    result.relative_rmse = squared_reference > 0.0 ?
        sqrt(squared_error / squared_reference) : INFINITY;
    result.cosine = squared_reference > 0.0 && squared_actual > 0.0 ?
        dot / sqrt(squared_reference * squared_actual) : 0.0;
    return result;
}

static void video_progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 1 || completed == total || completed % 6 == 0)
        fprintf(stderr, "VDN oracle video VAE: block %d/%d\n",
                completed, total);
}

static void audio_progress(int completed, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "VDN oracle audio VAE: stage %d/%d\n", completed, total);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s VIDEO_VAE AUDIO_VAE ORACLE\n", argv[0]);
        return 2;
    }
    int status = 1;
    char error[512] = {0};
    h3_st_header header;
    h3_video_frames frames;
    h3_audio_waveform waveform;
    memset(&header, 0, sizeof(header));
    memset(&frames, 0, sizeof(frames));
    memset(&waveform, 0, sizeof(waveform));
    float *video_latent = NULL, *audio_latent = NULL;
    float *audio_latent_c = NULL;
    float *video_expected = NULL, *audio_expected = NULL;
    const uint64_t video_latent_shape[] = {
        1, VIDEO_CHANNELS, VIDEO_T, VIDEO_H, VIDEO_W
    };
    const uint64_t audio_latent_shape[] = {
        AUDIO_BATCH, AUDIO_CHANNELS, AUDIO_T
    };
    const uint64_t video_shape[] = {
        OUTPUT_FRAMES, OUTPUT_H, OUTPUT_W, RGB_CHANNELS
    };
    const uint64_t audio_shape[] = {AUDIO_BATCH, AUDIO_SAMPLES};
    size_t bytes = 0;
    if (!h3_st_read_header(argv[3], &header, error, sizeof(error))) goto failed;
    video_latent = read_tensor(
        &header, "input.video_latent_normalized", H3_DTYPE_F32, 5,
        video_latent_shape, &bytes, error, sizeof(error));
    audio_latent = read_tensor(
        &header, "input.audio_latent_normalized", H3_DTYPE_F32, 3,
        audio_latent_shape, &bytes, error, sizeof(error));
    video_expected = read_tensor(
        &header, "video.rgb_f32", H3_DTYPE_F32, 4,
        video_shape, &bytes, error, sizeof(error));
    audio_expected = read_tensor(
        &header, "audio.pcm_f32", H3_DTYPE_F32, 2,
        audio_shape, &bytes, error, sizeof(error));
    if (!video_latent || !audio_latent || !video_expected || !audio_expected)
        goto failed;
    audio_latent_c = malloc(
        (size_t)AUDIO_BATCH * AUDIO_CHANNELS * AUDIO_T *
        sizeof(*audio_latent_c));
    if (!audio_latent_c) {
        snprintf(error, sizeof(error), "out of memory transposing audio latent");
        goto failed;
    }
    /* Upstream exposes [stereo,channel,time]; the public C decoder contract is
     * [channel,stereo,time], matching h3_dit_unpack_audio(). */
    for (int stereo = 0; stereo < AUDIO_BATCH; stereo++)
        for (int channel = 0; channel < AUDIO_CHANNELS; channel++)
            for (int time = 0; time < AUDIO_T; time++)
                audio_latent_c[
                    ((size_t)channel * AUDIO_BATCH + (size_t)stereo) *
                    AUDIO_T + (size_t)time] =
                    audio_latent[
                        ((size_t)stereo * AUDIO_CHANNELS + (size_t)channel) *
                        AUDIO_T + (size_t)time];

    if (!h3_video_vae_decode(
            argv[1], "unused-on-hip", video_latent, VIDEO_T, VIDEO_H, VIDEO_W,
            video_progress, NULL, &frames, error, sizeof(error)))
        goto failed;
    if (frames.frames != OUTPUT_FRAMES || frames.height != OUTPUT_H ||
        frames.width != OUTPUT_W) {
        snprintf(error, sizeof(error), "C video VAE returned wrong shape");
        goto failed;
    }
    if (!h3_audio_vae_decode(
            argv[2], "unused-on-hip", audio_latent_c, AUDIO_T,
            audio_progress, NULL, &waveform, error, sizeof(error)))
        goto failed;
    if (waveform.channels != AUDIO_BATCH || waveform.samples != AUDIO_SAMPLES ||
        waveform.sample_rate != 32000) {
        snprintf(error, sizeof(error), "C audio VAE returned wrong shape");
        goto failed;
    }

    const size_t video_elements = (size_t)OUTPUT_FRAMES * OUTPUT_H *
                                  OUTPUT_W * RGB_CHANNELS;
    const size_t audio_elements = (size_t)AUDIO_BATCH * AUDIO_SAMPLES;
    metrics video = calculate_metrics(
        frames.rgb, video_expected, video_elements);
    metrics audio = calculate_metrics(
        waveform.pcm, audio_expected, audio_elements);
    size_t rgb24_changed = 0;
    for (size_t index = 0; index < video_elements; index++) {
        long actual = lroundf(fminf(1.0f, fmaxf(0.0f, frames.rgb[index])) *
                              255.0f);
        long expected = lroundf(
            fminf(1.0f, fmaxf(0.0f, video_expected[index])) * 255.0f);
        rgb24_changed += actual != expected;
    }
    printf("OpenVDN video VAE: max_abs=%.9g relative_rmse=%.9g "
           "cosine=%.12g changed=%zu/%zu rgb24_changed=%zu nonfinite=%zu\n",
           video.max_abs, video.relative_rmse, video.cosine, video.changed,
           video_elements, rgb24_changed, video.nonfinite);
    printf("OpenVDN audio VAE: max_abs=%.9g relative_rmse=%.9g "
           "cosine=%.12g changed=%zu/%zu nonfinite=%zu\n",
           audio.max_abs, audio.relative_rmse, audio.cosine, audio.changed,
           audio_elements, audio.nonfinite);
    /* Existing independent real-weight decoder fixtures established 5%
     * relative-L2 bounds for both native VAEs. Keep those gates and add
     * cosine/non-finite checks before inspecting this VDN candidate. */
    if (video.nonfinite || audio.nonfinite || video.max_abs > 0.05 ||
        video.relative_rmse > 0.05 || video.cosine < 0.99 ||
        audio.max_abs > 0.001 || audio.relative_rmse > 0.05 ||
        audio.cosine < 0.99) {
        snprintf(error, sizeof(error), "dual VAE exceeds upstream parity bounds");
        goto failed;
    }
    puts("PASS: verified VDN latents close through both upstream/C VAEs");
    status = 0;
    goto cleanup;

failed:
    fprintf(stderr, "VDN VAE oracle failed: %s\n", error);
cleanup:
    h3_audio_waveform_free(&waveform);
    h3_video_frames_free(&frames);
    free(audio_expected);
    free(video_expected);
    free(audio_latent_c);
    free(audio_latent);
    free(video_latent);
    h3_st_free_header(&header);
    return status;
}
