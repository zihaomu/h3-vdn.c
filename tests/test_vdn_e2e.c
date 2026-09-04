#include "h3_audio_vae.h"
#include "h3_dit.h"
#include "h3_ffmpeg.h"
#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_vdn_dit.h"
#include "h3_vdn_prompt.h"
#include "h3_vdn_weights.h"
#include "h3_video_vae.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum {
    VIDEO_CHANNELS = 24,
    AUDIO_CHANNELS = 32,
    VIDEO_PATCH = 96,
    AUDIO_WIDTH = 32,
    DEFAULT_LATENT_H = 2,
    DEFAULT_LATENT_W = 4,
    DEFAULT_REQUESTED_FRAMES = 56,
    DEFAULT_NFE = 8
};

static int env_u32(const char *name, uint32_t fallback, uint32_t *value,
                   char *error, size_t error_size) {
    const char *text = getenv(name);
    if (!text || !*text) {
        *value = fallback;
        return 1;
    }
    char *end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(text, &end, 10);
    if (errno || !end || *end || !parsed || parsed > UINT32_MAX) {
        snprintf(error, error_size, "invalid %s=%s", name, text);
        return 0;
    }
    *value = (uint32_t)parsed;
    return 1;
}

static int checked_mul_size(size_t left, size_t right, size_t *result) {
    if (!result || (right && left > SIZE_MAX / right)) return 0;
    *result = left * right;
    return 1;
}

static uint64_t hash_bytes(const void *data, size_t bytes) {
    const uint8_t *values = data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < bytes; index++) {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void layer_progress(unsigned completed, unsigned total, void *opaque) {
    (void)opaque;
    if (completed == total)
        fprintf(stderr, "VDN E2E: completed %u transformer blocks\n", total);
}

static void nfe_progress(unsigned completed, unsigned total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "VDN E2E: NFE %u/%u\n", completed, total);
}

static void video_progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 1 || completed == total || completed % 6 == 0)
        fprintf(stderr, "VDN E2E: video VAE block %d/%d\n",
                completed, total);
}

static void audio_progress(int completed, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "VDN E2E: audio VAE stage %d/%d\n", completed, total);
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s H3_BASE STAGE_DMD PROMPT OUTPUT_MP4\n",
                argv[0]);
        return 2;
    }
    int status = 1;
    char error[512] = {0};
    uint32_t requested_frames = DEFAULT_REQUESTED_FRAMES;
    uint32_t latent_h = DEFAULT_LATENT_H;
    uint32_t latent_w = DEFAULT_LATENT_W;
    uint32_t evaluations = DEFAULT_NFE;
    if (!env_u32("VDN_E2E_FRAMES", requested_frames, &requested_frames,
                 error, sizeof(error)) ||
        !env_u32("VDN_E2E_LATENT_H", latent_h, &latent_h,
                 error, sizeof(error)) ||
        !env_u32("VDN_E2E_LATENT_W", latent_w, &latent_w,
                 error, sizeof(error)) ||
        !env_u32("VDN_E2E_NFE", evaluations, &evaluations,
                 error, sizeof(error))) {
        fprintf(stderr, "VDN native E2E failed: %s\n", error);
        return 2;
    }
    if (requested_frames > INT_MAX || latent_h < 2 || latent_w < 2 ||
        latent_h > INT_MAX / H3_VAE_SPATIAL_RATIO ||
        latent_w > INT_MAX / H3_VAE_SPATIAL_RATIO ||
        latent_h % 2 || latent_w % 2 || evaluations > H3_MAX_STEPS) {
        fprintf(stderr,
                "VDN native E2E failed: invalid frames/latent/NFE geometry\n");
        return 2;
    }
    h3_temporal_shape temporal = h3_temporal((int)requested_frames);
    uint32_t audio_latents = (uint32_t)temporal.audio_t;
    if (!env_u32("VDN_E2E_AUDIO_LATENTS", audio_latents, &audio_latents,
                 error, sizeof(error))) {
        fprintf(stderr, "VDN native E2E failed: %s\n", error);
        return 2;
    }
    if (temporal.frame_count <= 0 || temporal.video_t <= 0 ||
        temporal.audio_t <= 0 || audio_latents != (uint32_t)temporal.audio_t ||
        audio_latents > INT_MAX / 800) {
        fprintf(stderr,
                "VDN native E2E failed: audio latents must match frame geometry\n");
        return 1;
    }
    fprintf(stderr,
            "VDN E2E geometry: requested=%u output=%d latent=%dx%ux%u "
            "audio=%u NFE=%u\n",
            requested_frames, temporal.frame_count, temporal.video_t,
            latent_h, latent_w, audio_latents, evaluations);
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    h3_vdn_weight_store *store = NULL;
    h3_vdn_model_weights model;
    h3_text_embedding prompt;
    h3_vdn_layout layout;
    memset(&model, 0, sizeof(model));
    memset(&prompt, 0, sizeof(prompt));
    memset(&layout, 0, sizeof(layout));
    h3_gpu_tensor *refined = NULL, *video = NULL, *audio = NULL;
    float *video_rows = NULL, *audio_rows = NULL;
    float *video_latent = NULL, *audio_latent = NULL;
    h3_video_frames frames;
    h3_audio_waveform waveform;
    memset(&frames, 0, sizeof(frames));
    memset(&waveform, 0, sizeof(waveform));
    uint8_t *rgb = NULL;
    if (!gpu) goto failed;
    store = h3_vdn_weight_store_open(argv[1], argv[2], 1,
                                     error, sizeof(error));
    if (!store || !h3_vdn_prompt_load(argv[3], &prompt,
                                      error, sizeof(error)) ||
        !h3_vdn_model_weights_load(store, gpu, &model,
                                   error, sizeof(error))) goto failed;
    refined = h3_vdn_refine_prompt(gpu, &model, &prompt,
                                   error, sizeof(error));
    if (!refined || !h3_vdn_layout_build(
            &prompt, (uint32_t)temporal.video_t, latent_h, latent_w,
            audio_latents, &layout,
            error, sizeof(error))) goto failed;
    size_t video_row_elements, audio_row_elements;
    if (!checked_mul_size(layout.video_rows, VIDEO_PATCH,
                          &video_row_elements) ||
        !checked_mul_size(layout.audio_rows, AUDIO_WIDTH,
                          &audio_row_elements)) {
        snprintf(error, sizeof(error), "E2E latent row size overflow");
        goto failed;
    }
    video_rows = malloc(video_row_elements * sizeof(*video_rows));
    audio_rows = malloc(audio_row_elements * sizeof(*audio_rows));
    if (!video_rows || !audio_rows) {
        snprintf(error, sizeof(error), "out of memory allocating E2E latents");
        goto failed;
    }
    h3_rng rng;
    h3_rng_seed(&rng, 0);
    h3_rng_fill_normal(&rng, video_rows, video_row_elements);
    h3_rng_fill_normal(&rng, audio_rows, audio_row_elements);
    video = h3_gpu_tensor_from_f32(gpu, video_rows, video_row_elements);
    audio = h3_gpu_tensor_from_f32(gpu, audio_rows, audio_row_elements);
    if (!video || !audio || !h3_vdn_denoise(
            gpu, store, &model, refined, &layout, video, audio, evaluations,
            1, 5,
            layer_progress, nfe_progress, NULL, NULL, error, sizeof(error)) ||
        !h3_gpu_tensor_read_f32(video, video_rows, video_row_elements) ||
        !h3_gpu_tensor_read_f32(audio, audio_rows, audio_row_elements)) {
        if (!error[0]) snprintf(error, sizeof(error),
                                "cannot read denoised E2E latents: %s",
                                h3_gpu_error(gpu));
        goto failed;
    }
    uint64_t video_rows_hash = hash_bytes(
        video_rows, video_row_elements * sizeof(*video_rows));
    uint64_t audio_rows_hash = hash_bytes(
        audio_rows, audio_row_elements * sizeof(*audio_rows));
    size_t video_latent_elements, audio_latent_elements;
    size_t video_plane;
    if (!checked_mul_size((size_t)temporal.video_t, latent_h, &video_plane) ||
        !checked_mul_size(video_plane, latent_w, &video_plane) ||
        !checked_mul_size(VIDEO_CHANNELS, video_plane,
                          &video_latent_elements) ||
        !checked_mul_size((size_t)AUDIO_CHANNELS * 2, audio_latents,
                          &audio_latent_elements)) {
        snprintf(error, sizeof(error), "E2E unpacked latent size overflow");
        goto failed;
    }
    video_latent = malloc(video_latent_elements * sizeof(*video_latent));
    audio_latent = malloc(audio_latent_elements * sizeof(*audio_latent));
    if (!video_latent || !audio_latent ||
        !h3_dit_unpatchify_video(
            video_rows, VIDEO_CHANNELS, temporal.video_t,
            (int)latent_h, (int)latent_w,
            video_latent, video_latent_elements) ||
        !h3_dit_unpack_audio(audio_rows, AUDIO_CHANNELS, (int)audio_latents,
                             audio_latent, audio_latent_elements)) {
        snprintf(error, sizeof(error), "cannot unpack denoised E2E latents");
        goto failed;
    }

    /* Retire the DiT device state before loading the F32 decoder checkpoints. */
    h3_gpu_tensor_free(audio); audio = NULL;
    h3_gpu_tensor_free(video); video = NULL;
    h3_gpu_tensor_free(refined); refined = NULL;
    h3_vdn_layout_free(&layout);
    h3_vdn_model_weights_free(&model);
    h3_vdn_prompt_free(&prompt);
    h3_vdn_weight_store_free(store); store = NULL;
    h3_gpu_free(gpu); gpu = NULL;

    size_t video_path_size = strlen(argv[1]) + strlen("/vae") + 1;
    size_t audio_path_size = strlen(argv[1]) + strlen("/audio_vae") + 1;
    char *video_path = malloc(video_path_size);
    char *audio_path = malloc(audio_path_size);
    if (!video_path || !audio_path) {
        free(video_path); free(audio_path);
        snprintf(error, sizeof(error), "out of memory resolving VAE paths");
        goto failed;
    }
    snprintf(video_path, video_path_size, "%s/vae", argv[1]);
    snprintf(audio_path, audio_path_size, "%s/audio_vae", argv[1]);
    int decoded = h3_video_vae_decode(
        video_path, "unused-on-hip", video_latent, temporal.video_t,
        (int)latent_h, (int)latent_w, video_progress, NULL, &frames,
        error, sizeof(error));
    if (decoded)
        decoded = h3_audio_vae_decode(
            audio_path, "unused-on-hip", audio_latent, (int)audio_latents,
            audio_progress, NULL, &waveform, error, sizeof(error));
    free(audio_path); free(video_path);
    if (!decoded) goto failed;
    if (frames.frames != temporal.frame_count || waveform.channels != 2 ||
        waveform.samples != (int)audio_latents * 800 ||
        waveform.sample_rate != 32000) {
        snprintf(error, sizeof(error), "decoder E2E geometry mismatch");
        goto failed;
    }
    size_t pixels, rgb_bytes;
    if (frames.frames <= 0 || frames.height <= 0 || frames.width <= 0 ||
        !checked_mul_size((size_t)frames.frames, (size_t)frames.height,
                          &pixels) ||
        !checked_mul_size(pixels, (size_t)frames.width, &pixels) ||
        !checked_mul_size(pixels, 3, &rgb_bytes)) {
        snprintf(error, sizeof(error), "decoded E2E frame size overflow");
        goto failed;
    }
    rgb = malloc(rgb_bytes);
    if (!rgb) {
        snprintf(error, sizeof(error), "out of memory converting E2E RGB");
        goto failed;
    }
    for (size_t index = 0; index < rgb_bytes; index++) {
        float value = frames.rgb[index];
        if (!isfinite(value)) {
            snprintf(error, sizeof(error), "non-finite decoded E2E pixel");
            goto failed;
        }
        if (value < 0.0f) value = 0.0f;
        if (value > 1.0f) value = 1.0f;
        rgb[index] = (uint8_t)lroundf(value * 255.0f);
    }
    size_t frame_f32_bytes, waveform_bytes;
    if (!checked_mul_size(rgb_bytes, sizeof(*frames.rgb), &frame_f32_bytes) ||
        !checked_mul_size((size_t)waveform.samples,
                          (size_t)waveform.channels, &waveform_bytes) ||
        !checked_mul_size(waveform_bytes, sizeof(*waveform.pcm),
                          &waveform_bytes)) {
        snprintf(error, sizeof(error), "decoded E2E hash size overflow");
        goto failed;
    }
    printf("VDN E2E hashes: video_rows=%016llx audio_rows=%016llx "
           "video_f32=%016llx audio_pcm=%016llx rgb24=%016llx\n",
           (unsigned long long)video_rows_hash,
           (unsigned long long)audio_rows_hash,
           (unsigned long long)hash_bytes(frames.rgb, frame_f32_bytes),
           (unsigned long long)hash_bytes(waveform.pcm, waveform_bytes),
           (unsigned long long)hash_bytes(rgb, rgb_bytes));
    if (!h3_ffmpeg_write_av_rgb24_f32(
            argv[4], rgb, frames.frames, frames.width, frames.height, 24,
            waveform.pcm, waveform.samples, waveform.channels,
            waveform.sample_rate, error, sizeof(error))) goto failed;
    struct stat info;
    int width = 0, height = 0;
    if (stat(argv[4], &info) || info.st_size < 1024 ||
        !h3_ffprobe_visual_size(argv[4], &width, &height,
                                error, sizeof(error)) ||
        width != frames.width || height != frames.height) {
        if (!error[0]) snprintf(error, sizeof(error),
                                "invalid E2E MP4 output");
        goto failed;
    }
    printf("VDN native E2E passed: %s, %d frames %dx%d @24fps, "
           "stereo %d Hz/%d samples, %lld bytes\n",
           argv[4], frames.frames, frames.width, frames.height,
           waveform.sample_rate, waveform.samples, (long long)info.st_size);
    status = 0;
    goto cleanup;

failed:
    fprintf(stderr, "VDN native E2E failed: %s\n", error);
cleanup:
    free(rgb);
    h3_audio_waveform_free(&waveform);
    h3_video_frames_free(&frames);
    free(audio_latent); free(video_latent);
    free(audio_rows); free(video_rows);
    h3_gpu_tensor_free(audio); h3_gpu_tensor_free(video);
    h3_vdn_layout_free(&layout);
    h3_gpu_tensor_free(refined);
    h3_vdn_model_weights_free(&model);
    h3_vdn_prompt_free(&prompt);
    h3_vdn_weight_store_free(store);
    h3_gpu_free(gpu);
    return status;
}
