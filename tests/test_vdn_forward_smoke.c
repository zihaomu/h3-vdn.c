#include "h3_gpu.h"
#include "h3_vdn_dit.h"
#include "h3_vdn_prompt.h"
#include "h3_vdn_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    FRAMES = 17,
    LATENT_H = 2,
    LATENT_W = 4,
    AUDIO_LATENTS = 3,
    VIDEO_PATCH = 96,
    AUDIO_WIDTH = 32
};

static uint64_t hash_f32(const float *values, size_t count) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < count; index++) {
        uint32_t bits;
        memcpy(&bits, &values[index], sizeof(bits));
        for (unsigned byte = 0; byte < 4; byte++) {
            hash ^= (bits >> (byte * 8)) & UINT32_C(0xff);
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

static void layer_progress(unsigned completed, unsigned total, void *opaque) {
    (void)opaque;
    if (completed == 1 || completed == total || completed % 10 == 0)
        fprintf(stderr, "VDN full forward: block %u/%u\n", completed, total);
}

static void nfe_progress(unsigned completed, unsigned total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "VDN denoise: NFE %u/%u\n", completed, total);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s H3_BASE STAGE_DMD PROMPT\n", argv[0]);
        return 2;
    }
    int status = 1;
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    h3_vdn_weight_store *store = NULL;
    h3_vdn_model_weights model;
    h3_text_embedding prompt;
    h3_vdn_layout layout;
    h3_vdn_velocity velocity;
    memset(&model, 0, sizeof(model));
    memset(&prompt, 0, sizeof(prompt));
    memset(&layout, 0, sizeof(layout));
    memset(&velocity, 0, sizeof(velocity));
    h3_gpu_tensor *refined = NULL, *video = NULL, *audio = NULL;
    float *video_host = NULL, *audio_host = NULL;
    float *video_output = NULL, *audio_output = NULL;
    float *video_compare = NULL, *audio_compare = NULL;
    int denoise = getenv("VDN_SMOKE_DENOISE") != NULL;
    int compare_sdpa = !denoise &&
        getenv("VDN_SMOKE_COMPARE_SDPA") != NULL;
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
            &prompt, FRAMES, LATENT_H, LATENT_W, AUDIO_LATENTS,
            &layout, error, sizeof(error))) goto failed;
    size_t video_elements = (size_t)layout.video_rows * VIDEO_PATCH;
    size_t audio_elements = (size_t)layout.audio_rows * AUDIO_WIDTH;
    video_host = malloc(video_elements * sizeof(*video_host));
    audio_host = malloc(audio_elements * sizeof(*audio_host));
    video_output = malloc(video_elements * sizeof(*video_output));
    audio_output = malloc(audio_elements * sizeof(*audio_output));
    if (!video_host || !audio_host || !video_output || !audio_output) {
        snprintf(error, sizeof(error), "out of host memory for forward fixture");
        goto failed;
    }
    for (size_t index = 0; index < video_elements; index++)
        video_host[index] = sinf((float)index * 0.017f) * 0.75f;
    for (size_t index = 0; index < audio_elements; index++)
        audio_host[index] = cosf((float)index * 0.031f) * 0.6f;
    video = h3_gpu_tensor_from_f32(gpu, video_host, video_elements);
    audio = h3_gpu_tensor_from_f32(gpu, audio_host, audio_elements);
    int ok = video && audio;
    if (compare_sdpa) setenv("H3_VDN_SCALAR_SDPA", "1", 1);
    if (ok && denoise)
        ok = h3_vdn_denoise(
            gpu, store, &model, refined, &layout, video, audio, 8, 1, 5,
            layer_progress, nfe_progress, NULL, error, sizeof(error)) &&
             h3_gpu_tensor_read_f32(video, video_output, video_elements) &&
             h3_gpu_tensor_read_f32(audio, audio_output, audio_elements);
    else if (ok)
        ok = h3_vdn_forward(
                 gpu, store, &model, refined, &layout, video, audio,
                 0.125f, 0.375f, 1, 5, layer_progress, NULL,
                 &velocity, error, sizeof(error)) &&
             h3_gpu_tensor_read_f32(velocity.video, video_output,
                                    video_elements) &&
             h3_gpu_tensor_read_f32(velocity.audio, audio_output,
                                    audio_elements);
    if (!ok) {
        if (!error[0]) snprintf(error, sizeof(error),
                                "cannot read VDN forward outputs: %s",
                                h3_gpu_error(gpu));
        goto failed;
    }
    if (compare_sdpa) {
        video_compare = malloc(video_elements * sizeof(*video_compare));
        audio_compare = malloc(audio_elements * sizeof(*audio_compare));
        h3_vdn_velocity_free(&velocity);
        setenv("H3_VDN_SCALAR_SDPA", "0", 1);
        ok = video_compare && audio_compare &&
             h3_gpu_tensor_write_f32(video, video_host, video_elements) &&
             h3_gpu_tensor_write_f32(audio, audio_host, audio_elements) &&
             h3_vdn_forward(
                 gpu, store, &model, refined, &layout, video, audio,
                 0.125f, 0.375f, 1, 5, layer_progress, NULL,
                 &velocity, error, sizeof(error)) &&
             h3_gpu_tensor_read_f32(velocity.video, video_compare,
                                    video_elements) &&
             h3_gpu_tensor_read_f32(velocity.audio, audio_compare,
                                    audio_elements);
        if (!ok) goto failed;
        double squared = 0.0, reference_squared = 0.0;
        float maximum = 0.0f;
        size_t compared = video_elements + audio_elements;
        for (size_t index = 0; index < video_elements; index++) {
            float difference = fabsf(video_output[index] - video_compare[index]);
            if (difference > maximum) maximum = difference;
            squared += (double)difference * difference;
            reference_squared +=
                (double)video_output[index] * video_output[index];
        }
        for (size_t index = 0; index < audio_elements; index++) {
            float difference = fabsf(audio_output[index] - audio_compare[index]);
            if (difference > maximum) maximum = difference;
            squared += (double)difference * difference;
            reference_squared +=
                (double)audio_output[index] * audio_output[index];
        }
        double rmse = sqrt(squared / (double)compared);
        double relative_rmse = reference_squared > 0.0 ?
            sqrt(squared / reference_squared) : 0.0;
        printf("VDN SDPA scalar/wave32 comparison: max_abs=%.9g "
               "rmse=%.9g relative_rmse=%.9g wave_video=%016llx "
               "wave_audio=%016llx\n", maximum, rmse, relative_rmse,
               (unsigned long long)hash_f32(video_compare, video_elements),
               (unsigned long long)hash_f32(audio_compare, audio_elements));
        if (memcmp(video_output, video_compare,
                   video_elements * sizeof(*video_output)) ||
            memcmp(audio_output, audio_compare,
                   audio_elements * sizeof(*audio_output))) {
            snprintf(error, sizeof(error),
                     "scalar/wave32 VDN SDPA output is not bitwise identical");
            goto failed;
        }
    }
    size_t invalid = 0, nonzero = 0, changed = 0;
    for (size_t index = 0; index < video_elements; index++) {
        invalid += !isfinite(video_output[index]);
        nonzero += video_output[index] != 0.0f;
        changed += video_output[index] != video_host[index];
    }
    for (size_t index = 0; index < audio_elements; index++) {
        invalid += !isfinite(audio_output[index]);
        nonzero += audio_output[index] != 0.0f;
        changed += audio_output[index] != audio_host[index];
    }
    if (invalid || nonzero < (video_elements + audio_elements) / 2 ||
        (denoise && changed < (video_elements + audio_elements) / 2)) {
        snprintf(error, sizeof(error),
                 "invalid full-forward output: invalid=%zu nonzero=%zu changed=%zu",
                 invalid, nonzero, changed);
        goto failed;
    }
    h3_gpu_stats stats;
    if (!h3_gpu_get_stats(gpu, &stats)) goto failed;
    printf("VDN complete %s passed: sequence=%u, "
           "video=F32[%u,96] hash=%016llx, audio=F32[%u,32] "
           "hash=%016llx, peak=%.3f GiB\n",
           denoise ? "8-NFE latent denoise" : "50-layer forward",
           layout.sequence, layout.video_rows,
           (unsigned long long)hash_f32(video_output, video_elements),
           layout.audio_rows,
           (unsigned long long)hash_f32(audio_output, audio_elements),
           (double)stats.peak_live_bytes / (1024.0 * 1024.0 * 1024.0));
    status = 0;
    goto cleanup;

failed:
    fprintf(stderr, "VDN complete forward smoke failed: %s\n", error);
cleanup:
    free(audio_compare); free(video_compare);
    free(audio_output); free(video_output);
    free(audio_host); free(video_host);
    h3_vdn_velocity_free(&velocity);
    h3_gpu_tensor_free(audio); h3_gpu_tensor_free(video);
    h3_vdn_layout_free(&layout);
    h3_gpu_tensor_free(refined);
    h3_vdn_model_weights_free(&model);
    h3_vdn_prompt_free(&prompt);
    h3_vdn_weight_store_free(store);
    h3_gpu_free(gpu);
    return status;
}
