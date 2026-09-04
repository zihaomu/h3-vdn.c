#include "h3_gpu.h"
#include "h3_vdn_dit.h"
#include "h3_vdn_prompt.h"
#include "h3_vdn_weights.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    DEFAULT_FRAMES = 17,
    DEFAULT_LATENT_H = 2,
    DEFAULT_LATENT_W = 4,
    DEFAULT_AUDIO_LATENTS = 3,
    VIDEO_PATCH = 96,
    AUDIO_WIDTH = 32,
    HIDDEN = 5376,
    BLOCKS = 50
};

typedef struct {
    size_t elements;
    int capture;
    h3_gpu_tensor *snapshots[BLOCKS];
    uint16_t *reference;
    uint16_t *candidate;
} layer_comparison;

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int observe_layer(h3_gpu *gpu, unsigned completed, unsigned total,
                         const h3_gpu_tensor *hidden, void *opaque,
                         char *error, size_t error_size) {
    layer_comparison *comparison = opaque;
    if (!comparison || completed < 1 || completed > BLOCKS ||
        total != BLOCKS) {
        snprintf(error, error_size, "invalid layer comparison callback");
        return 0;
    }
    size_t index = completed - 1;
    if (comparison->capture) {
        comparison->snapshots[index] = h3_gpu_tensor_new_bf16(
            gpu, comparison->elements);
        if (!comparison->snapshots[index] || !h3_gpu_begin(gpu) ||
            !h3_gpu_copy_bf16(gpu, comparison->snapshots[index], 0, hidden,
                              0, comparison->elements) ||
            !h3_gpu_submit(gpu)) {
            snprintf(error, error_size, "cannot capture hidden layer %u: %s",
                     completed, h3_gpu_error(gpu));
            return 0;
        }
        return 1;
    }
    if (!comparison->snapshots[index] ||
        !h3_gpu_tensor_read_bf16(comparison->snapshots[index],
                                 comparison->reference,
                                 comparison->elements) ||
        !h3_gpu_tensor_read_bf16(hidden, comparison->candidate,
                                 comparison->elements)) {
        snprintf(error, error_size, "cannot read hidden layer %u: %s",
                 completed, h3_gpu_error(gpu));
        return 0;
    }
    float maximum = 0.0f;
    double squared_error = 0.0, squared_reference = 0.0;
    double squared_candidate = 0.0, dot_product = 0.0;
    size_t invalid = 0;
    for (size_t element = 0; element < comparison->elements; element++) {
        float reference = bf16_to_f32(comparison->reference[element]);
        float candidate = bf16_to_f32(comparison->candidate[element]);
        float difference = candidate - reference;
        if (fabsf(difference) > maximum) maximum = fabsf(difference);
        squared_error += (double)difference * difference;
        squared_reference += (double)reference * reference;
        squared_candidate += (double)candidate * candidate;
        dot_product += (double)reference * candidate;
        invalid += !isfinite(candidate);
    }
    double relative_rmse = squared_reference > 0.0 ?
        sqrt(squared_error / squared_reference) : INFINITY;
    double cosine = squared_reference > 0.0 && squared_candidate > 0.0 ?
        dot_product / sqrt(squared_reference * squared_candidate) : 0.0;
    printf("VDN Sage layer error[%u]: max_abs=%.9g relative_rmse=%.9g "
           "cosine=%.12g invalid=%zu\n", completed, maximum,
           relative_rmse, cosine, invalid);
    if (invalid) {
        snprintf(error, error_size,
                 "Sage hidden layer %u contains non-finite values",
                 completed);
        return 0;
    }
    return 1;
}

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
    h3_vdn_forward_timing forward_timing;
    h3_vdn_forward_timing baseline_forward_timing;
    h3_vdn_denoise_timing denoise_timing;
    memset(&model, 0, sizeof(model));
    memset(&prompt, 0, sizeof(prompt));
    memset(&layout, 0, sizeof(layout));
    memset(&velocity, 0, sizeof(velocity));
    memset(&forward_timing, 0, sizeof(forward_timing));
    memset(&baseline_forward_timing, 0, sizeof(baseline_forward_timing));
    memset(&denoise_timing, 0, sizeof(denoise_timing));
    h3_gpu_tensor *refined = NULL, *video = NULL, *audio = NULL;
    float *video_host = NULL, *audio_host = NULL;
    float *video_output = NULL, *audio_output = NULL;
    float *video_compare = NULL, *audio_compare = NULL;
    layer_comparison layers;
    memset(&layers, 0, sizeof(layers));
    uint32_t frames = DEFAULT_FRAMES;
    uint32_t latent_h = DEFAULT_LATENT_H;
    uint32_t latent_w = DEFAULT_LATENT_W;
    uint32_t audio_latents = DEFAULT_AUDIO_LATENTS;
    int denoise = getenv("VDN_SMOKE_DENOISE") != NULL;
    int compare_sdpa = !denoise &&
        getenv("VDN_SMOKE_COMPARE_SDPA") != NULL;
    int compare_sage = !denoise &&
        getenv("VDN_SMOKE_COMPARE_SAGE") != NULL;
    int compare_sage_layers = compare_sage &&
        getenv("VDN_SMOKE_SAGE_LAYER_ERRORS") != NULL;
    if (!env_u32("VDN_SMOKE_FRAMES", frames, &frames,
                 error, sizeof(error)) ||
        !env_u32("VDN_SMOKE_LATENT_H", latent_h, &latent_h,
                 error, sizeof(error)) ||
        !env_u32("VDN_SMOKE_LATENT_W", latent_w, &latent_w,
                 error, sizeof(error)) ||
        !env_u32("VDN_SMOKE_AUDIO_LATENTS", audio_latents, &audio_latents,
                 error, sizeof(error))) goto failed;
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
            &prompt, frames, latent_h, latent_w, audio_latents,
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
    if (compare_sage) {
        setenv("H3_VDN_SCALAR_SDPA", "0", 1);
        setenv("H3_VDN_SDPA", "wave32", 1);
    }
    if (ok && denoise)
        ok = h3_vdn_denoise(
            gpu, store, &model, refined, &layout, video, audio, 8, 1, 5,
            layer_progress, nfe_progress, NULL, &denoise_timing,
            error, sizeof(error)) &&
             h3_gpu_tensor_read_f32(video, video_output, video_elements) &&
             h3_gpu_tensor_read_f32(audio, audio_output, audio_elements);
    else if (ok) {
        layers.elements = (size_t)layout.sequence * HIDDEN;
        layers.capture = 1;
        if (compare_sage_layers) {
            layers.reference = malloc(
                layers.elements * sizeof(*layers.reference));
            layers.candidate = malloc(
                layers.elements * sizeof(*layers.candidate));
            ok = layers.reference && layers.candidate;
        }
        if (ok && compare_sage_layers)
            ok = h3_vdn_forward_observed(
                gpu, store, &model, refined, &layout, video, audio,
                0.125f, 0.375f, 1, 5, layer_progress, NULL,
                observe_layer, &layers, &velocity, &forward_timing,
                error, sizeof(error));
        else if (ok)
            ok = h3_vdn_forward(
                gpu, store, &model, refined, &layout, video, audio,
                0.125f, 0.375f, 1, 5, layer_progress, NULL,
                &velocity, &forward_timing, error, sizeof(error));
        ok = ok &&
             h3_gpu_tensor_read_f32(velocity.video, video_output,
                                    video_elements) &&
             h3_gpu_tensor_read_f32(velocity.audio, audio_output,
                                    audio_elements);
    }
    if (!ok) {
        if (!error[0]) snprintf(error, sizeof(error),
                                "cannot read VDN forward outputs: %s",
                                h3_gpu_error(gpu));
        goto failed;
    }
    if (compare_sdpa || compare_sage) {
        baseline_forward_timing = forward_timing;
        video_compare = malloc(video_elements * sizeof(*video_compare));
        audio_compare = malloc(audio_elements * sizeof(*audio_compare));
        h3_vdn_velocity_free(&velocity);
        setenv("H3_VDN_SCALAR_SDPA", "0", 1);
        if (compare_sage)
            setenv("H3_VDN_SDPA", "sage-i8-bf16", 1);
        ok = video_compare && audio_compare &&
             h3_gpu_tensor_write_f32(video, video_host, video_elements) &&
             h3_gpu_tensor_write_f32(audio, audio_host, audio_elements);
        layers.capture = 0;
        if (ok && compare_sage_layers)
            ok = h3_vdn_forward_observed(
                gpu, store, &model, refined, &layout, video, audio,
                0.125f, 0.375f, 1, 5, layer_progress, NULL,
                observe_layer, &layers, &velocity, &forward_timing,
                error, sizeof(error));
        else if (ok)
            ok = h3_vdn_forward(
                gpu, store, &model, refined, &layout, video, audio,
                0.125f, 0.375f, 1, 5, layer_progress, NULL,
                &velocity, &forward_timing, error, sizeof(error));
        ok = ok &&
             h3_gpu_tensor_read_f32(velocity.video, video_compare,
                                    video_elements) &&
             h3_gpu_tensor_read_f32(velocity.audio, audio_compare,
                                    audio_elements);
        if (!ok) goto failed;
        double squared = 0.0, reference_squared = 0.0;
        double candidate_squared = 0.0, dot_product = 0.0;
        float maximum = 0.0f;
        size_t candidate_invalid = 0;
        size_t compared = video_elements + audio_elements;
        for (size_t index = 0; index < video_elements; index++) {
            float difference = fabsf(video_output[index] - video_compare[index]);
            if (difference > maximum) maximum = difference;
            squared += (double)difference * difference;
            reference_squared +=
                (double)video_output[index] * video_output[index];
            candidate_squared +=
                (double)video_compare[index] * video_compare[index];
            dot_product +=
                (double)video_output[index] * video_compare[index];
            candidate_invalid += !isfinite(video_compare[index]);
        }
        for (size_t index = 0; index < audio_elements; index++) {
            float difference = fabsf(audio_output[index] - audio_compare[index]);
            if (difference > maximum) maximum = difference;
            squared += (double)difference * difference;
            reference_squared +=
                (double)audio_output[index] * audio_output[index];
            candidate_squared +=
                (double)audio_compare[index] * audio_compare[index];
            dot_product +=
                (double)audio_output[index] * audio_compare[index];
            candidate_invalid += !isfinite(audio_compare[index]);
        }
        double rmse = sqrt(squared / (double)compared);
        double relative_rmse = reference_squared > 0.0 ?
            sqrt(squared / reference_squared) : 0.0;
        double cosine = reference_squared > 0.0 && candidate_squared > 0.0 ?
            dot_product / sqrt(reference_squared * candidate_squared) : 0.0;
        printf("VDN SDPA %s comparison: max_abs=%.9g "
               "rmse=%.9g relative_rmse=%.9g cosine=%.12g "
               "invalid=%zu candidate_video=%016llx "
               "candidate_audio=%016llx baseline_wall=%.6f "
               "candidate_wall=%.6f\n",
               compare_sage ? "wave32/sage-i8-bf16" : "scalar/wave32",
               maximum, rmse, relative_rmse, cosine, candidate_invalid,
               (unsigned long long)hash_f32(video_compare, video_elements),
               (unsigned long long)hash_f32(audio_compare, audio_elements),
               baseline_forward_timing.total_seconds,
               forward_timing.total_seconds);
        if (candidate_invalid) {
            snprintf(error, sizeof(error),
                     "Sage VDN SDPA output contains non-finite values");
            goto failed;
        }
        if (!compare_sage &&
            (memcmp(video_output, video_compare,
                   video_elements * sizeof(*video_output)) ||
            memcmp(audio_output, audio_compare,
                   audio_elements * sizeof(*audio_output)))) {
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
    const char *profile_value = getenv("H3_PROFILE");
    if (profile_value && *profile_value && strcmp(profile_value, "0")) {
        if (denoise) {
            for (unsigned index = 0; index < denoise_timing.count; index++) {
                const h3_vdn_nfe_timing *entry =
                    &denoise_timing.entries[index];
                printf("VDN NFE timing[%u]: wall=%.6f forward=%.6f "
                       "blocks=%.6f scheduler=%.9f euler=%.6f "
                       "read=%llu h2d=%llu sdpa=%.6f\n",
                       index, entry->wall_seconds,
                       entry->forward.total_seconds,
                       entry->forward.blocks_seconds,
                       entry->scheduler_seconds, entry->euler_seconds,
                       (unsigned long long)entry->profile.weight_read_bytes,
                       (unsigned long long)entry->profile.weight_upload_bytes,
                       entry->profile.sdpa_seconds);
            }
        } else {
            printf("VDN forward timing: total=%.6f prepare=%.6f "
                   "input=%.6f timestep=%.6f blocks=%.6f "
                   "output=%.6f cleanup=%.6f\n",
                   forward_timing.total_seconds,
                   forward_timing.prepare_seconds,
                   forward_timing.input_projection_seconds,
                   forward_timing.timestep_seconds,
                   forward_timing.blocks_seconds,
                   forward_timing.output_head_seconds,
                   forward_timing.cleanup_seconds);
        }
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
    for (size_t index = 0; index < BLOCKS; index++)
        h3_gpu_tensor_free(layers.snapshots[index]);
    free(layers.candidate); free(layers.reference);
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
