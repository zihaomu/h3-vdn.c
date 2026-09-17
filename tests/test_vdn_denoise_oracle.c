#include "h3_gpu.h"
#include "h3_safetensors.h"
#include "h3_vdn_dit.h"
#include "h3_vdn_prompt.h"
#include "h3_vdn_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    NFE = 8,
    TEXT_ROWS = 800,
    AUDIO_ROWS = 6,
    VIDEO_ROWS = 34,
    HIDDEN = 5376,
    VIDEO_PATCH = 96,
    AUDIO_WIDTH = 32,
    FRAMES = 17,
    LATENT_HEIGHT = 2,
    LATENT_WIDTH = 4,
    AUDIO_LATENTS = 3
};

typedef struct {
    double max_abs;
    double relative_rmse;
    double cosine;
    size_t changed;
    size_t nonfinite;
} metrics;

typedef struct {
    h3_st_header *header;
    float *video_actual;
    float *video_expected;
    float *audio_actual;
    float *audio_expected;
    double worst_video_relative;
    double worst_audio_relative;
    double lowest_video_cosine;
    double lowest_audio_cosine;
    unsigned worst_video_nfe;
    unsigned worst_audio_nfe;
    size_t nonfinite;
} denoise_comparison;

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

static int observe_nfe(h3_gpu *gpu, unsigned completed, unsigned total,
                       const h3_gpu_tensor *video_rows,
                       const h3_gpu_tensor *audio_rows, void *opaque,
                       char *error, size_t error_size) {
    denoise_comparison *comparison = opaque;
    const size_t video_elements = (size_t)VIDEO_ROWS * VIDEO_PATCH;
    const size_t audio_elements = (size_t)AUDIO_ROWS * AUDIO_WIDTH;
    if (!comparison || completed < 1 || completed > NFE || total != NFE) {
        snprintf(error, error_size, "invalid denoise oracle callback");
        return 0;
    }
    char video_name[64], audio_name[64];
    snprintf(video_name, sizeof(video_name),
             "nfe_%u.video_rows", completed - 1);
    snprintf(audio_name, sizeof(audio_name),
             "nfe_%u.audio_rows", completed - 1);
    const h3_st_tensor *video_tensor = h3_st_find(
        comparison->header, video_name);
    const h3_st_tensor *audio_tensor = h3_st_find(
        comparison->header, audio_name);
    if (!video_tensor || video_tensor->dtype != H3_DTYPE_F32 ||
        video_tensor->ndim != 2 || video_tensor->shape[0] != VIDEO_ROWS ||
        video_tensor->shape[1] != VIDEO_PATCH || !audio_tensor ||
        audio_tensor->dtype != H3_DTYPE_F32 || audio_tensor->ndim != 2 ||
        audio_tensor->shape[0] != AUDIO_ROWS ||
        audio_tensor->shape[1] != AUDIO_WIDTH ||
        !h3_st_read_data(
            comparison->header, video_tensor, comparison->video_expected,
            video_elements * sizeof(*comparison->video_expected),
            error, error_size) ||
        !h3_st_read_data(
            comparison->header, audio_tensor, comparison->audio_expected,
            audio_elements * sizeof(*comparison->audio_expected),
            error, error_size) ||
        !h3_gpu_tensor_read_f32(
            video_rows, comparison->video_actual, video_elements) ||
        !h3_gpu_tensor_read_f32(
            audio_rows, comparison->audio_actual, audio_elements)) {
        if (!error[0])
            snprintf(error, error_size, "cannot read NFE %u rows: %s",
                     completed - 1, h3_gpu_error(gpu));
        return 0;
    }
    metrics video = calculate_metrics(
        comparison->video_actual, comparison->video_expected, video_elements);
    metrics audio = calculate_metrics(
        comparison->audio_actual, comparison->audio_expected, audio_elements);
    printf("OpenVDN denoise NFE %u video: max_abs=%.9g relative_rmse=%.9g "
           "cosine=%.12g changed=%zu nonfinite=%zu\n",
           completed - 1, video.max_abs, video.relative_rmse, video.cosine,
           video.changed, video.nonfinite);
    printf("OpenVDN denoise NFE %u audio: max_abs=%.9g relative_rmse=%.9g "
           "cosine=%.12g changed=%zu nonfinite=%zu\n",
           completed - 1, audio.max_abs, audio.relative_rmse, audio.cosine,
           audio.changed, audio.nonfinite);
    if (video.relative_rmse > comparison->worst_video_relative) {
        comparison->worst_video_relative = video.relative_rmse;
        comparison->worst_video_nfe = completed - 1;
    }
    if (audio.relative_rmse > comparison->worst_audio_relative) {
        comparison->worst_audio_relative = audio.relative_rmse;
        comparison->worst_audio_nfe = completed - 1;
    }
    if (video.cosine < comparison->lowest_video_cosine)
        comparison->lowest_video_cosine = video.cosine;
    if (audio.cosine < comparison->lowest_audio_cosine)
        comparison->lowest_audio_cosine = audio.cosine;
    comparison->nonfinite += video.nonfinite + audio.nonfinite;
    return !video.nonfinite && !audio.nonfinite;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s H3_BASE STAGE_DMD PROMPT "
                "BLOCK0_ORACLE DENOISE_ORACLE\n", argv[0]);
        return 2;
    }
    int status = 1;
    char error[512] = {0};
    h3_st_header input_header, denoise_header;
    h3_vdn_model_weights model;
    h3_text_embedding prompt;
    h3_vdn_layout layout;
    h3_vdn_denoise_timing timing;
    memset(&input_header, 0, sizeof(input_header));
    memset(&denoise_header, 0, sizeof(denoise_header));
    memset(&model, 0, sizeof(model));
    memset(&prompt, 0, sizeof(prompt));
    memset(&layout, 0, sizeof(layout));
    memset(&timing, 0, sizeof(timing));
    h3_gpu *gpu = NULL;
    h3_vdn_weight_store *store = NULL;
    h3_gpu_tensor *refined = NULL, *video = NULL, *audio = NULL;
    uint16_t *refined_host = NULL;
    float *video_host = NULL, *audio_host = NULL;
    denoise_comparison comparison;
    memset(&comparison, 0, sizeof(comparison));
    comparison.lowest_video_cosine = 1.0;
    comparison.lowest_audio_cosine = 1.0;
    const size_t video_elements = (size_t)VIDEO_ROWS * VIDEO_PATCH;
    const size_t audio_elements = (size_t)AUDIO_ROWS * AUDIO_WIDTH;

    if (!h3_st_read_header(argv[4], &input_header, error, sizeof(error)) ||
        !h3_st_read_header(argv[5], &denoise_header, error, sizeof(error)))
        goto failed;
    const uint64_t prompt_shape[] = {1, TEXT_ROWS, HIDDEN};
    const uint64_t video_shape[] = {VIDEO_ROWS, VIDEO_PATCH};
    const uint64_t audio_shape[] = {AUDIO_ROWS, AUDIO_WIDTH};
    size_t bytes = 0;
    refined_host = read_tensor(
        &input_header, "input.refined_prompt", H3_DTYPE_BF16, 3,
        prompt_shape, &bytes, error, sizeof(error));
    video_host = read_tensor(
        &input_header, "input.video_rows", H3_DTYPE_F32, 2,
        video_shape, &bytes, error, sizeof(error));
    audio_host = read_tensor(
        &input_header, "input.audio_rows", H3_DTYPE_F32, 2,
        audio_shape, &bytes, error, sizeof(error));
    if (!refined_host || !video_host || !audio_host) goto failed;
    comparison.header = &denoise_header;
    comparison.video_actual = malloc(
        video_elements * sizeof(*comparison.video_actual));
    comparison.video_expected = malloc(
        video_elements * sizeof(*comparison.video_expected));
    comparison.audio_actual = malloc(
        audio_elements * sizeof(*comparison.audio_actual));
    comparison.audio_expected = malloc(
        audio_elements * sizeof(*comparison.audio_expected));
    if (!comparison.video_actual || !comparison.video_expected ||
        !comparison.audio_actual || !comparison.audio_expected) {
        snprintf(error, sizeof(error), "out of host memory for denoise oracle");
        goto failed;
    }

    gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) goto failed;
    store = h3_vdn_weight_store_open(argv[1], argv[2], 1,
                                     error, sizeof(error));
    if (!store || !h3_vdn_prompt_load(argv[3], &prompt,
                                      error, sizeof(error)) ||
        !h3_vdn_model_weights_load(store, gpu, &model,
                                   error, sizeof(error)) ||
        !h3_vdn_layout_build(
            &prompt, FRAMES, LATENT_HEIGHT, LATENT_WIDTH, AUDIO_LATENTS,
            &layout, error, sizeof(error)))
        goto failed;
    refined = h3_gpu_tensor_from_bf16(
        gpu, refined_host, (size_t)TEXT_ROWS * HIDDEN);
    video = h3_gpu_tensor_from_f32(gpu, video_host, video_elements);
    audio = h3_gpu_tensor_from_f32(gpu, audio_host, audio_elements);
    if (!refined || !video || !audio ||
        setenv("H3_VDN_SDPA", "scalar", 1) != 0 ||
        !h3_vdn_denoise_observed(
            gpu, store, &model, refined, &layout, video, audio,
            NFE, 1, 5, NULL, NULL, NULL, observe_nfe, &comparison,
            &timing, error, sizeof(error)))
        goto failed;

    /* The independent upstream MATH/native eight-NFE envelope reaches
     * relative RMSE 0.03409 / 0.01505 for final video/audio rows, with
     * cosine 0.999419 / 0.999887. These gates were fixed before executing
     * the C candidate and retain margin for scalar accumulation order. */
    if (comparison.nonfinite || comparison.worst_video_relative > 0.08 ||
        comparison.worst_audio_relative > 0.05 ||
        comparison.lowest_video_cosine < 0.995 ||
        comparison.lowest_audio_cosine < 0.995) {
        snprintf(error, sizeof(error),
                 "eight-NFE latent parity exceeds upstream bounds");
        goto failed;
    }
    printf("PASS: eight-NFE latent parity (video worst NFE=%u rel=%.9g "
           "cosine=%.12g; audio worst NFE=%u rel=%.9g cosine=%.12g)\n",
           comparison.worst_video_nfe, comparison.worst_video_relative,
           comparison.lowest_video_cosine, comparison.worst_audio_nfe,
           comparison.worst_audio_relative, comparison.lowest_audio_cosine);
    status = 0;
    goto cleanup;

failed:
    fprintf(stderr, "VDN denoise oracle failed: %s\n", error);
cleanup:
    h3_gpu_tensor_free(audio);
    h3_gpu_tensor_free(video);
    h3_gpu_tensor_free(refined);
    h3_vdn_layout_free(&layout);
    h3_vdn_model_weights_free(&model);
    h3_vdn_prompt_free(&prompt);
    h3_vdn_weight_store_free(store);
    h3_gpu_free(gpu);
    free(comparison.audio_expected);
    free(comparison.audio_actual);
    free(comparison.video_expected);
    free(comparison.video_actual);
    free(audio_host);
    free(video_host);
    free(refined_host);
    h3_st_free_header(&denoise_header);
    h3_st_free_header(&input_header);
    return status;
}
