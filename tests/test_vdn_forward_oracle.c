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
    BLOCKS = 50,
    SEQUENCE = 840,
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
    h3_st_header *header;
    uint16_t *actual;
    uint16_t *expected;
    size_t elements;
    double worst_relative_rmse;
    double lowest_cosine;
    unsigned worst_layer;
    size_t nonfinite;
} forward_comparison;

typedef struct {
    double max_abs;
    double relative_rmse;
    double cosine;
    size_t changed;
    size_t nonfinite;
} metrics;

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

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

static metrics bf16_metrics(const uint16_t *actual,
                            const uint16_t *expected, size_t elements) {
    metrics result = {0};
    double squared_error = 0.0, squared_reference = 0.0;
    double squared_actual = 0.0, dot = 0.0;
    for (size_t index = 0; index < elements; index++) {
        double a = bf16_to_f32(actual[index]);
        double b = bf16_to_f32(expected[index]);
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

static metrics f32_metrics(const float *actual,
                           const float *expected, size_t elements) {
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

static int observe_layer(h3_gpu *gpu, unsigned completed, unsigned total,
                         const h3_gpu_tensor *hidden, void *opaque,
                         char *error, size_t error_size) {
    (void)gpu;
    forward_comparison *comparison = opaque;
    if (!comparison || completed < 1 || completed > BLOCKS ||
        total != BLOCKS) {
        snprintf(error, error_size, "invalid forward oracle callback");
        return 0;
    }
    char name[64];
    snprintf(name, sizeof(name), "block_%02u.output", completed - 1);
    const h3_st_tensor *tensor = h3_st_find(comparison->header, name);
    if (!tensor || tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != 3 ||
        tensor->shape[0] != 1 || tensor->shape[1] != SEQUENCE ||
        tensor->shape[2] != HIDDEN ||
        !h3_st_read_data(comparison->header, tensor, comparison->expected,
                         comparison->elements * sizeof(*comparison->expected),
                         error, error_size) ||
        !h3_gpu_tensor_read_bf16(hidden, comparison->actual,
                                 comparison->elements)) {
        if (!error[0])
            snprintf(error, error_size, "cannot read forward layer %u", completed);
        return 0;
    }
    metrics value = bf16_metrics(
        comparison->actual, comparison->expected, comparison->elements);
    printf("OpenVDN forward block %02u: max_abs=%.9g relative_rmse=%.9g "
           "cosine=%.12g changed=%zu/%zu nonfinite=%zu\n",
           completed - 1, value.max_abs, value.relative_rmse, value.cosine,
           value.changed, comparison->elements, value.nonfinite);
    if (value.relative_rmse > comparison->worst_relative_rmse) {
        comparison->worst_relative_rmse = value.relative_rmse;
        comparison->worst_layer = completed - 1;
    }
    if (value.cosine < comparison->lowest_cosine)
        comparison->lowest_cosine = value.cosine;
    comparison->nonfinite += value.nonfinite;
    return value.nonfinite == 0;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s H3_BASE STAGE_DMD PROMPT "
                "BLOCK0_ORACLE FORWARD_ORACLE\n", argv[0]);
        return 2;
    }
    int status = 1;
    char error[512] = {0};
    h3_st_header input_header, forward_header;
    h3_vdn_model_weights model;
    h3_text_embedding prompt;
    h3_vdn_layout layout;
    h3_vdn_velocity velocity;
    memset(&input_header, 0, sizeof(input_header));
    memset(&forward_header, 0, sizeof(forward_header));
    memset(&model, 0, sizeof(model));
    memset(&prompt, 0, sizeof(prompt));
    memset(&layout, 0, sizeof(layout));
    memset(&velocity, 0, sizeof(velocity));
    h3_gpu *gpu = NULL;
    h3_vdn_weight_store *store = NULL;
    h3_gpu_tensor *refined = NULL, *video = NULL, *audio = NULL;
    uint16_t *refined_host = NULL;
    float *video_host = NULL, *audio_host = NULL;
    float *video_actual = NULL, *audio_actual = NULL;
    float *video_expected = NULL, *audio_expected = NULL;
    forward_comparison comparison;
    memset(&comparison, 0, sizeof(comparison));
    comparison.lowest_cosine = 1.0;

    if (!h3_st_read_header(argv[4], &input_header, error, sizeof(error)) ||
        !h3_st_read_header(argv[5], &forward_header, error, sizeof(error)))
        goto failed;
    const uint64_t prompt_shape[] = {1, TEXT_ROWS, HIDDEN};
    const uint64_t video_shape[] = {VIDEO_ROWS, VIDEO_PATCH};
    const uint64_t audio_shape[] = {AUDIO_ROWS, AUDIO_WIDTH};
    const uint64_t video_output_shape[] = {1, VIDEO_ROWS, VIDEO_PATCH};
    const uint64_t audio_output_shape[] = {1, AUDIO_ROWS, AUDIO_WIDTH};
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
    video_expected = read_tensor(
        &forward_header, "final.video_velocity", H3_DTYPE_F32, 3,
        video_output_shape, &bytes, error, sizeof(error));
    audio_expected = read_tensor(
        &forward_header, "final.audio_velocity", H3_DTYPE_F32, 3,
        audio_output_shape, &bytes, error, sizeof(error));
    if (!refined_host || !video_host || !audio_host || !video_expected ||
        !audio_expected)
        goto failed;

    comparison.header = &forward_header;
    comparison.elements = (size_t)SEQUENCE * HIDDEN;
    comparison.actual = malloc(
        comparison.elements * sizeof(*comparison.actual));
    comparison.expected = malloc(
        comparison.elements * sizeof(*comparison.expected));
    video_actual = malloc(
        (size_t)VIDEO_ROWS * VIDEO_PATCH * sizeof(*video_actual));
    audio_actual = malloc(
        (size_t)AUDIO_ROWS * AUDIO_WIDTH * sizeof(*audio_actual));
    if (!comparison.actual || !comparison.expected || !video_actual ||
        !audio_actual) {
        snprintf(error, sizeof(error), "out of host memory for forward oracle");
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
    if (layout.sequence != SEQUENCE || layout.text_rows != TEXT_ROWS ||
        layout.audio_rows != AUDIO_ROWS || layout.video_rows != VIDEO_ROWS) {
        snprintf(error, sizeof(error), "unexpected forward oracle layout");
        goto failed;
    }
    refined = h3_gpu_tensor_from_bf16(
        gpu, refined_host, (size_t)TEXT_ROWS * HIDDEN);
    video = h3_gpu_tensor_from_f32(
        gpu, video_host, (size_t)VIDEO_ROWS * VIDEO_PATCH);
    audio = h3_gpu_tensor_from_f32(
        gpu, audio_host, (size_t)AUDIO_ROWS * AUDIO_WIDTH);
    if (!refined || !video || !audio ||
        setenv("H3_VDN_SDPA", "scalar", 1) != 0 ||
        !h3_vdn_forward_observed(
            gpu, store, &model, refined, &layout, video, audio,
            0.0f, 0.0f, 1, 5, NULL, NULL,
            observe_layer, &comparison, &velocity, NULL,
            error, sizeof(error)) ||
        !h3_gpu_tensor_read_f32(
            velocity.video, video_actual, (size_t)VIDEO_ROWS * VIDEO_PATCH) ||
        !h3_gpu_tensor_read_f32(
            velocity.audio, audio_actual, (size_t)AUDIO_ROWS * AUDIO_WIDTH)) {
        if (!error[0])
            snprintf(error, sizeof(error), "cannot execute forward oracle: %s",
                     h3_gpu_error(gpu));
        goto failed;
    }

    metrics video_value = f32_metrics(
        video_actual, video_expected, (size_t)VIDEO_ROWS * VIDEO_PATCH);
    metrics audio_value = f32_metrics(
        audio_actual, audio_expected, (size_t)AUDIO_ROWS * AUDIO_WIDTH);
    printf("OpenVDN final video velocity: max_abs=%.9g relative_rmse=%.9g "
           "cosine=%.12g changed=%zu nonfinite=%zu\n",
           video_value.max_abs, video_value.relative_rmse,
           video_value.cosine, video_value.changed, video_value.nonfinite);
    printf("OpenVDN final audio velocity: max_abs=%.9g relative_rmse=%.9g "
           "cosine=%.12g changed=%zu nonfinite=%zu\n",
           audio_value.max_abs, audio_value.relative_rmse,
           audio_value.cosine, audio_value.changed, audio_value.nonfinite);

    /* The two correct upstream MATH/native SDPA paths diverge by at most
     * 0.08450 relative RMSE / 0.99691 cosine in the block stack and by
     * 0.02473 video / 0.05698 audio relative RMSE at the heads.  These gates
     * are fixed before observing the C candidate and retain independent
     * margin for scalar accumulation order. */
    if (comparison.nonfinite || comparison.worst_relative_rmse > 0.15 ||
        comparison.lowest_cosine < 0.99 || video_value.nonfinite ||
        audio_value.nonfinite || video_value.relative_rmse > 0.05 ||
        audio_value.relative_rmse > 0.12 || video_value.cosine < 0.995 ||
        audio_value.cosine < 0.99) {
        snprintf(error, sizeof(error),
                 "50-layer forward exceeds upstream parity bounds "
                 "(worst block=%u rel=%.9g cosine=%.12g)",
                 comparison.worst_layer, comparison.worst_relative_rmse,
                 comparison.lowest_cosine);
        goto failed;
    }
    printf("PASS: streamed 50-layer and final velocity parity "
           "(worst block=%u relative_rmse=%.9g lowest_cosine=%.12g)\n",
           comparison.worst_layer, comparison.worst_relative_rmse,
           comparison.lowest_cosine);
    status = 0;
    goto cleanup;

failed:
    fprintf(stderr, "VDN forward oracle failed: %s\n", error);
cleanup:
    h3_vdn_velocity_free(&velocity);
    h3_gpu_tensor_free(audio);
    h3_gpu_tensor_free(video);
    h3_gpu_tensor_free(refined);
    h3_vdn_layout_free(&layout);
    h3_vdn_model_weights_free(&model);
    h3_vdn_prompt_free(&prompt);
    h3_vdn_weight_store_free(store);
    h3_gpu_free(gpu);
    free(comparison.expected);
    free(comparison.actual);
    free(audio_actual);
    free(video_actual);
    free(audio_expected);
    free(video_expected);
    free(audio_host);
    free(video_host);
    free(refined_host);
    h3_st_free_header(&forward_header);
    h3_st_free_header(&input_header);
    return status;
}
