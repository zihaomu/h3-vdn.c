#include "h3_gpu.h"
#include "h3_safetensors.h"
#include "h3_vdn_dit.h"
#include "h3_vdn_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    SEQUENCE = 840,
    TEXT_ROWS = 800,
    VIDEO_START = 806,
    AUDIO_ROWS = 6,
    VIDEO_ROWS = 34,
    FRAMES = 17,
    FRAME_HEIGHT = 1,
    FRAME_WIDTH = 2,
    HIDDEN = 5376,
    TIME_WIDTH = 2688,
    VIDEO_PATCH = 96,
    AUDIO_WIDTH = 32,
    ROPE_HALF = 48,
    MODALITIES = 3,
    MODULATION_SLOTS = 6
};

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int compare_bf16(const char *label, const uint16_t *actual,
                        const uint16_t *expected, size_t elements,
                        double relative_bound, double cosine_bound,
                        char *error, size_t error_size) {
    double squared_error = 0.0, squared_reference = 0.0;
    double squared_actual = 0.0, dot = 0.0, max_abs = 0.0;
    size_t nonfinite = 0, changed = 0;
    for (size_t index = 0; index < elements; index++) {
        double a = bf16_to_f32(actual[index]);
        double b = bf16_to_f32(expected[index]);
        if (!isfinite(a) || !isfinite(b)) {
            nonfinite++;
            continue;
        }
        double difference = a - b;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
        squared_error += difference * difference;
        squared_reference += b * b;
        squared_actual += a * a;
        dot += a * b;
        changed += actual[index] != expected[index];
    }
    double relative_rmse = squared_reference > 0.0 ?
        sqrt(squared_error / squared_reference) : sqrt(squared_error);
    double norm_product = squared_actual * squared_reference;
    double cosine = norm_product > 0.0 ? dot / sqrt(norm_product) :
        (squared_error == 0.0 ? 1.0 : 0.0);
    printf("%s: max_abs=%.9g relative_rmse=%.9g cosine=%.12g "
           "changed=%zu/%zu nonfinite=%zu\n",
           label, max_abs, relative_rmse, cosine, changed, elements,
           nonfinite);
    if (nonfinite || relative_rmse > relative_bound ||
        cosine < cosine_bound) {
        snprintf(error, error_size, "%s exceeds upstream parity bounds", label);
        return 0;
    }
    return 1;
}

static int compare_f32(const char *label, const float *actual,
                       const float *expected, size_t elements,
                       double relative_bound, double cosine_bound,
                       char *error, size_t error_size) {
    double squared_error = 0.0, squared_reference = 0.0;
    double squared_actual = 0.0, dot = 0.0, max_abs = 0.0;
    size_t nonfinite = 0, changed = 0;
    for (size_t index = 0; index < elements; index++) {
        double a = actual[index];
        double b = expected[index];
        if (!isfinite(a) || !isfinite(b)) {
            nonfinite++;
            continue;
        }
        double difference = a - b;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
        squared_error += difference * difference;
        squared_reference += b * b;
        squared_actual += a * a;
        dot += a * b;
        changed += actual[index] != expected[index];
    }
    double relative_rmse = squared_reference > 0.0 ?
        sqrt(squared_error / squared_reference) : sqrt(squared_error);
    double norm_product = squared_actual * squared_reference;
    double cosine = norm_product > 0.0 ? dot / sqrt(norm_product) :
        (squared_error == 0.0 ? 1.0 : 0.0);
    printf("%s: max_abs=%.9g relative_rmse=%.9g cosine=%.12g "
           "changed=%zu/%zu nonfinite=%zu\n",
           label, max_abs, relative_rmse, cosine, changed, elements,
           nonfinite);
    if (nonfinite || relative_rmse > relative_bound ||
        cosine < cosine_bound) {
        snprintf(error, error_size, "%s exceeds upstream parity bounds", label);
        return 0;
    }
    return 1;
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

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s H3_BASE STAGE_DMD BLOCK0_ORACLE\n", argv[0]);
        return 2;
    }
    int status = 1;
    char error[512] = {0};
    h3_st_header header;
    h3_vdn_model_weights model;
    h3_vdn_block_weights block;
    memset(&header, 0, sizeof(header));
    memset(&model, 0, sizeof(model));
    memset(&block, 0, sizeof(block));
    h3_gpu *gpu = NULL;
    h3_vdn_weight_store *store = NULL;
    h3_gpu_tensor *hidden = NULL, *time_embedding = NULL;
    h3_gpu_tensor *modulation = NULL, *row_map = NULL;
    h3_gpu_tensor *rope_cos = NULL, *rope_sin = NULL;
    h3_gpu_tensor *video_rows = NULL, *audio_rows = NULL;
    h3_gpu_tensor *refined_prompt = NULL;
    uint16_t *input = NULL, *expected = NULL, *actual = NULL;
    uint16_t *refined_prompt_host = NULL;
    uint16_t *modulation_host = NULL, *rope_cos_host = NULL, *rope_sin_host = NULL;
    uint16_t *modulation_actual = NULL;
    float *video_rows_host = NULL, *audio_rows_host = NULL;
    float *time_values_host = NULL, *time_embedding_host = NULL;
    float *time_embedding_actual = NULL;
    int64_t *map_i64 = NULL;
    uint32_t *map_u32 = NULL;
    uint16_t *chunks[MODULATION_SLOTS] = {0};

    if (!h3_st_read_header(argv[3], &header, error, sizeof(error))) goto failed;
    const uint64_t hidden_shape[] = {1, SEQUENCE, HIDDEN};
    const uint64_t map_shape[] = {SEQUENCE};
    const uint64_t rope_shape[] = {SEQUENCE, ROPE_HALF};
    const uint64_t chunk_shape[] = {MODALITIES, HIDDEN};
    const uint64_t time_shape[] = {1, TIME_WIDTH};
    const uint64_t timestep_shape[] = {1};
    const uint64_t video_shape[] = {VIDEO_ROWS, VIDEO_PATCH};
    const uint64_t audio_shape[] = {AUDIO_ROWS, AUDIO_WIDTH};
    const uint64_t prompt_shape[] = {1, TEXT_ROWS, HIDDEN};
    size_t bytes = 0;
    input = read_tensor(&header, "input.packed_hidden", H3_DTYPE_BF16,
                        3, hidden_shape, &bytes, error, sizeof(error));
    expected = read_tensor(&header, "block_0.output", H3_DTYPE_BF16,
                           3, hidden_shape, &bytes, error, sizeof(error));
    map_i64 = read_tensor(&header, "input.adaln_indices", H3_DTYPE_I64,
                          1, map_shape, &bytes, error, sizeof(error));
    rope_cos_host = read_tensor(&header, "input.rope_cos_half_bf16",
                                H3_DTYPE_BF16, 2, rope_shape, &bytes,
                                error, sizeof(error));
    rope_sin_host = read_tensor(&header, "input.rope_sin_half_bf16",
                                H3_DTYPE_BF16, 2, rope_shape, &bytes,
                                error, sizeof(error));
    time_embedding_host = read_tensor(
        &header, "input.time_embedding", H3_DTYPE_F32, 2, time_shape,
        &bytes, error, sizeof(error));
    time_values_host = read_tensor(
        &header, "input.unique_timesteps", H3_DTYPE_F32, 1,
        timestep_shape, &bytes, error, sizeof(error));
    video_rows_host = read_tensor(
        &header, "input.video_rows", H3_DTYPE_F32, 2, video_shape,
        &bytes, error, sizeof(error));
    audio_rows_host = read_tensor(
        &header, "input.audio_rows", H3_DTYPE_F32, 2, audio_shape,
        &bytes, error, sizeof(error));
    refined_prompt_host = read_tensor(
        &header, "input.refined_prompt", H3_DTYPE_BF16, 3, prompt_shape,
        &bytes, error, sizeof(error));
    if (!input || !expected || !map_i64 || !rope_cos_host || !rope_sin_host ||
        !time_embedding_host || !time_values_host || !video_rows_host ||
        !audio_rows_host || !refined_prompt_host)
        goto failed;
    for (unsigned slot = 0; slot < MODULATION_SLOTS; slot++) {
        char name[64];
        snprintf(name, sizeof(name), "block_0.modulation.%u", slot);
        chunks[slot] = read_tensor(&header, name, H3_DTYPE_BF16,
                                   2, chunk_shape, &bytes,
                                   error, sizeof(error));
        if (!chunks[slot]) goto failed;
    }
    size_t hidden_elements = (size_t)SEQUENCE * HIDDEN;
    size_t modulation_elements =
        (size_t)MODALITIES * MODULATION_SLOTS * HIDDEN;
    modulation_host = malloc(modulation_elements * sizeof(*modulation_host));
    modulation_actual = malloc(
        modulation_elements * sizeof(*modulation_actual));
    map_u32 = malloc((size_t)SEQUENCE * sizeof(*map_u32));
    actual = malloc(hidden_elements * sizeof(*actual));
    time_embedding_actual = malloc(
        TIME_WIDTH * sizeof(*time_embedding_actual));
    if (!modulation_host || !modulation_actual || !map_u32 || !actual ||
        !time_embedding_actual) {
        snprintf(error, sizeof(error), "out of host memory");
        goto failed;
    }
    for (unsigned row = 0; row < MODALITIES; row++) {
        for (unsigned slot = 0; slot < MODULATION_SLOTS; slot++) {
            memcpy(modulation_host +
                       ((size_t)row * MODULATION_SLOTS + slot) * HIDDEN,
                   chunks[slot] + (size_t)row * HIDDEN,
                   HIDDEN * sizeof(*modulation_host));
        }
    }
    for (size_t row = 0; row < SEQUENCE; row++) {
        if (map_i64[row] < 0 || map_i64[row] >= MODALITIES) {
            snprintf(error, sizeof(error), "invalid AdaLN map row %zu", row);
            goto failed;
        }
        map_u32[row] = (uint32_t)map_i64[row];
    }

    gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) goto failed;
    store = h3_vdn_weight_store_open(argv[1], argv[2], 1,
                                     error, sizeof(error));
    if (!store || !h3_vdn_model_weights_load(
            store, gpu, &model, error, sizeof(error)) ||
        !h3_vdn_block_weights_load(
            store, gpu, 0, &block, error, sizeof(error))) goto failed;
    hidden = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    video_rows = h3_gpu_tensor_from_f32(
        gpu, video_rows_host, (size_t)VIDEO_ROWS * VIDEO_PATCH);
    audio_rows = h3_gpu_tensor_from_f32(
        gpu, audio_rows_host, (size_t)AUDIO_ROWS * AUDIO_WIDTH);
    refined_prompt = h3_gpu_tensor_from_bf16(
        gpu, refined_prompt_host, (size_t)TEXT_ROWS * HIDDEN);
    if (!hidden || !video_rows || !audio_rows || !refined_prompt ||
        !h3_gpu_begin(gpu) ||
        !h3_gpu_copy_bf16(gpu, hidden, 0, refined_prompt, 0,
                          (size_t)TEXT_ROWS * HIDDEN) ||
        !h3_gpu_patch_linear_bf16_offset(
            gpu, hidden, (size_t)TEXT_ROWS * HIDDEN,
            audio_rows, 0, model.audio_in_weight, model.audio_in_bias,
            AUDIO_ROWS, AUDIO_WIDTH, HIDDEN) ||
        !h3_gpu_patch_linear_bf16_offset(
            gpu, hidden, (size_t)VIDEO_START * HIDDEN,
            video_rows, 0, model.video_in_weight, model.video_in_bias,
            VIDEO_ROWS, VIDEO_PATCH, HIDDEN) ||
        !h3_gpu_submit(gpu) ||
        !h3_gpu_tensor_read_bf16(hidden, actual, hidden_elements)) {
        snprintf(error, sizeof(error), "cannot construct C block input: %s",
                 h3_gpu_error(gpu));
        goto failed;
    }
    if (memcmp(actual, refined_prompt_host,
               (size_t)TEXT_ROWS * HIDDEN * sizeof(*actual)) != 0) {
        snprintf(error, sizeof(error),
                 "C packed text rows differ from refined prompt bytes");
        goto failed;
    }
    if (!compare_bf16(
            "OpenVDN C audio input projection",
            actual + (size_t)TEXT_ROWS * HIDDEN,
            input + (size_t)TEXT_ROWS * HIDDEN,
            (size_t)AUDIO_ROWS * HIDDEN, 0.01, 0.9999,
            error, sizeof(error)) ||
        !compare_bf16(
            "OpenVDN C video input projection",
            actual + (size_t)VIDEO_START * HIDDEN,
            input + (size_t)VIDEO_START * HIDDEN,
            (size_t)VIDEO_ROWS * HIDDEN, 0.01, 0.9999,
            error, sizeof(error)) ||
        !compare_bf16(
            "OpenVDN C packed hidden", actual, input, hidden_elements,
            0.01, 0.9999, error, sizeof(error)))
        goto failed;
    time_embedding = h3_vdn_time_embedding(
        gpu, &model, time_values_host, 1, error, sizeof(error));
    if (!time_embedding || !h3_gpu_tensor_read_f32(
            time_embedding, time_embedding_actual, TIME_WIDTH) ||
        !compare_f32(
            "OpenVDN C time embedding", time_embedding_actual,
            time_embedding_host, TIME_WIDTH, 0.01, 0.9999,
            error, sizeof(error)))
        goto failed;
    modulation = time_embedding ? h3_vdn_block_modulation(
        gpu, &block, time_embedding, 1, error, sizeof(error)) : NULL;
    row_map = h3_gpu_tensor_from_u32(gpu, map_u32, SEQUENCE);
    rope_cos = h3_gpu_tensor_from_bf16(
        gpu, rope_cos_host, (size_t)SEQUENCE * ROPE_HALF);
    rope_sin = h3_gpu_tensor_from_bf16(
        gpu, rope_sin_host, (size_t)SEQUENCE * ROPE_HALF);
    if (!modulation || !row_map ||
        !rope_cos || !rope_sin ||
        !h3_gpu_tensor_read_bf16(
            modulation, modulation_actual, modulation_elements)) {
        snprintf(error, sizeof(error), "cannot allocate block oracle tensors: %s",
                 h3_gpu_error(gpu));
        goto failed;
    }
    {
        double squared_error = 0.0, squared_reference = 0.0;
        double squared_actual = 0.0, dot = 0.0, max_abs = 0.0;
        size_t nonfinite = 0;
        for (size_t index = 0; index < modulation_elements; index++) {
            double a = bf16_to_f32(modulation_actual[index]);
            double b = bf16_to_f32(modulation_host[index]);
            if (!isfinite(a) || !isfinite(b)) {
                nonfinite++;
                continue;
            }
            double difference = a - b;
            if (fabs(difference) > max_abs) max_abs = fabs(difference);
            squared_error += difference * difference;
            squared_reference += b * b;
            squared_actual += a * a;
            dot += a * b;
        }
        double relative_rmse = sqrt(squared_error / squared_reference);
        double cosine = dot / sqrt(squared_actual * squared_reference);
        printf("OpenVDN block 0 modulation: max_abs=%.9g "
               "relative_rmse=%.9g cosine=%.12g nonfinite=%zu\n",
               max_abs, relative_rmse, cosine, nonfinite);
        if (nonfinite || relative_rmse > 0.01 || cosine < 0.9999) {
            snprintf(error, sizeof(error),
                     "block 0 modulation exceeds upstream parity bounds");
            goto failed;
        }
    }
    if (setenv("H3_VDN_SDPA", "scalar", 1) != 0) {
        snprintf(error, sizeof(error), "cannot select scalar SDPA");
        goto failed;
    }
    if (!h3_vdn_run_block(
            gpu, &block, hidden, modulation, row_map, rope_cos, rope_sin,
            SEQUENCE, TEXT_ROWS, VIDEO_START, FRAMES,
            FRAME_HEIGHT, FRAME_WIDTH, 1, 5, error, sizeof(error)) ||
        !h3_gpu_tensor_read_bf16(hidden, actual, hidden_elements)) goto failed;

    double squared_error = 0.0, squared_reference = 0.0;
    double squared_actual = 0.0, dot = 0.0, max_abs = 0.0;
    size_t nonfinite = 0, changed = 0;
    for (size_t index = 0; index < hidden_elements; index++) {
        double a = bf16_to_f32(actual[index]);
        double b = bf16_to_f32(expected[index]);
        if (!isfinite(a) || !isfinite(b)) {
            nonfinite++;
            continue;
        }
        double difference = a - b;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
        squared_error += difference * difference;
        squared_reference += b * b;
        squared_actual += a * a;
        dot += a * b;
        changed += actual[index] != expected[index];
    }
    double relative_rmse = sqrt(squared_error / squared_reference);
    double cosine = dot / sqrt(squared_actual * squared_reference);
    printf("OpenVDN block 0 oracle: max_abs=%.9g relative_rmse=%.9g "
           "cosine=%.12g changed=%zu/%zu nonfinite=%zu\n",
           max_abs, relative_rmse, cosine, changed, hidden_elements, nonfinite);
    /* Two correct upstream SDPA backends differ at this output by relative
     * RMSE 0.001649 and cosine 0.99999864. Keep independent margin for the
     * native scalar attention/rocBLAS ordering without accepting drift. */
    if (nonfinite || relative_rmse > 0.01 || cosine < 0.9999) {
        snprintf(error, sizeof(error),
                 "hybrid block 0 exceeds upstream parity bounds "
                 "(relative_rmse<=0.01, cosine>=0.9999)");
        goto failed;
    }
    status = 0;
    goto cleanup;

failed:
    fprintf(stderr, "VDN block 0 oracle failed: %s\n", error);
cleanup:
    h3_gpu_tensor_free(rope_sin);
    h3_gpu_tensor_free(rope_cos);
    h3_gpu_tensor_free(row_map);
    h3_gpu_tensor_free(modulation);
    h3_gpu_tensor_free(time_embedding);
    h3_gpu_tensor_free(refined_prompt);
    h3_gpu_tensor_free(audio_rows);
    h3_gpu_tensor_free(video_rows);
    h3_gpu_tensor_free(hidden);
    h3_vdn_block_weights_free(&block);
    h3_vdn_model_weights_free(&model);
    h3_vdn_weight_store_free(store);
    h3_gpu_free(gpu);
    free(time_embedding_actual);
    free(actual);
    free(map_u32);
    free(modulation_actual);
    free(modulation_host);
    free(refined_prompt_host);
    free(audio_rows_host);
    free(video_rows_host);
    free(time_values_host);
    free(time_embedding_host);
    for (unsigned slot = 0; slot < MODULATION_SLOTS; slot++) free(chunks[slot]);
    free(map_i64);
    free(rope_sin_host);
    free(rope_cos_host);
    free(expected);
    free(input);
    h3_st_free_header(&header);
    return status;
}
