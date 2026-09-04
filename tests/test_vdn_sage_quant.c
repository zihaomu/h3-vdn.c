#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SEQUENCE = 65, HEADS = 3, HEAD_DIM = 128 };
enum { WMMA_TILE = 16, WMMA_Q_START = 17, WMMA_K_START = 49,
       WMMA_HEAD = 1 };

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
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

static void quantize_cpu(const uint16_t *input, int8_t *output,
                         float *scales, uint32_t group_rows,
                         uint32_t groups) {
    for (uint32_t head = 0; head < HEADS; head++)
        for (uint32_t group = 0; group < groups; group++) {
            uint32_t row_begin = group * group_rows;
            uint32_t row_end = row_begin + group_rows;
            if (row_end > SEQUENCE) row_end = SEQUENCE;
            float amax = 0.0f;
            for (uint32_t row = row_begin; row < row_end; row++)
                for (uint32_t dimension = 0; dimension < HEAD_DIM;
                     dimension++) {
                    size_t index = ((size_t)row * HEADS + head) * HEAD_DIM +
                                   dimension;
                    amax = fmaxf(amax, fabsf(f32(input[index])));
                }
            float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
            float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
            scales[(size_t)head * groups + group] = scale;
            for (uint32_t row = row_begin; row < row_end; row++)
                for (uint32_t dimension = 0; dimension < HEAD_DIM;
                     dimension++) {
                    size_t index = ((size_t)row * HEADS + head) * HEAD_DIM +
                                   dimension;
                    long rounded = lrintf(f32(input[index]) * inverse);
                    if (rounded > 127) rounded = 127;
                    if (rounded < -127) rounded = -127;
                    output[index] = (int8_t)rounded;
                }
        }
}

int main(void) {
    const size_t elements = (size_t)SEQUENCE * HEADS * HEAD_DIM;
    const uint32_t q_groups = (SEQUENCE + 31) / 32;
    const uint32_t k_groups = (SEQUENCE + 63) / 64;
    uint16_t *query = malloc(elements * sizeof(*query));
    uint16_t *key = malloc(elements * sizeof(*key));
    uint16_t *mutated_value = malloc(elements * sizeof(*mutated_value));
    uint16_t *baseline_attention = malloc(elements * sizeof(*baseline_attention));
    uint16_t *sage_attention = malloc(elements * sizeof(*sage_attention));
    uint16_t *masked_attention = malloc(elements * sizeof(*masked_attention));
    int8_t *q_expected = calloc(elements, sizeof(*q_expected));
    int8_t *k_expected = calloc(elements, sizeof(*k_expected));
    int8_t *q_actual = malloc(elements * sizeof(*q_actual));
    int8_t *k_actual = malloc(elements * sizeof(*k_actual));
    float q_scales_expected[HEADS * q_groups];
    float k_scales_expected[HEADS * k_groups];
    float q_scales_actual[HEADS * q_groups];
    float k_scales_actual[HEADS * k_groups];
    float scores_actual[WMMA_TILE * WMMA_TILE];
    int ok = query && key && mutated_value && baseline_attention &&
             sage_attention && masked_attention && q_expected && k_expected &&
             q_actual && k_actual;
    if (!ok) {
        fprintf(stderr, "out of memory for Sage quant test\n");
        goto cleanup_host;
    }
    for (uint32_t row = 0; row < SEQUENCE; row++)
        for (uint32_t head = 0; head < HEADS; head++)
            for (uint32_t dimension = 0; dimension < HEAD_DIM; dimension++) {
                size_t index = ((size_t)row * HEADS + head) * HEAD_DIM +
                               dimension;
                int q_centered = (int)((index * 37 + 11) % 509) - 254;
                int k_centered = (int)((index * 19 + 7) % 257) - 128;
                query[index] = bf16((float)q_centered * 0.00390625f);
                key[index] = row == 64 ? bf16(0.0f) :
                    bf16((float)k_centered * 0.0078125f);
            }
    quantize_cpu(query, q_expected, q_scales_expected, 32, q_groups);
    quantize_cpu(key, k_expected, k_scales_expected, 64, k_groups);
    memcpy(mutated_value, key, elements * sizeof(*mutated_value));
    /* Frame 1 is masked for a frame-3 query with radius 1 and endpoint
     * anchors. Only V changes so groupwise K scaling cannot confound the
     * mask-leakage assertion. */
    for (uint32_t row = 9; row < 17; row++)
        for (uint32_t head = 0; head < HEADS; head++)
            for (uint32_t dimension = 0; dimension < HEAD_DIM; dimension++) {
                size_t index = ((size_t)row * HEADS + head) * HEAD_DIM +
                               dimension;
                mutated_value[index] = bf16(dimension & 1 ? 64.0f : -64.0f);
            }

    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    h3_gpu_tensor *q_input = NULL, *k_input = NULL;
    h3_gpu_tensor *q_output = NULL, *k_output = NULL;
    h3_gpu_tensor *q_scales = NULL, *k_scales = NULL;
    h3_gpu_tensor *scores = NULL;
    h3_gpu_tensor *mutated_value_input = NULL;
    h3_gpu_tensor *baseline_output = NULL, *sage_output = NULL;
    h3_gpu_tensor *masked_output = NULL;
    if (!gpu) {
        fprintf(stderr, "cannot create HIP context: %s\n", error);
        ok = 0;
        goto cleanup_gpu;
    }
    q_input = h3_gpu_tensor_from_bf16(gpu, query, elements);
    k_input = h3_gpu_tensor_from_bf16(gpu, key, elements);
    q_output = h3_gpu_tensor_new_i8(gpu, elements);
    k_output = h3_gpu_tensor_new_i8(gpu, elements);
    q_scales = h3_gpu_tensor_new_f32(gpu, (size_t)HEADS * q_groups);
    k_scales = h3_gpu_tensor_new_f32(gpu, (size_t)HEADS * k_groups);
    scores = h3_gpu_tensor_new_f32(gpu, WMMA_TILE * WMMA_TILE);
    mutated_value_input = h3_gpu_tensor_from_bf16(
        gpu, mutated_value, elements);
    baseline_output = h3_gpu_tensor_new_bf16(gpu, elements);
    sage_output = h3_gpu_tensor_new_bf16(gpu, elements);
    masked_output = h3_gpu_tensor_new_bf16(gpu, elements);
    if (!q_input || !k_input || !q_output || !k_output || !q_scales ||
        !k_scales || !h3_gpu_begin(gpu) ||
        !h3_gpu_vdn_sage_quant_qk_bf16(
            gpu, q_output, k_output, q_scales, k_scales, q_input, k_input,
            SEQUENCE, HEADS, HEAD_DIM) || !h3_gpu_submit(gpu) ||
        !h3_gpu_tensor_read_i8(q_output, q_actual, elements) ||
        !h3_gpu_tensor_read_i8(k_output, k_actual, elements) ||
        !h3_gpu_tensor_read_f32(
            q_scales, q_scales_actual, (size_t)HEADS * q_groups) ||
        !h3_gpu_tensor_read_f32(
            k_scales, k_scales_actual, (size_t)HEADS * k_groups)) {
        fprintf(stderr, "Sage quant GPU failure: %s\n", h3_gpu_error(gpu));
        ok = 0;
        goto cleanup_gpu;
    }
    size_t q_mismatches = 0, k_mismatches = 0;
    for (size_t index = 0; index < elements; index++) {
        if (q_actual[index] != q_expected[index]) q_mismatches++;
        if (k_actual[index] != k_expected[index]) k_mismatches++;
        if (q_actual[index] < -127 || k_actual[index] < -127) ok = 0;
    }
    if (q_mismatches || k_mismatches ||
        memcmp(q_scales_actual, q_scales_expected,
               sizeof(q_scales_actual)) ||
        memcmp(k_scales_actual, k_scales_expected,
               sizeof(k_scales_actual)) ||
        k_scales_actual[k_groups - 1] != 1.0f) {
        fprintf(stderr, "Sage quant mismatch: Q=%zu K=%zu\n",
                q_mismatches, k_mismatches);
        ok = 0;
    }
    const float attention_scale = 0.08838834764831845f;
    if (ok && (!scores || !h3_gpu_begin(gpu) ||
        !h3_gpu_vdn_sage_wmma_qk_tile_i8(
            gpu, scores, q_output, k_output, q_scales, k_scales,
            SEQUENCE, HEADS, HEAD_DIM, WMMA_Q_START, WMMA_K_START,
            WMMA_HEAD, attention_scale) || !h3_gpu_submit(gpu) ||
        !h3_gpu_tensor_read_f32(scores, scores_actual,
                                WMMA_TILE * WMMA_TILE))) {
        fprintf(stderr, "Sage WMMA tile GPU failure: %s\n",
                h3_gpu_error(gpu));
        ok = 0;
    }
    float max_absolute_error = 0.0f;
    if (ok)
        for (uint32_t q = 0; q < WMMA_TILE; q++)
            for (uint32_t k = 0; k < WMMA_TILE; k++) {
                uint32_t query_row = WMMA_Q_START + q;
                uint32_t key_row = WMMA_K_START + k;
                int32_t dot = 0;
                for (uint32_t dimension = 0; dimension < HEAD_DIM;
                     dimension++) {
                    size_t q_index = ((size_t)query_row * HEADS +
                                      WMMA_HEAD) * HEAD_DIM + dimension;
                    size_t k_index = ((size_t)key_row * HEADS +
                                      WMMA_HEAD) * HEAD_DIM + dimension;
                    dot += (int32_t)q_actual[q_index] *
                           (int32_t)k_actual[k_index];
                }
                float expected = (float)dot *
                    q_scales_actual[(size_t)WMMA_HEAD * q_groups +
                                    query_row / 32] *
                    k_scales_actual[(size_t)WMMA_HEAD * k_groups +
                                    key_row / 64] * attention_scale;
                size_t score_index = (size_t)q * WMMA_TILE + k;
                float absolute_error = fabsf(scores_actual[score_index] -
                                             expected);
                if (absolute_error > max_absolute_error)
                    max_absolute_error = absolute_error;
                if (absolute_error > 1.0e-5f) ok = 0;
            }
    if (!ok && max_absolute_error > 0.0f)
        fprintf(stderr, "Sage WMMA score mismatch: max-abs=%g\n",
                (double)max_absolute_error);
    const uint32_t video_start = 1;
    const uint32_t frames = 7;
    const uint32_t tokens_per_frame = 8;
    if (ok && (!mutated_value_input || !baseline_output || !sage_output ||
        !masked_output || !h3_gpu_begin(gpu) ||
        !h3_gpu_vdn_window_sdpa_bf16(
            gpu, baseline_output, q_input, k_input, k_input, SEQUENCE, HEADS,
            HEAD_DIM, video_start, frames, tokens_per_frame, 1, 0, 1,
            attention_scale) ||
        !h3_gpu_vdn_sage_attention_i8_bf16(
            gpu, sage_output, q_output, k_output, q_scales, k_scales,
            k_input, SEQUENCE, HEADS, HEAD_DIM, video_start, frames,
            tokens_per_frame, 1, 0, 1, attention_scale) ||
        !h3_gpu_vdn_sage_attention_i8_bf16(
            gpu, masked_output, q_output, k_output, q_scales, k_scales,
            mutated_value_input, SEQUENCE, HEADS, HEAD_DIM, video_start,
            frames, tokens_per_frame, 1, 0, 1, attention_scale) ||
        !h3_gpu_submit(gpu) ||
        !h3_gpu_tensor_read_bf16(baseline_output, baseline_attention,
                                 elements) ||
        !h3_gpu_tensor_read_bf16(sage_output, sage_attention, elements) ||
        !h3_gpu_tensor_read_bf16(masked_output, masked_attention, elements))) {
        fprintf(stderr, "Sage fused attention GPU failure: %s\n",
                h3_gpu_error(gpu));
        ok = 0;
    }
    double squared_error = 0.0, squared_reference = 0.0;
    double dot_product = 0.0, squared_candidate = 0.0;
    float attention_max_absolute_error = 0.0f;
    if (ok)
        for (size_t index = 0; index < elements; index++) {
            float reference = f32(baseline_attention[index]);
            float candidate = f32(sage_attention[index]);
            float difference = candidate - reference;
            if (!isfinite(candidate)) ok = 0;
            if (fabsf(difference) > attention_max_absolute_error)
                attention_max_absolute_error = fabsf(difference);
            squared_error += (double)difference * difference;
            squared_reference += (double)reference * reference;
            squared_candidate += (double)candidate * candidate;
            dot_product += (double)reference * candidate;
        }
    size_t mask_mismatches = 0;
    const uint32_t protected_query = video_start + 3 * tokens_per_frame;
    if (ok)
        for (uint32_t head = 0; head < HEADS; head++)
            for (uint32_t dimension = 0; dimension < HEAD_DIM; dimension++) {
                size_t index = ((size_t)protected_query * HEADS + head) *
                               HEAD_DIM + dimension;
                if (sage_attention[index] != masked_attention[index])
                    mask_mismatches++;
            }
    double relative_rmse = squared_reference > 0.0 ?
        sqrt(squared_error / squared_reference) : INFINITY;
    double cosine = squared_reference > 0.0 && squared_candidate > 0.0 ?
        dot_product / sqrt(squared_reference * squared_candidate) : 0.0;
    if (mask_mismatches || !isfinite(relative_rmse) || !isfinite(cosine) ||
        cosine < 0.95) {
        fprintf(stderr, "Sage fused attention mismatch: mask=%zu "
                "rel-rmse=%g cosine=%g\n", mask_mismatches, relative_rmse,
                cosine);
        ok = 0;
    }
    if (ok)
        printf("VDN Sage Q/K INT8 quant passed: Q=%016llx K=%016llx "
               "Q-groups=%u K-groups=%u WMMA-max-abs=%g "
               "attention-max-abs=%g rel-rmse=%g cosine=%.9f "
               "mask-mismatches=%zu\n",
               (unsigned long long)hash_bytes(q_actual, elements),
               (unsigned long long)hash_bytes(k_actual, elements),
               q_groups, k_groups, (double)max_absolute_error,
               (double)attention_max_absolute_error, relative_rmse, cosine,
               mask_mismatches);

cleanup_gpu:
    h3_gpu_tensor_free(masked_output);
    h3_gpu_tensor_free(sage_output);
    h3_gpu_tensor_free(baseline_output);
    h3_gpu_tensor_free(mutated_value_input);
    h3_gpu_tensor_free(scores);
    h3_gpu_tensor_free(k_scales); h3_gpu_tensor_free(q_scales);
    h3_gpu_tensor_free(k_output); h3_gpu_tensor_free(q_output);
    h3_gpu_tensor_free(k_input); h3_gpu_tensor_free(q_input);
    h3_gpu_free(gpu);
cleanup_host:
    free(masked_attention); free(sage_attention); free(baseline_attention);
    free(mutated_value);
    free(k_actual); free(q_actual); free(k_expected); free(q_expected);
    free(key); free(query);
    return ok ? 0 : 1;
}
