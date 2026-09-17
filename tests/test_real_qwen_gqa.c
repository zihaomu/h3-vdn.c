#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    SEQUENCE = 18,
    QUERY_HEADS = 64,
    KV_HEADS = 8,
    HEAD_DIM = 128
};

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_qwen_gqa.c: %s\n", message);
    exit(1);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static void reference_gqa(const uint16_t *query, const uint16_t *key,
                          const uint16_t *value, uint16_t *output) {
    float scores[SEQUENCE];
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (uint32_t row = 0; row < SEQUENCE; row++) {
        for (uint32_t q_head = 0; q_head < QUERY_HEADS; q_head++) {
            uint32_t kv_head = q_head / (QUERY_HEADS / KV_HEADS);
            size_t q_base = ((size_t)row * QUERY_HEADS + q_head) * HEAD_DIM;
            float maximum = -INFINITY;
            for (uint32_t key_row = 0; key_row <= row; key_row++) {
                size_t k_base = ((size_t)key_row * KV_HEADS + kv_head) *
                                HEAD_DIM;
                float dot = 0.0f;
                for (uint32_t dimension = 0; dimension < HEAD_DIM; dimension++)
                    dot = fmaf(bf16_to_f32(query[q_base + dimension]),
                               bf16_to_f32(key[k_base + dimension]), dot);
                float score = bf16_to_f32(f32_to_bf16(
                    bf16_to_f32(f32_to_bf16(dot)) * scale));
                scores[key_row] = score;
                maximum = fmaxf(maximum, score);
            }
            float denominator = 0.0f;
            for (uint32_t key_row = 0; key_row <= row; key_row++) {
                scores[key_row] = expf(scores[key_row] - maximum);
                denominator += scores[key_row];
            }
            for (uint32_t dimension = 0; dimension < HEAD_DIM; dimension++) {
                float sum = 0.0f;
                for (uint32_t key_row = 0; key_row <= row; key_row++) {
                    size_t source = ((size_t)key_row * KV_HEADS + kv_head) *
                                    HEAD_DIM + dimension;
                    float probability = bf16_to_f32(f32_to_bf16(
                        scores[key_row] / denominator));
                    sum = fmaf(probability, bf16_to_f32(value[source]), sum);
                }
                output[q_base + dimension] = f32_to_bf16(sum);
            }
        }
    }
}

static double compare(const char *label, const uint16_t *got,
                      const uint16_t *want, size_t count) {
    double maximum = 0.0, scale = 0.0, squares = 0.0, reference = 0.0;
    size_t nonfinite = 0;
    for (size_t index = 0; index < count; index++) {
        double actual = bf16_to_f32(got[index]);
        double expected = bf16_to_f32(want[index]);
        if (!isfinite(actual) || !isfinite(expected)) {
            nonfinite++;
            continue;
        }
        double delta = actual - expected;
        if (fabs(delta) > maximum) maximum = fabs(delta);
        if (fabs(expected) > scale) scale = fabs(expected);
        squares += delta * delta;
        reference += expected * expected;
    }
    double rel_max = maximum / (scale > 1e-12 ? scale : 1e-12);
    double rel_l2 = sqrt(squares / (reference > 1e-24 ? reference : 1e-24));
    printf("%-30s max abs %.7g, rel-max %.7g, rel-L2 %.7g, "
           "nonfinite %zu\n", label, maximum, rel_max, rel_l2, nonfinite);
    return nonfinite ? INFINITY : rel_l2;
}

static uint16_t *read_bf16(const h3_st_header *fixture, const char *name,
                           const uint64_t *shape, int ndim, size_t count) {
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != ndim ||
        h3_st_tensor_elements(tensor) != count)
        die("real-shape GQA tensor is absent or malformed");
    for (int index = 0; index < ndim; index++)
        if (tensor->shape[index] != shape[index])
            die("real-shape GQA tensor has the wrong shape");
    uint16_t *values = malloc(count * sizeof(*values));
    if (!values) die("out of memory loading real-shape GQA fixture");
    char error[512];
    if (!h3_st_read_data(fixture, tensor, values,
                         count * sizeof(*values), error, sizeof(error)))
        die(error);
    return values;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MULTIMODAL_LAYER1_FIXTURE\n", argv[0]);
        return 2;
    }
    const size_t query_count =
        (size_t)SEQUENCE * QUERY_HEADS * HEAD_DIM;
    const size_t kv_count = (size_t)SEQUENCE * KV_HEADS * HEAD_DIM;
    const uint64_t query_shape[] = {SEQUENCE, QUERY_HEADS, HEAD_DIM};
    const uint64_t kv_shape[] = {SEQUENCE, KV_HEADS, HEAD_DIM};
    h3_st_header fixture;
    char error[512];
    if (!h3_st_read_header(argv[1], &fixture, error, sizeof(error)))
        die(error);
    uint16_t *query = read_bf16(
        &fixture, "diagnostic.text.layer_01.rope_q", query_shape, 3,
        query_count);
    uint16_t *key = read_bf16(
        &fixture, "diagnostic.text.layer_01.rope_k", kv_shape, 3,
        kv_count);
    uint16_t *value = read_bf16(
        &fixture, "diagnostic.text.layer_01.value", kv_shape, 3,
        kv_count);
    uint16_t *want = read_bf16(
        &fixture, "diagnostic.text.layer_01.attention_heads", query_shape, 3,
        query_count);

    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);
    h3_gpu_tensor *query_gpu = h3_gpu_tensor_from_bf16(
        gpu, query, query_count);
    h3_gpu_tensor *key_gpu = h3_gpu_tensor_from_bf16(gpu, key, kv_count);
    h3_gpu_tensor *value_gpu = h3_gpu_tensor_from_bf16(
        gpu, value, kv_count);
    h3_gpu_tensor *output_gpu = h3_gpu_tensor_new_bf16(gpu, query_count);
    if (!query_gpu || !key_gpu || !value_gpu || !output_gpu)
        die(h3_gpu_error(gpu));
    if (!h3_gpu_begin(gpu) ||
        !h3_gpu_gqa_causal_bf16(
            gpu, output_gpu, query_gpu, key_gpu, value_gpu, SEQUENCE,
            QUERY_HEADS, KV_HEADS, HEAD_DIM,
            1.0f / sqrtf((float)HEAD_DIM)) ||
        !h3_gpu_submit(gpu))
        die(h3_gpu_error(gpu));
    uint16_t *got = malloc(query_count * sizeof(*got));
    if (!got || !h3_gpu_tensor_read_bf16(output_gpu, got, query_count))
        die("cannot read real-shape GQA output");

    uint16_t *cpu = malloc(query_count * sizeof(*cpu));
    if (!cpu) die("out of memory computing real-shape GQA reference");
    reference_gqa(query, key, value, cpu);
    compare("CPU serial diagnostic", cpu, want, query_count);
    double gpu_error = compare("HIP reference vs upstream", got, want,
                               query_count);
    compare("HIP vs CPU serial", got, cpu, query_count);

    free(cpu);
    free(got);
    h3_gpu_tensor_free(output_gpu);
    h3_gpu_tensor_free(value_gpu);
    h3_gpu_tensor_free(key_gpu);
    h3_gpu_tensor_free(query_gpu);
    h3_gpu_free(gpu);
    h3_st_free_header(&fixture);
    free(want);
    free(value);
    free(key);
    free(query);
    if (gpu_error >= 0.03)
        die("real-shape GQA exceeds the upstream error bound");
    puts("ok: HIP real-shape layer-1 GQA matches the upstream eager oracle");
    return 0;
}
