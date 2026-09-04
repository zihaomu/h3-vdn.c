#include "h3_gpu.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HEAD_DIM = 128 };

static double monotonic_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1.0e9;
}

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
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

static int parse_u32(const char *text, uint32_t *value) {
    if (!text || !*text) return 0;
    errno = 0;
    char *end = NULL;
    unsigned long parsed = strtoul(text, &end, 10);
    if (errno || end == text || *end || !parsed || parsed > UINT32_MAX)
        return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static int run(h3_gpu *gpu, h3_gpu_tensor *query_i8,
               h3_gpu_tensor *key_i8, h3_gpu_tensor *query_scales,
               h3_gpu_tensor *key_scales, const h3_gpu_tensor *query,
               const h3_gpu_tensor *key, uint32_t sequence, uint32_t heads,
               uint32_t iterations, double *seconds) {
    if (!h3_gpu_begin(gpu) || !h3_gpu_vdn_sage_quant_qk_bf16(
            gpu, query_i8, key_i8, query_scales, key_scales, query, key,
            sequence, heads, HEAD_DIM) || !h3_gpu_submit(gpu)) return 0;
    double start = monotonic_seconds();
    for (uint32_t iteration = 0; iteration < iterations; iteration++)
        if (!h3_gpu_begin(gpu) || !h3_gpu_vdn_sage_quant_qk_bf16(
                gpu, query_i8, key_i8, query_scales, key_scales, query, key,
                sequence, heads, HEAD_DIM) || !h3_gpu_submit(gpu)) return 0;
    *seconds = monotonic_seconds() - start;
    return 1;
}

int main(int argc, char **argv) {
    uint32_t sequence = 5338;
    uint32_t heads = 56;
    uint32_t iterations = 5;
    if (argc != 1 && argc != 4) {
        fprintf(stderr, "usage: %s [SEQUENCE HEADS ITERATIONS]\n", argv[0]);
        return 2;
    }
    if (argc == 4 &&
        (!parse_u32(argv[1], &sequence) || !parse_u32(argv[2], &heads) ||
         !parse_u32(argv[3], &iterations))) {
        fprintf(stderr, "invalid Sage quant benchmark argument\n");
        return 2;
    }
    uint64_t elements_wide = (uint64_t)sequence * heads * HEAD_DIM;
    if (elements_wide > SIZE_MAX / sizeof(uint16_t)) {
        fprintf(stderr, "Sage quant benchmark shape is too large\n");
        return 2;
    }
    size_t elements = (size_t)elements_wide;
    uint16_t *input = malloc(elements * sizeof(*input));
    int8_t *result = malloc(elements);
    if (!input || !result) {
        fprintf(stderr, "out of host memory for Sage quant benchmark\n");
        free(result); free(input);
        return 1;
    }
    for (size_t index = 0; index < elements; index++) {
        int centered = (int)((index * 37 + 11) % 509) - 254;
        input[index] = bf16((float)centered * 0.00390625f);
    }
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    h3_gpu_tensor *query = NULL, *key = NULL;
    h3_gpu_tensor *query_i8 = NULL, *key_i8 = NULL;
    h3_gpu_tensor *query_scales = NULL, *key_scales = NULL;
    int ok = gpu != NULL;
    uint32_t q_groups = (uint32_t)(((uint64_t)sequence + 31) / 32);
    uint32_t k_groups = (uint32_t)(((uint64_t)sequence + 63) / 64);
    if (ok) {
        query = h3_gpu_tensor_from_bf16(gpu, input, elements);
        key = h3_gpu_tensor_from_bf16(gpu, input, elements);
        query_i8 = h3_gpu_tensor_new_i8(gpu, elements);
        key_i8 = h3_gpu_tensor_new_i8(gpu, elements);
        query_scales = h3_gpu_tensor_new_f32(
            gpu, (size_t)heads * q_groups);
        key_scales = h3_gpu_tensor_new_f32(gpu, (size_t)heads * k_groups);
        ok = query && key && query_i8 && key_i8 && query_scales && key_scales;
    }
    double seconds = 0.0;
    if (ok)
        ok = run(gpu, query_i8, key_i8, query_scales, key_scales, query, key,
                 sequence, heads, iterations, &seconds) &&
             h3_gpu_tensor_read_i8(query_i8, result, elements);
    if (ok) {
        double average = seconds / iterations;
        double gib = (double)elements * 6.0 /
                     (1024.0 * 1024.0 * 1024.0);
        printf("VDN Sage Q/K quant: sequence=%u heads=%u dim=%u "
               "iterations=%u average=%.6fs traffic=%.3fGiB "
               "effective=%.2fGiB/s Q-hash=%016llx\n",
               sequence, heads, HEAD_DIM, iterations, average, gib,
               gib / average,
               (unsigned long long)hash_bytes(result, elements));
    } else {
        fprintf(stderr, "Sage quant benchmark failed: %s\n",
                error[0] ? error : h3_gpu_error(gpu));
    }
    h3_gpu_tensor_free(key_scales); h3_gpu_tensor_free(query_scales);
    h3_gpu_tensor_free(key_i8); h3_gpu_tensor_free(query_i8);
    h3_gpu_tensor_free(key); h3_gpu_tensor_free(query);
    h3_gpu_free(gpu);
    free(result); free(input);
    return ok ? 0 : 1;
}
