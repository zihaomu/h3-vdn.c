#include "h3_gpu.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HEAD_DIM = 128, FRAMES = 17 };

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

typedef struct {
    h3_gpu *gpu;
    h3_gpu_tensor *output;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *query_i8;
    h3_gpu_tensor *key_i8;
    h3_gpu_tensor *query_scales;
    h3_gpu_tensor *key_scales;
    uint32_t sequence;
    uint32_t heads;
    uint32_t video_start;
    uint32_t tokens_per_frame;
    float scale;
} benchmark;

static int enqueue(benchmark *bench, int sage) {
    if (!h3_gpu_begin(bench->gpu)) return 0;
    if (sage) {
        if (!h3_gpu_vdn_sage_quant_qk_bf16(
                bench->gpu, bench->query_i8, bench->key_i8,
                bench->query_scales, bench->key_scales, bench->query,
                bench->key, bench->sequence, bench->heads, HEAD_DIM) ||
            !h3_gpu_vdn_sage_attention_i8_bf16(
                bench->gpu, bench->output, bench->query_i8, bench->key_i8,
                bench->query_scales, bench->key_scales, bench->value,
                bench->sequence, bench->heads, HEAD_DIM, bench->video_start,
                FRAMES, bench->tokens_per_frame, 1, 5, 1, bench->scale))
            return 0;
    } else if (!h3_gpu_vdn_window_sdpa_bf16(
                   bench->gpu, bench->output, bench->query, bench->key,
                   bench->value, bench->sequence, bench->heads, HEAD_DIM,
                   bench->video_start, FRAMES, bench->tokens_per_frame,
                   1, 5, 1, bench->scale)) return 0;
    return h3_gpu_submit(bench->gpu);
}

static int run_path(benchmark *bench, int sage, uint32_t iterations,
                    double *seconds, uint16_t *result, size_t elements) {
    if (!enqueue(bench, sage)) return 0;
    double start = monotonic_seconds();
    for (uint32_t iteration = 0; iteration < iterations; iteration++)
        if (!enqueue(bench, sage)) return 0;
    *seconds = monotonic_seconds() - start;
    return h3_gpu_tensor_read_bf16(bench->output, result, elements);
}

int main(int argc, char **argv) {
    uint32_t sequence = 5338, heads = 56, video_start = 986;
    uint32_t tokens_per_frame = 256, iterations = 3;
    if (argc != 1 && argc != 6) {
        fprintf(stderr, "usage: %s [SEQUENCE HEADS VIDEO_START "
                        "TOKENS_PER_FRAME ITERATIONS]\n", argv[0]);
        return 2;
    }
    if (argc == 6 &&
        (!parse_u32(argv[1], &sequence) || !parse_u32(argv[2], &heads) ||
         !parse_u32(argv[3], &video_start) ||
         !parse_u32(argv[4], &tokens_per_frame) ||
         !parse_u32(argv[5], &iterations))) {
        fprintf(stderr, "invalid Sage attention benchmark argument\n");
        return 2;
    }
    uint64_t video_rows = (uint64_t)FRAMES * tokens_per_frame;
    uint64_t elements_wide = (uint64_t)sequence * heads * HEAD_DIM;
    if (video_start > sequence || video_rows > sequence - video_start ||
        elements_wide > SIZE_MAX / sizeof(uint16_t)) {
        fprintf(stderr, "invalid Sage attention benchmark shape\n");
        return 2;
    }
    size_t elements = (size_t)elements_wide;
    uint16_t *input = malloc(elements * sizeof(*input));
    uint16_t *baseline = malloc(elements * sizeof(*baseline));
    uint16_t *candidate = malloc(elements * sizeof(*candidate));
    if (!input || !baseline || !candidate) {
        fprintf(stderr, "out of host memory for Sage attention benchmark\n");
        free(candidate); free(baseline); free(input);
        return 1;
    }
    for (size_t index = 0; index < elements; index++) {
        int centered = (int)((index * 37 + 11) % 509) - 254;
        input[index] = bf16((float)centered * 0.0009765625f);
    }

    char error[512] = {0};
    benchmark bench = {0};
    bench.gpu = h3_gpu_create(NULL, error, sizeof(error));
    bench.sequence = sequence;
    bench.heads = heads;
    bench.video_start = video_start;
    bench.tokens_per_frame = tokens_per_frame;
    bench.scale = 1.0f / sqrtf((float)HEAD_DIM);
    uint32_t q_groups = (sequence + 31) / 32;
    uint32_t k_groups = (sequence + 63) / 64;
    int ok = bench.gpu != NULL;
    if (ok) {
        bench.query = h3_gpu_tensor_from_bf16(bench.gpu, input, elements);
        bench.key = h3_gpu_tensor_from_bf16(bench.gpu, input, elements);
        bench.value = h3_gpu_tensor_from_bf16(bench.gpu, input, elements);
        bench.output = h3_gpu_tensor_new_bf16(bench.gpu, elements);
        bench.query_i8 = h3_gpu_tensor_new_i8(bench.gpu, elements);
        bench.key_i8 = h3_gpu_tensor_new_i8(bench.gpu, elements);
        bench.query_scales = h3_gpu_tensor_new_f32(
            bench.gpu, (size_t)heads * q_groups);
        bench.key_scales = h3_gpu_tensor_new_f32(
            bench.gpu, (size_t)heads * k_groups);
        ok = bench.query && bench.key && bench.value && bench.output &&
             bench.query_i8 && bench.key_i8 && bench.query_scales &&
             bench.key_scales;
    }
    double baseline_seconds = 0.0, candidate_seconds = 0.0;
    const char *sage_first_value = getenv("H3_SAGE_BENCH_SAGE_FIRST");
    int sage_first = sage_first_value && *sage_first_value &&
                     strcmp(sage_first_value, "0");
    if (ok && sage_first)
        ok = run_path(&bench, 1, iterations, &candidate_seconds, candidate,
                      elements) &&
             run_path(&bench, 0, iterations, &baseline_seconds, baseline,
                      elements);
    else if (ok)
        ok = run_path(&bench, 0, iterations, &baseline_seconds, baseline,
                      elements) &&
             run_path(&bench, 1, iterations, &candidate_seconds, candidate,
                      elements);

    float max_absolute_error = 0.0f;
    double squared_error = 0.0, squared_reference = 0.0;
    double squared_candidate = 0.0, dot_product = 0.0;
    size_t nonfinite = 0;
    if (ok)
        for (size_t index = 0; index < elements; index++) {
            float reference = f32(baseline[index]);
            float approximation = f32(candidate[index]);
            float difference = approximation - reference;
            if (!isfinite(approximation)) nonfinite++;
            if (fabsf(difference) > max_absolute_error)
                max_absolute_error = fabsf(difference);
            squared_error += (double)difference * difference;
            squared_reference += (double)reference * reference;
            squared_candidate += (double)approximation * approximation;
            dot_product += (double)reference * approximation;
        }
    double relative_rmse = squared_reference > 0.0 ?
        sqrt(squared_error / squared_reference) : INFINITY;
    double cosine = squared_reference > 0.0 && squared_candidate > 0.0 ?
        dot_product / sqrt(squared_reference * squared_candidate) : 0.0;
    if (ok) {
        printf("VDN Sage attention: sequence=%u heads=%u iterations=%u "
               "order=%s baseline=%.6fs sage-taxed=%.6fs speedup=%.3fx "
               "max-abs=%g rel-rmse=%g cosine=%.9f nonfinite=%zu "
               "baseline-hash=%016llx sage-hash=%016llx\n",
               sequence, heads, iterations, sage_first ? "S-B" : "B-S",
               baseline_seconds / iterations,
               candidate_seconds / iterations,
               candidate_seconds > 0.0 ?
                   baseline_seconds / candidate_seconds : 0.0,
               (double)max_absolute_error, relative_rmse, cosine, nonfinite,
               (unsigned long long)hash_bytes(
                   baseline, elements * sizeof(*baseline)),
               (unsigned long long)hash_bytes(
                   candidate, elements * sizeof(*candidate)));
        if (nonfinite) ok = 0;
    } else {
        fprintf(stderr, "Sage attention benchmark failed: %s\n",
                error[0] ? error : h3_gpu_error(bench.gpu));
    }
    h3_gpu_tensor_free(bench.key_scales);
    h3_gpu_tensor_free(bench.query_scales);
    h3_gpu_tensor_free(bench.key_i8);
    h3_gpu_tensor_free(bench.query_i8);
    h3_gpu_tensor_free(bench.output);
    h3_gpu_tensor_free(bench.value);
    h3_gpu_tensor_free(bench.key);
    h3_gpu_tensor_free(bench.query);
    h3_gpu_free(bench.gpu);
    free(candidate); free(baseline); free(input);
    return ok ? 0 : 1;
}
