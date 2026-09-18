#include "h3_gpu.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HEAD_DIM = 128, WARMUP_ITERATIONS = 3 };

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
    uint32_t sequence;
    uint32_t heads;
    float scale;
} benchmark;

static int enqueue(benchmark *bench) {
    return h3_gpu_begin(bench->gpu) &&
           h3_gpu_sdpa_bf16(
               bench->gpu, bench->output, bench->query, bench->key,
               bench->value, bench->sequence, bench->heads, HEAD_DIM,
               bench->scale) &&
           h3_gpu_submit(bench->gpu);
}

static int run_path(benchmark *bench, const char *mode, uint32_t iterations,
                    double *seconds, uint16_t *result, size_t elements) {
    if (setenv("H3_BF16_SDPA", mode, 1) != 0) return 0;
    for (uint32_t warmup = 0; warmup < WARMUP_ITERATIONS; ++warmup)
        if (!enqueue(bench)) return 0;
    const double start = monotonic_seconds();
    for (uint32_t iteration = 0; iteration < iterations; ++iteration)
        if (!enqueue(bench)) return 0;
    *seconds = monotonic_seconds() - start;
    return h3_gpu_tensor_read_bf16(bench->output, result, elements);
}

int main(int argc, char **argv) {
    uint32_t sequence = 9300, heads = 56, iterations = 10;
    if (argc != 1 && argc != 4) {
        fprintf(stderr, "usage: %s [SEQUENCE HEADS ITERATIONS]\n", argv[0]);
        return 2;
    }
    if (argc == 4 &&
        (!parse_u32(argv[1], &sequence) || !parse_u32(argv[2], &heads) ||
         !parse_u32(argv[3], &iterations))) {
        fprintf(stderr, "invalid dense Sage benchmark argument\n");
        return 2;
    }
    const uint64_t elements_wide =
        (uint64_t)sequence * heads * HEAD_DIM;
    if (elements_wide > SIZE_MAX / sizeof(uint16_t)) {
        fprintf(stderr, "dense Sage benchmark shape overflows\n");
        return 2;
    }
    const size_t elements = (size_t)elements_wide;
    uint16_t *input = malloc(elements * sizeof(*input));
    uint16_t *baseline = malloc(elements * sizeof(*baseline));
    uint16_t *candidate = malloc(elements * sizeof(*candidate));
    uint16_t *repeat = malloc(elements * sizeof(*repeat));
    if (!input || !baseline || !candidate || !repeat) {
        fprintf(stderr, "out of host memory for dense Sage benchmark\n");
        free(repeat); free(candidate); free(baseline); free(input);
        return 1;
    }
    for (size_t index = 0; index < elements; ++index) {
        const int centered = (int)((index * 37 + 11) % 509) - 254;
        input[index] = bf16((float)centered * 0.0009765625f);
    }

    char error[512] = {0};
    benchmark bench = {0};
    bench.gpu = h3_gpu_create(NULL, error, sizeof(error));
    bench.sequence = sequence;
    bench.heads = heads;
    bench.scale = 1.0f / sqrtf((float)HEAD_DIM);
    int ok = bench.gpu != NULL;
    if (ok) {
        bench.query = h3_gpu_tensor_from_bf16(bench.gpu, input, elements);
        bench.key = h3_gpu_tensor_from_bf16(bench.gpu, input, elements);
        bench.value = h3_gpu_tensor_from_bf16(bench.gpu, input, elements);
        bench.output = h3_gpu_tensor_new_bf16(bench.gpu, elements);
        ok = bench.query && bench.key && bench.value && bench.output;
    }

    double baseline_seconds = 0.0, candidate_seconds = 0.0;
    const char *sage_first_value = getenv("H3_SAGE_BENCH_SAGE_FIRST");
    const int sage_first = sage_first_value && *sage_first_value &&
                           strcmp(sage_first_value, "0");
    if (ok && sage_first)
        ok = run_path(&bench, "sage-e33", iterations, &candidate_seconds,
                      candidate, elements) &&
             run_path(&bench, "rocblas", iterations, &baseline_seconds,
                      baseline, elements);
    else if (ok)
        ok = run_path(&bench, "rocblas", iterations, &baseline_seconds,
                      baseline, elements) &&
             run_path(&bench, "sage-e33", iterations, &candidate_seconds,
                      candidate, elements);
    if (ok) {
        double ignored = 0.0;
        ok = run_path(&bench, "sage-e33", 1, &ignored, repeat, elements) &&
             memcmp(candidate, repeat, elements * sizeof(*candidate)) == 0;
    }

    float max_absolute = 0.0f;
    double error2 = 0.0, reference2 = 0.0, candidate2 = 0.0, dot = 0.0;
    size_t nonfinite = 0;
    if (ok)
        for (size_t index = 0; index < elements; ++index) {
            const float reference = f32(baseline[index]);
            const float observed = f32(candidate[index]);
            const float difference = observed - reference;
            if (!isfinite(observed)) ++nonfinite;
            max_absolute = fmaxf(max_absolute, fabsf(difference));
            error2 += (double)difference * difference;
            reference2 += (double)reference * reference;
            candidate2 += (double)observed * observed;
            dot += (double)reference * observed;
        }
    const double relative = reference2 > 0.0 ?
        sqrt(error2 / reference2) : INFINITY;
    const double cosine = reference2 > 0.0 && candidate2 > 0.0 ?
        dot / sqrt(reference2 * candidate2) : 0.0;
    h3_gpu_stats stats = {0};
    if (bench.gpu) (void)h3_gpu_get_stats(bench.gpu, &stats);
    if (ok) {
        printf("H3 dense Sage attention: sequence=%u heads=%u warmup=%u "
               "iterations=%u order=%s rocblas=%.9fs sage=%.9fs "
               "speedup=%.6fx max-abs=%.9g rel-L2=%.9g "
               "cosine=%.12f nonfinite=%zu rocblas-hash=%016llx "
               "sage-hash=%016llx repeat=bitwise peak-live=%llu\n",
               sequence, heads, WARMUP_ITERATIONS, iterations,
               sage_first ? "S-B" : "B-S",
               baseline_seconds / iterations,
               candidate_seconds / iterations,
               candidate_seconds > 0.0 ?
                   baseline_seconds / candidate_seconds : 0.0,
               (double)max_absolute, relative, cosine, nonfinite,
               (unsigned long long)hash_bytes(
                   baseline, elements * sizeof(*baseline)),
               (unsigned long long)hash_bytes(
                   candidate, elements * sizeof(*candidate)),
               (unsigned long long)stats.peak_live_bytes);
        if (nonfinite) ok = 0;
    } else {
        fprintf(stderr, "dense Sage attention benchmark failed: %s\n",
                bench.gpu ? h3_gpu_error(bench.gpu) : error);
    }
    unsetenv("H3_BF16_SDPA");
    h3_gpu_tensor_free(bench.output);
    h3_gpu_tensor_free(bench.value);
    h3_gpu_tensor_free(bench.key);
    h3_gpu_tensor_free(bench.query);
    h3_gpu_free(bench.gpu);
    free(repeat); free(candidate); free(baseline); free(input);
    return ok ? 0 : 1;
}
