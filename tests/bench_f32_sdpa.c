#include "h3_gpu.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HEAD_DIM = 64 };

static double monotonic_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1.0e9;
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

static int run(h3_gpu *gpu, h3_gpu_tensor *output,
               const h3_gpu_tensor *query, const h3_gpu_tensor *key,
               const h3_gpu_tensor *value, uint32_t sequence,
               uint32_t heads, uint32_t iterations, double *seconds) {
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    if (!h3_gpu_begin(gpu) ||
        !h3_gpu_sdpa_f32(gpu, output, query, key, value, sequence, heads,
                         HEAD_DIM, scale) ||
        !h3_gpu_submit(gpu)) return 0;
    double start = monotonic_seconds();
    for (uint32_t iteration = 0; iteration < iterations; iteration++) {
        if (!h3_gpu_begin(gpu) ||
            !h3_gpu_sdpa_f32(gpu, output, query, key, value, sequence, heads,
                             HEAD_DIM, scale) ||
            !h3_gpu_submit(gpu)) return 0;
    }
    *seconds = monotonic_seconds() - start;
    return 1;
}

static int run_path(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t iterations, int wave32,
                    double *seconds, float *result, size_t elements) {
    setenv("H3_F32_SDPA_WAVE32", wave32 ? "1" : "0", 1);
    return run(gpu, output, query, key, value, sequence, heads, iterations,
               seconds) && h3_gpu_tensor_read_f32(output, result, elements);
}

int main(int argc, char **argv) {
    uint32_t sequence = 2273;
    uint32_t heads = 32;
    uint32_t iterations = 3;
    if (argc != 1 && argc != 4) {
        fprintf(stderr, "usage: %s [SEQUENCE HEADS ITERATIONS]\n", argv[0]);
        return 2;
    }
    if (argc == 4 &&
        (!parse_u32(argv[1], &sequence) || !parse_u32(argv[2], &heads) ||
         !parse_u32(argv[3], &iterations))) {
        fprintf(stderr, "invalid unsigned benchmark argument\n");
        return 2;
    }
    uint64_t elements_wide = (uint64_t)sequence * heads * HEAD_DIM;
    if (elements_wide > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "F32 SDPA benchmark shape is too large\n");
        return 2;
    }
    size_t elements = (size_t)elements_wide;
    float *input = malloc(elements * sizeof(*input));
    float *scalar = malloc(elements * sizeof(*scalar));
    float *wave = malloc(elements * sizeof(*wave));
    if (!input || !scalar || !wave) {
        fprintf(stderr, "out of host memory for F32 SDPA benchmark\n");
        free(wave); free(scalar); free(input);
        return 1;
    }
    for (size_t index = 0; index < elements; index++) {
        int centered = (int)(index % 509) - 254;
        input[index] = (float)centered * 0.0009765625f;
    }

    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    h3_gpu_tensor *query = NULL, *key = NULL, *value = NULL, *output = NULL;
    int ok = gpu != NULL;
    if (ok) {
        query = h3_gpu_tensor_from_f32(gpu, input, elements);
        key = h3_gpu_tensor_from_f32(gpu, input, elements);
        value = h3_gpu_tensor_from_f32(gpu, input, elements);
        output = h3_gpu_tensor_new_f32(gpu, elements);
        ok = query && key && value && output;
    }
    double scalar_seconds = 0.0, wave_seconds = 0.0;
    const char *wave_first_value = getenv("H3_SDPA_BENCH_WAVE_FIRST");
    int wave_first = wave_first_value && *wave_first_value &&
                     strcmp(wave_first_value, "0");
    if (ok && wave_first)
        ok = run_path(gpu, output, query, key, value, sequence, heads,
                      iterations, 1, &wave_seconds, wave, elements) &&
             run_path(gpu, output, query, key, value, sequence, heads,
                      iterations, 0, &scalar_seconds, scalar, elements);
    else if (ok)
        ok = run_path(gpu, output, query, key, value, sequence, heads,
                      iterations, 0, &scalar_seconds, scalar, elements) &&
             run_path(gpu, output, query, key, value, sequence, heads,
                      iterations, 1, &wave_seconds, wave, elements);
    size_t mismatches = 0;
    float max_abs = 0.0f;
    double squared_error = 0.0;
    if (ok) for (size_t index = 0; index < elements; index++) {
        if (memcmp(&scalar[index], &wave[index], sizeof(float))) mismatches++;
        float difference = fabsf(scalar[index] - wave[index]);
        if (difference > max_abs) max_abs = difference;
        squared_error += (double)difference * difference;
    }
    if (ok) {
        printf("F32 D64 SDPA: sequence=%u heads=%u iterations=%u "
               "scalar=%.6fs wave32=%.6fs speedup=%.3fx mismatches=%zu "
               "max_abs=%.9g rmse=%.9g scalar_hash=%016llx "
               "wave_hash=%016llx\n",
               sequence, heads, iterations,
               scalar_seconds / iterations, wave_seconds / iterations,
               wave_seconds > 0.0 ? scalar_seconds / wave_seconds : 0.0,
               mismatches, max_abs, sqrt(squared_error / (double)elements),
               (unsigned long long)hash_bytes(
                   scalar, elements * sizeof(*scalar)),
               (unsigned long long)hash_bytes(wave, elements * sizeof(*wave)));
        if (mismatches) ok = 0;
    } else {
        fprintf(stderr, "F32 SDPA benchmark failed: %s\n",
                error[0] ? error : h3_gpu_error(gpu));
    }
    unsetenv("H3_F32_SDPA_WAVE32");
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(query);
    h3_gpu_free(gpu);
    free(wave); free(scalar); free(input);
    return ok ? 0 : 1;
}
