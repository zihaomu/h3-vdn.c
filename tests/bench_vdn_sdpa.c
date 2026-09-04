#include "h3_gpu.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HEADS = 56, HEAD_DIM = 128, FRAMES = 17 };

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

static uint64_t hash_bf16(const uint16_t *values, size_t count) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < count; index++) {
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
    if (errno || end == text || *end || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

int main(int argc, char **argv) {
    uint32_t sequence = 5338;
    uint32_t video_start = 986;
    uint32_t tokens_per_frame = 256;
    uint32_t iterations = 3;
    if (argc != 1 && argc != 5) {
        fprintf(stderr, "usage: %s [SEQUENCE VIDEO_START TOKENS_PER_FRAME ITERATIONS]\n",
                argv[0]);
        return 2;
    }
    if (argc == 5 &&
        (!parse_u32(argv[1], &sequence) ||
         !parse_u32(argv[2], &video_start) ||
         !parse_u32(argv[3], &tokens_per_frame) ||
         !parse_u32(argv[4], &iterations))) {
        fprintf(stderr, "invalid unsigned benchmark argument\n");
        return 2;
    }
    uint64_t video_rows = (uint64_t)FRAMES * tokens_per_frame;
    uint64_t elements_wide = (uint64_t)sequence * HEADS * HEAD_DIM;
    if (!sequence || !tokens_per_frame || !iterations ||
        video_start > sequence || video_rows > sequence - video_start ||
        elements_wide > SIZE_MAX) {
        fprintf(stderr, "invalid VDN SDPA benchmark shape\n");
        return 2;
    }
    size_t elements = (size_t)elements_wide;
    uint16_t *values = malloc(elements * sizeof(*values));
    uint16_t *result = malloc(elements * sizeof(*result));
    if (!values || !result) {
        fprintf(stderr, "out of host memory for VDN SDPA benchmark\n");
        free(result);
        free(values);
        return 1;
    }
    for (size_t index = 0; index < elements; index++) {
        int centered = (int)(index % 257) - 128;
        values[index] = bf16((float)centered * 0.001f);
    }

    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    h3_gpu_tensor *query = NULL, *key = NULL, *value = NULL, *output = NULL;
    int ok = gpu != NULL;
    if (ok) {
        h3_gpu_profile_set_label(gpu, "VDN SDPA benchmark");
        query = h3_gpu_tensor_from_bf16(gpu, values, elements);
        key = h3_gpu_tensor_from_bf16(gpu, values, elements);
        value = h3_gpu_tensor_from_bf16(gpu, values, elements);
        output = h3_gpu_tensor_new_bf16(gpu, elements);
        ok = query && key && value && output;
    }
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    if (ok)
        ok = h3_gpu_begin(gpu) && h3_gpu_vdn_window_sdpa_bf16(
                 gpu, output, query, key, value, sequence, HEADS, HEAD_DIM,
                 video_start, FRAMES, tokens_per_frame, 1, 5, 1, scale) &&
             h3_gpu_submit(gpu);
    double start = monotonic_seconds();
    for (uint32_t iteration = 0; iteration < iterations && ok; iteration++)
        ok = h3_gpu_begin(gpu) && h3_gpu_vdn_window_sdpa_bf16(
                 gpu, output, query, key, value, sequence, HEADS, HEAD_DIM,
                 video_start, FRAMES, tokens_per_frame, 1, 5, 1, scale) &&
             h3_gpu_submit(gpu);
    double seconds = monotonic_seconds() - start;
    if (ok) ok = h3_gpu_tensor_read_bf16(output, result, elements);
    if (ok) {
        printf("VDN window SDPA: sequence=%u video_start=%u frames=%u "
               "tokens_per_frame=%u heads=%u dim=%u iterations=%u "
               "total=%.6fs average=%.6fs hash=%016llx\n",
               sequence, video_start, FRAMES, tokens_per_frame, HEADS,
               HEAD_DIM, iterations, seconds, seconds / iterations,
               (unsigned long long)hash_bf16(result, elements));
    } else {
        fprintf(stderr, "VDN SDPA benchmark failed: %s\n",
                error[0] ? error : h3_gpu_error(gpu));
    }
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(query);
    h3_gpu_free(gpu);
    free(result);
    free(values);
    return ok ? 0 : 1;
}
