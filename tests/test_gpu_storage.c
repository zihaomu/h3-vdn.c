#include "h3_backend.h"
#include "h3_gpu.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        goto failed; \
    } \
} while (0)

static int same_f32(const float *left, const float *right, size_t count) {
    for (size_t index = 0; index < count; index++)
        if (left[index] != right[index]) return 0;
    return 1;
}

int main(void) {
    char error[512] = {0};
    h3_device_info device;
    h3_gpu *gpu = NULL;
    h3_gpu_tensor *first = NULL;
    h3_gpu_tensor *second = NULL;
    h3_gpu_tensor *bf16 = NULL;
    h3_gpu_tensor *loaded = NULL;
    char path[] = "/tmp/h3-gpu-storage-XXXXXX";
    int descriptor = -1;
    int result = 1;

    CHECK(h3_backend_probe(0, &device, error, sizeof(error)));
    gpu = h3_gpu_create(NULL, error, sizeof(error));
    CHECK(gpu != NULL);

    const float values[] = {1.25f, -2.5f, 3.75f, 9.0f};
    float got[4] = {0};
    first = h3_gpu_tensor_from_f32(gpu, values, 4);
    second = h3_gpu_tensor_new_f32(gpu, 4);
    CHECK(first && second);
    CHECK(h3_gpu_tensor_elements(first) == 4);
    CHECK(h3_gpu_tensor_dtype(first) == H3_GPU_F32);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_copy_f32(gpu, second, 0, first, 0, 4));
    CHECK(h3_gpu_continue(gpu));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(second, got, 4));
    CHECK(same_f32(values, got, 4));

    const float replacement[] = {7.0f, 8.0f};
    CHECK(h3_gpu_tensor_write_f32_range(second, 1, replacement, 2));
    CHECK(h3_gpu_tensor_read_f32(second, got, 4));
    const float expected[] = {1.25f, 7.0f, 8.0f, 9.0f};
    CHECK(same_f32(expected, got, 4));

    const uint16_t bf16_values[] = {0x3f80, 0xc020, 0x4070, 0x4110};
    uint16_t bf16_got[4] = {0};
    bf16 = h3_gpu_tensor_from_bf16(gpu, bf16_values, 4);
    CHECK(bf16 != NULL);
    CHECK(h3_gpu_tensor_read_bf16(bf16, bf16_got, 4));
    CHECK(!memcmp(bf16_values, bf16_got, sizeof(bf16_values)));

    descriptor = mkstemp(path);
    CHECK(descriptor >= 0);
    const uint32_t prefix = UINT32_C(0x12345678);
    CHECK(write(descriptor, &prefix, sizeof(prefix)) == (ssize_t)sizeof(prefix));
    CHECK(write(descriptor, bf16_values, sizeof(bf16_values)) ==
          (ssize_t)sizeof(bf16_values));
    CHECK(close(descriptor) == 0);
    descriptor = -1;
    loaded = h3_gpu_tensor_load_bf16(gpu, path, sizeof(prefix), 4);
    CHECK(loaded != NULL);
    memset(bf16_got, 0, sizeof(bf16_got));
    CHECK(h3_gpu_tensor_read_bf16(loaded, bf16_got, 4));
    CHECK(!memcmp(bf16_values, bf16_got, sizeof(bf16_values)));

    h3_gpu_stats stats;
    CHECK(h3_gpu_get_stats(gpu, &stats));
    CHECK(stats.tensor_allocations == 4);
    CHECK(stats.live_bytes == 4 * sizeof(float) * 2 +
                              4 * sizeof(uint16_t) * 2);
    CHECK(stats.peak_live_bytes == stats.live_bytes);
    CHECK(stats.blit_copies == 1);
    result = 0;

failed:
    if (result && gpu) fprintf(stderr, "GPU error: %s\n", h3_gpu_error(gpu));
    if (descriptor >= 0) close(descriptor);
    unlink(path);
    h3_gpu_tensor_free(loaded);
    h3_gpu_tensor_free(bf16);
    h3_gpu_tensor_free(second);
    h3_gpu_tensor_free(first);
    h3_gpu_free(gpu);
    if (!result) puts("GPU storage tests passed");
    return result;
}
