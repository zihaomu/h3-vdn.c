#include "h3_gpu.h"
#include "h3_vdn_weights.h"
#include "h3_weights.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        ok = 0;                                                              \
        goto cleanup;                                                        \
    }                                                                        \
} while (0)

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000))
        bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    else if (bits & UINT32_C(0xffff))
        bits |= UINT32_C(0x10000);
    return (uint16_t)(bits >> 16);
}

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int read_element(const h3_weight_store *store, const char *name,
                        size_t index, uint16_t *value) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor || !header || tensor->dtype != H3_DTYPE_BF16 ||
        index >= h3_st_tensor_elements(tensor)) return 0;
    int descriptor = open(header->path, O_RDONLY);
    if (descriptor < 0) return 0;
    ssize_t count = pread(descriptor, value, sizeof(*value),
                          (off_t)(tensor->file_offset + index * sizeof(*value)));
    close(descriptor);
    return count == (ssize_t)sizeof(*value);
}

static int merge_first(const h3_weight_store *adapter, const char *target,
                       const char *adapter_name, uint32_t input_dim,
                       uint32_t rank, float *weight) {
    char a_name[288];
    char b_name[288];
    snprintf(a_name, sizeof(a_name), "%s.lora_A.%s.weight",
             target, adapter_name);
    snprintf(b_name, sizeof(b_name), "%s.lora_B.%s.weight",
             target, adapter_name);
    float sum = 0.0f;
    for (uint32_t inner = 0; inner < rank; inner++) {
        uint16_t a;
        uint16_t b;
        if (!read_element(adapter, a_name, (size_t)inner * input_dim, &a) ||
            !read_element(adapter, b_name, inner, &b)) return 0;
        sum += from_bf16(a) * from_bf16(b);
    }
    *weight = from_bf16(to_bf16(*weight + sum));
    return 1;
}

static int expected_first(const h3_weight_store *base,
                          const char *base_name,
                          const h3_weight_store *first,
                          const char *first_name,
                          const h3_weight_store *second,
                          const char *second_name,
                          const char *target, uint32_t input_dim,
                          uint32_t first_rank, uint32_t second_rank,
                          float *result) {
    uint16_t value;
    if (!read_element(base, base_name, 0, &value)) return 0;
    *result = from_bf16(value);
    if (first && !merge_first(first, target, first_name, input_dim,
                              first_rank, result)) return 0;
    if (second && !merge_first(second, target, second_name, input_dim,
                               second_rank, result)) return 0;
    return 1;
}

static char *path_join(const char *root, const char *relative) {
    size_t length = strlen(root) + strlen(relative) + 2;
    char *path = malloc(length);
    if (path) snprintf(path, length, "%s/%s", root, relative);
    return path;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s H3_BASE STAGE_DMD\n", argv[0]);
        return 2;
    }
    int ok = 1;
    char error[512];
    h3_gpu *gpu = NULL;
    h3_vdn_weight_store *vdn = NULL;
    h3_weight_store *base = NULL;
    h3_weight_store *default_adapter = NULL;
    h3_weight_store *turbo_adapter = NULL;
    h3_vdn_block_weights block;
    h3_vdn_model_weights model;
    memset(&block, 0, sizeof(block));
    memset(&model, 0, sizeof(model));
    char *base_path = path_join(argv[1], "transformer");
    char *default_path = path_join(argv[2], "adapters/default");
    char *turbo_path = path_join(argv[2], "adapters/turbo");
    CHECK(base_path && default_path && turbo_path);

    gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "cannot create HIP context: %s\n", error);
        ok = 0;
        goto cleanup;
    }
    vdn = h3_vdn_weight_store_open(argv[1], argv[2], 1,
                                   error, sizeof(error));
    if (!vdn) {
        fprintf(stderr, "cannot open VDN stores: %s\n", error);
        ok = 0;
        goto cleanup;
    }
    if (!h3_vdn_block_weights_load(vdn, gpu, 0, &block,
                                   error, sizeof(error))) {
        fprintf(stderr, "cannot load VDN block 0: %s\n", error);
        ok = 0;
        goto cleanup;
    }
    CHECK(h3_gpu_tensor_elements(block.q) == (size_t)7168 * 5376);
    CHECK(h3_gpu_tensor_elements(block.fc1) == (size_t)28672 * 5376);
    CHECK(h3_gpu_tensor_elements(block.adaln_weight) ==
          (size_t)96768 * 2688);
    CHECK(h3_gpu_tensor_elements(block.linear.to_out) ==
          (size_t)5376 * 7168);
    CHECK(h3_gpu_tensor_dtype(block.q) == H3_GPU_BF16);

    h3_gpu_stats stats;
    CHECK(h3_gpu_get_stats(gpu, &stats));
    CHECK(stats.live_bytes > UINT64_C(1000000000));
    CHECK(stats.live_bytes < UINT64_C(2000000000));
    printf("VDN block 0 resident weights: %.3f GiB (peak %.3f GiB)\n",
           (double)stats.live_bytes / (1024.0 * 1024.0 * 1024.0),
           (double)stats.peak_live_bytes / (1024.0 * 1024.0 * 1024.0));

    base = h3_weight_store_open(base_path, error, sizeof(error));
    default_adapter = h3_weight_store_open(
        default_path, error, sizeof(error));
    turbo_adapter = h3_weight_store_open(turbo_path, error, sizeof(error));
    CHECK(base && default_adapter && turbo_adapter);

    float expected;
    uint16_t actual_bits;
    CHECK(expected_first(
        base, "transformer_blocks.0.attn.to_q.weight",
        default_adapter, "default", turbo_adapter, "turbo",
        "transformer_blocks.0.attn.orig.to_q", 5376, 64, 64, &expected));
    CHECK(h3_gpu_tensor_read_bf16(block.q, &actual_bits, 1));
    CHECK(fabsf(from_bf16(actual_bits) - expected) <= 0.015625f);

    CHECK(expected_first(
        base, "transformer_blocks.0.adaln_proj.linear.weight",
        NULL, NULL, turbo_adapter, "turbo",
        "transformer_blocks.0.adaln_proj.linear", 2688, 0, 16, &expected));
    CHECK(h3_gpu_tensor_read_bf16(block.adaln_weight, &actual_bits, 1));
    CHECK(fabsf(from_bf16(actual_bits) - expected) <= 0.015625f);

    h3_vdn_block_weights_free(&block);
    h3_vdn_weight_cache_stats cache_stats;
    CHECK(h3_vdn_weight_store_cache_stats(vdn, &cache_stats));
    if (cache_stats.budget_bytes) {
        CHECK(cache_stats.resident_blocks == 1);
        CHECK(cache_stats.hits == 0 && cache_stats.misses == 1);
        CHECK(h3_vdn_block_weights_load(
            vdn, gpu, 0, &block, error, sizeof(error)));
        CHECK(h3_vdn_weight_store_cache_stats(vdn, &cache_stats));
        CHECK(cache_stats.hits == 1 && cache_stats.misses == 1);
        CHECK(h3_gpu_tensor_read_bf16(block.q, &actual_bits, 1));
        h3_vdn_block_weights_free(&block);
    }
    CHECK(h3_vdn_model_weights_load(vdn, gpu, &model, error, sizeof(error)));
    CHECK(h3_gpu_tensor_elements(model.context_weight) ==
          (size_t)5376 * 5120);
    CHECK(h3_gpu_tensor_dtype(model.video_in_weight) == H3_GPU_F32);
    CHECK(h3_gpu_tensor_elements(model.refiner[1].fc1) ==
          (size_t)28672 * 5376);
    CHECK(h3_gpu_tensor_elements(model.final_adaln_weight) ==
          (size_t)10752 * 2688);
    CHECK(expected_first(
        base, "token_refiner.refiner_blocks.0.attn.to_q.weight",
        default_adapter, "default", turbo_adapter, "turbo",
        "token_refiner.refiner_blocks.0.attn.to_q", 5376, 64, 64,
        &expected));
    CHECK(h3_gpu_tensor_read_bf16(model.refiner[0].q, &actual_bits, 1));
    CHECK(fabsf(from_bf16(actual_bits) - expected) <= 0.015625f);
    CHECK(expected_first(
        base, "norm_out.linear.weight", NULL, NULL,
        turbo_adapter, "turbo", "norm_out.linear", 2688, 0, 16,
        &expected));
    CHECK(h3_gpu_tensor_read_bf16(model.final_adaln_weight, &actual_bits, 1));
    CHECK(fabsf(from_bf16(actual_bits) - expected) <= 0.015625f);
    CHECK(h3_gpu_get_stats(gpu, &stats));
    printf("VDN top-level + refiner resident weights: %.3f GiB\n",
           (double)stats.live_bytes / (1024.0 * 1024.0 * 1024.0));

cleanup:
    h3_weight_store_free(turbo_adapter);
    h3_weight_store_free(default_adapter);
    h3_weight_store_free(base);
    h3_vdn_model_weights_free(&model);
    h3_vdn_block_weights_free(&block);
    h3_vdn_weight_store_free(vdn);
    vdn = NULL;
    if (gpu) {
        h3_gpu_stats final_stats;
        if (!h3_gpu_get_stats(gpu, &final_stats) || final_stats.live_bytes != 0) {
            fprintf(stderr, "VDN block tensors leaked GPU memory\n");
            ok = 0;
        }
    }
    h3_gpu_free(gpu);
    free(turbo_path);
    free(default_path);
    free(base_path);
    if (!ok) return 1;
    puts("VDN real block loader and default+turbo LoRA parity passed");
    return 0;
}
