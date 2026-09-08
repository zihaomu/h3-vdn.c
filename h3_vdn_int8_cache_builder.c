#include "h3_gpu.h"
#include "h3_sha256.h"
#include "h3_vdn_int8_cache.h"
#include "h3_vdn_weights.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum {
    ADALN_OUT = 96768,
    ADALN_IN = 2688,
    FC1_OUT = 28672,
    FC1_IN = 5376,
    FC2_OUT = 5376,
    FC2_IN = 14336
};

static double monotonic_seconds(void) {
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0) return 0.0;
    return (double)time.tv_sec + (double)time.tv_nsec / 1e9;
}

static int enabled_environment(const char *name) {
    const char *value = getenv(name);
    return value && *value && strcmp(value, "0");
}

static int normalize_output(const char *input, char output[4096]) {
    if (!input || !*input) return 0;
    size_t length = strlen(input);
    while (length > 1 && input[length - 1] == '/') length--;
    if (length == 1 && input[0] == '/') return 0;
    if (length >= 4096) return 0;
    memcpy(output, input, length);
    output[length] = '\0';
    return 1;
}

static int parent_directory(const char *path, char output[4096]) {
    size_t length = strlen(path);
    if (length >= 4096) return 0;
    memcpy(output, path, length + 1);
    char *slash = strrchr(output, '/');
    if (!slash) {
        memcpy(output, ".", 2);
    } else if (slash == output) {
        slash[1] = '\0';
    } else {
        *slash = '\0';
    }
    return 1;
}

static int sync_parent(const char *path, char *error, size_t error_size) {
    char parent[4096];
    if (!parent_directory(path, parent)) {
        snprintf(error, error_size, "cache parent path is too long");
        return 0;
    }
    int descriptor = open(parent, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (descriptor < 0) {
        snprintf(error, error_size, "cannot open cache parent %s: %s",
                 parent, strerror(errno));
        return 0;
    }
    int ok = fsync(descriptor) == 0;
    if (!ok)
        snprintf(error, error_size, "cannot fsync cache parent %s: %s",
                 parent, strerror(errno));
    close(descriptor);
    return ok;
}

static int publish_directory(const char *temporary, const char *output,
                             char *error, size_t error_size) {
#ifdef __linux__
    if (renameat2(AT_FDCWD, temporary, AT_FDCWD, output,
                  RENAME_NOREPLACE) == 0) return 1;
#else
    struct stat status;
    if (lstat(output, &status) != 0 && errno == ENOENT &&
        rename(temporary, output) == 0) return 1;
#endif
    snprintf(error, error_size, "cannot atomically publish %s as %s: %s",
             temporary, output, strerror(errno));
    return 0;
}

static int allocate_quantized_block(h3_gpu *gpu,
                                    h3_vdn_int8_block *quantized) {
    memset(quantized, 0, sizeof(*quantized));
    quantized->adaln_weight = h3_gpu_tensor_new_i8(
        gpu, (size_t)ADALN_OUT * ADALN_IN);
    quantized->adaln_scales = h3_gpu_tensor_new_f32(gpu, ADALN_OUT);
    quantized->fc1_weight = h3_gpu_tensor_new_i8(
        gpu, (size_t)FC1_OUT * FC1_IN);
    quantized->fc1_scales = h3_gpu_tensor_new_f32(gpu, FC1_OUT);
    quantized->fc2_weight = h3_gpu_tensor_new_i8(
        gpu, (size_t)FC2_OUT * FC2_IN);
    quantized->fc2_scales = h3_gpu_tensor_new_f32(gpu, FC2_OUT);
    return quantized->adaln_weight && quantized->adaln_scales &&
           quantized->fc1_weight && quantized->fc1_scales &&
           quantized->fc2_weight && quantized->fc2_scales;
}

static int quantize_block(h3_gpu *gpu,
                          const h3_vdn_block_weights *source,
                          h3_vdn_int8_block *quantized) {
    return allocate_quantized_block(gpu, quantized) &&
           h3_gpu_begin(gpu) &&
           h3_gpu_quantize_weight_int8(
               gpu, quantized->adaln_weight, quantized->adaln_scales,
               source->adaln_weight, ADALN_OUT, ADALN_IN) &&
           h3_gpu_quantize_weight_int8(
               gpu, quantized->fc1_weight, quantized->fc1_scales,
               source->fc1, FC1_OUT, FC1_IN) &&
           h3_gpu_quantize_weight_int8(
               gpu, quantized->fc2_weight, quantized->fc2_scales,
               source->fc2, FC2_OUT, FC2_IN) &&
           h3_gpu_submit(gpu);
}

static int build_cache(const char *base_model_dir,
                       const char *checkpoint_dir,
                       const char *temporary_directory, int use_turbo,
                       char *error, size_t error_size) {
    h3_vdn_weight_store *store = h3_vdn_weight_store_open(
        base_model_dir, checkpoint_dir, use_turbo, error, error_size);
    if (!store) return 0;

    uint8_t source_sha256[H3_VDN_INT8_CACHE_DIGEST_BYTES];
    double source_started = monotonic_seconds();
    fprintf(stderr, "Hashing the effective INT8 source tensor set...\n");
    if (!h3_vdn_weight_store_int8_source_sha256(
            store, source_sha256, error, error_size)) {
        h3_vdn_weight_store_free(store);
        return 0;
    }
    char source_hex[65];
    h3_sha256_hex(source_sha256, source_hex);
    fprintf(stderr, "Source SHA-256: %s (%.3f s)\n", source_hex,
            monotonic_seconds() - source_started);

    h3_gpu *gpu = h3_gpu_create(NULL, error, error_size);
    if (!gpu) {
        h3_vdn_weight_store_free(store);
        return 0;
    }
    h3_vdn_int8_cache_manifest manifest = {0};
    int ok = 1;
    for (unsigned block = 0; ok && block < H3_VDN_INT8_CACHE_BLOCKS;
         block++) {
        double started = monotonic_seconds();
        h3_vdn_block_weights source = {0};
        h3_vdn_int8_block quantized = {0};
        ok = h3_vdn_block_weights_load(
                 store, gpu, block, &source, error, error_size) &&
             quantize_block(gpu, &source, &quantized);
        if (!ok && error && error_size && !error[0])
            snprintf(error, error_size, "cannot quantize VDN block %u: %s",
                     block, h3_gpu_error(gpu));
        if (ok)
            ok = h3_vdn_int8_cache_write_block(
                temporary_directory, block, &quantized,
                &manifest.block_bytes[block],
                manifest.block_sha256[block], error, error_size);
        h3_vdn_int8_block_free(&quantized);
        h3_vdn_block_weights_free(&source);
        if (ok)
            fprintf(stderr, "Cached block %02u/%02u: %.3f GiB in %.3f s\n",
                    block + 1, H3_VDN_INT8_CACHE_BLOCKS,
                    (double)manifest.block_bytes[block] /
                        (1024.0 * 1024.0 * 1024.0),
                    monotonic_seconds() - started);
    }
    h3_gpu_free(gpu);
    h3_vdn_weight_store_free(store);
    if (!ok) return 0;

    if (!h3_vdn_int8_cache_write_manifest(
            temporary_directory, use_turbo, source_sha256, &manifest,
            error, error_size)) return 0;
    fprintf(stderr, "Validating every cache file and tensor schema...\n");
    double validation_started = monotonic_seconds();
    h3_vdn_int8_cache *validation = h3_vdn_int8_cache_open(
        temporary_directory, use_turbo, source_sha256, error, error_size);
    if (!validation) return 0;
    h3_vdn_int8_cache_free(validation);
    fprintf(stderr, "Full cache validation passed in %.3f s\n",
            monotonic_seconds() - validation_started);
    return 1;
}

static int verify_cache(const char *base_model_dir,
                        const char *checkpoint_dir,
                        const char *cache_directory, int use_turbo,
                        char *error, size_t error_size) {
    h3_vdn_weight_store *store = h3_vdn_weight_store_open(
        base_model_dir, checkpoint_dir, use_turbo, error, error_size);
    if (!store) return 0;
    uint8_t source_sha256[H3_VDN_INT8_CACHE_DIGEST_BYTES];
    double source_started = monotonic_seconds();
    int ok = h3_vdn_weight_store_int8_source_sha256(
        store, source_sha256, error, error_size);
    h3_vdn_weight_store_free(store);
    if (!ok) return 0;
    char source_hex[65];
    h3_sha256_hex(source_sha256, source_hex);
    fprintf(stderr, "Source SHA-256: %s (%.3f s)\n", source_hex,
            monotonic_seconds() - source_started);

    double cache_started = monotonic_seconds();
    h3_vdn_int8_cache *cache = h3_vdn_int8_cache_open(
        cache_directory, use_turbo, source_sha256, error, error_size);
    if (!cache) return 0;
    h3_vdn_int8_cache_free(cache);
    fprintf(stderr, "VDN INT8 cache verification passed in %.3f s: %s\n",
            monotonic_seconds() - cache_started, cache_directory);
    return 1;
}

static int parse_turbo(const char *text, int *use_turbo) {
    if (!text || !use_turbo) return 0;
    if (!strcmp(text, "0")) {
        *use_turbo = 0;
        return 1;
    }
    if (!strcmp(text, "1")) {
        *use_turbo = 1;
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--verify")) {
        if (argc != 5 && argc != 6) {
            fprintf(stderr,
                    "usage: %s --verify BASE_MODEL_DIR CHECKPOINT_DIR "
                    "CACHE_DIR [USE_TURBO]\n",
                    argv[0]);
            return 2;
        }
        int use_turbo = 1;
        if (argc == 6 && !parse_turbo(argv[5], &use_turbo)) {
            fprintf(stderr, "USE_TURBO must be 0 or 1\n");
            return 2;
        }
        char error[1024] = {0};
        if (!verify_cache(argv[2], argv[3], argv[4], use_turbo,
                          error, sizeof(error))) {
            fprintf(stderr, "cache verification failed: %s\n",
                    error[0] ? error : "unknown error");
            return 1;
        }
        return 0;
    }
    if (argc != 4 && argc != 5) {
        fprintf(stderr,
                "usage: %s BASE_MODEL_DIR CHECKPOINT_DIR OUTPUT_CACHE_DIR "
                "[USE_TURBO]\n"
                "USE_TURBO defaults to 1 and must be 0 or 1.\n",
                argv[0]);
        return 2;
    }
    int use_turbo = 1;
    if (argc == 5 && !parse_turbo(argv[4], &use_turbo)) {
        fprintf(stderr, "USE_TURBO must be 0 or 1\n");
        return 2;
    }
    if (enabled_environment("H3_VDN_RESIDENT_GIB") ||
        enabled_environment("H3_VDN_INT8_CACHE")) {
        fprintf(stderr,
                "unset H3_VDN_RESIDENT_GIB and H3_VDN_INT8_CACHE while "
                "building a cache\n");
        return 2;
    }

    char output[4096], temporary[4096], error[1024] = {0};
    if (!normalize_output(argv[3], output)) {
        fprintf(stderr, "invalid OUTPUT_CACHE_DIR\n");
        return 2;
    }
    struct stat status;
    if (lstat(output, &status) == 0 || errno != ENOENT) {
        fprintf(stderr, "refusing to overwrite OUTPUT_CACHE_DIR: %s\n",
                output);
        return 2;
    }
    int length = snprintf(temporary, sizeof(temporary), "%s.tmp.%ld",
                          output, (long)getpid());
    if (length < 0 || (size_t)length >= sizeof(temporary)) {
        fprintf(stderr, "temporary cache path is too long\n");
        return 2;
    }
    if (mkdir(temporary, 0700) != 0) {
        fprintf(stderr, "cannot create temporary cache %s: %s\n",
                temporary, strerror(errno));
        return 1;
    }

    double started = monotonic_seconds();
    if (!build_cache(argv[1], argv[2], temporary, use_turbo,
                     error, sizeof(error))) {
        fprintf(stderr, "cache generation failed: %s\n",
                error[0] ? error : "unknown error");
        fprintf(stderr, "partial cache retained for inspection: %s\n",
                temporary);
        return 1;
    }
    if (!publish_directory(temporary, output, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    if (!sync_parent(output, error, sizeof(error))) {
        fprintf(stderr, "cache was published, but durability sync failed: %s\n",
                error);
        return 1;
    }
    fprintf(stderr, "Published VDN INT8 cache: %s (%.3f s total)\n",
            output, monotonic_seconds() - started);
    return 0;
}
