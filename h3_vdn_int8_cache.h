#ifndef H3_VDN_INT8_CACHE_H
#define H3_VDN_INT8_CACHE_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

#define H3_VDN_INT8_CACHE_BLOCKS 50
#define H3_VDN_INT8_CACHE_DIGEST_BYTES 32

typedef struct h3_vdn_int8_cache h3_vdn_int8_cache;

typedef struct {
    uint64_t block_bytes[H3_VDN_INT8_CACHE_BLOCKS];
    uint8_t block_sha256[H3_VDN_INT8_CACHE_BLOCKS]
                        [H3_VDN_INT8_CACHE_DIGEST_BYTES];
} h3_vdn_int8_cache_manifest;

typedef struct {
    h3_gpu_tensor *adaln_weight;
    h3_gpu_tensor *adaln_scales;
    h3_gpu_tensor *fc1_weight;
    h3_gpu_tensor *fc1_scales;
    h3_gpu_tensor *fc2_weight;
    h3_gpu_tensor *fc2_scales;
} h3_vdn_int8_block;

int h3_vdn_int8_cache_read_manifest(
    const char *directory, int use_turbo,
    const uint8_t source_sha256[H3_VDN_INT8_CACHE_DIGEST_BYTES],
    h3_vdn_int8_cache_manifest *manifest,
    char *error, size_t error_size);

h3_vdn_int8_cache *h3_vdn_int8_cache_open(
    const char *directory, int use_turbo,
    const uint8_t source_sha256[H3_VDN_INT8_CACHE_DIGEST_BYTES],
    char *error, size_t error_size);
void h3_vdn_int8_cache_free(h3_vdn_int8_cache *cache);

int h3_vdn_int8_cache_load_block(
    const h3_vdn_int8_cache *cache, h3_gpu *gpu, unsigned block,
    h3_vdn_int8_block *weights, char *error, size_t error_size);
void h3_vdn_int8_block_free(h3_vdn_int8_block *weights);

int h3_vdn_int8_cache_write_block(
    const char *directory, unsigned block,
    const h3_vdn_int8_block *weights, uint64_t *file_bytes,
    uint8_t file_sha256[H3_VDN_INT8_CACHE_DIGEST_BYTES],
    char *error, size_t error_size);
int h3_vdn_int8_cache_write_manifest(
    const char *directory, int use_turbo,
    const uint8_t source_sha256[H3_VDN_INT8_CACHE_DIGEST_BYTES],
    const h3_vdn_int8_cache_manifest *manifest,
    char *error, size_t error_size);

#endif
