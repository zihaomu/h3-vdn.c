#ifndef H3_SAGE_DENSE_BRIDGE_H
#define H3_SAGE_DENSE_BRIDGE_H

#include <hip/hip_runtime_api.h>

#include <stddef.h>
#include <stdint.h>

/* H3-owned C boundary for SageAttention-AMD's generic dense E33 path.
 * The bridge owns no device memory and keeps C++ Sage types out of H3's
 * public GPU interface. */
typedef struct {
    uint32_t sequence;
    uint32_t heads;
    uint32_t head_dim;
} h3_sage_dense_geometry;

typedef struct {
    const void *query_bf16;
    const void *key_bf16;
    const void *value_bf16;
    void *output_bf16;
    h3_sage_dense_geometry geometry;
    float scale;
    void *workspace;
    size_t workspace_bytes;
    hipStream_t stream;
} h3_sage_dense_params;

hipError_t h3_sage_dense_workspace_size(
    const h3_sage_dense_geometry *geometry, size_t *bytes);

hipError_t h3_sage_dense_prepare(
    const h3_sage_dense_geometry *geometry, void *workspace,
    size_t workspace_bytes, hipStream_t stream);

hipError_t h3_sage_dense_launch_prepared(
    const h3_sage_dense_params *params);

#endif
