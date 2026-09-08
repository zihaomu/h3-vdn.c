#ifndef H3_VDN_SAGE_BRIDGE_H
#define H3_VDN_SAGE_BRIDGE_H

#include <hip/hip_runtime_api.h>

#include <stddef.h>
#include <stdint.h>

/* H3-owned boundary around the C++ SageAttention-AMD API. This header keeps
 * third-party types out of H3's public C headers and contains no planner or
 * kernel implementation. */
typedef struct {
    uint32_t sequence;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t video_start;
    uint32_t frames;
    uint32_t tokens_per_frame;
    uint32_t radius;
    uint32_t chunk;
    int anchor_both;
} h3_vdn_sage_bridge_geometry;

typedef struct {
    const void *query_bf16;
    const void *key_bf16;
    const void *value_bf16;
    void *output_bf16;
    h3_vdn_sage_bridge_geometry geometry;
    float scale;
    void *workspace;
    size_t workspace_bytes;
    hipStream_t stream;
} h3_vdn_sage_bridge_params;

hipError_t h3_vdn_sage_bridge_workspace_size(
    const h3_vdn_sage_bridge_geometry *geometry, size_t *bytes);

hipError_t h3_vdn_sage_bridge_prepare(
    const h3_vdn_sage_bridge_geometry *geometry, void *workspace,
    size_t workspace_bytes, hipStream_t stream);

hipError_t h3_vdn_sage_bridge_launch_prepared(
    const h3_vdn_sage_bridge_params *params);

#endif
