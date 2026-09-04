#ifndef H3_VDN_SAGE_GFX12_H
#define H3_VDN_SAGE_GFX12_H

#include "h3_vdn_sage.h"

#include <hip/hip_runtime_api.h>

#include <stdint.h>

hipError_t h3_vdn_sage_gfx12_launch_i8_bf16(
    const int8_t *query, const int8_t *key,
    const float *query_scales, const float *key_scales,
    const void *value_bf16, void *output_bf16,
    const h3_vdn_q_task *tasks, uint32_t task_count,
    uint32_t sequence, uint32_t heads,
    uint32_t query_groups, uint32_t key_groups,
    float scale, hipStream_t stream);

#endif
