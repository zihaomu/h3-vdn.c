/* SPDX-License-Identifier: Apache-2.0
 *
 * H3 VDN SageAttention gfx12 hot kernel. The raw WMMA fragment packing was
 * independently adapted from the Apache-2.0 SageAttention gfx12 work at
 * thu-ml/SageAttention PR #368, commit
 * 66f5e64c9e36084c863a4480e570069245e58f90.
 */

#include "h3_vdn_sage_gfx12.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kHeadDim = 128;
constexpr int kTile = 16;
[[maybe_unused]] constexpr int kDepthTiles = kHeadDim / kTile;

using i32x2 = std::int32_t __attribute__((ext_vector_type(2)));
using i32x8 = std::int32_t __attribute__((ext_vector_type(8)));
using f32x8 = float __attribute__((ext_vector_type(8)));
using u16x8 = std::uint16_t __attribute__((ext_vector_type(8)));
using u32x4 = std::uint32_t __attribute__((ext_vector_type(4)));

[[maybe_unused]] __device__ __forceinline__ std::uint32_t pack_bf16_rne(
    float low, float high) {
    std::uint32_t low_bits = __builtin_bit_cast(std::uint32_t, low);
    std::uint32_t high_bits = __builtin_bit_cast(std::uint32_t, high);
    low_bits += 0x7fffu + ((low_bits >> 16) & 1u);
    high_bits += 0x7fffu + ((high_bits >> 16) & 1u);
    return (low_bits >> 16) | (high_bits & 0xffff0000u);
}

[[maybe_unused]] __device__ __forceinline__ i32x2 load_i8_fragment(
    const std::int8_t *__restrict__ matrix,
    std::uint32_t row,
    std::uint32_t head,
    std::uint32_t heads,
    int lane,
    int depth_base,
    bool valid) {
    i32x2 result = {0, 0};
    if (!valid) return result;
    const int k_base = 8 * (lane >> 4);
    const std::size_t offset =
        (static_cast<std::size_t>(row) * heads + head) * kHeadDim +
        depth_base + k_base;
    const uint2 packed =
        *reinterpret_cast<const uint2 *>(matrix + offset);
    result[0] = static_cast<std::int32_t>(packed.x);
    result[1] = static_cast<std::int32_t>(packed.y);
    return result;
}

__device__ __forceinline__ int wmma_bf16_k_for_lane_element(int lane,
                                                             int element) {
    const int reg = element >> 1;
    const int half = element & 1;
    return ((reg >> 1) << 3) + (((lane >> 4) & 1) << 2) +
           ((reg & 1) << 1) + half;
}

[[maybe_unused]] __device__ __forceinline__ u16x8
make_probability_fragment(u32x4 words, int lane) {
    const bool upper_lane = lane >= 16;
    const std::uint32_t first = upper_lane ? words[2] : words[0];
    const std::uint32_t second = upper_lane ? words[3] : words[1];
    const std::uint32_t peer_source_first =
        upper_lane ? words[0] : words[2];
    const std::uint32_t peer_source_second =
        upper_lane ? words[1] : words[3];
    const std::uint32_t peer_first =
        __shfl_xor(peer_source_first, 16, 32);
    const std::uint32_t peer_second =
        __shfl_xor(peer_source_second, 16, 32);
    u32x4 result = {upper_lane ? peer_first : first,
                    upper_lane ? peer_second : second,
                    upper_lane ? first : peer_first,
                    upper_lane ? second : peer_second};
    return __builtin_bit_cast(u16x8, result);
}

[[maybe_unused]] __device__ __forceinline__ u16x8 load_value_fragment(
    const hip_bfloat16 *__restrict__ value,
    std::uint32_t key_tile,
    std::uint32_t head,
    std::uint32_t sequence,
    std::uint32_t heads,
    int depth_tile,
    int lane) {
    u16x8 result;
    const std::uint32_t dimension =
        static_cast<std::uint32_t>(depth_tile * kTile + (lane & 15));
#pragma unroll
    for (int element = 0; element < 8; ++element) {
        const std::uint32_t key =
            key_tile + static_cast<std::uint32_t>(
                           wmma_bf16_k_for_lane_element(lane, element));
        result[element] =
            key < sequence
                ? value[(static_cast<std::size_t>(key) * heads + head) *
                            kHeadDim +
                        dimension]
                      .data
                : 0;
    }
    return result;
}

[[maybe_unused]] __device__ __forceinline__ bool key_in_interval(
    const h3_vdn_key_interval &interval,
    std::uint32_t key,
    std::uint32_t sequence) {
    return key < sequence && key >= interval.begin && key < interval.end;
}

__global__ __launch_bounds__(64) void vdn_sage_i8_bf16_gfx12_kernel(
    const std::int8_t *__restrict__ query,
    const std::int8_t *__restrict__ key,
    const hip_bfloat16 *__restrict__ value,
    hip_bfloat16 *__restrict__ output,
    const float *__restrict__ q_scales,
    const float *__restrict__ k_scales,
    const h3_vdn_q_task *__restrict__ tasks,
    std::uint32_t task_count,
    std::uint32_t sequence,
    std::uint32_t heads,
    std::uint32_t q_groups,
    std::uint32_t k_groups,
    float attention_scale) {
    const std::uint32_t task_index = blockIdx.x;
    const std::uint32_t head = blockIdx.y;
    if (task_index >= task_count || head >= heads || threadIdx.x >= 64)
        return;

#if defined(__gfx12__) || defined(__gfx1200__) || defined(__gfx1201__)
    const int wave = static_cast<int>(threadIdx.x >> 5);
    const int lane = static_cast<int>(threadIdx.x & 31);
    const h3_vdn_q_task *__restrict__ task = tasks + task_index;
    const std::uint32_t task_q_begin = task->q_begin;
    const std::uint32_t task_q_count = task->q_count;
    const std::uint32_t task_interval_count = task->interval_count;
    const std::uint32_t q_local =
        static_cast<std::uint32_t>(wave * kTile + (lane & 15));
    const std::uint32_t query_row = task_q_begin + q_local;
    const bool query_valid = q_local < task_q_count && query_row < sequence;

    i32x2 q_fragments[kDepthTiles];
#pragma unroll
    for (int depth_tile = 0; depth_tile < kDepthTiles; ++depth_tile) {
        q_fragments[depth_tile] = load_i8_fragment(
            query, query_row, head, heads, lane, depth_tile * kTile,
            query_valid);
    }

    f32x8 output_fragments[kDepthTiles];
#pragma unroll
    for (int depth_tile = 0; depth_tile < kDepthTiles; ++depth_tile) {
#pragma unroll
        for (int element = 0; element < 8; ++element)
            output_fragments[depth_tile][element] = 0.0f;
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const float q_scale =
        query_valid
            ? q_scales[static_cast<std::size_t>(head) * q_groups +
                       query_row / 32]
            : 1.0f;

    for (std::uint32_t interval_index = 0;
         interval_index < task_interval_count; ++interval_index) {
        const h3_vdn_key_interval interval = task->allowed[interval_index];
        const std::uint64_t first_tile = interval.begin & ~15u;
        for (std::uint64_t key_tile_wide = first_tile;
             key_tile_wide < interval.end; key_tile_wide += kTile) {
            const std::uint32_t key_tile =
                static_cast<std::uint32_t>(key_tile_wide);
            i32x8 score_accumulator = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int depth_tile = 0; depth_tile < kDepthTiles;
                 ++depth_tile) {
                const std::uint32_t key_row =
                    key_tile + static_cast<std::uint32_t>(lane & 15);
                const i32x2 k_fragment = load_i8_fragment(
                    key, key_row, head, heads, lane, depth_tile * kTile,
                    key_row < sequence);
                score_accumulator =
                    __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, k_fragment, true, q_fragments[depth_tile],
                        score_accumulator, true);
            }

            const float k_scale =
                k_scales[static_cast<std::size_t>(head) * k_groups +
                         key_tile / 64];
            const float score_scale =
                q_scale * k_scale * attention_scale;
            f32x8 scores;
            float local_max = -INFINITY;
#pragma unroll
            for (int element = 0; element < 8; ++element) {
                const std::uint32_t key_row =
                    key_tile + static_cast<std::uint32_t>(
                                   ((lane >> 4) << 3) + element);
                const bool allowed =
                    query_valid && key_in_interval(interval, key_row,
                                                   sequence);
                scores[element] =
                    allowed
                        ? static_cast<float>(score_accumulator[element]) *
                              score_scale
                        : -INFINITY;
                local_max = fmaxf(local_max, scores[element]);
            }
            const float tile_max =
                fmaxf(local_max, __shfl_xor(local_max, 16, 32));
            const float next_max = fmaxf(running_max, tile_max);
            const float alpha = running_sum == 0.0f
                                    ? 0.0f
                                    : __expf(running_max - next_max);
            running_max = next_max;
            running_sum *= alpha;

            const int output_row_base = (lane >> 4) << 3;
#pragma unroll
            for (int depth_tile = 0; depth_tile < kDepthTiles;
                 ++depth_tile) {
#pragma unroll
                for (int element = 0; element < 8; ++element) {
                    const float row_alpha =
                        __shfl(alpha, output_row_base + element, 32);
                    output_fragments[depth_tile][element] *= row_alpha;
                }
            }

            u32x4 probability_words;
            float local_sum = 0.0f;
            const float probability_max = query_valid ? running_max : 0.0f;
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int low_element = 2 * pair;
                const int high_element = low_element + 1;
                const float low_probability =
                    __expf(scores[low_element] - probability_max);
                const float high_probability =
                    __expf(scores[high_element] - probability_max);
                local_sum += low_probability + high_probability;
                probability_words[pair] =
                    pack_bf16_rne(low_probability, high_probability);
            }
            running_sum +=
                local_sum + __shfl_xor(local_sum, 16, 32);
            const u16x8 probability_fragment =
                make_probability_fragment(probability_words, lane);

#pragma unroll
            for (int depth_tile = 0; depth_tile < kDepthTiles;
                 ++depth_tile) {
                const u16x8 value_fragment = load_value_fragment(
                    value, key_tile, head, sequence, heads, depth_tile,
                    lane);
                output_fragments[depth_tile] =
                    __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                        probability_fragment, value_fragment,
                        output_fragments[depth_tile]);
            }
        }
    }

    const int output_row_base = (lane >> 4) << 3;
    const int task_output_row_base = wave * kTile + output_row_base;
#pragma unroll
    for (int element = 0; element < 8; ++element) {
        const float denominator =
            __shfl(running_sum, output_row_base + element, 32);
        const float inverse_denominator = 1.0f / denominator;
        const std::uint32_t row =
            task_q_begin +
            static_cast<std::uint32_t>(task_output_row_base + element);
        if (row < task_q_begin + task_q_count && row < sequence) {
#pragma unroll
            for (int depth_tile = 0; depth_tile < kDepthTiles;
                 ++depth_tile) {
                const std::uint32_t dimension =
                    static_cast<std::uint32_t>(
                        depth_tile * kTile + (lane & 15));
                output[(static_cast<std::size_t>(row) * heads + head) *
                           kHeadDim +
                       dimension] = hip_bfloat16(
                    output_fragments[depth_tile][element] *
                    inverse_denominator);
            }
        }
    }
#else
    (void)query;
    (void)key;
    (void)value;
    (void)output;
    (void)q_scales;
    (void)k_scales;
    (void)tasks;
    (void)task_count;
    (void)sequence;
    (void)heads;
    (void)q_groups;
    (void)k_groups;
    (void)attention_scale;
#endif
}

}  // namespace

hipError_t h3_vdn_sage_gfx12_launch_i8_bf16(
    const int8_t *query, const int8_t *key,
    const float *query_scales, const float *key_scales,
    const void *value_bf16, void *output_bf16,
    const h3_vdn_q_task *tasks, uint32_t task_count,
    uint32_t sequence, uint32_t heads,
    uint32_t query_groups, uint32_t key_groups,
    float scale, hipStream_t stream) {
    if (!query || !key || !query_scales || !key_scales || !value_bf16 ||
        !output_bf16 || !tasks || !task_count || !sequence || !heads ||
        !query_groups || !key_groups || !std::isfinite(scale) ||
        scale <= 0.0f) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL(
        vdn_sage_i8_bf16_gfx12_kernel, dim3(task_count, heads), dim3(64),
        0, stream, query, key,
        static_cast<const hip_bfloat16 *>(value_bf16),
        static_cast<hip_bfloat16 *>(output_bf16), query_scales, key_scales,
        tasks, task_count, sequence, heads, query_groups, key_groups, scale);
    return hipGetLastError();
}
