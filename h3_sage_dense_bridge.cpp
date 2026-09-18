#include "h3_sage_dense_bridge.h"

#include <sage_attention.hpp>

#include <cstdint>
#include <new>
#include <vector>

namespace {

std::size_t task_count(const h3_sage_dense_geometry &geometry) {
    return (static_cast<std::size_t>(geometry.sequence) + 31u) / 32u;
}

sageattention::descriptor descriptor(
    const h3_sage_dense_geometry &geometry) {
    return {{1,
             geometry.sequence,
             geometry.sequence,
             geometry.heads,
             geometry.heads,
             geometry.head_dim,
             sageattention::layout::nhd},
            {sageattention::data_type::bf16,
             sageattention::data_type::bf16,
             sageattention::qk_mode::bf16,
             sageattention::pv_mode::bf16,
             sageattention::kernel_id::e33_bf16_qk_gfx12_d128},
            task_count(geometry)};
}

std::vector<sageattention::q_task> dense_tasks(
    const h3_sage_dense_geometry &geometry) {
    std::vector<sageattention::q_task> tasks;
    tasks.reserve(task_count(geometry));
    for (std::uint64_t begin = 0; begin < geometry.sequence; begin += 32) {
        const std::uint32_t row = static_cast<std::uint32_t>(begin);
        const std::uint32_t count =
            geometry.sequence - row < 32 ? geometry.sequence - row : 32;
        tasks.push_back(
            {row, count, 1, {{0, geometry.sequence}}});
    }
    return tasks;
}

}  // namespace

hipError_t h3_sage_dense_workspace_size(
    const h3_sage_dense_geometry *geometry, size_t *bytes) {
    if (!geometry || !bytes) return hipErrorInvalidValue;
    const sageattention::descriptor operation = descriptor(*geometry);
    const hipError_t status = sageattention::validate_descriptor(operation);
    if (status != hipSuccess) return status;
    *bytes = sageattention::workspace_size(operation);
    return *bytes ? hipSuccess : hipErrorNotSupported;
}

hipError_t h3_sage_dense_prepare(
    const h3_sage_dense_geometry *geometry, void *workspace,
    size_t workspace_bytes, hipStream_t stream) {
    if (!geometry || !workspace) return hipErrorInvalidValue;
    try {
        const sageattention::descriptor operation = descriptor(*geometry);
        const std::vector<sageattention::q_task> tasks =
            dense_tasks(*geometry);
        return sageattention::prepare_workspace(
            operation, {tasks.data(), tasks.size()}, workspace,
            workspace_bytes, stream);
    } catch (const std::bad_alloc &) {
        return hipErrorOutOfMemory;
    } catch (...) {
        return hipErrorUnknown;
    }
}

hipError_t h3_sage_dense_launch_prepared(
    const h3_sage_dense_params *params) {
    if (!params) return hipErrorInvalidValue;
    const sageattention::params operation = {
        params->query_bf16,
        params->key_bf16,
        params->value_bf16,
        params->output_bf16,
        descriptor(params->geometry),
        params->scale,
        params->workspace,
        params->workspace_bytes,
        params->stream};
    return sageattention::launch_prepared(operation);
}
