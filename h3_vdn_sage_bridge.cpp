#include "h3_vdn_sage_bridge.h"

#include <h3_vdn_sage.hpp>

namespace {

h3_vdn_sage_geometry upstream_geometry(
    const h3_vdn_sage_bridge_geometry &source) {
    return {source.sequence,
            source.heads,
            source.head_dim,
            source.video_start,
            source.frames,
            source.tokens_per_frame,
            source.radius,
            source.chunk,
            source.anchor_both != 0};
}

}  // namespace

hipError_t h3_vdn_sage_bridge_workspace_size(
    const h3_vdn_sage_bridge_geometry *geometry, size_t *bytes) {
    if (!geometry || !bytes) return hipErrorInvalidValue;
    const h3_vdn_sage_geometry upstream = upstream_geometry(*geometry);
    const hipError_t status = h3_vdn_sage_validate_geometry(upstream);
    if (status != hipSuccess) return status;
    *bytes = h3_vdn_sage_workspace_size(
        upstream, h3_vdn_sage_pv_mode::bf16);
    return *bytes ? hipSuccess : hipErrorNotSupported;
}

hipError_t h3_vdn_sage_bridge_prepare(
    const h3_vdn_sage_bridge_geometry *geometry, void *workspace,
    size_t workspace_bytes, hipStream_t stream) {
    if (!geometry) return hipErrorInvalidValue;
    return h3_vdn_sage_prepare_workspace(
        upstream_geometry(*geometry), h3_vdn_sage_pv_mode::bf16,
        workspace, workspace_bytes, stream);
}

hipError_t h3_vdn_sage_bridge_launch_prepared(
    const h3_vdn_sage_bridge_params *params) {
    if (!params) return hipErrorInvalidValue;
    const h3_vdn_sage_params upstream = {
        params->query_bf16,
        params->key_bf16,
        params->value_bf16,
        params->output_bf16,
        upstream_geometry(params->geometry),
        params->scale,
        h3_vdn_sage_pv_mode::bf16,
        params->workspace,
        params->workspace_bytes,
        params->stream};
    return h3_vdn_sage_launch_prepared(upstream);
}
