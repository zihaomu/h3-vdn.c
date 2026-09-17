#include "h3_weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_h3_diffusers_weights.c: %s\n", message);
    exit(1);
}

static void require_tensor(const h3_weight_store *weights, const char *name,
                           h3_dtype dtype, int ndim, uint64_t first,
                           uint64_t second) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(weights, name, &header);
    if (!tensor || !header) fail(name);
    if (tensor->dtype != dtype || tensor->ndim != ndim ||
        tensor->shape[0] != first || (ndim == 2 && tensor->shape[1] != second))
        fail("resolved tensor has the wrong schema");
}

int main(int argc, char **argv) {
    const char *directory = argc > 1 ? argv[1] : "MiniMax-H3/transformer";
    char error[512];
    h3_weight_store *weights = h3_weight_store_open(
        directory, error, sizeof(error));
    if (!weights) fail(error);
    if (h3_weight_store_shards(weights) != 14)
        fail("official transformer must expose all 14 shards");

    /* Legacy requests must resolve to the official Diffusers names. */
    require_tensor(weights, "condition_proj.weight", H3_DTYPE_BF16, 2,
                   5376, 5120);
    require_tensor(weights, "video_patch_proj.weight", H3_DTYPE_F32, 2,
                   5376, 96);
    require_tensor(weights, "time_embedder.proj_in.weight", H3_DTYPE_F32, 2,
                   5376, 256);
    require_tensor(weights, "blocks.0.attn.q_norm.weight", H3_DTYPE_BF16, 1,
                   128, 0);
    require_tensor(weights, "blocks.49.attn.out_proj.weight",
                   H3_DTYPE_BF16, 2, 5376, 7168);
    require_tensor(weights, "blocks.0.mlp.fc1.weight", H3_DTYPE_BF16, 2,
                   28672, 5376);
    require_tensor(weights, "token_refiner.blocks.1.mlp.fc2.weight",
                   H3_DTYPE_BF16, 2, 5376, 14336);
    require_tensor(weights, "final_layer.adaln_proj.linear.weight",
                   H3_DTYPE_BF16, 2, 10752, 2688);
    require_tensor(weights, "final_layer.video_out.weight", H3_DTYPE_F32, 2,
                   96, 5376);

    /* Q/K/V remain independent payloads and are concatenated by h3_dit.c. */
    require_tensor(weights, "transformer_blocks.0.attn.to_q.weight",
                   H3_DTYPE_BF16, 2, 7168, 5376);
    require_tensor(weights, "transformer_blocks.0.attn.to_k.weight",
                   H3_DTYPE_BF16, 2, 7168, 5376);
    require_tensor(weights, "transformer_blocks.0.attn.to_v.weight",
                   H3_DTYPE_BF16, 2, 7168, 5376);
    if (h3_weight_find(weights, "blocks.0.attn.qkv_proj.weight", NULL))
        fail("modern checkpoint unexpectedly exposes legacy fused QKV");

    h3_weight_store_free(weights);
    puts("ok: official Diffusers transformer resolves 14 shards and legacy aliases");
    return 0;
}
