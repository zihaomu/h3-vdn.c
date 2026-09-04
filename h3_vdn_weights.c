#include "h3_vdn_weights.h"

#include "h3_weights.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VDN_BLOCKS = 50,
    VDN_HIDDEN = 5376,
    VDN_HEADS = 56,
    VDN_HEAD_DIM = 128,
    VDN_INNER = VDN_HEADS * VDN_HEAD_DIM,
    VDN_FFN = 14336,
    VDN_ADALN_WIDTH = 96768,
    VDN_TIME_WIDTH = 2688,
    VDN_ALPHA_WIDTH = 128,
    VDN_VIDEO_PATCH = 96,
    VDN_AUDIO_WIDTH = 32,
    VDN_CONTEXT_WIDTH = 5120,
    VDN_TIME_INPUT = 256,
    VDN_FINAL_ADALN_WIDTH = 10752
};

struct h3_vdn_weight_store {
    h3_weight_store *base;
    h3_weight_store *linear;
    h3_weight_store *default_adapter;
    h3_weight_store *turbo_adapter;
    int use_turbo;
};

static int vdn_fail(char *error, size_t error_size, const char *format, ...) {
    if (error && error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

static char *vdn_path(const char *root, const char *relative) {
    if (!root || !relative) return NULL;
    size_t root_length = strlen(root);
    size_t relative_length = strlen(relative);
    if (root_length > SIZE_MAX - relative_length - 2) return NULL;
    char *path = malloc(root_length + relative_length + 2);
    if (path)
        snprintf(path, root_length + relative_length + 2,
                 "%s/%s", root, relative);
    return path;
}

h3_vdn_weight_store *h3_vdn_weight_store_open(
        const char *base_model_dir, const char *checkpoint_dir, int use_turbo,
        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!base_model_dir || !*base_model_dir ||
        !checkpoint_dir || !*checkpoint_dir ||
        (use_turbo != 0 && use_turbo != 1)) {
        vdn_fail(error, error_size, "invalid VDN weight store arguments");
        return NULL;
    }
    h3_vdn_weight_store *store = calloc(1, sizeof(*store));
    if (!store) {
        vdn_fail(error, error_size, "out of memory creating VDN weight store");
        return NULL;
    }
    char *base = vdn_path(base_model_dir, "transformer");
    char *linear = vdn_path(checkpoint_dir, "linear_branch");
    char *default_adapter = vdn_path(checkpoint_dir, "adapters/default");
    char *turbo_adapter = use_turbo ?
        vdn_path(checkpoint_dir, "adapters/turbo") : NULL;
    if (!base || !linear || !default_adapter || (use_turbo && !turbo_adapter)) {
        vdn_fail(error, error_size, "out of memory resolving VDN weight paths");
        goto failed;
    }
    store->base = h3_weight_store_open(base, error, error_size);
    if (!store->base) goto failed;
    store->linear = h3_weight_store_open(linear, error, error_size);
    if (!store->linear) goto failed;
    store->default_adapter = h3_weight_store_open(
        default_adapter, error, error_size);
    if (!store->default_adapter) goto failed;
    if (use_turbo) {
        store->turbo_adapter = h3_weight_store_open(
            turbo_adapter, error, error_size);
        if (!store->turbo_adapter) goto failed;
    }
    store->use_turbo = use_turbo;
    free(turbo_adapter);
    free(default_adapter);
    free(linear);
    free(base);
    return store;

failed:
    free(turbo_adapter);
    free(default_adapter);
    free(linear);
    free(base);
    h3_vdn_weight_store_free(store);
    return NULL;
}

void h3_vdn_weight_store_free(h3_vdn_weight_store *store) {
    if (!store) return;
    h3_weight_store_free(store->turbo_adapter);
    h3_weight_store_free(store->default_adapter);
    h3_weight_store_free(store->linear);
    h3_weight_store_free(store->base);
    free(store);
}

static h3_gpu_tensor *load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                const char *name, int ndim,
                                const uint64_t *shape,
                                char *error, size_t error_size) {
    return h3_weight_load_bf16(store, gpu, name, ndim, shape,
                              error, error_size);
}

static h3_gpu_tensor *load1(const h3_weight_store *store, h3_gpu *gpu,
                            const char *name, uint64_t width,
                            char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return load_bf16(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *load2(const h3_weight_store *store, h3_gpu *gpu,
                            const char *name, uint64_t rows, uint64_t columns,
                            char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return load_bf16(store, gpu, name, 2, shape, error, error_size);
}

static h3_gpu_tensor *load3(const h3_weight_store *store, h3_gpu *gpu,
                            const char *name, uint64_t first, uint64_t second,
                            uint64_t third, char *error, size_t error_size) {
    uint64_t shape[] = {first, second, third};
    return load_bf16(store, gpu, name, 3, shape, error, error_size);
}

static h3_gpu_tensor *load4(const h3_weight_store *store, h3_gpu *gpu,
                            const char *name, uint64_t first, uint64_t second,
                            uint64_t third, uint64_t fourth,
                            char *error, size_t error_size) {
    uint64_t shape[] = {first, second, third, fourth};
    return load_bf16(store, gpu, name, 4, shape, error, error_size);
}

static h3_gpu_tensor *load_f32_1(const h3_weight_store *store, h3_gpu *gpu,
                                 const char *name, uint64_t width,
                                 char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *load_f32_2(const h3_weight_store *store, h3_gpu *gpu,
                                 const char *name, uint64_t rows,
                                 uint64_t columns,
                                 char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(store, gpu, name, 2, shape, error, error_size);
}

static int merge_adapter(h3_gpu *gpu, h3_gpu_tensor *weight,
                         const h3_weight_store *adapter,
                         const char *target, const char *adapter_name,
                         uint32_t input_dim, uint32_t output_dim,
                         uint32_t rank, float scale,
                         char *error, size_t error_size) {
    char a_name[288];
    char b_name[288];
    int a_length = snprintf(a_name, sizeof(a_name), "%s.lora_A.%s.weight",
                            target, adapter_name);
    int b_length = snprintf(b_name, sizeof(b_name), "%s.lora_B.%s.weight",
                            target, adapter_name);
    if (a_length < 0 || (size_t)a_length >= sizeof(a_name) ||
        b_length < 0 || (size_t)b_length >= sizeof(b_name))
        return vdn_fail(error, error_size, "VDN LoRA target name is too long");
    h3_gpu_tensor *a = load2(adapter, gpu, a_name, rank, input_dim,
                             error, error_size);
    h3_gpu_tensor *b = a ? load2(adapter, gpu, b_name, output_dim, rank,
                                 error, error_size) : NULL;
    if (!a || !b) {
        h3_gpu_tensor_free(b);
        h3_gpu_tensor_free(a);
        return 0;
    }
    int ok = h3_gpu_begin(gpu) &&
             h3_gpu_lora_merge_bf16(gpu, weight, weight, a, b,
                                     input_dim, output_dim, rank, scale) &&
             h3_gpu_submit(gpu);
    if (!ok)
        vdn_fail(error, error_size, "cannot merge %s adapter into %s: %s",
                 adapter_name, target, h3_gpu_error(gpu));
    h3_gpu_tensor_free(b);
    h3_gpu_tensor_free(a);
    return ok;
}

static h3_gpu_tensor *load_effective(
        h3_vdn_weight_store *store, h3_gpu *gpu,
        const char *base_name, const char *target,
        uint32_t rows, uint32_t columns,
        int use_default, int use_turbo, uint32_t turbo_rank,
        char *error, size_t error_size) {
    h3_gpu_tensor *weight = load2(store->base, gpu, base_name, rows, columns,
                                  error, error_size);
    if (!weight) return NULL;
    if (use_default && !merge_adapter(
            gpu, weight, store->default_adapter, target, "default",
            columns, rows, 64, 1.0f, error, error_size)) goto failed;
    if (use_turbo && !merge_adapter(
            gpu, weight, store->turbo_adapter, target, "turbo",
            columns, rows, turbo_rank, 1.0f, error, error_size)) goto failed;
    return weight;
failed:
    h3_gpu_tensor_free(weight);
    return NULL;
}

void h3_vdn_block_weights_free(h3_vdn_block_weights *weights) {
    if (!weights) return;
#define FREE(field) h3_gpu_tensor_free(weights->field)
    FREE(norm1); FREE(norm2); FREE(q); FREE(k); FREE(v); FREE(q_norm);
    FREE(k_norm); FREE(out); FREE(fc1); FREE(fc2); FREE(adaln_weight);
    FREE(adaln_bias);
    FREE(linear.alpha_a_log); FREE(linear.alpha_down);
    FREE(linear.alpha_dt_bias); FREE(linear.alpha_up); FREE(linear.beta);
    FREE(linear.norm); FREE(linear.gate_down); FREE(linear.gate_up_bias);
    FREE(linear.gate_up); FREE(linear.k_spatial); FREE(linear.k_temporal);
    FREE(linear.v_spatial); FREE(linear.v_temporal);
    FREE(linear.softmax_gate_bias); FREE(linear.softmax_gate_weight);
    FREE(linear.to_out);
#undef FREE
    memset(weights, 0, sizeof(*weights));
}

int h3_vdn_block_weights_load(h3_vdn_weight_store *store, h3_gpu *gpu,
                              unsigned block, h3_vdn_block_weights *weights,
                              char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!store || !gpu || !weights || block >= VDN_BLOCKS)
        return vdn_fail(error, error_size,
                        "invalid VDN block weight load arguments");
    memset(weights, 0, sizeof(*weights));
    char base[224];
    char target[224];
    char linear[256];
#define BASE1(field, suffix, width) do {                                      \
    snprintf(base, sizeof(base), "transformer_blocks.%u.%s", block, suffix);\
    weights->field = load1(store->base, gpu, base, width, error, error_size);  \
    if (!weights->field) goto failed;                                         \
} while (0)
#define EFFECTIVE(field, base_suffix, target_suffix, rows, columns, def, turbo, rank) do { \
    snprintf(base, sizeof(base), "transformer_blocks.%u.%s", block, base_suffix);       \
    snprintf(target, sizeof(target), "transformer_blocks.%u.%s", block, target_suffix); \
    weights->field = load_effective(store, gpu, base, target, rows, columns,             \
        def, (turbo) && store->use_turbo, rank, error, error_size);                       \
    if (!weights->field) goto failed;                                                     \
} while (0)
#define LINEAR1(field, suffix, width) do {                                  \
    snprintf(linear, sizeof(linear), "transformer_blocks.%u.attn.%s",      \
             block, suffix);                                                 \
    weights->linear.field = load1(store->linear, gpu, linear, width,         \
                                  error, error_size);                         \
    if (!weights->linear.field) goto failed;                                 \
} while (0)
#define LINEAR2(field, suffix, rows, columns) do {                           \
    snprintf(linear, sizeof(linear), "transformer_blocks.%u.attn.%s",      \
             block, suffix);                                                 \
    weights->linear.field = load2(store->linear, gpu, linear, rows, columns, \
                                  error, error_size);                         \
    if (!weights->linear.field) goto failed;                                 \
} while (0)
    BASE1(norm1, "norm1.weight", VDN_HIDDEN);
    BASE1(norm2, "norm2.weight", VDN_HIDDEN);
    BASE1(q_norm, "attn.norm_q.weight", VDN_HEAD_DIM);
    BASE1(k_norm, "attn.norm_k.weight", VDN_HEAD_DIM);
    EFFECTIVE(q, "attn.to_q.weight", "attn.orig.to_q", VDN_INNER,
              VDN_HIDDEN, 1, 1, 64);
    EFFECTIVE(k, "attn.to_k.weight", "attn.orig.to_k", VDN_INNER,
              VDN_HIDDEN, 1, 1, 64);
    EFFECTIVE(v, "attn.to_v.weight", "attn.orig.to_v", VDN_INNER,
              VDN_HIDDEN, 1, 1, 64);
    EFFECTIVE(out, "attn.to_out.0.weight", "attn.orig.to_out.0",
              VDN_HIDDEN, VDN_INNER, 1, 1, 64);
    EFFECTIVE(fc1, "ff.net.0.proj.weight", "ff.net.0.proj",
              VDN_FFN * 2, VDN_HIDDEN, 0, 1, 64);
    EFFECTIVE(fc2, "ff.net.2.weight", "ff.net.2",
              VDN_HIDDEN, VDN_FFN, 0, 1, 64);
    EFFECTIVE(adaln_weight, "adaln_proj.linear.weight", "adaln_proj.linear",
              VDN_ADALN_WIDTH, VDN_TIME_WIDTH, 0, 1, 16);
    snprintf(base, sizeof(base),
             "transformer_blocks.%u.adaln_proj.linear.bias", block);
    weights->adaln_bias = load1(store->base, gpu, base, VDN_ADALN_WIDTH,
                                error, error_size);
    if (!weights->adaln_bias) goto failed;

    LINEAR1(alpha_a_log, "linear_attention.alpha.A_log", VDN_HEADS);
    LINEAR2(alpha_down, "linear_attention.alpha.down.weight",
            VDN_ALPHA_WIDTH, VDN_HIDDEN);
    LINEAR1(alpha_dt_bias, "linear_attention.alpha.dt_bias", VDN_INNER);
    LINEAR2(alpha_up, "linear_attention.alpha.up.weight",
            VDN_INNER, VDN_ALPHA_WIDTH);
    LINEAR2(beta, "linear_attention.beta_proj.weight", VDN_HEADS, VDN_HIDDEN);
    LINEAR1(norm, "linear_attention.norm.weight", VDN_HEAD_DIM);
    LINEAR2(gate_down, "linear_attention.output_gate.down.weight",
            VDN_ALPHA_WIDTH, VDN_HIDDEN);
    LINEAR1(gate_up_bias, "linear_attention.output_gate.up.bias", VDN_INNER);
    LINEAR2(gate_up, "linear_attention.output_gate.up.weight",
            VDN_INNER, VDN_ALPHA_WIDTH);
    snprintf(linear, sizeof(linear),
             "transformer_blocks.%u.attn.linear_attention.short_conv.k_sp.weight",
             block);
    weights->linear.k_spatial = load4(store->linear, gpu, linear,
                                      VDN_INNER, 1, 5, 5, error, error_size);
    if (!weights->linear.k_spatial) goto failed;
    snprintf(linear, sizeof(linear),
             "transformer_blocks.%u.attn.linear_attention.short_conv.v_sp.weight",
             block);
    weights->linear.v_spatial = load4(store->linear, gpu, linear,
                                      VDN_INNER, 1, 5, 5, error, error_size);
    if (!weights->linear.v_spatial) goto failed;
    snprintf(linear, sizeof(linear),
             "transformer_blocks.%u.attn.linear_attention.short_conv.k_tm.weight",
             block);
    weights->linear.k_temporal = load3(store->linear, gpu, linear,
                                       VDN_INNER, 1, 5, error, error_size);
    if (!weights->linear.k_temporal) goto failed;
    snprintf(linear, sizeof(linear),
             "transformer_blocks.%u.attn.linear_attention.short_conv.v_tm.weight",
             block);
    weights->linear.v_temporal = load3(store->linear, gpu, linear,
                                       VDN_INNER, 1, 5, error, error_size);
    if (!weights->linear.v_temporal) goto failed;
    LINEAR1(softmax_gate_bias, "softmax_gate.up.bias", VDN_HEADS);
    LINEAR2(softmax_gate_weight, "softmax_gate.up.weight",
            VDN_HEADS, VDN_HIDDEN);
    LINEAR2(to_out, "to_out_linear.weight", VDN_HIDDEN, VDN_INNER);
#undef LINEAR2
#undef LINEAR1
#undef EFFECTIVE
#undef BASE1
    return 1;

failed:
    h3_vdn_block_weights_free(weights);
    return 0;
}

static void refiner_free(h3_vdn_refiner_weights *weights) {
    if (!weights) return;
#define FREE_REFINER(field) h3_gpu_tensor_free(weights->field)
    FREE_REFINER(norm1); FREE_REFINER(norm2); FREE_REFINER(q);
    FREE_REFINER(k); FREE_REFINER(v); FREE_REFINER(q_norm);
    FREE_REFINER(k_norm); FREE_REFINER(out); FREE_REFINER(fc1);
    FREE_REFINER(fc2);
#undef FREE_REFINER
    memset(weights, 0, sizeof(*weights));
}

void h3_vdn_model_weights_free(h3_vdn_model_weights *weights) {
    if (!weights) return;
#define FREE_MODEL(field) h3_gpu_tensor_free(weights->field)
    FREE_MODEL(video_in_weight); FREE_MODEL(video_in_bias);
    FREE_MODEL(audio_in_weight); FREE_MODEL(audio_in_bias);
    FREE_MODEL(context_weight); FREE_MODEL(context_bias);
    FREE_MODEL(time_linear1_weight); FREE_MODEL(time_linear1_bias);
    FREE_MODEL(time_linear2_weight); FREE_MODEL(time_linear2_bias);
    refiner_free(&weights->refiner[0]);
    refiner_free(&weights->refiner[1]);
    FREE_MODEL(refiner_final_norm); FREE_MODEL(final_norm);
    FREE_MODEL(final_adaln_weight); FREE_MODEL(final_adaln_bias);
    FREE_MODEL(video_out_weight); FREE_MODEL(video_out_bias);
    FREE_MODEL(audio_out_weight); FREE_MODEL(audio_out_bias);
#undef FREE_MODEL
    memset(weights, 0, sizeof(*weights));
}

static int load_refiner(h3_vdn_weight_store *store, h3_gpu *gpu,
                        unsigned block, h3_vdn_refiner_weights *weights,
                        char *error, size_t error_size) {
    char name[224];
    char target[224];
#define REFINER1(field, suffix, width) do {                                   \
    snprintf(name, sizeof(name),                                              \
             "token_refiner.refiner_blocks.%u.%s", block, suffix);         \
    weights->field = load1(store->base, gpu, name, width, error, error_size); \
    if (!weights->field) goto failed;                                         \
} while (0)
#define REFINER_EFFECTIVE(field, suffix, rows, columns, def, turbo) do {       \
    snprintf(name, sizeof(name),                                              \
             "token_refiner.refiner_blocks.%u.%s.weight", block, suffix);  \
    snprintf(target, sizeof(target),                                          \
             "token_refiner.refiner_blocks.%u.%s", block, suffix);         \
    weights->field = load_effective(store, gpu, name, target, rows, columns,  \
        def, (turbo) && store->use_turbo, 64, error, error_size);             \
    if (!weights->field) goto failed;                                         \
} while (0)
    REFINER1(norm1, "norm1.weight", VDN_HIDDEN);
    REFINER1(norm2, "norm2.weight", VDN_HIDDEN);
    REFINER1(q_norm, "attn.norm_q.weight", VDN_HEAD_DIM);
    REFINER1(k_norm, "attn.norm_k.weight", VDN_HEAD_DIM);
    REFINER_EFFECTIVE(q, "attn.to_q", VDN_INNER, VDN_HIDDEN, 1, 1);
    REFINER_EFFECTIVE(k, "attn.to_k", VDN_INNER, VDN_HIDDEN, 1, 1);
    REFINER_EFFECTIVE(v, "attn.to_v", VDN_INNER, VDN_HIDDEN, 1, 1);
    REFINER_EFFECTIVE(out, "attn.to_out.0", VDN_HIDDEN, VDN_INNER, 1, 1);
    REFINER_EFFECTIVE(fc1, "ff.net.0.proj", VDN_FFN * 2, VDN_HIDDEN, 0, 1);
    REFINER_EFFECTIVE(fc2, "ff.net.2", VDN_HIDDEN, VDN_FFN, 0, 1);
#undef REFINER_EFFECTIVE
#undef REFINER1
    return 1;
failed:
    refiner_free(weights);
    return 0;
}

int h3_vdn_model_weights_load(h3_vdn_weight_store *store, h3_gpu *gpu,
                              h3_vdn_model_weights *weights,
                              char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!store || !gpu || !weights)
        return vdn_fail(error, error_size,
                        "invalid VDN model weight load arguments");
    memset(weights, 0, sizeof(*weights));
#define LOAD_F32_1(field, name, width) do {                                    \
    weights->field = load_f32_1(store->base, gpu, name, width,                \
                                 error, error_size);                           \
    if (!weights->field) goto failed;                                         \
} while (0)
#define LOAD_F32_2(field, name, rows, columns) do {                            \
    weights->field = load_f32_2(store->base, gpu, name, rows, columns,        \
                                 error, error_size);                           \
    if (!weights->field) goto failed;                                         \
} while (0)
#define LOAD_BF16_1(field, name, width) do {                                   \
    weights->field = load1(store->base, gpu, name, width, error, error_size); \
    if (!weights->field) goto failed;                                         \
} while (0)
#define LOAD_BF16_2(field, name, rows, columns) do {                           \
    weights->field = load2(store->base, gpu, name, rows, columns,             \
                            error, error_size);                                \
    if (!weights->field) goto failed;                                         \
} while (0)
    LOAD_F32_2(video_in_weight, "proj_in.weight", VDN_HIDDEN, VDN_VIDEO_PATCH);
    LOAD_F32_1(video_in_bias, "proj_in.bias", VDN_HIDDEN);
    LOAD_F32_2(audio_in_weight, "audio_proj_in.weight", VDN_HIDDEN,
               VDN_AUDIO_WIDTH);
    LOAD_F32_1(audio_in_bias, "audio_proj_in.bias", VDN_HIDDEN);
    LOAD_BF16_2(context_weight, "context_embedder.weight", VDN_HIDDEN,
                VDN_CONTEXT_WIDTH);
    LOAD_BF16_1(context_bias, "context_embedder.bias", VDN_HIDDEN);
    LOAD_F32_2(time_linear1_weight, "time_embedder.linear_1.weight",
               VDN_HIDDEN, VDN_TIME_INPUT);
    LOAD_F32_1(time_linear1_bias, "time_embedder.linear_1.bias", VDN_HIDDEN);
    LOAD_F32_2(time_linear2_weight, "time_embedder.linear_2.weight",
               VDN_TIME_WIDTH, VDN_HIDDEN);
    LOAD_F32_1(time_linear2_bias, "time_embedder.linear_2.bias", VDN_TIME_WIDTH);
    if (!load_refiner(store, gpu, 0, &weights->refiner[0], error, error_size) ||
        !load_refiner(store, gpu, 1, &weights->refiner[1], error, error_size))
        goto failed;
    LOAD_BF16_1(refiner_final_norm, "token_refiner.final_norm.weight", VDN_HIDDEN);
    LOAD_BF16_1(final_norm, "norm_out.norm.weight", VDN_HIDDEN);
    weights->final_adaln_weight = load_effective(
        store, gpu, "norm_out.linear.weight", "norm_out.linear",
        VDN_FINAL_ADALN_WIDTH, VDN_TIME_WIDTH, 0, store->use_turbo, 16,
        error, error_size);
    if (!weights->final_adaln_weight) goto failed;
    LOAD_BF16_1(final_adaln_bias, "norm_out.linear.bias", VDN_FINAL_ADALN_WIDTH);
    LOAD_F32_2(video_out_weight, "proj_out.weight", VDN_VIDEO_PATCH, VDN_HIDDEN);
    LOAD_F32_1(video_out_bias, "proj_out.bias", VDN_VIDEO_PATCH);
    LOAD_F32_2(audio_out_weight, "audio_proj_out.weight", VDN_AUDIO_WIDTH,
               VDN_HIDDEN);
    LOAD_F32_1(audio_out_bias, "audio_proj_out.bias", VDN_AUDIO_WIDTH);
#undef LOAD_BF16_2
#undef LOAD_BF16_1
#undef LOAD_F32_2
#undef LOAD_F32_1
    return 1;
failed:
    h3_vdn_model_weights_free(weights);
    return 0;
}
