#include "h3_vdn_dit.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    VDN_PROMPT_ROWS = 800,
    VDN_TEXT_WIDTH = 5120,
    VDN_HIDDEN = 5376,
    VDN_HEADS = 56,
    VDN_HEAD_DIM = 128,
    VDN_INNER = VDN_HEADS * VDN_HEAD_DIM,
    VDN_FFN = 14336,
    VDN_TIME_INPUT = 256,
    VDN_TIME_WIDTH = 2688,
    VDN_MODALITIES = 3,
    VDN_ADALN_SLOTS = 6,
    VDN_ROPE_HALF = 48,
    VDN_VIDEO_PATCH = 96,
    VDN_AUDIO_WIDTH = 32,
    VDN_BLOCKS = 50
};

static uint16_t dit_f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += UINT32_C(0x7fff) + lsb;
    return (uint16_t)(bits >> 16);
}

static void dit_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int dit_op(h3_gpu *gpu, int ok, const char *label,
                  char *error, size_t error_size) {
    if (ok) return 1;
    dit_fail(error, error_size, "%s: %s", label, h3_gpu_error(gpu));
    return 0;
}

void h3_vdn_layout_free(h3_vdn_layout *layout) {
    if (!layout) return;
    h3_layout_free(&layout->packed);
    free(layout->token_tags);
    free(layout->rope_cos);
    free(layout->rope_sin);
    memset(layout, 0, sizeof(*layout));
}

int h3_vdn_layout_build(const h3_text_embedding *prompt,
                        uint32_t latent_frames, uint32_t latent_height,
                        uint32_t latent_width, uint32_t audio_latents,
                        h3_vdn_layout *layout,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!layout) {
        dit_fail(error, error_size, "missing VDN layout output");
        return 0;
    }
    memset(layout, 0, sizeof(*layout));
    if (!prompt || !prompt->values || !prompt->tags ||
        prompt->tokens != VDN_PROMPT_ROWS ||
        prompt->width != VDN_TEXT_WIDTH || !latent_frames ||
        latent_height < 2 || latent_width < 2 ||
        latent_height % 2 || latent_width % 2 ||
        latent_frames > INT32_MAX || latent_height > INT32_MAX ||
        latent_width > INT32_MAX || audio_latents > INT32_MAX) {
        dit_fail(error, error_size, "invalid VDN packed-layout arguments");
        return 0;
    }
    h3_layout_spec spec;
    memset(&spec, 0, sizeof(spec));
    spec.text_len = VDN_PROMPT_ROWS;
    spec.latent_t = (int)latent_frames;
    spec.latent_h = (int)latent_height;
    spec.latent_w = (int)latent_width;
    spec.audio_t = (int)audio_latents;
    /* h3_layout_build only uses frame_count for optional keyframe validation.
     * Reconstruct the released 2+5k latent geometry for completeness. */
    spec.frame_count = latent_frames <= 2 ? 5 :
        5 + (int)((latent_frames - 2) / 5) * 17;
    if (!h3_layout_build(&spec, &layout->packed, error, error_size)) return 0;
    if (layout->packed.seq_len > UINT32_MAX ||
        layout->packed.audio_target_rows > UINT32_MAX ||
        layout->packed.img_target_rows > UINT32_MAX) {
        dit_fail(error, error_size, "VDN packed layout exceeds 32-bit GPU limits");
        goto failed;
    }
    uint64_t frame_rows_wide =
        (uint64_t)(latent_height / 2) * (latent_width / 2);
    if (frame_rows_wide > UINT32_MAX ||
        (uint64_t)latent_frames * frame_rows_wide !=
            layout->packed.img_target_rows) {
        dit_fail(error, error_size, "inconsistent VDN video layout geometry");
        goto failed;
    }
    layout->sequence = (uint32_t)layout->packed.seq_len;
    layout->text_rows = VDN_PROMPT_ROWS;
    layout->audio_start = VDN_PROMPT_ROWS;
    layout->audio_rows = (uint32_t)layout->packed.audio_target_rows;
    layout->video_start = layout->audio_start + layout->audio_rows;
    layout->video_rows = (uint32_t)layout->packed.img_target_rows;
    layout->frames = latent_frames;
    layout->frame_height = latent_height / 2;
    layout->frame_width = latent_width / 2;
    if ((uint64_t)layout->video_start + layout->video_rows !=
        layout->sequence) {
        dit_fail(error, error_size, "unexpected VDN segment ordering");
        goto failed;
    }
    size_t rope_elements = (size_t)layout->sequence * VDN_ROPE_HALF;
    if (layout->sequence && rope_elements / layout->sequence != VDN_ROPE_HALF) {
        dit_fail(error, error_size, "VDN RoPE table size overflow");
        goto failed;
    }
    layout->token_tags = malloc((size_t)layout->sequence);
    layout->rope_cos = malloc(rope_elements * sizeof(*layout->rope_cos));
    layout->rope_sin = malloc(rope_elements * sizeof(*layout->rope_sin));
    if (!layout->token_tags || !layout->rope_cos || !layout->rope_sin) {
        dit_fail(error, error_size, "out of memory building VDN layout tables");
        goto failed;
    }
    memcpy(layout->token_tags, prompt->tags, VDN_PROMPT_ROWS);
    memset(layout->token_tags + layout->audio_start, 2,
           layout->audio_rows);
    memset(layout->token_tags + layout->video_start, 0,
           layout->video_rows);
    float inverse[VDN_ROPE_HALF / 3];
    for (uint32_t frequency = 0; frequency < VDN_ROPE_HALF / 3;
         frequency++) {
        inverse[frequency] = expf(-logf(10000.0f) * (float)frequency /
                                  (float)(VDN_ROPE_HALF / 3));
    }
    for (uint32_t row = 0; row < layout->sequence; row++) {
        const h3_position *position = &layout->packed.positions[row];
        const double axes[3] = {position->t, position->h, position->w};
        for (uint32_t axis = 0; axis < 3; axis++) {
            for (uint32_t frequency = 0;
                 frequency < VDN_ROPE_HALF / 3; frequency++) {
                uint32_t column = axis * (VDN_ROPE_HALF / 3) + frequency;
                float angle = (float)axes[axis] * inverse[frequency];
                size_t index = (size_t)row * VDN_ROPE_HALF + column;
                layout->rope_cos[index] = dit_f32_to_bf16(cosf(angle));
                layout->rope_sin[index] = dit_f32_to_bf16(sinf(angle));
            }
        }
    }
    return 1;

failed:
    h3_vdn_layout_free(layout);
    return 0;
}

static int run_refiner(h3_gpu *gpu, const h3_vdn_refiner_weights *weights,
                       h3_gpu_tensor *hidden, h3_gpu_tensor *norm,
                       h3_gpu_tensor *query_raw, h3_gpu_tensor *key_raw,
                       h3_gpu_tensor *value, h3_gpu_tensor *query,
                       h3_gpu_tensor *key, h3_gpu_tensor *attended,
                       h3_gpu_tensor *branch, h3_gpu_tensor *fc1,
                       h3_gpu_tensor *activated, h3_gpu_tensor *dummy_rope,
                       char *error, size_t error_size) {
#define OP(call, label) do {                                                   \
    if (!dit_op(gpu, (call), label, error, error_size)) return 0;             \
} while (0)
    OP(h3_gpu_rms_norm_bf16(gpu, norm, hidden, weights->norm1,
                            VDN_PROMPT_ROWS, VDN_HIDDEN, 1e-5f),
       "VDN refiner attention RMSNorm");
    OP(h3_gpu_linear_bf16(gpu, query_raw, norm, weights->q, NULL,
                          VDN_PROMPT_ROWS, VDN_HIDDEN, VDN_INNER),
       "VDN refiner Q projection");
    OP(h3_gpu_linear_bf16(gpu, key_raw, norm, weights->k, NULL,
                          VDN_PROMPT_ROWS, VDN_HIDDEN, VDN_INNER),
       "VDN refiner K projection");
    OP(h3_gpu_linear_bf16(gpu, value, norm, weights->v, NULL,
                          VDN_PROMPT_ROWS, VDN_HIDDEN, VDN_INNER),
       "VDN refiner V projection");
    OP(h3_gpu_vdn_qk_rope_bf16(
           gpu, query, key, query_raw, key_raw, weights->q_norm,
           weights->k_norm, dummy_rope, dummy_rope, VDN_PROMPT_ROWS,
           VDN_HEADS, VDN_HEAD_DIM, 0, 1e-5f),
       "VDN refiner QK normalization");
    OP(h3_gpu_sdpa_bf16(gpu, attended, query, key, value,
                        VDN_PROMPT_ROWS, VDN_HEADS, VDN_HEAD_DIM,
                        1.0f / sqrtf((float)VDN_HEAD_DIM)),
       "VDN refiner full attention");
    OP(h3_gpu_linear_bf16(gpu, branch, attended, weights->out, NULL,
                          VDN_PROMPT_ROWS, VDN_INNER, VDN_HIDDEN),
       "VDN refiner attention output");
    OP(h3_gpu_add_bf16(gpu, hidden, hidden, branch,
                       VDN_PROMPT_ROWS * VDN_HIDDEN),
       "VDN refiner attention residual");
    OP(h3_gpu_rms_norm_bf16(gpu, norm, hidden, weights->norm2,
                            VDN_PROMPT_ROWS, VDN_HIDDEN, 1e-5f),
       "VDN refiner MLP RMSNorm");
    OP(h3_gpu_linear_bf16(gpu, fc1, norm, weights->fc1, NULL,
                          VDN_PROMPT_ROWS, VDN_HIDDEN, VDN_FFN * 2),
       "VDN refiner MLP input");
    OP(h3_gpu_swiglu_bf16(gpu, activated, fc1, VDN_PROMPT_ROWS, VDN_FFN),
       "VDN refiner SwiGLU");
    OP(h3_gpu_linear_bf16(gpu, branch, activated, weights->fc2, NULL,
                          VDN_PROMPT_ROWS, VDN_FFN, VDN_HIDDEN),
       "VDN refiner MLP output");
    OP(h3_gpu_add_bf16(gpu, hidden, hidden, branch,
                       VDN_PROMPT_ROWS * VDN_HIDDEN),
       "VDN refiner MLP residual");
#undef OP
    return 1;
}

h3_gpu_tensor *h3_vdn_refine_prompt(
        h3_gpu *gpu, const h3_vdn_model_weights *weights,
        const h3_text_embedding *prompt, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!gpu || !weights || !prompt || !prompt->values ||
        prompt->tokens != VDN_PROMPT_ROWS || prompt->width != VDN_TEXT_WIDTH) {
        dit_fail(error, error_size, "invalid VDN prompt refinement arguments");
        return NULL;
    }
    size_t hidden_elements = (size_t)VDN_PROMPT_ROWS * VDN_HIDDEN;
    size_t inner_elements = (size_t)VDN_PROMPT_ROWS * VDN_INNER;
    h3_gpu_tensor *source = h3_gpu_tensor_from_bf16(
        gpu, prompt->values, (size_t)VDN_PROMPT_ROWS * VDN_TEXT_WIDTH);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    h3_gpu_tensor *norm = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    h3_gpu_tensor *query_raw = h3_gpu_tensor_new_bf16(gpu, inner_elements);
    h3_gpu_tensor *key_raw = h3_gpu_tensor_new_bf16(gpu, inner_elements);
    h3_gpu_tensor *value = h3_gpu_tensor_new_bf16(gpu, inner_elements);
    h3_gpu_tensor *query = h3_gpu_tensor_new_bf16(gpu, inner_elements);
    h3_gpu_tensor *key = h3_gpu_tensor_new_bf16(gpu, inner_elements);
    h3_gpu_tensor *attended = h3_gpu_tensor_new_bf16(gpu, inner_elements);
    h3_gpu_tensor *branch = h3_gpu_tensor_new_bf16(gpu, hidden_elements);
    h3_gpu_tensor *fc1 = h3_gpu_tensor_new_bf16(
        gpu, (size_t)VDN_PROMPT_ROWS * VDN_FFN * 2);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_bf16(
        gpu, (size_t)VDN_PROMPT_ROWS * VDN_FFN);
    h3_gpu_tensor *dummy_rope = h3_gpu_tensor_new_bf16(gpu, 1);
    int ok = source && hidden && norm && query_raw && key_raw && value &&
             query && key && attended && branch && fc1 && activated &&
             dummy_rope;
    if (!ok) {
        dit_fail(error, error_size,
                 "cannot allocate VDN refiner activations: %s",
                 h3_gpu_error(gpu));
        goto cleanup;
    }
    ok = dit_op(gpu, h3_gpu_begin(gpu), "begin VDN prompt refinement",
                error, error_size) &&
         dit_op(gpu, h3_gpu_linear_bf16(
                    gpu, hidden, source, weights->context_weight,
                    weights->context_bias, VDN_PROMPT_ROWS, VDN_TEXT_WIDTH,
                    VDN_HIDDEN), "VDN context projection", error, error_size) &&
         run_refiner(gpu, &weights->refiner[0], hidden, norm, query_raw,
                     key_raw, value, query, key, attended, branch, fc1,
                     activated, dummy_rope, error, error_size) &&
         run_refiner(gpu, &weights->refiner[1], hidden, norm, query_raw,
                     key_raw, value, query, key, attended, branch, fc1,
                     activated, dummy_rope, error, error_size) &&
         dit_op(gpu, h3_gpu_rms_norm_bf16(
                    gpu, hidden, hidden, weights->refiner_final_norm,
                    VDN_PROMPT_ROWS, VDN_HIDDEN, 1e-5f),
                "VDN refiner final RMSNorm", error, error_size) &&
         dit_op(gpu, h3_gpu_submit(gpu), "submit VDN prompt refinement",
                error, error_size);
cleanup:
    h3_gpu_tensor_free(dummy_rope);
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(fc1);
    h3_gpu_tensor_free(branch);
    h3_gpu_tensor_free(attended);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(key_raw);
    h3_gpu_tensor_free(query_raw);
    h3_gpu_tensor_free(norm);
    h3_gpu_tensor_free(source);
    if (!ok) {
        h3_gpu_tensor_free(hidden);
        return NULL;
    }
    return hidden;
}

h3_gpu_tensor *h3_vdn_time_embedding(
        h3_gpu *gpu, const h3_vdn_model_weights *weights,
        const float *timesteps, uint32_t rows,
        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!gpu || !weights || !timesteps || !rows) {
        dit_fail(error, error_size, "invalid VDN timestep arguments");
        return NULL;
    }
    size_t input_elements = (size_t)rows * VDN_TIME_INPUT;
    float *features = malloc(input_elements * sizeof(*features));
    if (!features) {
        dit_fail(error, error_size, "out of memory constructing VDN timestep features");
        return NULL;
    }
    for (uint32_t row = 0; row < rows; row++)
        for (uint32_t index = 0; index < VDN_TIME_INPUT / 2; index++) {
            float frequency = expf(-logf(10000.0f) * (float)index /
                                   (float)(VDN_TIME_INPUT / 2));
            float angle = timesteps[row] * frequency;
            features[(size_t)row * VDN_TIME_INPUT + index] = cosf(angle);
            features[(size_t)row * VDN_TIME_INPUT +
                     VDN_TIME_INPUT / 2 + index] = sinf(angle);
        }
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(gpu, features, input_elements);
    free(features);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * VDN_HIDDEN);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * VDN_HIDDEN);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * VDN_TIME_WIDTH);
    int ok = input && hidden && activated && output;
    if (!ok) {
        dit_fail(error, error_size, "cannot allocate VDN timestep activations: %s",
                 h3_gpu_error(gpu));
        goto cleanup;
    }
    ok = dit_op(gpu, h3_gpu_begin(gpu), "begin VDN timestep embedding",
                error, error_size) &&
         dit_op(gpu, h3_gpu_linear_f32(
                    gpu, hidden, input, weights->time_linear1_weight,
                    weights->time_linear1_bias, rows, VDN_TIME_INPUT,
                    VDN_HIDDEN), "VDN timestep input projection",
                error, error_size) &&
         dit_op(gpu, h3_gpu_silu_f32(
                    gpu, activated, hidden, rows * VDN_HIDDEN),
                "VDN timestep SiLU", error, error_size) &&
         dit_op(gpu, h3_gpu_linear_f32(
                    gpu, output, activated, weights->time_linear2_weight,
                    weights->time_linear2_bias, rows, VDN_HIDDEN,
                    VDN_TIME_WIDTH), "VDN timestep output projection",
                error, error_size) &&
         dit_op(gpu, h3_gpu_submit(gpu), "submit VDN timestep embedding",
                error, error_size);
cleanup:
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(input);
    if (!ok) {
        h3_gpu_tensor_free(output);
        return NULL;
    }
    return output;
}

h3_gpu_tensor *h3_vdn_block_modulation(
        h3_gpu *gpu, const h3_vdn_block_weights *weights,
        const h3_gpu_tensor *time_embedding, uint32_t time_rows,
        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!gpu || !weights || !time_embedding || !time_rows) {
        dit_fail(error, error_size, "invalid VDN modulation arguments");
        return NULL;
    }
    size_t time_elements = (size_t)time_rows * VDN_TIME_WIDTH;
    size_t modulation_width =
        (size_t)VDN_MODALITIES * VDN_ADALN_SLOTS * VDN_HIDDEN;
    if (time_elements > UINT32_MAX || modulation_width > UINT32_MAX) {
        dit_fail(error, error_size, "VDN modulation exceeds 32-bit GPU limits");
        return NULL;
    }
    h3_gpu_tensor *activated = h3_gpu_tensor_new_f32(gpu, time_elements);
    h3_gpu_tensor *input = h3_gpu_tensor_new_bf16(gpu, time_elements);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(
        gpu, (size_t)time_rows * modulation_width);
    int ok = activated && input && output;
    if (!ok) {
        dit_fail(error, error_size, "cannot allocate VDN modulation tensors: %s",
                 h3_gpu_error(gpu));
        goto cleanup;
    }
    ok = dit_op(gpu, h3_gpu_begin(gpu), "begin VDN modulation",
                error, error_size) &&
         dit_op(gpu, h3_gpu_silu_f32(
                    gpu, activated, time_embedding, (uint32_t)time_elements),
                "VDN modulation SiLU", error, error_size) &&
         dit_op(gpu, h3_gpu_cast_f32_to_bf16(
                    gpu, input, activated, (uint32_t)time_elements),
                "VDN modulation BF16 cast", error, error_size) &&
         dit_op(gpu, h3_gpu_linear_bf16(
                    gpu, output, input, weights->adaln_weight,
                    weights->adaln_bias, time_rows, VDN_TIME_WIDTH,
                    (uint32_t)modulation_width),
                "VDN block AdaLN projection", error, error_size) &&
         dit_op(gpu, h3_gpu_submit(gpu), "submit VDN modulation",
                error, error_size);
cleanup:
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(activated);
    if (!ok) {
        h3_gpu_tensor_free(output);
        return NULL;
    }
    return output;
}

int h3_vdn_run_block(
        h3_gpu *gpu, const h3_vdn_block_weights *weights,
        h3_gpu_tensor *hidden, const h3_gpu_tensor *modulation,
        const h3_gpu_tensor *row_map, const h3_gpu_tensor *rope_cos,
        const h3_gpu_tensor *rope_sin, uint32_t sequence,
        uint32_t text_rows, uint32_t video_start, uint32_t frames,
        uint32_t frame_height, uint32_t frame_width,
        uint32_t radius, uint32_t chunk,
        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    uint64_t tokens_per_frame_wide =
        (uint64_t)frame_height * frame_width;
    uint64_t video_rows_wide = (uint64_t)frames * tokens_per_frame_wide;
    uint64_t video_end_wide = (uint64_t)video_start + video_rows_wide;
    uint64_t packed_elements_wide = (uint64_t)sequence * VDN_HIDDEN;
    uint64_t qkv_elements_wide = (uint64_t)sequence * VDN_INNER;
    if (!gpu || !weights || !hidden || !modulation || !row_map ||
        !rope_cos || !rope_sin || !sequence || !text_rows ||
        text_rows > video_start || frames <= 2 || !frame_height ||
        !frame_width || tokens_per_frame_wide > UINT32_MAX ||
        video_rows_wide > UINT32_MAX || video_end_wide > sequence ||
        packed_elements_wide > UINT32_MAX || qkv_elements_wide > SIZE_MAX ||
        h3_gpu_tensor_dtype(hidden) != H3_GPU_BF16 ||
        h3_gpu_tensor_elements(hidden) < (size_t)packed_elements_wide) {
        dit_fail(error, error_size, "invalid VDN block execution arguments");
        return 0;
    }
    uint32_t tokens_per_frame = (uint32_t)tokens_per_frame_wide;
    uint32_t inner_frames = frames - 2;
    uint32_t inner_rows = inner_frames * tokens_per_frame;
    size_t packed_elements = (size_t)packed_elements_wide;
    size_t qkv_elements = (size_t)qkv_elements_wide;
    size_t inner_features = (size_t)inner_rows * VDN_INNER;
    size_t inner_hidden = (size_t)inner_rows * VDN_HIDDEN;
    size_t video_feature_offset =
        (size_t)(video_start + tokens_per_frame) * VDN_INNER;
    size_t video_hidden_offset =
        (size_t)(video_start + tokens_per_frame) * VDN_HIDDEN;
    size_t state_elements =
        (size_t)inner_frames * VDN_HEADS * VDN_HEAD_DIM * VDN_HEAD_DIM;
    size_t text_state_elements =
        (size_t)VDN_HEADS * VDN_HEAD_DIM * VDN_HEAD_DIM;

    h3_gpu_tensor *norm_attn = NULL, *query_raw = NULL, *key_raw = NULL;
    h3_gpu_tensor *value_raw = NULL, *query = NULL, *key = NULL;
    h3_gpu_tensor *attended = NULL, *softmax_logits = NULL;
    h3_gpu_tensor *softmax_gated = NULL, *branch = NULL;
    h3_gpu_tensor *video_x = NULL, *video_q_raw = NULL;
    h3_gpu_tensor *video_k_raw = NULL, *video_v_raw = NULL;
    h3_gpu_tensor *linear_q = NULL, *linear_k = NULL, *linear_v = NULL;
    h3_gpu_tensor *video_beta = NULL, *gate_down = NULL;
    h3_gpu_tensor *gate_logits = NULL, *frame_mean = NULL;
    h3_gpu_tensor *alpha_down_w = NULL, *alpha_up_w = NULL;
    h3_gpu_tensor *alpha_hidden = NULL, *alpha_delta = NULL;
    h3_gpu_tensor *alpha = NULL, *a = NULL, *b = NULL;
    h3_gpu_tensor *transition = NULL, *injection = NULL;
    h3_gpu_tensor *text_k = NULL, *text_v = NULL, *text_beta = NULL;
    h3_gpu_tensor *text_a = NULL, *text_b = NULL, *text_alpha = NULL;
    h3_gpu_tensor *text_transition = NULL, *text_injection = NULL;
    h3_gpu_tensor *prefix = NULL, *suffix = NULL, *linear_readout = NULL;
    h3_gpu_tensor *linear_projected = NULL, *video_branch = NULL;
    h3_gpu_tensor *norm_mlp = NULL, *fc1 = NULL, *activated = NULL;
    h3_gpu_tensor *mlp_branch = NULL;
#define BF(field, count) (field = h3_gpu_tensor_new_bf16(gpu, (count)))
#define F32(field, count) (field = h3_gpu_tensor_new_f32(gpu, (count)))
    BF(norm_attn, packed_elements); BF(query_raw, qkv_elements);
    BF(key_raw, qkv_elements); BF(value_raw, qkv_elements);
    BF(query, qkv_elements); BF(key, qkv_elements); BF(attended, qkv_elements);
    BF(softmax_logits, (size_t)sequence * VDN_HEADS);
    BF(softmax_gated, qkv_elements); BF(branch, packed_elements);
    BF(video_x, inner_hidden); BF(video_q_raw, inner_features);
    BF(video_k_raw, inner_features); BF(video_v_raw, inner_features);
    BF(linear_q, inner_features); BF(linear_k, inner_features);
    BF(linear_v, inner_features); BF(video_beta, (size_t)inner_rows * VDN_HEADS);
    BF(gate_down, (size_t)inner_rows * VDN_HEAD_DIM);
    BF(gate_logits, inner_features); F32(frame_mean, (size_t)inner_frames * VDN_HIDDEN);
    F32(alpha_down_w, (size_t)VDN_HEAD_DIM * VDN_HIDDEN);
    F32(alpha_up_w, (size_t)VDN_INNER * VDN_HEAD_DIM);
    F32(alpha_hidden, (size_t)inner_frames * VDN_HEAD_DIM);
    F32(alpha_delta, (size_t)inner_frames * VDN_INNER);
    F32(alpha, (size_t)inner_frames * VDN_INNER);
    F32(a, state_elements); F32(b, state_elements);
    F32(transition, state_elements); F32(injection, state_elements);
    BF(text_k, (size_t)text_rows * VDN_INNER);
    BF(text_v, (size_t)text_rows * VDN_INNER);
    BF(text_beta, (size_t)text_rows * VDN_HEADS);
    F32(text_a, text_state_elements); F32(text_b, text_state_elements);
    float ones[VDN_INNER];
    for (size_t index = 0; index < VDN_INNER; index++) ones[index] = 1.0f;
    text_alpha = h3_gpu_tensor_from_f32(gpu, ones, VDN_INNER);
    F32(text_transition, text_state_elements);
    F32(text_injection, text_state_elements);
    F32(prefix, state_elements); F32(suffix, state_elements);
    BF(linear_readout, inner_features); BF(linear_projected, inner_hidden);
    BF(video_branch, inner_hidden); BF(norm_mlp, packed_elements);
    BF(fc1, (size_t)sequence * VDN_FFN * 2);
    BF(activated, (size_t)sequence * VDN_FFN);
    BF(mlp_branch, packed_elements);
#undef F32
#undef BF
    int ok = norm_attn && query_raw && key_raw && value_raw && query && key &&
        attended && softmax_logits && softmax_gated && branch && video_x &&
        video_q_raw && video_k_raw && video_v_raw && linear_q && linear_k &&
        linear_v && video_beta && gate_down && gate_logits && frame_mean &&
        alpha_down_w && alpha_up_w && alpha_hidden && alpha_delta && alpha &&
        a && b && transition && injection && text_k && text_v && text_beta &&
        text_a && text_b && text_alpha && text_transition && text_injection &&
        prefix && suffix && linear_readout && linear_projected && video_branch &&
        norm_mlp && fc1 && activated && mlp_branch;
    if (!ok) {
        dit_fail(error, error_size, "cannot allocate VDN block activations: %s",
                 h3_gpu_error(gpu));
        goto cleanup;
    }
#define OP(call, label) do {                                                   \
    if (!dit_op(gpu, (call), label, error, error_size)) {                     \
        ok = 0; goto cleanup;                                                  \
    }                                                                         \
} while (0)
    OP(h3_gpu_begin(gpu), "begin VDN transformer block");
    OP(h3_gpu_adaln_bf16(gpu, norm_attn, hidden, weights->norm1, modulation,
                         row_map, sequence, VDN_HIDDEN, VDN_ADALN_SLOTS,
                         0, 1, 1e-5f), "VDN attention AdaLN");
    OP(h3_gpu_linear_bf16(gpu, query_raw, norm_attn, weights->q, NULL,
                          sequence, VDN_HIDDEN, VDN_INNER), "VDN Q projection");
    OP(h3_gpu_linear_bf16(gpu, key_raw, norm_attn, weights->k, NULL,
                          sequence, VDN_HIDDEN, VDN_INNER), "VDN K projection");
    OP(h3_gpu_linear_bf16(gpu, value_raw, norm_attn, weights->v, NULL,
                          sequence, VDN_HIDDEN, VDN_INNER), "VDN V projection");
    OP(h3_gpu_vdn_qk_rope_bf16(
           gpu, query, key, query_raw, key_raw, weights->q_norm,
           weights->k_norm, rope_cos, rope_sin, sequence, VDN_HEADS,
           VDN_HEAD_DIM, VDN_ROPE_HALF, 1e-5f), "VDN QK norm/RoPE");
    OP(h3_gpu_vdn_window_sdpa_bf16(
           gpu, attended, query, key, value_raw, sequence, VDN_HEADS,
           VDN_HEAD_DIM, video_start, frames, tokens_per_frame, radius, chunk,
           1, 1.0f / sqrtf((float)VDN_HEAD_DIM)), "VDN window attention");
    OP(h3_gpu_linear_bf16(gpu, softmax_logits, norm_attn,
                          weights->linear.softmax_gate_weight,
                          weights->linear.softmax_gate_bias, sequence,
                          VDN_HIDDEN, VDN_HEADS), "VDN softmax gate projection");
    OP(h3_gpu_vdn_softmax_gate_bf16(
           gpu, softmax_gated, attended, softmax_logits, sequence,
           VDN_HEADS, VDN_HEAD_DIM), "VDN softmax gate");
    OP(h3_gpu_linear_bf16(gpu, branch, softmax_gated, weights->out, NULL,
                          sequence, VDN_INNER, VDN_HIDDEN),
       "VDN softmax output projection");

    OP(h3_gpu_copy_bf16(gpu, video_x, 0, norm_attn, video_hidden_offset,
                        inner_hidden), "copy VDN linear video hidden");
    OP(h3_gpu_copy_bf16(gpu, video_q_raw, 0, query_raw,
                        video_feature_offset, inner_features),
       "copy VDN linear raw Q");
    OP(h3_gpu_copy_bf16(gpu, video_k_raw, 0, key_raw,
                        video_feature_offset, inner_features),
       "copy VDN linear raw K");
    OP(h3_gpu_copy_bf16(gpu, video_v_raw, 0, value_raw,
                        video_feature_offset, inner_features),
       "copy VDN linear raw V");
    OP(h3_gpu_vdn_linear_features_bf16(
           gpu, linear_q, linear_k, linear_v, video_q_raw, video_k_raw,
           video_v_raw, weights->linear.k_spatial, weights->linear.k_temporal,
           weights->linear.v_spatial, weights->linear.v_temporal,
           inner_frames, frame_height, frame_width, VDN_HEADS, VDN_HEAD_DIM,
           1e-6f), "VDN linear video features");
    OP(h3_gpu_linear_bf16(gpu, video_beta, video_x, weights->linear.beta, NULL,
                          inner_rows, VDN_HIDDEN, VDN_HEADS),
       "VDN video beta projection");
    OP(h3_gpu_linear_bf16(gpu, gate_down, video_x,
                          weights->linear.gate_down, NULL, inner_rows,
                          VDN_HIDDEN, VDN_HEAD_DIM), "VDN output gate down");
    OP(h3_gpu_linear_bf16(gpu, gate_logits, gate_down,
                          weights->linear.gate_up,
                          weights->linear.gate_up_bias, inner_rows,
                          VDN_HEAD_DIM, VDN_INNER), "VDN output gate up");
    OP(h3_gpu_vdn_frame_mean_bf16(gpu, frame_mean, video_x, 0,
                                  inner_frames, tokens_per_frame, VDN_HIDDEN),
       "VDN frame mean");
    OP(h3_gpu_cast_bf16_to_f32(
           gpu, alpha_down_w, weights->linear.alpha_down,
           VDN_HEAD_DIM * VDN_HIDDEN), "VDN alpha down weight promotion");
    OP(h3_gpu_cast_bf16_to_f32(
           gpu, alpha_up_w, weights->linear.alpha_up,
           VDN_INNER * VDN_HEAD_DIM), "VDN alpha up weight promotion");
    OP(h3_gpu_linear_f32(gpu, alpha_hidden, frame_mean, alpha_down_w, NULL,
                         inner_frames, VDN_HIDDEN, VDN_HEAD_DIM),
       "VDN alpha down projection");
    OP(h3_gpu_linear_f32(gpu, alpha_delta, alpha_hidden, alpha_up_w, NULL,
                         inner_frames, VDN_HEAD_DIM, VDN_INNER),
       "VDN alpha up projection");
    OP(h3_gpu_vdn_alpha_f32(
           gpu, alpha, alpha_delta, weights->linear.alpha_dt_bias,
           weights->linear.alpha_a_log, inner_frames, VDN_HEADS,
           VDN_HEAD_DIM), "VDN FP32 alpha");
    OP(h3_gpu_vdn_frame_stats_bf16(
           gpu, a, b, linear_k, linear_v, video_beta, inner_frames,
           tokens_per_frame, VDN_HEADS, VDN_HEAD_DIM),
       "VDN video frame statistics");
    OP(h3_gpu_vdn_solve_f32(gpu, transition, injection, a, b, alpha,
                            inner_frames, VDN_HEADS, VDN_HEAD_DIM),
       "VDN video Cholesky solve");

    OP(h3_gpu_vdn_text_features_bf16(
           gpu, text_k, text_v, key_raw, value_raw, text_rows,
           VDN_HEADS, VDN_HEAD_DIM, 1e-6f), "VDN text features");
    OP(h3_gpu_linear_bf16(gpu, text_beta, norm_attn,
                          weights->linear.beta, NULL, text_rows,
                          VDN_HIDDEN, VDN_HEADS), "VDN text beta projection");
    OP(h3_gpu_vdn_frame_stats_bf16(
           gpu, text_a, text_b, text_k, text_v, text_beta, 1, text_rows,
           VDN_HEADS, VDN_HEAD_DIM), "VDN text statistics");
    OP(h3_gpu_vdn_solve_f32(
           gpu, text_transition, text_injection, text_a, text_b, text_alpha,
           1, VDN_HEADS, VDN_HEAD_DIM), "VDN text-state Cholesky solve");
    OP(h3_gpu_vdn_scan_f32(
           gpu, prefix, suffix, transition, injection, text_injection, 0.5f,
           inner_frames, VDN_HEADS, VDN_HEAD_DIM), "VDN bidirectional scan");
    OP(h3_gpu_vdn_readout_bf16(
           gpu, linear_readout, linear_q, prefix, suffix, alpha,
           text_injection, 0.5f, weights->linear.norm, gate_logits,
           inner_frames, tokens_per_frame, VDN_HEADS, VDN_HEAD_DIM,
           radius, chunk, 1e-6f), "VDN linear readout");
    OP(h3_gpu_linear_bf16(gpu, linear_projected, linear_readout,
                          weights->linear.to_out, NULL, inner_rows,
                          VDN_INNER, VDN_HIDDEN),
       "VDN linear output projection");
    OP(h3_gpu_copy_bf16(gpu, video_branch, 0, branch,
                        video_hidden_offset, inner_hidden),
       "copy VDN window branch rows");
    OP(h3_gpu_add_bf16(gpu, video_branch, video_branch, linear_projected,
                       (uint32_t)inner_hidden), "merge VDN attention branches");
    OP(h3_gpu_copy_bf16(gpu, branch, video_hidden_offset, video_branch, 0,
                        inner_hidden), "store VDN merged video rows");

    OP(h3_gpu_gate_bf16(gpu, hidden, hidden, branch, modulation, row_map,
                        sequence, VDN_HIDDEN, VDN_ADALN_SLOTS, 2),
       "VDN attention residual gate");
    OP(h3_gpu_adaln_bf16(gpu, norm_mlp, hidden, weights->norm2, modulation,
                         row_map, sequence, VDN_HIDDEN, VDN_ADALN_SLOTS,
                         3, 4, 1e-5f), "VDN MLP AdaLN");
    OP(h3_gpu_linear_bf16(gpu, fc1, norm_mlp, weights->fc1, NULL,
                          sequence, VDN_HIDDEN, VDN_FFN * 2),
       "VDN MLP input projection");
    OP(h3_gpu_swiglu_bf16(gpu, activated, fc1, sequence, VDN_FFN),
       "VDN SwiGLU");
    OP(h3_gpu_linear_bf16(gpu, mlp_branch, activated, weights->fc2, NULL,
                          sequence, VDN_FFN, VDN_HIDDEN),
       "VDN MLP output projection");
    OP(h3_gpu_gate_bf16(gpu, hidden, hidden, mlp_branch, modulation, row_map,
                        sequence, VDN_HIDDEN, VDN_ADALN_SLOTS, 5),
       "VDN MLP residual gate");
    OP(h3_gpu_submit(gpu), "submit VDN transformer block");
#undef OP
cleanup:
#define FREE(field) h3_gpu_tensor_free(field)
    FREE(mlp_branch); FREE(activated); FREE(fc1); FREE(norm_mlp);
    FREE(video_branch); FREE(linear_projected); FREE(linear_readout);
    FREE(suffix); FREE(prefix); FREE(text_injection); FREE(text_transition);
    FREE(text_alpha); FREE(text_b); FREE(text_a); FREE(text_beta);
    FREE(text_v); FREE(text_k); FREE(injection); FREE(transition); FREE(b);
    FREE(a); FREE(alpha); FREE(alpha_delta); FREE(alpha_hidden);
    FREE(alpha_up_w); FREE(alpha_down_w); FREE(frame_mean); FREE(gate_logits);
    FREE(gate_down); FREE(video_beta); FREE(linear_v); FREE(linear_k);
    FREE(linear_q); FREE(video_v_raw); FREE(video_k_raw); FREE(video_q_raw);
    FREE(video_x); FREE(branch); FREE(softmax_gated); FREE(softmax_logits);
    FREE(attended); FREE(key); FREE(query); FREE(value_raw); FREE(key_raw);
    FREE(query_raw); FREE(norm_attn);
#undef FREE
    return ok;
}

void h3_vdn_velocity_free(h3_vdn_velocity *velocity) {
    if (!velocity) return;
    h3_gpu_tensor_free(velocity->video);
    h3_gpu_tensor_free(velocity->audio);
    memset(velocity, 0, sizeof(*velocity));
}

static h3_gpu_tensor *final_modulation(
        h3_gpu *gpu, const h3_vdn_model_weights *weights,
        const h3_gpu_tensor *time_embedding, uint32_t time_rows,
        char *error, size_t error_size) {
    size_t time_elements = (size_t)time_rows * VDN_TIME_WIDTH;
    if (time_elements > UINT32_MAX) {
        dit_fail(error, error_size, "VDN final modulation is too large");
        return NULL;
    }
    h3_gpu_tensor *activated = h3_gpu_tensor_new_f32(gpu, time_elements);
    h3_gpu_tensor *input = h3_gpu_tensor_new_bf16(gpu, time_elements);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(
        gpu, (size_t)time_rows * 2 * VDN_HIDDEN);
    int ok = activated && input && output;
    if (!ok) {
        dit_fail(error, error_size,
                 "cannot allocate VDN final modulation tensors: %s",
                 h3_gpu_error(gpu));
        goto cleanup;
    }
    ok = dit_op(gpu, h3_gpu_begin(gpu), "begin VDN final modulation",
                error, error_size) &&
         dit_op(gpu, h3_gpu_silu_f32(
                    gpu, activated, time_embedding, (uint32_t)time_elements),
                "VDN final modulation SiLU", error, error_size) &&
         dit_op(gpu, h3_gpu_cast_f32_to_bf16(
                    gpu, input, activated, (uint32_t)time_elements),
                "VDN final modulation BF16 cast", error, error_size) &&
         dit_op(gpu, h3_gpu_linear_bf16(
                    gpu, output, input, weights->final_adaln_weight,
                    weights->final_adaln_bias, time_rows, VDN_TIME_WIDTH,
                    2 * VDN_HIDDEN), "VDN final AdaLN projection",
                error, error_size) &&
         dit_op(gpu, h3_gpu_submit(gpu), "submit VDN final modulation",
                error, error_size);
cleanup:
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(activated);
    if (!ok) {
        h3_gpu_tensor_free(output);
        return NULL;
    }
    return output;
}

int h3_vdn_forward(h3_gpu *gpu, h3_vdn_weight_store *store,
                   const h3_vdn_model_weights *weights,
                   const h3_gpu_tensor *refined_prompt,
                   const h3_vdn_layout *layout,
                   const h3_gpu_tensor *video_rows,
                   const h3_gpu_tensor *audio_rows,
                   float video_timestep, float audio_timestep,
                   uint32_t radius, uint32_t chunk,
                   h3_vdn_layer_progress progress, void *progress_opaque,
                   h3_vdn_velocity *velocity,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (velocity) memset(velocity, 0, sizeof(*velocity));
    size_t packed_elements = layout ?
        (size_t)layout->sequence * VDN_HIDDEN : 0;
    size_t video_elements = layout ?
        (size_t)layout->video_rows * VDN_VIDEO_PATCH : 0;
    size_t audio_elements = layout ?
        (size_t)layout->audio_rows * VDN_AUDIO_WIDTH : 0;
    size_t rope_elements = layout ?
        (size_t)layout->sequence * VDN_ROPE_HALF : 0;
    if (!gpu || !store || !weights || !refined_prompt || !layout ||
        !video_rows || (layout && layout->audio_rows && !audio_rows) ||
        !velocity || !layout->sequence || layout->text_rows != VDN_PROMPT_ROWS ||
        layout->video_start != layout->audio_start + layout->audio_rows ||
        layout->sequence != layout->video_start + layout->video_rows ||
        !isfinite(video_timestep) || !isfinite(audio_timestep) ||
        video_timestep < 0.0f || video_timestep >= 1.0f ||
        audio_timestep < 0.0f || audio_timestep >= 1.0f ||
        h3_gpu_tensor_dtype(refined_prompt) != H3_GPU_BF16 ||
        h3_gpu_tensor_elements(refined_prompt) <
            (size_t)VDN_PROMPT_ROWS * VDN_HIDDEN ||
        h3_gpu_tensor_dtype(video_rows) != H3_GPU_F32 ||
        h3_gpu_tensor_elements(video_rows) < video_elements ||
        (layout->audio_rows &&
         (h3_gpu_tensor_dtype(audio_rows) != H3_GPU_F32 ||
          h3_gpu_tensor_elements(audio_rows) < audio_elements)) ||
        !layout->token_tags || !layout->rope_cos || !layout->rope_sin ||
        packed_elements > UINT32_MAX) {
        dit_fail(error, error_size, "invalid VDN forward arguments");
        return 0;
    }
    int same_time = video_timestep == audio_timestep;
    float timesteps[2];
    uint32_t video_time_row = 0, audio_time_row = 0;
    uint32_t time_rows = same_time ? 1u : 2u;
    if (same_time) {
        timesteps[0] = video_timestep;
    } else if (video_timestep < audio_timestep) {
        timesteps[0] = video_timestep;
        timesteps[1] = audio_timestep;
        audio_time_row = 1;
    } else {
        timesteps[0] = audio_timestep;
        timesteps[1] = video_timestep;
        video_time_row = 1;
    }
    uint32_t *time_map_host = malloc(
        (size_t)layout->sequence * sizeof(*time_map_host));
    uint32_t *adaln_map_host = malloc(
        (size_t)layout->sequence * sizeof(*adaln_map_host));
    if (!time_map_host || !adaln_map_host) {
        free(time_map_host); free(adaln_map_host);
        dit_fail(error, error_size, "out of memory building VDN timestep maps");
        return 0;
    }
    for (uint32_t row = 0; row < layout->sequence; row++) {
        uint32_t time_row = row >= layout->audio_start &&
                            row < layout->video_start ?
                            audio_time_row : video_time_row;
        time_map_host[row] = time_row;
        adaln_map_host[row] = time_row * VDN_MODALITIES +
                              layout->token_tags[row];
    }

    h3_gpu_tensor *time_map = h3_gpu_tensor_from_u32(
        gpu, time_map_host, layout->sequence);
    h3_gpu_tensor *adaln_map = h3_gpu_tensor_from_u32(
        gpu, adaln_map_host, layout->sequence);
    free(adaln_map_host); free(time_map_host);
    h3_gpu_tensor *rope_cos = h3_gpu_tensor_from_bf16(
        gpu, layout->rope_cos, rope_elements);
    h3_gpu_tensor *rope_sin = h3_gpu_tensor_from_bf16(
        gpu, layout->rope_sin, rope_elements);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(gpu, packed_elements);
    h3_gpu_tensor *time_embedding = NULL, *modulation = NULL;
    h3_gpu_tensor *final_mod = NULL, *normalized = NULL;
    h3_gpu_tensor *video_bf16 = NULL, *audio_bf16 = NULL;
    h3_gpu_tensor *video_f32 = NULL, *audio_f32 = NULL;
    h3_vdn_block_weights block;
    memset(&block, 0, sizeof(block));
    int ok = time_map && adaln_map && rope_cos && rope_sin && hidden;
    if (!ok) {
        dit_fail(error, error_size,
                 "cannot allocate VDN packed forward tensors: %s",
                 h3_gpu_error(gpu));
        goto cleanup;
    }
    ok = dit_op(gpu, h3_gpu_begin(gpu), "begin VDN input packing",
                error, error_size) &&
         dit_op(gpu, h3_gpu_copy_bf16(
                    gpu, hidden, 0, refined_prompt, 0,
                    (size_t)layout->text_rows * VDN_HIDDEN),
                "pack VDN text rows", error, error_size);
    if (ok && layout->audio_rows)
        ok = dit_op(gpu, h3_gpu_patch_linear_bf16_offset(
                        gpu, hidden,
                        (size_t)layout->audio_start * VDN_HIDDEN,
                        audio_rows, 0, weights->audio_in_weight,
                        weights->audio_in_bias, layout->audio_rows,
                        VDN_AUDIO_WIDTH, VDN_HIDDEN),
                    "pack VDN audio rows", error, error_size);
    if (ok)
        ok = dit_op(gpu, h3_gpu_patch_linear_bf16_offset(
                        gpu, hidden,
                        (size_t)layout->video_start * VDN_HIDDEN,
                        video_rows, 0, weights->video_in_weight,
                        weights->video_in_bias, layout->video_rows,
                        VDN_VIDEO_PATCH, VDN_HIDDEN),
                    "pack VDN video rows", error, error_size) &&
             dit_op(gpu, h3_gpu_submit(gpu), "submit VDN input packing",
                    error, error_size);
    if (!ok) goto cleanup;

    time_embedding = h3_vdn_time_embedding(
        gpu, weights, timesteps, time_rows, error, error_size);
    if (!time_embedding) { ok = 0; goto cleanup; }
    for (unsigned layer = 0; layer < VDN_BLOCKS; layer++) {
        if (!h3_vdn_block_weights_load(store, gpu, layer, &block,
                                       error, error_size)) {
            ok = 0; goto cleanup;
        }
        modulation = h3_vdn_block_modulation(
            gpu, &block, time_embedding, time_rows, error, error_size);
        if (!modulation || !h3_vdn_run_block(
                gpu, &block, hidden, modulation, adaln_map, rope_cos,
                rope_sin, layout->sequence, layout->text_rows,
                layout->video_start, layout->frames, layout->frame_height,
                layout->frame_width, radius, chunk, error, error_size)) {
            ok = 0; goto cleanup;
        }
        h3_gpu_tensor_free(modulation);
        modulation = NULL;
        h3_vdn_block_weights_free(&block);
        if (progress) progress(layer + 1, VDN_BLOCKS, progress_opaque);
    }

    final_mod = final_modulation(gpu, weights, time_embedding, time_rows,
                                 error, error_size);
    normalized = h3_gpu_tensor_new_bf16(gpu, packed_elements);
    video_bf16 = h3_gpu_tensor_new_bf16(
        gpu, (size_t)layout->video_rows * VDN_HIDDEN);
    video_f32 = h3_gpu_tensor_new_f32(
        gpu, (size_t)layout->video_rows * VDN_HIDDEN);
    velocity->video = h3_gpu_tensor_new_f32(gpu, video_elements);
    if (layout->audio_rows) {
        audio_bf16 = h3_gpu_tensor_new_bf16(
            gpu, (size_t)layout->audio_rows * VDN_HIDDEN);
        audio_f32 = h3_gpu_tensor_new_f32(
            gpu, (size_t)layout->audio_rows * VDN_HIDDEN);
        velocity->audio = h3_gpu_tensor_new_f32(gpu, audio_elements);
    }
    ok = final_mod && normalized && video_bf16 && video_f32 &&
         velocity->video && (!layout->audio_rows ||
         (audio_bf16 && audio_f32 && velocity->audio));
    if (!ok) {
        dit_fail(error, error_size,
                 "cannot allocate VDN output-head tensors: %s",
                 h3_gpu_error(gpu));
        goto cleanup;
    }
    ok = dit_op(gpu, h3_gpu_begin(gpu), "begin VDN output heads",
                error, error_size) &&
         dit_op(gpu, h3_gpu_adaln_bf16(
                    gpu, normalized, hidden, weights->final_norm, final_mod,
                    time_map, layout->sequence, VDN_HIDDEN, 2, 0, 1, 1e-5f),
                "VDN final AdaLN", error, error_size) &&
         dit_op(gpu, h3_gpu_copy_bf16(
                    gpu, video_bf16, 0, normalized,
                    (size_t)layout->video_start * VDN_HIDDEN,
                    (size_t)layout->video_rows * VDN_HIDDEN),
                "select VDN video head rows", error, error_size) &&
         dit_op(gpu, h3_gpu_cast_bf16_to_f32(
                    gpu, video_f32, video_bf16,
                    layout->video_rows * VDN_HIDDEN),
                "promote VDN video head rows", error, error_size) &&
         dit_op(gpu, h3_gpu_linear_f32(
                    gpu, velocity->video, video_f32,
                    weights->video_out_weight, weights->video_out_bias,
                    layout->video_rows, VDN_HIDDEN, VDN_VIDEO_PATCH),
                "VDN video output head", error, error_size);
    if (ok && layout->audio_rows)
        ok = dit_op(gpu, h3_gpu_copy_bf16(
                        gpu, audio_bf16, 0, normalized,
                        (size_t)layout->audio_start * VDN_HIDDEN,
                        (size_t)layout->audio_rows * VDN_HIDDEN),
                    "select VDN audio head rows", error, error_size) &&
             dit_op(gpu, h3_gpu_cast_bf16_to_f32(
                        gpu, audio_f32, audio_bf16,
                        layout->audio_rows * VDN_HIDDEN),
                    "promote VDN audio head rows", error, error_size) &&
             dit_op(gpu, h3_gpu_linear_f32(
                        gpu, velocity->audio, audio_f32,
                        weights->audio_out_weight, weights->audio_out_bias,
                        layout->audio_rows, VDN_HIDDEN, VDN_AUDIO_WIDTH),
                    "VDN audio output head", error, error_size);
    if (ok)
        ok = dit_op(gpu, h3_gpu_submit(gpu), "submit VDN output heads",
                    error, error_size);

cleanup:
    h3_vdn_block_weights_free(&block);
    h3_gpu_tensor_free(audio_f32); h3_gpu_tensor_free(video_f32);
    h3_gpu_tensor_free(audio_bf16); h3_gpu_tensor_free(video_bf16);
    h3_gpu_tensor_free(normalized); h3_gpu_tensor_free(final_mod);
    h3_gpu_tensor_free(modulation); h3_gpu_tensor_free(time_embedding);
    h3_gpu_tensor_free(hidden); h3_gpu_tensor_free(rope_sin);
    h3_gpu_tensor_free(rope_cos); h3_gpu_tensor_free(adaln_map);
    h3_gpu_tensor_free(time_map);
    if (!ok) h3_vdn_velocity_free(velocity);
    return ok;
}

int h3_vdn_denoise(h3_gpu *gpu, h3_vdn_weight_store *store,
                   const h3_vdn_model_weights *weights,
                   const h3_gpu_tensor *refined_prompt,
                   const h3_vdn_layout *layout,
                   h3_gpu_tensor *video_rows, h3_gpu_tensor *audio_rows,
                   unsigned evaluations, uint32_t radius, uint32_t chunk,
                   h3_vdn_layer_progress layer_progress,
                   h3_vdn_nfe_progress nfe_progress, void *progress_opaque,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!gpu || !store || !weights || !refined_prompt || !layout ||
        !video_rows || (layout->audio_rows && !audio_rows) ||
        evaluations < 1 || evaluations > H3_MAX_STEPS) {
        dit_fail(error, error_size, "invalid VDN denoising arguments");
        return 0;
    }
    h3_sigma_schedule schedule;
    if (!h3_serving_schedule_build((int)evaluations, &schedule)) {
        dit_fail(error, error_size, "cannot build VDN paired sigma schedule");
        return 0;
    }
    uint64_t video_elements_wide =
        (uint64_t)layout->video_rows * VDN_VIDEO_PATCH;
    uint64_t audio_elements_wide =
        (uint64_t)layout->audio_rows * VDN_AUDIO_WIDTH;
    if (video_elements_wide > UINT32_MAX ||
        audio_elements_wide > UINT32_MAX) {
        dit_fail(error, error_size, "VDN latent rows exceed 32-bit GPU limits");
        return 0;
    }
    uint32_t video_elements = (uint32_t)video_elements_wide;
    uint32_t audio_elements = (uint32_t)audio_elements_wide;
    for (unsigned step = 0; step < evaluations; step++) {
        float video_timestep = 1.0f - schedule.video[step];
        float audio_timestep = 1.0f - schedule.audio[step];
        h3_vdn_velocity velocity;
        memset(&velocity, 0, sizeof(velocity));
        if (!h3_vdn_forward(
                gpu, store, weights, refined_prompt, layout, video_rows,
                audio_rows, video_timestep, audio_timestep, radius, chunk,
                layer_progress, progress_opaque, &velocity,
                error, error_size)) return 0;
        /* Match MiniMaxH3Scheduler exactly: sigma_from_t is recovered from
         * the rounded model timestep, while the ratio uses the sigma grid. */
        float video_sigma_from_t = 1.0f - video_timestep;
        float audio_sigma_from_t = 1.0f - audio_timestep;
        float video_ratio = schedule.video[step + 1] / schedule.video[step];
        float audio_ratio = schedule.audio[step + 1] / schedule.audio[step];
        float video_scale = (1.0f - video_ratio) * video_sigma_from_t;
        float audio_scale = (1.0f - audio_ratio) * audio_sigma_from_t;
        int ok = dit_op(gpu, h3_gpu_begin(gpu), "begin VDN Euler step",
                        error, error_size) &&
                 dit_op(gpu, h3_gpu_euler_f32(
                            gpu, video_rows, velocity.video, video_elements,
                            video_scale), "VDN video Euler step",
                        error, error_size);
        if (ok && audio_elements)
            ok = dit_op(gpu, h3_gpu_euler_f32(
                            gpu, audio_rows, velocity.audio, audio_elements,
                            audio_scale), "VDN audio Euler step",
                        error, error_size);
        if (ok)
            ok = dit_op(gpu, h3_gpu_submit(gpu), "submit VDN Euler step",
                        error, error_size);
        h3_vdn_velocity_free(&velocity);
        if (!ok) return 0;
        if (nfe_progress) nfe_progress(step + 1, evaluations, progress_opaque);
    }
    return 1;
}
