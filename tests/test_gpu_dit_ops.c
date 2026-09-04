#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000))
        bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    else if (bits & UINT32_C(0xffff))
        bits |= UINT32_C(0x10000);
    return (uint16_t)(bits >> 16);
}

static float f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int compare(const uint16_t *actual, const float *expected,
                   size_t count, float tolerance, const char *label) {
    for (size_t index = 0; index < count; index++) {
        float got = f32(actual[index]);
        if (fabsf(got - expected[index]) > tolerance) {
            fprintf(stderr, "%s[%zu]: got %.7f expected %.7f\n",
                    label, index, got, expected[index]);
            return 0;
        }
    }
    return 1;
}

#define CHECK(expression)                                                      \
    do {                                                                       \
        if (!(expression)) {                                                   \
            fprintf(stderr, "failed: %s: %s\n", #expression,               \
                    h3_gpu_error(gpu));                                        \
            ok = 0;                                                            \
            goto done;                                                         \
        }                                                                      \
    } while (0)

int main(void) {
    int ok = 1;
    char error[512];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "cannot create HIP context: %s\n", error);
        return 1;
    }

    h3_gpu_tensor *owned[32] = {0};
    size_t owned_count = 0;
#define OWN(value) (owned[owned_count++] = (value))

    const float residual_f[] = {1, 2, 3, 4, -1, 0, 1, 2};
    const float branch_f[] = {2, -2, 1, 0.5f, 1, 2, -1, -2};
    const float norm_f[] = {1, 0.5f, 1.5f, -1};
    uint16_t residual[8], branch[8], norm[4], modulation[24];
    for (size_t i = 0; i < 8; i++) {
        residual[i] = bf16(residual_f[i]);
        branch[i] = bf16(branch_f[i]);
    }
    for (size_t i = 0; i < 4; i++) norm[i] = bf16(norm_f[i]);
    /* Two modulation rows, slots = [gate, shift, scale]. */
    for (size_t row = 0; row < 2; row++)
        for (size_t column = 0; column < 4; column++) {
            modulation[row * 12 + column] = bf16(row ? -0.25f : 0.5f);
            modulation[row * 12 + 4 + column] =
                bf16(0.1f * (float)(column + 1));
            modulation[row * 12 + 8 + column] =
                bf16(row ? 0.0f : 0.2f);
        }
    const uint32_t row_map_host[] = {1, 0};
    h3_gpu_tensor *residual_t = OWN(h3_gpu_tensor_from_bf16(gpu, residual, 8));
    h3_gpu_tensor *branch_t = OWN(h3_gpu_tensor_from_bf16(gpu, branch, 8));
    h3_gpu_tensor *norm_t = OWN(h3_gpu_tensor_from_bf16(gpu, norm, 4));
    h3_gpu_tensor *mod_t = OWN(h3_gpu_tensor_from_bf16(gpu, modulation, 24));
    h3_gpu_tensor *map_t = OWN(h3_gpu_tensor_from_u32(gpu, row_map_host, 2));
    h3_gpu_tensor *gate_t = OWN(h3_gpu_tensor_new_bf16(gpu, 8));
    h3_gpu_tensor *adaln_t = OWN(h3_gpu_tensor_new_bf16(gpu, 8));
    h3_gpu_tensor *fused_gate_t = OWN(h3_gpu_tensor_new_bf16(gpu, 8));
    h3_gpu_tensor *fused_adaln_t = OWN(h3_gpu_tensor_new_bf16(gpu, 8));
    CHECK(residual_t && branch_t && norm_t && mod_t && map_t && gate_t &&
          adaln_t && fused_gate_t && fused_adaln_t);

    /* Grouped layout for one head: Q[4], K[4], V[4] per row. */
    const float qkv_f[] = {
        1, 2, 3, 4, 4, 3, 2, 1, 1, 0, -1, 2,
        2, 1, 0, -1, 1, -1, 2, -2, 0.5f, 1.5f, -0.5f, 2.5f
    };
    uint16_t qkv[24], one[4], cosine[4], sine[4];
    for (size_t i = 0; i < 24; i++) qkv[i] = bf16(qkv_f[i]);
    for (size_t i = 0; i < 4; i++) one[i] = bf16(1.0f);
    for (size_t i = 0; i < 4; i++) {
        cosine[i] = bf16(1.0f);
        sine[i] = bf16(0.0f);
    }
    h3_gpu_tensor *qkv_t = OWN(h3_gpu_tensor_from_bf16(gpu, qkv, 24));
    h3_gpu_tensor *one_t = OWN(h3_gpu_tensor_from_bf16(gpu, one, 4));
    h3_gpu_tensor *cos_t = OWN(h3_gpu_tensor_from_bf16(gpu, cosine, 4));
    h3_gpu_tensor *sin_t = OWN(h3_gpu_tensor_from_bf16(gpu, sine, 4));
    h3_gpu_tensor *query_t = OWN(h3_gpu_tensor_new_bf16(gpu, 8));
    h3_gpu_tensor *key_t = OWN(h3_gpu_tensor_new_bf16(gpu, 8));
    h3_gpu_tensor *value_t = OWN(h3_gpu_tensor_new_bf16(gpu, 8));
    h3_gpu_tensor *attention_t = OWN(h3_gpu_tensor_new_bf16(gpu, 8));
    CHECK(qkv_t && one_t && cos_t && sin_t && query_t && key_t && value_t &&
          attention_t);

    const float patch_input_f[] = {1, 2, 3, -1, 0.5f, 2};
    const float patch_weight_f[] = {1, 0, -1, 2, 1, 0};
    const float patch_bias_f[] = {0.25f, -0.5f};
    h3_gpu_tensor *patch_input_t = OWN(
        h3_gpu_tensor_from_f32(gpu, patch_input_f, 6));
    h3_gpu_tensor *patch_weight_t = OWN(
        h3_gpu_tensor_from_f32(gpu, patch_weight_f, 6));
    h3_gpu_tensor *patch_bias_t = OWN(
        h3_gpu_tensor_from_f32(gpu, patch_bias_f, 2));
    h3_gpu_tensor *patch_output_t = OWN(h3_gpu_tensor_new_bf16(gpu, 4));
    CHECK(patch_input_t && patch_weight_t && patch_bias_t && patch_output_t);

    const float mlp_input_f[] = {1.0f, -2.0f};
    const float mlp_fc1_f[] = {1, 0, 0, 1, 2, 1, -1, 1};
    const float mlp_fc2_f[] = {1, 2, -1, 0.5f};
    uint16_t mlp_input[2], mlp_fc1[8], mlp_fc2[4];
    for (size_t i = 0; i < 2; i++) mlp_input[i] = bf16(mlp_input_f[i]);
    for (size_t i = 0; i < 8; i++) mlp_fc1[i] = bf16(mlp_fc1_f[i]);
    for (size_t i = 0; i < 4; i++) mlp_fc2[i] = bf16(mlp_fc2_f[i]);
    h3_gpu_tensor *mlp_input_t = OWN(
        h3_gpu_tensor_from_bf16(gpu, mlp_input, 2));
    h3_gpu_tensor *mlp_fc1_t = OWN(
        h3_gpu_tensor_from_bf16(gpu, mlp_fc1, 8));
    h3_gpu_tensor *mlp_fc2_t = OWN(
        h3_gpu_tensor_from_bf16(gpu, mlp_fc2, 4));
    h3_gpu_tensor *mlp_output_t = OWN(h3_gpu_tensor_new_bf16(gpu, 2));
    CHECK(mlp_input_t && mlp_fc1_t && mlp_fc2_t && mlp_output_t);

    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_gate_bf16(gpu, gate_t, residual_t, branch_t, mod_t, map_t,
                           2, 4, 3, 0));
    CHECK(h3_gpu_adaln_bf16(gpu, adaln_t, gate_t, norm_t, mod_t, map_t,
                            2, 4, 3, 1, 2, 1.0e-5f));
    CHECK(h3_gpu_gate_adaln_bf16(
        gpu, fused_gate_t, fused_adaln_t, residual_t, branch_t, norm_t,
        mod_t, mod_t, map_t, 2, 4, 3, 0, 1, 2, 1.0e-5f));
    CHECK(h3_gpu_grouped_qkv_rope_bf16(
        gpu, query_t, key_t, value_t, qkv_t, one_t, one_t, cos_t, sin_t,
        2, 1, 4, 2, 1.0e-5f));
    CHECK(h3_gpu_sdpa_bf16(gpu, attention_t, query_t, key_t, value_t,
                           2, 1, 4, 0.5f));
    CHECK(h3_gpu_patch_linear_bf16(gpu, patch_output_t, patch_input_t,
                                   patch_weight_t, patch_bias_t, 2, 3, 2));
    CHECK(h3_gpu_mlp_bf16(gpu, mlp_output_t, mlp_input_t, mlp_fc1_t,
                          mlp_fc2_t, 1, 2, 2, 2));
    CHECK(h3_gpu_submit(gpu));

    uint16_t standalone_gate[8], standalone_adaln[8];
    uint16_t fused_gate[8], fused_adaln[8];
    CHECK(h3_gpu_tensor_read_bf16(gate_t, standalone_gate, 8));
    CHECK(h3_gpu_tensor_read_bf16(adaln_t, standalone_adaln, 8));
    CHECK(h3_gpu_tensor_read_bf16(fused_gate_t, fused_gate, 8));
    CHECK(h3_gpu_tensor_read_bf16(fused_adaln_t, fused_adaln, 8));
    if (memcmp(standalone_gate, fused_gate, sizeof(fused_gate)) ||
        memcmp(standalone_adaln, fused_adaln, sizeof(fused_adaln))) {
        fprintf(stderr, "fused gate/AdaLN differs from standalone path\n");
        ok = 0;
        goto done;
    }

    uint16_t query[8], key[8], value[8], attention[8];
    CHECK(h3_gpu_tensor_read_bf16(query_t, query, 8));
    CHECK(h3_gpu_tensor_read_bf16(key_t, key, 8));
    CHECK(h3_gpu_tensor_read_bf16(value_t, value, 8));
    CHECK(h3_gpu_tensor_read_bf16(attention_t, attention, 8));
    float query_expected[8], key_expected[8], value_expected[8];
    for (size_t row = 0; row < 2; row++) {
        float q_square = 0.0f, k_square = 0.0f;
        for (size_t d = 0; d < 4; d++) {
            q_square += qkv_f[row * 12 + d] * qkv_f[row * 12 + d];
            k_square += qkv_f[row * 12 + 4 + d] *
                        qkv_f[row * 12 + 4 + d];
        }
        float qi = 1.0f / sqrtf(q_square / 4.0f + 1.0e-5f);
        float ki = 1.0f / sqrtf(k_square / 4.0f + 1.0e-5f);
        for (size_t d = 0; d < 4; d++) {
            query_expected[row * 4 + d] = qkv_f[row * 12 + d] * qi;
            key_expected[row * 4 + d] = qkv_f[row * 12 + 4 + d] * ki;
            value_expected[row * 4 + d] = qkv_f[row * 12 + 8 + d];
        }
    }
    CHECK(compare(query, query_expected, 8, 0.016f, "QKV query"));
    CHECK(compare(key, key_expected, 8, 0.016f, "QKV key"));
    CHECK(compare(value, value_expected, 8, 0.016f, "QKV value"));
    float attention_expected[8];
    for (size_t q = 0; q < 2; q++) {
        float score[2];
        for (size_t k = 0; k < 2; k++) {
            score[k] = 0.0f;
            for (size_t d = 0; d < 4; d++)
                score[k] += f32(query[q * 4 + d]) * f32(key[k * 4 + d]);
            score[k] *= 0.5f;
        }
        float maximum = fmaxf(score[0], score[1]);
        float probability[2] = {expf(score[0] - maximum),
                                expf(score[1] - maximum)};
        float denominator = probability[0] + probability[1];
        for (size_t d = 0; d < 4; d++)
            attention_expected[q * 4 + d] =
                (probability[0] * f32(value[d]) +
                 probability[1] * f32(value[4 + d])) / denominator;
    }
    CHECK(compare(attention, attention_expected, 8, 0.02f, "SDPA"));

    uint16_t patch[4];
    const float patch_expected[] = {-1.75f, 3.5f, -2.75f, -2.0f};
    CHECK(h3_gpu_tensor_read_bf16(patch_output_t, patch, 4));
    CHECK(compare(patch, patch_expected, 4, 0.02f, "patch projection"));

    uint16_t mlp_output[2];
    float gate = -2.0f;
    float activated = f32(bf16(gate / (1.0f + expf(-gate)) * -3.0f));
    const float mlp_expected[] = {activated * 2.0f, activated * 0.5f};
    CHECK(h3_gpu_tensor_read_bf16(mlp_output_t, mlp_output, 2));
    CHECK(compare(mlp_output, mlp_expected, 2, 0.025f, "fused MLP"));

done:
    for (size_t index = owned_count; index; index--)
        h3_gpu_tensor_free(owned[index - 1]);
    h3_gpu_free(gpu);
    if (!ok) return 1;
    puts("PASS: HIP DiT AdaLN/gate, QKV/RoPE, SDPA and patch projection");
    return 0;
}
