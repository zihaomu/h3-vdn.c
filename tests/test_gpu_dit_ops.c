#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

static double relative_l2_bf16(const uint16_t *actual,
                               const uint16_t *expected, size_t count,
                               double *maximum) {
    double error_square = 0.0;
    double expected_square = 0.0;
    *maximum = 0.0;
    for (size_t index = 0; index < count; index++) {
        double reference = f32(expected[index]);
        double delta = (double)f32(actual[index]) - reference;
        error_square += delta * delta;
        expected_square += reference * reference;
        *maximum = fmax(*maximum, fabs(delta));
    }
    return sqrt(error_square / fmax(expected_square, 1e-24));
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
    float *large_patch_input = NULL;
    uint16_t *large_patch_output = NULL;

    h3_gpu_tensor *owned[40] = {0};
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

    /* Cross the HIP grid.y limit that the official VDN geometry exceeds.
     * Width one keeps this launch-boundary regression small. */
    enum { LARGE_PATCH_ROWS = 65536 };
    large_patch_input = malloc(
        (size_t)LARGE_PATCH_ROWS * sizeof(*large_patch_input));
    large_patch_output = malloc(
        (size_t)LARGE_PATCH_ROWS * sizeof(*large_patch_output));
    CHECK(large_patch_input && large_patch_output);
    for (size_t row = 0; row < LARGE_PATCH_ROWS; row++)
        large_patch_input[row] = (float)((int)(row % 17) - 8);
    const float large_patch_weight[] = {0.5f};
    const float large_patch_bias[] = {0.25f};
    h3_gpu_tensor *large_patch_input_t = OWN(h3_gpu_tensor_from_f32(
        gpu, large_patch_input, LARGE_PATCH_ROWS));
    h3_gpu_tensor *large_patch_weight_t = OWN(h3_gpu_tensor_from_f32(
        gpu, large_patch_weight, 1));
    h3_gpu_tensor *large_patch_bias_t = OWN(h3_gpu_tensor_from_f32(
        gpu, large_patch_bias, 1));
    h3_gpu_tensor *large_patch_output_t = OWN(
        h3_gpu_tensor_new_bf16(gpu, LARGE_PATCH_ROWS));
    CHECK(large_patch_input_t && large_patch_weight_t && large_patch_bias_t &&
          large_patch_output_t);

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

    enum { FULL_SEQUENCE = 67, FULL_HEADS = 2, FULL_DIMENSION = 128,
           FULL_ELEMENTS = FULL_SEQUENCE * FULL_HEADS * FULL_DIMENSION };
    uint16_t full_query[FULL_ELEMENTS], full_key[FULL_ELEMENTS];
    uint16_t full_value[FULL_ELEMENTS], full_scalar[FULL_ELEMENTS];
    uint16_t full_tiled[FULL_ELEMENTS], full_repeat[FULL_ELEMENTS];
    uint16_t full_matrix[FULL_ELEMENTS], full_matrix_repeat[FULL_ELEMENTS];
    uint16_t full_auto[FULL_ELEMENTS];
    uint16_t full_sage[FULL_ELEMENTS], full_sage_repeat[FULL_ELEMENTS];
    for (size_t index = 0; index < FULL_ELEMENTS; index++) {
        int centered = (int)(index % 37) - 18;
        full_query[index] = bf16((float)centered / 23.0f);
        centered = (int)((index * 7 + 3) % 41) - 20;
        full_key[index] = bf16((float)centered / 29.0f);
        centered = (int)((index * 11 + 5) % 43) - 21;
        full_value[index] = bf16((float)centered / 19.0f);
    }
    h3_gpu_tensor *full_query_t = OWN(h3_gpu_tensor_from_bf16(
        gpu, full_query, FULL_ELEMENTS));
    h3_gpu_tensor *full_key_t = OWN(h3_gpu_tensor_from_bf16(
        gpu, full_key, FULL_ELEMENTS));
    h3_gpu_tensor *full_value_t = OWN(h3_gpu_tensor_from_bf16(
        gpu, full_value, FULL_ELEMENTS));
    h3_gpu_tensor *full_output_t = OWN(h3_gpu_tensor_new_bf16(
        gpu, FULL_ELEMENTS));
    h3_gpu_tensor *full_repeat_t = OWN(h3_gpu_tensor_new_bf16(
        gpu, FULL_ELEMENTS));
    CHECK(full_query_t && full_key_t && full_value_t && full_output_t &&
          full_repeat_t);

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
    CHECK(h3_gpu_patch_linear_bf16(
        gpu, large_patch_output_t, large_patch_input_t, large_patch_weight_t,
        large_patch_bias_t, LARGE_PATCH_ROWS, 1, 1));
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
    CHECK(h3_gpu_tensor_read_bf16(
        large_patch_output_t, large_patch_output, LARGE_PATCH_ROWS));
    for (size_t row = 0; row < LARGE_PATCH_ROWS; row++) {
        uint16_t expected_value = bf16(large_patch_input[row] * 0.5f + 0.25f);
        if (large_patch_output[row] != expected_value) {
            fprintf(stderr, "large patch projection[%zu]: got %.7f expected %.7f\n",
                    row, f32(large_patch_output[row]), f32(expected_value));
            ok = 0;
            goto done;
        }
    }

    uint16_t mlp_output[2];
    const float value0 = 1.0f;
    const float value1 = -2.0f;
    const float gate0 = 0.0f;
    const float gate1 = -3.0f;
    const float activated0 = f32(bf16(
        value0 * gate0 / (1.0f + expf(-gate0))));
    const float activated1 = f32(bf16(
        value1 * gate1 / (1.0f + expf(-gate1))));
    const float mlp_expected[] = {
        activated0 + activated1 * 2.0f,
        -activated0 + activated1 * 0.5f,
    };
    CHECK(h3_gpu_tensor_read_bf16(mlp_output_t, mlp_output, 2));
    CHECK(compare(mlp_output, mlp_expected, 2, 0.025f, "fused MLP"));

    setenv("H3_BF16_SDPA_ROCBLAS", "0", 1);
    setenv("H3_BF16_SDPA_SCALAR", "1", 1);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_bf16(
        gpu, full_output_t, full_query_t, full_key_t, full_value_t,
        FULL_SEQUENCE, FULL_HEADS, FULL_DIMENSION,
        1.0f / sqrtf((float)FULL_DIMENSION)));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_bf16(
        full_output_t, full_scalar, FULL_ELEMENTS));
    unsetenv("H3_BF16_SDPA_SCALAR");
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_bf16(
        gpu, full_output_t, full_query_t, full_key_t, full_value_t,
        FULL_SEQUENCE, FULL_HEADS, FULL_DIMENSION,
        1.0f / sqrtf((float)FULL_DIMENSION)));
    CHECK(h3_gpu_sdpa_bf16(
        gpu, full_repeat_t, full_query_t, full_key_t, full_value_t,
        FULL_SEQUENCE, FULL_HEADS, FULL_DIMENSION,
        1.0f / sqrtf((float)FULL_DIMENSION)));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_bf16(
        full_output_t, full_tiled, FULL_ELEMENTS));
    CHECK(h3_gpu_tensor_read_bf16(
        full_repeat_t, full_repeat, FULL_ELEMENTS));
    if (memcmp(full_tiled, full_repeat, sizeof(full_tiled))) {
        fprintf(stderr, "D=128 tiled SDPA is not repeatable\n");
        ok = 0;
        goto done;
    }
    double full_maximum = 0.0;
    double full_relative = relative_l2_bf16(
        full_tiled, full_scalar, FULL_ELEMENTS, &full_maximum);
    printf("D=128 tiled/scalar SDPA: max-abs %.8g rel-L2 %.8g; "
           "repeat bitwise\n", full_maximum, full_relative);
    if (full_relative >= 0.02) {
        fprintf(stderr, "D=128 tiled SDPA exceeds scalar tolerance\n");
        ok = 0;
        goto done;
    }

    setenv("H3_BF16_SDPA_ROCBLAS", "1", 1);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_bf16(
        gpu, full_output_t, full_query_t, full_key_t, full_value_t,
        FULL_SEQUENCE, FULL_HEADS, FULL_DIMENSION,
        1.0f / sqrtf((float)FULL_DIMENSION)));
    CHECK(h3_gpu_sdpa_bf16(
        gpu, full_repeat_t, full_query_t, full_key_t, full_value_t,
        FULL_SEQUENCE, FULL_HEADS, FULL_DIMENSION,
        1.0f / sqrtf((float)FULL_DIMENSION)));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_bf16(
        full_output_t, full_matrix, FULL_ELEMENTS));
    CHECK(h3_gpu_tensor_read_bf16(
        full_repeat_t, full_matrix_repeat, FULL_ELEMENTS));
    unsetenv("H3_BF16_SDPA_ROCBLAS");
    if (memcmp(full_matrix, full_matrix_repeat, sizeof(full_matrix))) {
        fprintf(stderr, "D=128 matrix SDPA is not repeatable\n");
        ok = 0;
        goto done;
    }
    double matrix_maximum = 0.0;
    double matrix_relative = relative_l2_bf16(
        full_matrix, full_scalar, FULL_ELEMENTS, &matrix_maximum);
    printf("D=128 matrix/scalar SDPA: max-abs %.8g rel-L2 %.8g; "
           "repeat bitwise\n", matrix_maximum, matrix_relative);
    if (matrix_relative >= 0.02) {
        fprintf(stderr, "D=128 matrix SDPA exceeds scalar tolerance\n");
        ok = 0;
        goto done;
    }

    setenv("H3_BF16_SDPA", "auto", 1);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_bf16(
        gpu, full_output_t, full_query_t, full_key_t, full_value_t,
        FULL_SEQUENCE, FULL_HEADS, FULL_DIMENSION,
        1.0f / sqrtf((float)FULL_DIMENSION)));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_bf16(
        full_output_t, full_auto, FULL_ELEMENTS));
    unsetenv("H3_BF16_SDPA");
    if (memcmp(full_auto, full_matrix, sizeof(full_auto))) {
        fprintf(stderr, "D=128 auto SDPA did not preserve matrix default\n");
        ok = 0;
        goto done;
    }
    printf("D=128 auto SDPA preserves matrix output bytes\n");

    setenv("H3_BF16_SDPA", "sage-e33", 1);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_sdpa_bf16(
        gpu, full_output_t, full_query_t, full_key_t, full_value_t,
        FULL_SEQUENCE, FULL_HEADS, FULL_DIMENSION,
        1.0f / sqrtf((float)FULL_DIMENSION)));
    CHECK(h3_gpu_sdpa_bf16(
        gpu, full_repeat_t, full_query_t, full_key_t, full_value_t,
        FULL_SEQUENCE, FULL_HEADS, FULL_DIMENSION,
        1.0f / sqrtf((float)FULL_DIMENSION)));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_bf16(
        full_output_t, full_sage, FULL_ELEMENTS));
    CHECK(h3_gpu_tensor_read_bf16(
        full_repeat_t, full_sage_repeat, FULL_ELEMENTS));
    unsetenv("H3_BF16_SDPA");
    if (memcmp(full_sage, full_sage_repeat, sizeof(full_sage))) {
        fprintf(stderr, "D=128 Sage E33 SDPA is not repeatable\n");
        ok = 0;
        goto done;
    }
    double sage_maximum = 0.0;
    double sage_relative = relative_l2_bf16(
        full_sage, full_scalar, FULL_ELEMENTS, &sage_maximum);
    printf("D=128 Sage-E33/scalar SDPA: max-abs %.8g rel-L2 %.8g; "
           "repeat bitwise\n", sage_maximum, sage_relative);
    if (sage_relative >= 0.02) {
        fprintf(stderr, "D=128 Sage E33 SDPA exceeds scalar tolerance\n");
        ok = 0;
        goto done;
    }

done:
    unsetenv("H3_BF16_SDPA");
    unsetenv("H3_BF16_SDPA_ROCBLAS");
    unsetenv("H3_BF16_SDPA_SCALAR");
    free(large_patch_output);
    free(large_patch_input);
    for (size_t index = owned_count; index; index--)
        h3_gpu_tensor_free(owned[index - 1]);
    h3_gpu_free(gpu);
    if (!ok) return 1;
    puts("PASS: HIP DiT AdaLN/gate, QKV/RoPE, SDPA and patch projection");
    return 0;
}
