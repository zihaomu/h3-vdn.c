#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000))
        bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    else if (bits & UINT32_C(0xffff))
        bits |= UINT32_C(0x10000);
    return (uint16_t)(bits >> 16);
}

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int close_enough(float actual, float expected, float tolerance,
                        const char *label, size_t index) {
    float error = fabsf(actual - expected);
    if (error <= tolerance) return 1;
    fprintf(stderr, "%s[%zu]: got %.8f, expected %.8f (error %.8f)\n",
            label, index, actual, expected, error);
    return 0;
}

static int check_f32(const float *actual, const float *expected, size_t count,
                     float tolerance, const char *label) {
    for (size_t index = 0; index < count; index++)
        if (!close_enough(actual[index], expected[index], tolerance, label,
                          index))
            return 0;
    return 1;
}

static int check_bf16(const uint16_t *actual, const float *expected,
                      size_t count, float tolerance, const char *label) {
    for (size_t index = 0; index < count; index++)
        if (!close_enough(from_bf16(actual[index]), expected[index], tolerance,
                          label, index))
            return 0;
    return 1;
}

#define REQUIRE(expression)                                                    \
    do {                                                                       \
        if (!(expression)) {                                                   \
            fprintf(stderr, "failed: %s (%s)\n", #expression,              \
                    gpu ? h3_gpu_error(gpu) : "no GPU context");              \
            ok = 0;                                                            \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

int main(void) {
    int ok = 1;
    char error[512];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "cannot create GPU context: %s\n", error);
        return 1;
    }

    const float cast_input[] = {-3.25f, -1.0f, 0.0f, 0.5f, 2.0f, 9.75f};
    const size_t cast_count = sizeof(cast_input) / sizeof(cast_input[0]);
    h3_gpu_tensor *cast_source = NULL;
    h3_gpu_tensor *cast_bf16 = NULL;
    h3_gpu_tensor *cast_roundtrip = NULL;
    h3_gpu_tensor *silu_output = NULL;
    h3_gpu_tensor *linear_input = NULL;
    h3_gpu_tensor *linear_weight = NULL;
    h3_gpu_tensor *linear_bias = NULL;
    h3_gpu_tensor *linear_output = NULL;
    h3_gpu_tensor *norm_input = NULL;
    h3_gpu_tensor *norm_weight = NULL;
    h3_gpu_tensor *norm_bias = NULL;
    h3_gpu_tensor *rms_output = NULL;
    h3_gpu_tensor *layer_output = NULL;
    h3_gpu_tensor *bf_left = NULL;
    h3_gpu_tensor *bf_right = NULL;
    h3_gpu_tensor *bf_output = NULL;
    h3_gpu_tensor *bf_linear_input = NULL;
    h3_gpu_tensor *bf_linear_weight = NULL;
    h3_gpu_tensor *bf_linear_bias = NULL;
    h3_gpu_tensor *bf_linear_output = NULL;
    h3_gpu_tensor *lora_base = NULL;
    h3_gpu_tensor *lora_a = NULL;
    h3_gpu_tensor *lora_b = NULL;
    h3_gpu_tensor *lora_output = NULL;
    h3_gpu_tensor *sample = NULL;

    cast_source = h3_gpu_tensor_from_f32(gpu, cast_input, cast_count);
    cast_bf16 = h3_gpu_tensor_new_bf16(gpu, cast_count);
    cast_roundtrip = h3_gpu_tensor_new_f32(gpu, cast_count);
    silu_output = h3_gpu_tensor_new_f32(gpu, cast_count);
    REQUIRE(cast_source && cast_bf16 && cast_roundtrip && silu_output);

    const float matrix_input[] = {1.0f, 2.0f, -1.0f,
                                  0.5f, -2.0f, 3.0f};
    const float matrix_weight[] = {2.0f, -1.0f, 0.5f,
                                   -3.0f, 2.0f, 1.0f};
    const float matrix_bias[] = {0.25f, -0.5f};
    linear_input = h3_gpu_tensor_from_f32(gpu, matrix_input, 6);
    linear_weight = h3_gpu_tensor_from_f32(gpu, matrix_weight, 6);
    linear_bias = h3_gpu_tensor_from_f32(gpu, matrix_bias, 2);
    linear_output = h3_gpu_tensor_new_f32(gpu, 4);
    REQUIRE(linear_input && linear_weight && linear_bias && linear_output);

    const float norm_values[] = {1.0f, 2.0f, 3.0f, 4.0f,
                                 -2.0f, 0.0f, 2.0f, 4.0f};
    const float norm_scales[] = {1.0f, 0.5f, 2.0f, -1.0f};
    const float norm_offsets[] = {0.1f, -0.2f, 0.3f, 0.0f};
    norm_input = h3_gpu_tensor_from_f32(gpu, norm_values, 8);
    norm_weight = h3_gpu_tensor_from_f32(gpu, norm_scales, 4);
    norm_bias = h3_gpu_tensor_from_f32(gpu, norm_offsets, 4);
    rms_output = h3_gpu_tensor_new_f32(gpu, 8);
    layer_output = h3_gpu_tensor_new_f32(gpu, 8);
    REQUIRE(norm_input && norm_weight && norm_bias && rms_output &&
            layer_output);

    uint16_t bf_left_values[4];
    uint16_t bf_right_values[4];
    const float left_values[] = {1.0f, -2.0f, 4.0f, 0.5f};
    const float right_values[] = {3.0f, 1.0f, -1.5f, 2.0f};
    for (size_t index = 0; index < 4; index++) {
        bf_left_values[index] = to_bf16(left_values[index]);
        bf_right_values[index] = to_bf16(right_values[index]);
    }
    bf_left = h3_gpu_tensor_from_bf16(gpu, bf_left_values, 4);
    bf_right = h3_gpu_tensor_from_bf16(gpu, bf_right_values, 4);
    bf_output = h3_gpu_tensor_new_bf16(gpu, 4);
    REQUIRE(bf_left && bf_right && bf_output);

    uint16_t bf_matrix_input[6];
    uint16_t bf_matrix_weight[6];
    uint16_t bf_matrix_bias[2];
    for (size_t index = 0; index < 6; index++) {
        bf_matrix_input[index] = to_bf16(matrix_input[index]);
        bf_matrix_weight[index] = to_bf16(matrix_weight[index]);
    }
    for (size_t index = 0; index < 2; index++)
        bf_matrix_bias[index] = to_bf16(matrix_bias[index]);
    bf_linear_input = h3_gpu_tensor_from_bf16(gpu, bf_matrix_input, 6);
    bf_linear_weight = h3_gpu_tensor_from_bf16(gpu, bf_matrix_weight, 6);
    bf_linear_bias = h3_gpu_tensor_from_bf16(gpu, bf_matrix_bias, 2);
    bf_linear_output = h3_gpu_tensor_new_bf16(gpu, 4);
    REQUIRE(bf_linear_input && bf_linear_weight && bf_linear_bias &&
            bf_linear_output);

    const float lora_base_values[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f
    };
    const float lora_a_values[] = {
        1.0f, 2.0f, -1.0f, 0.5f, -0.5f, 1.5f
    };
    const float lora_b_values[] = {2.0f, -1.0f, 0.25f, 3.0f};
    uint16_t lora_base_bf16[6];
    uint16_t lora_a_bf16[6];
    uint16_t lora_b_bf16[4];
    for (size_t index = 0; index < 6; index++) {
        lora_base_bf16[index] = to_bf16(lora_base_values[index]);
        lora_a_bf16[index] = to_bf16(lora_a_values[index]);
    }
    for (size_t index = 0; index < 4; index++)
        lora_b_bf16[index] = to_bf16(lora_b_values[index]);
    lora_base = h3_gpu_tensor_from_bf16(gpu, lora_base_bf16, 6);
    lora_a = h3_gpu_tensor_from_bf16(gpu, lora_a_bf16, 6);
    lora_b = h3_gpu_tensor_from_bf16(gpu, lora_b_bf16, 4);
    lora_output = h3_gpu_tensor_new_bf16(gpu, 6);
    REQUIRE(lora_base && lora_a && lora_b && lora_output);

    const float initial_sample[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    sample = h3_gpu_tensor_from_f32(gpu, initial_sample, 5);
    REQUIRE(sample);

    REQUIRE(h3_gpu_begin(gpu));
    REQUIRE(h3_gpu_cast_f32_to_bf16(gpu, cast_bf16, cast_source,
                                    (uint32_t)cast_count));
    REQUIRE(h3_gpu_cast_bf16_to_f32(gpu, cast_roundtrip, cast_bf16,
                                    (uint32_t)cast_count));
    REQUIRE(h3_gpu_silu_f32(gpu, silu_output, cast_source,
                            (uint32_t)cast_count));
    REQUIRE(h3_gpu_linear_f32(gpu, linear_output, linear_input, linear_weight,
                              linear_bias, 2, 3, 2));
    REQUIRE(h3_gpu_rms_norm_f32(gpu, rms_output, norm_input, norm_weight,
                                2, 4, 1.0e-5f));
    REQUIRE(h3_gpu_layer_norm_f32(gpu, layer_output, norm_input, norm_weight,
                                  norm_bias, 2, 4, 1.0e-5f));
    REQUIRE(h3_gpu_add_bf16(gpu, bf_output, bf_left, bf_right, 4));
    REQUIRE(h3_gpu_sub_bf16(gpu, bf_output, bf_output, bf_right, 4));
    REQUIRE(h3_gpu_linear_bf16(gpu, bf_linear_output, bf_linear_input,
                               bf_linear_weight, bf_linear_bias, 2, 3, 2));
    REQUIRE(h3_gpu_lora_merge_bf16(gpu, lora_output, lora_base, lora_a,
                                   lora_b, 3, 2, 2, 0.5f));
    REQUIRE(h3_gpu_euler_bf16(gpu, sample, 1, bf_left, bf_right, 4,
                              0.5f, 0.25f));
    REQUIRE(h3_gpu_submit(gpu));

    float f32_result[8];
    REQUIRE(h3_gpu_tensor_read_f32(cast_roundtrip, f32_result, cast_count));
    REQUIRE(check_f32(f32_result, cast_input, cast_count, 0.03125f,
                      "BF16 roundtrip"));

    REQUIRE(h3_gpu_tensor_read_f32(silu_output, f32_result, cast_count));
    float silu_expected[6];
    for (size_t index = 0; index < cast_count; index++)
        silu_expected[index] = cast_input[index] /
                               (1.0f + expf(-cast_input[index]));
    REQUIRE(check_f32(f32_result, silu_expected, cast_count, 1.0e-5f,
                      "SiLU F32"));

    const float linear_expected[] = {-0.25f, -0.5f, 4.75f, -3.0f};
    REQUIRE(h3_gpu_tensor_read_f32(linear_output, f32_result, 4));
    REQUIRE(check_f32(f32_result, linear_expected, 4, 1.0e-5f,
                      "linear F32"));

    float rms_expected[8];
    float layer_expected[8];
    for (size_t row = 0; row < 2; row++) {
        float sum_square = 0.0f;
        float mean = 0.0f;
        for (size_t column = 0; column < 4; column++) {
            float value = norm_values[row * 4 + column];
            sum_square += value * value;
            mean += value;
        }
        float rms_inverse = 1.0f / sqrtf(sum_square / 4.0f + 1.0e-5f);
        mean /= 4.0f;
        float variance = 0.0f;
        for (size_t column = 0; column < 4; column++) {
            float centered = norm_values[row * 4 + column] - mean;
            variance += centered * centered;
        }
        float layer_inverse = 1.0f / sqrtf(variance / 4.0f + 1.0e-5f);
        for (size_t column = 0; column < 4; column++) {
            size_t index = row * 4 + column;
            rms_expected[index] = norm_values[index] * rms_inverse *
                                  norm_scales[column];
            layer_expected[index] =
                (norm_values[index] - mean) * layer_inverse *
                    norm_scales[column] +
                norm_offsets[column];
        }
    }
    REQUIRE(h3_gpu_tensor_read_f32(rms_output, f32_result, 8));
    REQUIRE(check_f32(f32_result, rms_expected, 8, 2.0e-5f,
                      "RMSNorm F32"));
    REQUIRE(h3_gpu_tensor_read_f32(layer_output, f32_result, 8));
    REQUIRE(check_f32(f32_result, layer_expected, 8, 2.0e-5f,
                      "LayerNorm F32"));

    uint16_t bf16_result[4];
    REQUIRE(h3_gpu_tensor_read_bf16(bf_output, bf16_result, 4));
    REQUIRE(check_bf16(bf16_result, left_values, 4, 0.03125f,
                       "BF16 add/sub"));
    REQUIRE(h3_gpu_tensor_read_bf16(bf_linear_output, bf16_result, 4));
    REQUIRE(check_bf16(bf16_result, linear_expected, 4, 0.03125f,
                       "linear BF16"));
    uint16_t lora_result[6];
    const float lora_expected[] = {
        1.75f, 4.25f, 1.25f, 4.875f, 4.5f, 8.125f
    };
    REQUIRE(h3_gpu_tensor_read_bf16(lora_output, lora_result, 6));
    REQUIRE(check_bf16(lora_result, lora_expected, 6, 0.03125f,
                       "LoRA BF16 merge"));

    REQUIRE(h3_gpu_tensor_read_f32(sample, f32_result, 5));
    float sample_expected[5];
    memcpy(sample_expected, initial_sample, sizeof(initial_sample));
    for (size_t index = 0; index < 4; index++) {
        float velocity = left_values[index] +
                         0.25f * (left_values[index] - right_values[index]);
        sample_expected[index + 1] += 0.5f * velocity;
    }
    REQUIRE(check_f32(f32_result, sample_expected, 5, 1.0e-5f,
                      "Euler BF16"));

    if (h3_gpu_linear_f32(gpu, linear_output, linear_input, linear_weight,
                          linear_bias, 2, 3, 2)) {
        fprintf(stderr, "linear unexpectedly ran outside h3_gpu_begin\n");
        ok = 0;
        goto cleanup;
    }
    if (strstr(h3_gpu_error(gpu), "requires h3_gpu_begin") == NULL) {
        fprintf(stderr, "missing explicit begin error: %s\n",
                h3_gpu_error(gpu));
        ok = 0;
        goto cleanup;
    }

cleanup:
    h3_gpu_tensor_free(sample);
    h3_gpu_tensor_free(lora_output);
    h3_gpu_tensor_free(lora_b);
    h3_gpu_tensor_free(lora_a);
    h3_gpu_tensor_free(lora_base);
    h3_gpu_tensor_free(bf_linear_output);
    h3_gpu_tensor_free(bf_linear_bias);
    h3_gpu_tensor_free(bf_linear_weight);
    h3_gpu_tensor_free(bf_linear_input);
    h3_gpu_tensor_free(bf_output);
    h3_gpu_tensor_free(bf_right);
    h3_gpu_tensor_free(bf_left);
    h3_gpu_tensor_free(layer_output);
    h3_gpu_tensor_free(rms_output);
    h3_gpu_tensor_free(norm_bias);
    h3_gpu_tensor_free(norm_weight);
    h3_gpu_tensor_free(norm_input);
    h3_gpu_tensor_free(linear_output);
    h3_gpu_tensor_free(linear_bias);
    h3_gpu_tensor_free(linear_weight);
    h3_gpu_tensor_free(linear_input);
    h3_gpu_tensor_free(silu_output);
    h3_gpu_tensor_free(cast_roundtrip);
    h3_gpu_tensor_free(cast_bf16);
    h3_gpu_tensor_free(cast_source);
    h3_gpu_free(gpu);

    if (!ok) return 1;
    puts("PASS: HIP core operators, rocBLAS linear and LoRA merge parity");
    return 0;
}
