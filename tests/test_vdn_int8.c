#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ROWS = 3, INPUT_DIM = 37, OUTPUT_DIM = 11 };

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void quantize_rows(const uint16_t *input, int8_t *output,
                          float *scales, uint32_t rows, uint32_t columns) {
    for (uint32_t row = 0; row < rows; row++) {
        size_t offset = (size_t)row * columns;
        float amax = 0.0f;
        for (uint32_t column = 0; column < columns; column++)
            amax = fmaxf(amax, fabsf(f32(input[offset + column])));
        float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
        scales[row] = scale;
        for (uint32_t column = 0; column < columns; column++) {
            long value = lrintf(f32(input[offset + column]) * inverse);
            if (value > 127) value = 127;
            if (value < -127) value = -127;
            output[offset + column] = (int8_t)value;
        }
    }
}

int main(void) {
    const size_t input_elements = (size_t)ROWS * INPUT_DIM;
    const size_t weight_elements = (size_t)OUTPUT_DIM * INPUT_DIM;
    const size_t output_elements = (size_t)ROWS * OUTPUT_DIM;
    uint16_t input[input_elements];
    uint16_t weight[weight_elements];
    int8_t input_expected[input_elements], weight_expected[weight_elements];
    int8_t input_actual[input_elements], weight_actual[weight_elements];
    float input_scales_expected[ROWS], weight_scales_expected[OUTPUT_DIM];
    float input_scales_actual[ROWS], weight_scales_actual[OUTPUT_DIM];
    uint16_t output_expected[output_elements], output_actual[output_elements];

    for (size_t index = 0; index < input_elements; index++) {
        int centered = (int)((index * 29 + 5) % 61) - 30;
        input[index] = bf16((float)centered * 0.03125f);
    }
    for (size_t index = 0; index < weight_elements; index++) {
        int centered = (int)((index * 17 + 3) % 251) - 125;
        weight[index] = bf16((float)centered * 0.0078125f);
    }
    memset(input + (size_t)(ROWS - 1) * INPUT_DIM, 0,
           INPUT_DIM * sizeof(*input));
    memset(weight + (size_t)(OUTPUT_DIM - 1) * INPUT_DIM, 0,
           INPUT_DIM * sizeof(*weight));
    quantize_rows(input, input_expected, input_scales_expected,
                  ROWS, INPUT_DIM);
    quantize_rows(weight, weight_expected, weight_scales_expected,
                  OUTPUT_DIM, INPUT_DIM);
    for (uint32_t row = 0; row < ROWS; row++)
        for (uint32_t output = 0; output < OUTPUT_DIM; output++) {
            int32_t sum = 0;
            for (uint32_t input_column = 0; input_column < INPUT_DIM;
                 input_column++)
                sum += (int32_t)input_expected[
                           (size_t)row * INPUT_DIM + input_column] *
                       (int32_t)weight_expected[
                           (size_t)output * INPUT_DIM + input_column];
            output_expected[(size_t)row * OUTPUT_DIM + output] = bf16(
                (float)sum * input_scales_expected[row] *
                weight_scales_expected[output]);
        }

    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "cannot create HIP context: %s\n", error);
        return 1;
    }
    h3_gpu_tensor *input_t = h3_gpu_tensor_from_bf16(
        gpu, input, input_elements);
    h3_gpu_tensor *weight_t = h3_gpu_tensor_from_bf16(
        gpu, weight, weight_elements);
    h3_gpu_tensor *input_i8_t = h3_gpu_tensor_new_i8(gpu, input_elements);
    h3_gpu_tensor *weight_i8_t = h3_gpu_tensor_new_i8(gpu, weight_elements);
    h3_gpu_tensor *input_scales_t = h3_gpu_tensor_new_f32(gpu, ROWS);
    h3_gpu_tensor *weight_scales_t = h3_gpu_tensor_new_f32(gpu, OUTPUT_DIM);
    h3_gpu_tensor *output_t = h3_gpu_tensor_new_bf16(gpu, output_elements);
    int ok = input_t && weight_t && input_i8_t && weight_i8_t &&
             input_scales_t && weight_scales_t && output_t &&
             h3_gpu_begin(gpu) &&
             h3_gpu_quantize_weight_int8(
                 gpu, weight_i8_t, weight_scales_t, weight_t,
                 OUTPUT_DIM, INPUT_DIM) &&
             h3_gpu_linear_int8_bf16(
                 gpu, output_t, input_i8_t, input_scales_t, input_t,
                 weight_i8_t, weight_scales_t, ROWS, INPUT_DIM, OUTPUT_DIM,
                 0) &&
             h3_gpu_submit(gpu) &&
             h3_gpu_tensor_read_i8(
                 input_i8_t, input_actual, input_elements) &&
             h3_gpu_tensor_read_i8(
                 weight_i8_t, weight_actual, weight_elements) &&
             h3_gpu_tensor_read_f32(
                 input_scales_t, input_scales_actual, ROWS) &&
             h3_gpu_tensor_read_f32(
                 weight_scales_t, weight_scales_actual, OUTPUT_DIM) &&
             h3_gpu_tensor_read_bf16(
                 output_t, output_actual, output_elements);
    if (!ok) {
        fprintf(stderr, "VDN INT8 GPU failure: %s\n", h3_gpu_error(gpu));
    } else if (memcmp(input_expected, input_actual, sizeof(input_actual)) ||
               memcmp(weight_expected, weight_actual, sizeof(weight_actual)) ||
               memcmp(input_scales_expected, input_scales_actual,
                      sizeof(input_scales_actual)) ||
               memcmp(weight_scales_expected, weight_scales_actual,
                      sizeof(weight_scales_actual)) ||
               memcmp(output_expected, output_actual, sizeof(output_actual))) {
        size_t input_mismatches = 0, weight_mismatches = 0;
        size_t output_mismatches = 0;
        for (size_t index = 0; index < input_elements; index++)
            input_mismatches += input_expected[index] != input_actual[index];
        for (size_t index = 0; index < weight_elements; index++)
            weight_mismatches += weight_expected[index] != weight_actual[index];
        for (size_t index = 0; index < output_elements; index++)
            output_mismatches += output_expected[index] != output_actual[index];
        fprintf(stderr, "VDN INT8 mismatch: input=%zu weight=%zu output=%zu\n",
                input_mismatches, weight_mismatches, output_mismatches);
        ok = 0;
    }
    if (ok && (input_scales_actual[ROWS - 1] != 1.0f ||
               weight_scales_actual[OUTPUT_DIM - 1] != 1.0f)) {
        fprintf(stderr, "VDN INT8 zero-row scale is not one\n");
        ok = 0;
    }

    h3_gpu_tensor_free(output_t);
    h3_gpu_tensor_free(weight_scales_t);
    h3_gpu_tensor_free(input_scales_t);
    h3_gpu_tensor_free(weight_i8_t);
    h3_gpu_tensor_free(input_i8_t);
    h3_gpu_tensor_free(weight_t);
    h3_gpu_tensor_free(input_t);
    h3_gpu_free(gpu);
    if (!ok) return 1;
    puts("PASS: VDN INT8 row quantization and rocBLAS linear oracle");
    return 0;
}
