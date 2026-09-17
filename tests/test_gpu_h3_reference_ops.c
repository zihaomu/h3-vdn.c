#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TOKENS = 3,
    VOCAB = 5,
    WIDTH = 4,
    SEQUENCE = 4,
    QUERY_HEADS = 4,
    KV_HEADS = 2,
    HEAD_DIM = 4,
    QUERY_COUNT = SEQUENCE * QUERY_HEADS * HEAD_DIM,
    KV_COUNT = SEQUENCE * KV_HEADS * HEAD_DIM,
    ROPE_COUNT = SEQUENCE * (HEAD_DIM / 2),
    NORM_COUNT = 2 * 2 * HEAD_DIM,
    VISION_SEQUENCE = 2,
    VISION_HEADS = 2,
    VISION_HEAD_DIM = 4,
    VISION_COUNT = VISION_SEQUENCE * VISION_HEADS * VISION_HEAD_DIM,
    VISION_ROPE_COUNT = VISION_SEQUENCE * (VISION_HEAD_DIM / 2),
    AUDIO_BATCH = 1,
    AUDIO_LENGTH = 2,
    AUDIO_HEADS = 2,
    AUDIO_HEAD_DIM = 4,
    AUDIO_WIDTH = AUDIO_HEADS * AUDIO_HEAD_DIM,
    AUDIO_COUNT = AUDIO_BATCH * AUDIO_LENGTH * AUDIO_WIDTH,
    AUDIO_OUTPUT_DIM = 2,
    AUDIO_POOL_COUNT = AUDIO_BATCH * AUDIO_LENGTH * AUDIO_OUTPUT_DIM,
    GEGLU_COUNT = 7,
    SWIGLU_ROWS = 2,
    SWIGLU_WIDTH = 3,
    SWIGLU_COUNT = SWIGLU_ROWS * SWIGLU_WIDTH,
    PAD_DEPTH = 2,
    PAD_HEIGHT = 3,
    PAD_WIDTH = 4,
    PAD_CHANNELS = 2,
    PAD_INPUT_COUNT = PAD_DEPTH * PAD_HEIGHT * PAD_WIDTH * PAD_CHANNELS,
    PAD_OUT_DEPTH = PAD_DEPTH + 1,
    PAD_OUT_HEIGHT = PAD_HEIGHT + 2,
    PAD_OUT_WIDTH = PAD_WIDTH + 2,
    PAD_OUTPUT_COUNT = PAD_OUT_DEPTH * PAD_OUT_HEIGHT * PAD_OUT_WIDTH *
                       PAD_CHANNELS,
    CONV_OUT_CHANNELS = 3,
    CONV_OUT_DEPTH = PAD_OUT_DEPTH - 2 + 1,
    CONV_OUT_HEIGHT = PAD_OUT_HEIGHT - 2 + 1,
    CONV_OUT_WIDTH = PAD_OUT_WIDTH - 2 + 1,
    CONV_WEIGHT_COUNT = CONV_OUT_CHANNELS * PAD_CHANNELS * 2 * 2 * 2,
    CONV_OUTPUT_COUNT = CONV_OUT_DEPTH * CONV_OUT_HEIGHT * CONV_OUT_WIDTH *
                        CONV_OUT_CHANNELS,
    GROUP_DEPTH = 2,
    GROUP_HEIGHT = 2,
    GROUP_WIDTH = 2,
    GROUP_CHANNELS = 4,
    GROUPS = 2,
    GROUP_COUNT = GROUP_DEPTH * GROUP_HEIGHT * GROUP_WIDTH * GROUP_CHANNELS
};

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_gpu_h3_reference_ops.c: %s\n", message);
    exit(1);
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t rounding = UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)((bits + rounding) >> 16);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void require_gpu(h3_gpu *gpu, int result, const char *operation) {
    if (result) return;
    fprintf(stderr, "FAIL tests/test_gpu_h3_reference_ops.c: %s: %s\n",
            operation, h3_gpu_error(gpu));
    exit(1);
}

static h3_gpu_tensor *from_bf16(h3_gpu *gpu, const uint16_t *values,
                                 size_t count) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(gpu, values, count);
    if (!tensor) fail(h3_gpu_error(gpu));
    return tensor;
}

static h3_gpu_tensor *fresh_bf16(h3_gpu *gpu, size_t count) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_bf16(gpu, count);
    if (!tensor) fail(h3_gpu_error(gpu));
    return tensor;
}

static h3_gpu_tensor *from_f32(h3_gpu *gpu, const float *values,
                                size_t count) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_f32(gpu, values, count);
    if (!tensor) fail(h3_gpu_error(gpu));
    return tensor;
}

static h3_gpu_tensor *fresh_f32(h3_gpu *gpu, size_t count) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_f32(gpu, count);
    if (!tensor) fail(h3_gpu_error(gpu));
    return tensor;
}

static void check_exact(const char *label, const uint16_t *got,
                        const uint16_t *want, size_t count) {
    for (size_t index = 0; index < count; index++) {
        if (got[index] != want[index]) {
            fprintf(stderr,
                    "FAIL %s at %zu: got %.9g (0x%04x), want %.9g (0x%04x)\n",
                    label, index, bf16_to_f32(got[index]), got[index],
                    bf16_to_f32(want[index]), want[index]);
            exit(1);
        }
    }
}

static void check_close(const char *label, const uint16_t *got,
                        const uint16_t *want, size_t count, float tolerance) {
    float maximum = 0.0f;
    size_t maximum_index = 0;
    for (size_t index = 0; index < count; index++) {
        float error = fabsf(bf16_to_f32(got[index]) -
                            bf16_to_f32(want[index]));
        if (error > maximum) {
            maximum = error;
            maximum_index = index;
        }
    }
    printf("%-24s max-abs %.9g\n", label, maximum);
    if (maximum > tolerance) {
        fprintf(stderr, "FAIL %s at %zu exceeds %.9g\n", label,
                maximum_index, tolerance);
        exit(1);
    }
}

static void check_f32_close(const char *label, const float *got,
                            const float *want, size_t count,
                            float tolerance) {
    float maximum = 0.0f;
    size_t maximum_index = 0;
    for (size_t index = 0; index < count; index++) {
        float error = fabsf(got[index] - want[index]);
        if (error > maximum) {
            maximum = error;
            maximum_index = index;
        }
    }
    printf("%-24s max-abs %.9g\n", label, maximum);
    if (maximum > tolerance) {
        fprintf(stderr, "FAIL %s at %zu exceeds %.9g\n", label,
                maximum_index, tolerance);
        exit(1);
    }
}

static void reference_head_norm(uint16_t *values, const uint16_t *weight,
                                uint32_t head_rows, uint32_t head_dim,
                                float epsilon) {
    for (uint32_t row = 0; row < head_rows; row++) {
        size_t base = (size_t)row * head_dim;
        float sum = 0.0f;
        for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
            float value = bf16_to_f32(values[base + dimension]);
            sum = fmaf(value, value, sum);
        }
        float inverse = 1.0f / sqrtf(sum / (float)head_dim + epsilon);
        for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
            float value = bf16_to_f32(values[base + dimension]);
            values[base + dimension] = f32_to_bf16(
                value * inverse * bf16_to_f32(weight[dimension]));
        }
    }
}

static void reference_rope(uint16_t *query, uint16_t *key,
                           const float *cosine, const float *sine) {
    const uint32_t half = HEAD_DIM / 2;
    for (uint32_t row = 0; row < SEQUENCE; row++) {
        for (uint32_t head = 0; head < QUERY_HEADS; head++) {
            size_t base = ((size_t)row * QUERY_HEADS + head) * HEAD_DIM;
            for (uint32_t dimension = 0; dimension < half; dimension++) {
                float first = bf16_to_f32(query[base + dimension]);
                float second = bf16_to_f32(query[base + half + dimension]);
                float c = cosine[(size_t)row * half + dimension];
                float s = sine[(size_t)row * half + dimension];
                query[base + dimension] = f32_to_bf16(first * c - second * s);
                query[base + half + dimension] =
                    f32_to_bf16(second * c + first * s);
            }
        }
        for (uint32_t head = 0; head < KV_HEADS; head++) {
            size_t base = ((size_t)row * KV_HEADS + head) * HEAD_DIM;
            for (uint32_t dimension = 0; dimension < half; dimension++) {
                float first = bf16_to_f32(key[base + dimension]);
                float second = bf16_to_f32(key[base + half + dimension]);
                float c = cosine[(size_t)row * half + dimension];
                float s = sine[(size_t)row * half + dimension];
                key[base + dimension] = f32_to_bf16(first * c - second * s);
                key[base + half + dimension] =
                    f32_to_bf16(second * c + first * s);
            }
        }
    }
}

static void reference_gqa(const uint16_t *query, const uint16_t *key,
                          const uint16_t *value, uint16_t *output,
                          float scale) {
    for (uint32_t row = 0; row < SEQUENCE; row++) {
        for (uint32_t q_head = 0; q_head < QUERY_HEADS; q_head++) {
            uint32_t kv_head = q_head / (QUERY_HEADS / KV_HEADS);
            size_t q_base = ((size_t)row * QUERY_HEADS + q_head) * HEAD_DIM;
            float scores[SEQUENCE];
            float maximum = -INFINITY;
            for (uint32_t key_row = 0; key_row <= row; key_row++) {
                size_t k_base = ((size_t)key_row * KV_HEADS + kv_head) *
                                HEAD_DIM;
                float dot = 0.0f;
                for (uint32_t dimension = 0; dimension < HEAD_DIM; dimension++)
                    dot = fmaf(bf16_to_f32(query[q_base + dimension]),
                               bf16_to_f32(key[k_base + dimension]), dot);
                float score = bf16_to_f32(f32_to_bf16(
                    bf16_to_f32(f32_to_bf16(dot)) * scale));
                scores[key_row] = score;
                maximum = fmaxf(maximum, score);
            }
            float denominator = 0.0f;
            for (uint32_t key_row = 0; key_row <= row; key_row++) {
                scores[key_row] = expf(scores[key_row] - maximum);
                denominator += scores[key_row];
            }
            for (uint32_t dimension = 0; dimension < HEAD_DIM; dimension++) {
                float sum = 0.0f;
                for (uint32_t key_row = 0; key_row <= row; key_row++) {
                    size_t v_index = ((size_t)key_row * KV_HEADS + kv_head) *
                                     HEAD_DIM + dimension;
                    float probability = bf16_to_f32(f32_to_bf16(
                        scores[key_row] / denominator));
                    sum = fmaf(probability,
                               bf16_to_f32(value[v_index]), sum);
                }
                output[q_base + dimension] = f32_to_bf16(sum);
            }
        }
    }
}

static void reference_vision_qkv(
        const uint16_t *qkv, const uint16_t *cosine, const uint16_t *sine,
        uint16_t *query, uint16_t *key, uint16_t *value) {
    const uint32_t half = VISION_HEAD_DIM / 2;
    const size_t inner = VISION_HEADS * VISION_HEAD_DIM;
    for (uint32_t row = 0; row < VISION_SEQUENCE; row++) {
        size_t row_base = (size_t)row * inner * 3;
        for (uint32_t head = 0; head < VISION_HEADS; head++) {
            size_t q_base = row_base + (size_t)head * VISION_HEAD_DIM;
            size_t k_base = q_base + inner;
            size_t v_base = q_base + inner * 2;
            size_t output_base = ((size_t)row * VISION_HEADS + head) *
                                 VISION_HEAD_DIM;
            for (uint32_t dimension = 0; dimension < VISION_HEAD_DIM;
                 dimension++) {
                uint32_t pair = dimension < half ? dimension + half :
                                                    dimension - half;
                size_t rope_index = (size_t)row * half + dimension % half;
                float c = bf16_to_f32(cosine[rope_index]);
                float s = bf16_to_f32(sine[rope_index]);
                float q0 = bf16_to_f32(qkv[q_base + dimension]);
                float k0 = bf16_to_f32(qkv[k_base + dimension]);
                float q1 = bf16_to_f32(qkv[q_base + pair]);
                float k1 = bf16_to_f32(qkv[k_base + pair]);
                query[output_base + dimension] = f32_to_bf16(
                    dimension < half ? q0 * c - q1 * s : q0 * c + q1 * s);
                key[output_base + dimension] = f32_to_bf16(
                    dimension < half ? k0 * c - k1 * s : k0 * c + k1 * s);
                value[output_base + dimension] = qkv[v_base + dimension];
            }
        }
    }
}

static int reflect_coordinate(int coordinate, int length) {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= length) return 2 * length - coordinate - 2;
    return coordinate;
}

static void reference_pad(const float *input, float *output) {
    for (uint32_t out_t = 0; out_t < PAD_OUT_DEPTH; out_t++) {
        for (uint32_t out_y = 0; out_y < PAD_OUT_HEIGHT; out_y++) {
            for (uint32_t out_x = 0; out_x < PAD_OUT_WIDTH; out_x++) {
                for (uint32_t channel = 0; channel < PAD_CHANNELS; channel++) {
                    size_t destination = (((size_t)out_t * PAD_OUT_HEIGHT +
                        out_y) * PAD_OUT_WIDTH + out_x) * PAD_CHANNELS + channel;
                    if (out_t == 0) {
                        output[destination] = 0.0f;
                        continue;
                    }
                    uint32_t source_t = out_t - 1;
                    uint32_t source_y = (uint32_t)reflect_coordinate(
                        (int)out_y - 1, PAD_HEIGHT);
                    uint32_t source_x = (uint32_t)reflect_coordinate(
                        (int)out_x - 1, PAD_WIDTH);
                    size_t source = (((size_t)source_t * PAD_HEIGHT + source_y) *
                        PAD_WIDTH + source_x) * PAD_CHANNELS + channel;
                    output[destination] = input[source];
                }
            }
        }
    }
}

static void reference_conv3d(const float *input, const float *weight,
                             const float *bias, float *output) {
    for (uint32_t out_t = 0; out_t < CONV_OUT_DEPTH; out_t++) {
        for (uint32_t out_y = 0; out_y < CONV_OUT_HEIGHT; out_y++) {
            for (uint32_t out_x = 0; out_x < CONV_OUT_WIDTH; out_x++) {
                for (uint32_t out_channel = 0;
                     out_channel < CONV_OUT_CHANNELS; out_channel++) {
                    float sum = bias[out_channel];
                    for (uint32_t in_channel = 0;
                         in_channel < PAD_CHANNELS; in_channel++) {
                        for (uint32_t kt = 0; kt < 2; kt++) {
                            for (uint32_t ky = 0; ky < 2; ky++) {
                                for (uint32_t kx = 0; kx < 2; kx++) {
                                    size_t input_index =
                                        (((size_t)(out_t + kt) *
                                           PAD_OUT_HEIGHT + out_y + ky) *
                                          PAD_OUT_WIDTH + out_x + kx) *
                                         PAD_CHANNELS + in_channel;
                                    size_t weight_index =
                                        ((((size_t)out_channel * PAD_CHANNELS +
                                            in_channel) * 2 + kt) * 2 + ky) *
                                         2 + kx;
                                    sum = fmaf(input[input_index],
                                               weight[weight_index], sum);
                                }
                            }
                        }
                    }
                    size_t output_index =
                        (((size_t)out_t * CONV_OUT_HEIGHT + out_y) *
                         CONV_OUT_WIDTH + out_x) * CONV_OUT_CHANNELS +
                        out_channel;
                    output[output_index] = sum;
                }
            }
        }
    }
}

static void reference_group_norm_silu(const float *input, const float *weight,
                                      const float *bias, float *output) {
    uint32_t channels_per_group = GROUP_CHANNELS / GROUPS;
    uint32_t elements = GROUP_HEIGHT * GROUP_WIDTH * channels_per_group;
    for (uint32_t temporal = 0; temporal < GROUP_DEPTH; temporal++) {
        for (uint32_t group = 0; group < GROUPS; group++) {
            float sum = 0.0f;
            for (uint32_t index = 0; index < elements; index++) {
                uint32_t spatial = index / channels_per_group;
                uint32_t channel = group * channels_per_group +
                                   index % channels_per_group;
                size_t source = ((size_t)temporal * GROUP_HEIGHT * GROUP_WIDTH +
                                 spatial) * GROUP_CHANNELS + channel;
                sum += input[source];
            }
            float mean = sum / (float)elements;
            float square_sum = 0.0f;
            for (uint32_t index = 0; index < elements; index++) {
                uint32_t spatial = index / channels_per_group;
                uint32_t channel = group * channels_per_group +
                                   index % channels_per_group;
                size_t source = ((size_t)temporal * GROUP_HEIGHT * GROUP_WIDTH +
                                 spatial) * GROUP_CHANNELS + channel;
                float centered = input[source] - mean;
                square_sum = fmaf(centered, centered, square_sum);
            }
            float inverse = 1.0f / sqrtf(square_sum / (float)elements + 1e-6f);
            for (uint32_t index = 0; index < elements; index++) {
                uint32_t spatial = index / channels_per_group;
                uint32_t channel = group * channels_per_group +
                                   index % channels_per_group;
                size_t destination =
                    ((size_t)temporal * GROUP_HEIGHT * GROUP_WIDTH + spatial) *
                    GROUP_CHANNELS + channel;
                float normalized = (input[destination] - mean) * inverse *
                                   weight[channel] + bias[channel];
                output[destination] = normalized /
                                      (1.0f + expf(-normalized));
            }
        }
    }
}

int main(void) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) fail(error);

    uint16_t embedding_weight[VOCAB * WIDTH];
    for (size_t index = 0; index < VOCAB * WIDTH; index++)
        embedding_weight[index] = f32_to_bf16((float)((int)index - 8) / 8.0f);
    uint32_t token_ids[TOKENS] = {0, 4, 7};
    uint16_t embedding_want[TOKENS * WIDTH];
    for (size_t column = 0; column < WIDTH; column++) {
        embedding_want[column] = embedding_weight[column];
        embedding_want[WIDTH + column] = embedding_weight[4 * WIDTH + column];
        embedding_want[2 * WIDTH + column] = f32_to_bf16(0.0f);
    }

    uint16_t norm_input[NORM_COUNT], norm_weight[HEAD_DIM];
    for (size_t index = 0; index < NORM_COUNT; index++)
        norm_input[index] = f32_to_bf16((float)((int)(index % 9) - 4) / 3.0f);
    for (size_t index = 0; index < HEAD_DIM; index++)
        norm_weight[index] = f32_to_bf16(0.75f + (float)index * 0.125f);
    uint16_t norm_want[NORM_COUNT];
    memcpy(norm_want, norm_input, sizeof(norm_want));
    reference_head_norm(norm_want, norm_weight, 4, HEAD_DIM, 1e-6f);

    uint16_t query[QUERY_COUNT], key[KV_COUNT], value[KV_COUNT];
    for (size_t index = 0; index < QUERY_COUNT; index++)
        query[index] = f32_to_bf16((float)((int)(index % 17) - 8) / 9.0f);
    for (size_t index = 0; index < KV_COUNT; index++) {
        key[index] = f32_to_bf16((float)((int)(index % 13) - 6) / 7.0f);
        value[index] = f32_to_bf16((float)((int)(index % 11) - 5) / 6.0f);
    }
    float rope_cos[ROPE_COUNT], rope_sin[ROPE_COUNT];
    for (uint32_t row = 0; row < SEQUENCE; row++) {
        for (uint32_t dimension = 0; dimension < HEAD_DIM / 2; dimension++) {
            float angle = (float)(row + 1) * (float)(dimension + 1) * 0.13f;
            rope_cos[row * (HEAD_DIM / 2) + dimension] = cosf(angle);
            rope_sin[row * (HEAD_DIM / 2) + dimension] = sinf(angle);
        }
    }
    uint16_t query_want[QUERY_COUNT], key_want[KV_COUNT];
    uint16_t gqa_want[QUERY_COUNT];
    memcpy(query_want, query, sizeof(query_want));
    memcpy(key_want, key, sizeof(key_want));
    reference_rope(query_want, key_want, rope_cos, rope_sin);
    reference_gqa(query_want, key_want, value, gqa_want, 0.5f);

    uint16_t vision_qkv[VISION_COUNT * 3];
    uint16_t vision_cos[VISION_ROPE_COUNT], vision_sin[VISION_ROPE_COUNT];
    uint16_t vision_query_want[VISION_COUNT], vision_key_want[VISION_COUNT];
    uint16_t vision_value_want[VISION_COUNT];
    for (size_t index = 0; index < VISION_COUNT * 3; index++)
        vision_qkv[index] = f32_to_bf16(
            (float)((int)(index % 19) - 9) / 10.0f);
    for (uint32_t row = 0; row < VISION_SEQUENCE; row++) {
        for (uint32_t dimension = 0; dimension < VISION_HEAD_DIM / 2;
             dimension++) {
            float angle = (float)(row + 1) * (float)(dimension + 1) * 0.17f;
            size_t index = (size_t)row * (VISION_HEAD_DIM / 2) + dimension;
            vision_cos[index] = f32_to_bf16(cosf(angle));
            vision_sin[index] = f32_to_bf16(sinf(angle));
        }
    }
    reference_vision_qkv(
        vision_qkv, vision_cos, vision_sin, vision_query_want,
        vision_key_want, vision_value_want);

    float audio_qkv[AUDIO_COUNT * 3];
    float q_bias[AUDIO_WIDTH], k_bias[AUDIO_WIDTH], v_bias[AUDIO_WIDTH];
    float audio_query_want[AUDIO_COUNT], audio_key_want[AUDIO_COUNT];
    float audio_value_want[AUDIO_COUNT], audio_pool_want[AUDIO_POOL_COUNT];
    for (size_t index = 0; index < AUDIO_COUNT * 3; index++)
        audio_qkv[index] = (float)((int)(index % 23) - 11) / 11.0f;
    for (size_t index = 0; index < AUDIO_WIDTH; index++) {
        q_bias[index] = (float)index * 0.01f;
        k_bias[index] = -(float)index * 0.02f;
        v_bias[index] = 0.125f + (float)index * 0.005f;
    }
    for (uint32_t row = 0; row < AUDIO_BATCH * AUDIO_LENGTH; row++) {
        size_t base = (size_t)row * AUDIO_WIDTH * 3;
        for (uint32_t column = 0; column < AUDIO_WIDTH; column++) {
            size_t index = (size_t)row * AUDIO_WIDTH + column;
            audio_query_want[index] = audio_qkv[base + column] + q_bias[column];
            audio_key_want[index] =
                audio_qkv[base + AUDIO_WIDTH + column] + k_bias[column];
            audio_value_want[index] =
                audio_qkv[base + AUDIO_WIDTH * 2 + column] + v_bias[column];
        }
    }
    for (uint32_t row = 0; row < AUDIO_BATCH * AUDIO_LENGTH; row++) {
        for (uint32_t column = 0; column < AUDIO_OUTPUT_DIM; column++) {
            float sum = 0.0f;
            uint32_t pool = AUDIO_HEAD_DIM / AUDIO_OUTPUT_DIM;
            for (uint32_t head = 0; head < AUDIO_HEADS; head++) {
                size_t base = ((size_t)row * AUDIO_HEADS + head) *
                              AUDIO_HEAD_DIM + (size_t)column * pool;
                for (uint32_t item = 0; item < pool; item++)
                    sum += audio_value_want[base + item];
            }
            audio_pool_want[(size_t)row * AUDIO_OUTPUT_DIM + column] =
                sum / (float)(AUDIO_HEADS * pool);
        }
    }
    float geglu_gate[GEGLU_COUNT], geglu_linear[GEGLU_COUNT];
    float geglu_want[GEGLU_COUNT];
    for (size_t index = 0; index < GEGLU_COUNT; index++) {
        float x = (float)((int)index - 3) * 0.75f;
        geglu_gate[index] = x;
        geglu_linear[index] = 0.5f + (float)index * 0.125f;
        float cube = x * x * x;
        float gelu = 0.5f * x *
            (1.0f + tanhf(0.7978845608028654f *
                          (x + 0.044715f * cube)));
        geglu_want[index] = gelu * geglu_linear[index];
    }
    float swiglu_fused[SWIGLU_COUNT * 2], swiglu_want[SWIGLU_COUNT];
    for (size_t row = 0; row < SWIGLU_ROWS; row++) {
        for (size_t column = 0; column < SWIGLU_WIDTH; column++) {
            size_t index = row * SWIGLU_WIDTH + column;
            size_t base = row * SWIGLU_WIDTH * 2;
            float gate = (float)((int)index - 2) * 0.625f;
            float up_value = 0.25f + (float)index * 0.375f;
            swiglu_fused[base + column] = gate;
            swiglu_fused[base + SWIGLU_WIDTH + column] = up_value;
            swiglu_want[index] = gate / (1.0f + expf(-gate)) * up_value;
        }
    }

    float pad_input[PAD_INPUT_COUNT], pad_want[PAD_OUTPUT_COUNT];
    float conv_weight[CONV_WEIGHT_COUNT], conv_bias[CONV_OUT_CHANNELS];
    float conv_want[CONV_OUTPUT_COUNT];
    for (size_t index = 0; index < PAD_INPUT_COUNT; index++)
        pad_input[index] = (float)((int)(index % 29) - 14) / 13.0f;
    for (size_t index = 0; index < CONV_WEIGHT_COUNT; index++)
        conv_weight[index] = (float)((int)(index % 17) - 8) / 31.0f;
    for (size_t index = 0; index < CONV_OUT_CHANNELS; index++)
        conv_bias[index] = (float)((int)index - 1) * 0.125f;
    reference_pad(pad_input, pad_want);
    reference_conv3d(pad_want, conv_weight, conv_bias, conv_want);

    float group_input[GROUP_COUNT], group_weight[GROUP_CHANNELS];
    float group_bias[GROUP_CHANNELS], group_want[GROUP_COUNT];
    for (size_t index = 0; index < GROUP_COUNT; index++)
        group_input[index] = (float)((int)(index % 21) - 10) / 8.0f;
    for (size_t index = 0; index < GROUP_CHANNELS; index++) {
        group_weight[index] = 0.75f + (float)index * 0.125f;
        group_bias[index] = (float)((int)index - 2) * 0.05f;
    }
    reference_group_norm_silu(
        group_input, group_weight, group_bias, group_want);

    h3_gpu_tensor *embedding_weight_gpu = from_bf16(
        gpu, embedding_weight, VOCAB * WIDTH);
    h3_gpu_tensor *token_ids_gpu = h3_gpu_tensor_from_u32(
        gpu, token_ids, TOKENS);
    h3_gpu_tensor *embedding_gpu = fresh_bf16(gpu, TOKENS * WIDTH);
    h3_gpu_tensor *norm_gpu = from_bf16(gpu, norm_input, NORM_COUNT);
    h3_gpu_tensor *norm_weight_gpu = from_bf16(gpu, norm_weight, HEAD_DIM);
    h3_gpu_tensor *query_gpu = from_bf16(gpu, query, QUERY_COUNT);
    h3_gpu_tensor *key_gpu = from_bf16(gpu, key, KV_COUNT);
    h3_gpu_tensor *value_gpu = from_bf16(gpu, value, KV_COUNT);
    h3_gpu_tensor *cos_gpu = h3_gpu_tensor_from_f32(gpu, rope_cos, ROPE_COUNT);
    h3_gpu_tensor *sin_gpu = h3_gpu_tensor_from_f32(gpu, rope_sin, ROPE_COUNT);
    h3_gpu_tensor *gqa_gpu = fresh_bf16(gpu, QUERY_COUNT);
    h3_gpu_tensor *vision_qkv_gpu = from_bf16(
        gpu, vision_qkv, VISION_COUNT * 3);
    h3_gpu_tensor *vision_cos_gpu = from_bf16(
        gpu, vision_cos, VISION_ROPE_COUNT);
    h3_gpu_tensor *vision_sin_gpu = from_bf16(
        gpu, vision_sin, VISION_ROPE_COUNT);
    h3_gpu_tensor *vision_query_gpu = fresh_bf16(gpu, VISION_COUNT);
    h3_gpu_tensor *vision_key_gpu = fresh_bf16(gpu, VISION_COUNT);
    h3_gpu_tensor *vision_value_gpu = fresh_bf16(gpu, VISION_COUNT);
    h3_gpu_tensor *audio_qkv_gpu = from_f32(
        gpu, audio_qkv, AUDIO_COUNT * 3);
    h3_gpu_tensor *q_bias_gpu = from_f32(gpu, q_bias, AUDIO_WIDTH);
    h3_gpu_tensor *k_bias_gpu = from_f32(gpu, k_bias, AUDIO_WIDTH);
    h3_gpu_tensor *v_bias_gpu = from_f32(gpu, v_bias, AUDIO_WIDTH);
    h3_gpu_tensor *audio_query_gpu = fresh_f32(gpu, AUDIO_COUNT);
    h3_gpu_tensor *audio_key_gpu = fresh_f32(gpu, AUDIO_COUNT);
    h3_gpu_tensor *audio_value_gpu = fresh_f32(gpu, AUDIO_COUNT);
    h3_gpu_tensor *audio_pool_gpu = fresh_f32(gpu, AUDIO_POOL_COUNT);
    h3_gpu_tensor *geglu_gate_gpu = from_f32(gpu, geglu_gate, GEGLU_COUNT);
    h3_gpu_tensor *geglu_linear_gpu = from_f32(
        gpu, geglu_linear, GEGLU_COUNT);
    h3_gpu_tensor *geglu_gpu = fresh_f32(gpu, GEGLU_COUNT);
    h3_gpu_tensor *swiglu_fused_gpu = from_f32(
        gpu, swiglu_fused, SWIGLU_COUNT * 2);
    h3_gpu_tensor *swiglu_gpu = fresh_f32(gpu, SWIGLU_COUNT);
    h3_gpu_tensor *pad_input_gpu = from_f32(gpu, pad_input, PAD_INPUT_COUNT);
    h3_gpu_tensor *pad_gpu = fresh_f32(gpu, PAD_OUTPUT_COUNT);
    h3_gpu_tensor *conv_weight_gpu = from_f32(
        gpu, conv_weight, CONV_WEIGHT_COUNT);
    h3_gpu_tensor *conv_bias_gpu = from_f32(
        gpu, conv_bias, CONV_OUT_CHANNELS);
    h3_gpu_tensor *conv_gpu = fresh_f32(gpu, CONV_OUTPUT_COUNT);
    h3_gpu_tensor *group_input_gpu = from_f32(
        gpu, group_input, GROUP_COUNT);
    h3_gpu_tensor *group_weight_gpu = from_f32(
        gpu, group_weight, GROUP_CHANNELS);
    h3_gpu_tensor *group_bias_gpu = from_f32(
        gpu, group_bias, GROUP_CHANNELS);
    h3_gpu_tensor *group_gpu = fresh_f32(gpu, GROUP_COUNT);
    if (!token_ids_gpu || !cos_gpu || !sin_gpu) fail(h3_gpu_error(gpu));

    require_gpu(gpu, h3_gpu_begin(gpu), "begin reference operators");
    require_gpu(gpu, h3_gpu_embedding_bf16(
        gpu, embedding_gpu, embedding_weight_gpu, token_ids_gpu,
        TOKENS, VOCAB, WIDTH), "embedding");
    require_gpu(gpu, h3_gpu_head_rms_norm_bf16(
        gpu, norm_gpu, norm_weight_gpu, 2, 2, HEAD_DIM, 1e-6f),
        "head RMSNorm");
    require_gpu(gpu, h3_gpu_rope_text_bf16(
        gpu, query_gpu, key_gpu, cos_gpu, sin_gpu, SEQUENCE,
        QUERY_HEADS, KV_HEADS, HEAD_DIM), "text RoPE");
    require_gpu(gpu, h3_gpu_gqa_causal_bf16(
        gpu, gqa_gpu, query_gpu, key_gpu, value_gpu, SEQUENCE,
        QUERY_HEADS, KV_HEADS, HEAD_DIM, 0.5f), "causal GQA");
    require_gpu(gpu, h3_gpu_vision_qkv_rope_bf16(
        gpu, vision_query_gpu, vision_key_gpu, vision_value_gpu,
        vision_qkv_gpu, vision_cos_gpu, vision_sin_gpu, VISION_SEQUENCE,
        VISION_HEADS, VISION_HEAD_DIM, VISION_HEAD_DIM / 2),
        "vision QKV/RoPE");
    require_gpu(gpu, h3_gpu_audio_qkv_split_f32(
        gpu, audio_query_gpu, audio_key_gpu, audio_value_gpu, audio_qkv_gpu,
        q_bias_gpu, k_bias_gpu, v_bias_gpu, AUDIO_BATCH, AUDIO_LENGTH,
        AUDIO_HEADS, AUDIO_HEAD_DIM), "audio QKV split");
    require_gpu(gpu, h3_gpu_audio_attention_pool_f32(
        gpu, audio_pool_gpu, audio_value_gpu, AUDIO_BATCH, AUDIO_LENGTH,
        AUDIO_HEADS, AUDIO_HEAD_DIM, AUDIO_OUTPUT_DIM),
        "audio attention pool");
    require_gpu(gpu, h3_gpu_geglu_f32(
        gpu, geglu_gpu, geglu_gate_gpu, geglu_linear_gpu, GEGLU_COUNT),
        "audio GeGLU");
    require_gpu(gpu, h3_gpu_swiglu_f32(
        gpu, swiglu_gpu, swiglu_fused_gpu, SWIGLU_ROWS, SWIGLU_WIDTH),
        "video VAE SwiGLU");
    require_gpu(gpu, h3_gpu_vae_encoder_pad_f32(
        gpu, pad_gpu, pad_input_gpu, 1, PAD_DEPTH, PAD_HEIGHT, PAD_WIDTH,
        PAD_CHANNELS, 1, 1, 1, 1, 1), "VAE encoder padding");
    require_gpu(gpu, h3_gpu_conv3d_f32(
        gpu, conv_gpu, pad_gpu, conv_weight_gpu, conv_bias_gpu, 1,
        PAD_OUT_DEPTH, PAD_OUT_HEIGHT, PAD_OUT_WIDTH, PAD_CHANNELS,
        CONV_OUT_CHANNELS, 2, 2, 2, 1, 1, 1), "VAE encoder Conv3D");
    require_gpu(gpu, h3_gpu_vae_encoder_group_norm_silu_f32(
        gpu, group_gpu, group_input_gpu, group_weight_gpu, group_bias_gpu, 1,
        GROUP_DEPTH, GROUP_HEIGHT, GROUP_WIDTH, GROUP_CHANNELS, GROUPS, 1e-6f),
        "VAE encoder GroupNorm/SiLU");
    require_gpu(gpu, h3_gpu_submit(gpu), "submit reference operators");

    uint16_t embedding_got[TOKENS * WIDTH], norm_got[NORM_COUNT];
    uint16_t query_got[QUERY_COUNT], key_got[KV_COUNT], gqa_got[QUERY_COUNT];
    uint16_t vision_query_got[VISION_COUNT], vision_key_got[VISION_COUNT];
    uint16_t vision_value_got[VISION_COUNT];
    float audio_query_got[AUDIO_COUNT], audio_key_got[AUDIO_COUNT];
    float audio_value_got[AUDIO_COUNT], audio_pool_got[AUDIO_POOL_COUNT];
    float geglu_got[GEGLU_COUNT];
    float swiglu_got[SWIGLU_COUNT];
    float pad_got[PAD_OUTPUT_COUNT], conv_got[CONV_OUTPUT_COUNT];
    float group_got[GROUP_COUNT];
    if (!h3_gpu_tensor_read_bf16(embedding_gpu, embedding_got,
                                  TOKENS * WIDTH) ||
        !h3_gpu_tensor_read_bf16(norm_gpu, norm_got, NORM_COUNT) ||
        !h3_gpu_tensor_read_bf16(query_gpu, query_got, QUERY_COUNT) ||
        !h3_gpu_tensor_read_bf16(key_gpu, key_got, KV_COUNT) ||
        !h3_gpu_tensor_read_bf16(gqa_gpu, gqa_got, QUERY_COUNT) ||
        !h3_gpu_tensor_read_bf16(
            vision_query_gpu, vision_query_got, VISION_COUNT) ||
        !h3_gpu_tensor_read_bf16(
            vision_key_gpu, vision_key_got, VISION_COUNT) ||
        !h3_gpu_tensor_read_bf16(
            vision_value_gpu, vision_value_got, VISION_COUNT) ||
        !h3_gpu_tensor_read_f32(audio_query_gpu, audio_query_got,
                                AUDIO_COUNT) ||
        !h3_gpu_tensor_read_f32(audio_key_gpu, audio_key_got, AUDIO_COUNT) ||
        !h3_gpu_tensor_read_f32(audio_value_gpu, audio_value_got,
                                AUDIO_COUNT) ||
        !h3_gpu_tensor_read_f32(audio_pool_gpu, audio_pool_got,
                                AUDIO_POOL_COUNT) ||
        !h3_gpu_tensor_read_f32(geglu_gpu, geglu_got, GEGLU_COUNT) ||
        !h3_gpu_tensor_read_f32(swiglu_gpu, swiglu_got, SWIGLU_COUNT) ||
        !h3_gpu_tensor_read_f32(pad_gpu, pad_got, PAD_OUTPUT_COUNT) ||
        !h3_gpu_tensor_read_f32(conv_gpu, conv_got, CONV_OUTPUT_COUNT) ||
        !h3_gpu_tensor_read_f32(group_gpu, group_got, GROUP_COUNT))
        fail("cannot read reference operator outputs");
    check_exact("embedding", embedding_got, embedding_want, TOKENS * WIDTH);
    check_close("head RMSNorm", norm_got, norm_want, NORM_COUNT, 0.015625f);
    check_exact("query RoPE", query_got, query_want, QUERY_COUNT);
    check_exact("key RoPE", key_got, key_want, KV_COUNT);
    check_close("causal GQA", gqa_got, gqa_want, QUERY_COUNT, 0.015625f);
    check_exact("vision query RoPE", vision_query_got, vision_query_want,
                VISION_COUNT);
    check_exact("vision key RoPE", vision_key_got, vision_key_want,
                VISION_COUNT);
    check_exact("vision value split", vision_value_got, vision_value_want,
                VISION_COUNT);
    check_f32_close("audio query split", audio_query_got, audio_query_want,
                    AUDIO_COUNT, 1e-7f);
    check_f32_close("audio key split", audio_key_got, audio_key_want,
                    AUDIO_COUNT, 1e-7f);
    check_f32_close("audio value split", audio_value_got, audio_value_want,
                    AUDIO_COUNT, 1e-7f);
    check_f32_close("audio attention pool", audio_pool_got, audio_pool_want,
                    AUDIO_POOL_COUNT, 1e-7f);
    check_f32_close("audio GeGLU", geglu_got, geglu_want, GEGLU_COUNT,
                    2e-6f);
    check_f32_close("video VAE SwiGLU", swiglu_got, swiglu_want,
                    SWIGLU_COUNT, 2e-6f);
    check_f32_close("VAE encoder padding", pad_got, pad_want,
                    PAD_OUTPUT_COUNT, 0.0f);
    check_f32_close("VAE encoder Conv3D", conv_got, conv_want,
                    CONV_OUTPUT_COUNT, 2e-6f);
    check_f32_close("VAE GroupNorm/SiLU", group_got, group_want,
                    GROUP_COUNT, 2e-6f);

    h3_gpu_tensor_free(group_gpu);
    h3_gpu_tensor_free(group_bias_gpu);
    h3_gpu_tensor_free(group_weight_gpu);
    h3_gpu_tensor_free(group_input_gpu);
    h3_gpu_tensor_free(conv_gpu);
    h3_gpu_tensor_free(conv_bias_gpu);
    h3_gpu_tensor_free(conv_weight_gpu);
    h3_gpu_tensor_free(pad_gpu);
    h3_gpu_tensor_free(pad_input_gpu);
    h3_gpu_tensor_free(geglu_gpu);
    h3_gpu_tensor_free(swiglu_gpu);
    h3_gpu_tensor_free(swiglu_fused_gpu);
    h3_gpu_tensor_free(geglu_linear_gpu);
    h3_gpu_tensor_free(geglu_gate_gpu);
    h3_gpu_tensor_free(audio_pool_gpu);
    h3_gpu_tensor_free(audio_value_gpu);
    h3_gpu_tensor_free(audio_key_gpu);
    h3_gpu_tensor_free(audio_query_gpu);
    h3_gpu_tensor_free(v_bias_gpu);
    h3_gpu_tensor_free(k_bias_gpu);
    h3_gpu_tensor_free(q_bias_gpu);
    h3_gpu_tensor_free(audio_qkv_gpu);
    h3_gpu_tensor_free(vision_value_gpu);
    h3_gpu_tensor_free(vision_key_gpu);
    h3_gpu_tensor_free(vision_query_gpu);
    h3_gpu_tensor_free(vision_sin_gpu);
    h3_gpu_tensor_free(vision_cos_gpu);
    h3_gpu_tensor_free(vision_qkv_gpu);
    h3_gpu_tensor_free(gqa_gpu);
    h3_gpu_tensor_free(sin_gpu);
    h3_gpu_tensor_free(cos_gpu);
    h3_gpu_tensor_free(value_gpu);
    h3_gpu_tensor_free(key_gpu);
    h3_gpu_tensor_free(query_gpu);
    h3_gpu_tensor_free(norm_weight_gpu);
    h3_gpu_tensor_free(norm_gpu);
    h3_gpu_tensor_free(embedding_gpu);
    h3_gpu_tensor_free(token_ids_gpu);
    h3_gpu_tensor_free(embedding_weight_gpu);
    h3_gpu_free(gpu);
    puts("ok: original H3 HIP reference operators match CPU oracles");
    return 0;
}
