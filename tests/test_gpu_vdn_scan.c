#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float fp32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void step(const float state[4], const float transition[4],
                 const float injection[4], float output[4]) {
    for (unsigned row = 0; row < 2; row++)
        for (unsigned column = 0; column < 2; column++) {
            float value = injection[row * 2 + column];
            for (unsigned inner = 0; inner < 2; inner++)
                value += state[row * 2 + inner] *
                         transition[inner * 2 + column];
            output[row * 2 + column] = value;
        }
}

int main(void) {
    enum { F = 3, H = 1, D = 2, S = 1, W = 4 };
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) return 1;
    uint16_t mean_input[(1 + F * 2) * W];
    for (size_t i = 0; i < sizeof(mean_input) / sizeof(mean_input[0]); i++)
        mean_input[i] = bf16(((float)(int)i - 8.0f) * 0.1f);
    float delta[F * H * D] = {-0.4f, 0.1f, 0.7f, -0.2f, 0.3f, 0.9f};
    uint16_t dt[D] = {bf16(-0.1f), bf16(0.2f)};
    uint16_t alog[H] = {bf16(logf(1.7f))};
    h3_gpu_tensor *mi = h3_gpu_tensor_from_bf16(
        gpu, mean_input, sizeof(mean_input) / sizeof(mean_input[0]));
    h3_gpu_tensor *mo = h3_gpu_tensor_new_f32(gpu, F * W);
    h3_gpu_tensor *de = h3_gpu_tensor_from_f32(gpu, delta, F * H * D);
    h3_gpu_tensor *dtb = h3_gpu_tensor_from_bf16(gpu, dt, D);
    h3_gpu_tensor *alo = h3_gpu_tensor_from_bf16(gpu, alog, H);
    h3_gpu_tensor *alpha_t = h3_gpu_tensor_new_f32(gpu, F * H * D);
    int ok = mi && mo && de && dtb && alo && alpha_t && h3_gpu_begin(gpu) &&
        h3_gpu_vdn_frame_mean_bf16(gpu, mo, mi, W, F, 2, W) &&
        h3_gpu_vdn_alpha_f32(gpu, alpha_t, de, dtb, alo, F, H, D) &&
        h3_gpu_submit(gpu);
    float means[F * W], alpha[F * H * D];
    ok = ok && h3_gpu_tensor_read_f32(mo, means, F * W) &&
         h3_gpu_tensor_read_f32(alpha_t, alpha, F * H * D);
    for (unsigned frame = 0; frame < F && ok; frame++) {
        for (unsigned column = 0; column < W; column++) {
            float expected = 0.5f *
                (fp32(mean_input[W + (frame * 2) * W + column]) +
                 fp32(mean_input[W + (frame * 2 + 1) * W + column]));
            if (fabsf(means[frame * W + column] - expected) > 1e-6f) ok = 0;
        }
        for (unsigned channel = 0; channel < D; channel++) {
            size_t index = (size_t)frame * D + channel;
            float x = delta[index] + fp32(dt[channel]);
            float expected = expf(-expf(fp32(alog[0])) * log1pf(expf(x)));
            if (fabsf(alpha[index] - expected) > 2e-6f) ok = 0;
        }
    }

    float transition[F * 4] = {
        0.8f, 0.1f, -0.05f, 0.7f,
        0.75f, -0.04f, 0.08f, 0.72f,
        0.68f, 0.03f, -0.02f, 0.77f
    };
    float injection[F * 4] = {
        0.1f, -0.03f, 0.04f, 0.08f,
        -0.02f, 0.05f, 0.07f, -0.01f,
        0.03f, 0.02f, -0.04f, 0.06f
    };
    float text[4] = {0.2f, -0.1f, 0.05f, 0.3f};
    h3_gpu_tensor *tr = h3_gpu_tensor_from_f32(gpu, transition, F * 4);
    h3_gpu_tensor *in = h3_gpu_tensor_from_f32(gpu, injection, F * 4);
    h3_gpu_tensor *tx = h3_gpu_tensor_from_f32(gpu, text, 4);
    h3_gpu_tensor *pr = h3_gpu_tensor_new_f32(gpu, F * 4);
    h3_gpu_tensor *su = h3_gpu_tensor_new_f32(gpu, F * 4);
    ok = ok && tr && in && tx && pr && su && h3_gpu_begin(gpu) &&
         h3_gpu_vdn_scan_f32(gpu, pr, su, tr, in, tx, 0.5f, F, H, D) &&
         h3_gpu_submit(gpu);
    float prefix[F * 4], suffix[F * 4], expected_prefix[F * 4];
    float expected_suffix[F * 4], state[4];
    ok = ok && h3_gpu_tensor_read_f32(pr, prefix, F * 4) &&
         h3_gpu_tensor_read_f32(su, suffix, F * 4);
    for (unsigned i = 0; i < 4; i++) state[i] = 0.5f * text[i];
    for (unsigned frame = 0; frame < F; frame++) {
        step(state, transition + frame * 4, injection + frame * 4,
             expected_prefix + frame * 4);
        memcpy(state, expected_prefix + frame * 4, sizeof(state));
    }
    for (unsigned i = 0; i < 4; i++) state[i] = 0.5f * text[i];
    for (unsigned reverse = F; reverse > 0; reverse--) {
        unsigned frame = reverse - 1;
        step(state, transition + frame * 4, injection + frame * 4,
             expected_suffix + frame * 4);
        memcpy(state, expected_suffix + frame * 4, sizeof(state));
    }
    for (size_t i = 0; i < F * 4 && ok; i++)
        if (fabsf(prefix[i] - expected_prefix[i]) > 2e-6f ||
            fabsf(suffix[i] - expected_suffix[i]) > 2e-6f) ok = 0;

    uint16_t query[F * H * D] = {
        bf16(0.4f), bf16(-0.2f), bf16(0.1f), bf16(0.5f),
        bf16(-0.3f), bf16(0.2f)
    };
    uint16_t gates[F * H * D] = {
        bf16(-0.2f), bf16(0.3f), bf16(0.1f), bf16(-0.4f),
        bf16(0.5f), bf16(-0.1f)
    };
    uint16_t norm[D] = {bf16(0.9f), bf16(1.1f)};
    h3_gpu_tensor *qt = h3_gpu_tensor_from_bf16(gpu, query, F * H * D);
    h3_gpu_tensor *gt = h3_gpu_tensor_from_bf16(gpu, gates, F * H * D);
    h3_gpu_tensor *nt = h3_gpu_tensor_from_bf16(gpu, norm, D);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, F * H * D);
    ok = ok && qt && gt && nt && out && h3_gpu_begin(gpu) &&
         h3_gpu_vdn_readout_bf16(gpu, out, qt, pr, su, alpha_t, tx, 0.5f,
             nt, gt, F, S, H, D, 0, 0, 1e-6f) && h3_gpu_submit(gpu);
    uint16_t actual[F * H * D];
    ok = ok && h3_gpu_tensor_read_bf16(out, actual, F * H * D);
    for (unsigned frame = 0; frame < F && ok; frame++) {
        const float *before = frame ? expected_prefix + (frame - 1) * 4 : text;
        const float *after = frame + 1 < F ?
            expected_suffix + (frame + 1) * 4 : text;
        float readout[D], sum = 1e-6f;
        for (unsigned row = 0; row < D; row++) {
            readout[row] = 0.0f;
            for (unsigned column = 0; column < D; column++) {
                float b = before[row * D + column] * alpha[frame * D + column];
                float a = after[row * D + column] * alpha[frame * D + column];
                if (!frame) b *= 0.5f;
                if (frame + 1 == F) a *= 0.5f;
                readout[row] += fp32(query[frame * D + column]) * (b + a);
            }
            sum += readout[row] * readout[row];
        }
        for (unsigned row = 0; row < D; row++) {
            float gate = 1.0f / (1.0f + expf(-fp32(gates[frame * D + row])));
            float expected = readout[row] / sqrtf(sum / (float)D) *
                             fp32(norm[row]) * gate;
            if (fabsf(fp32(actual[frame * D + row]) - expected) > 0.015f) {
                fprintf(stderr, "readout mismatch frame=%u row=%u %g/%g\n",
                        frame, row, fp32(actual[frame * D + row]), expected);
                ok = 0;
            }
        }
    }
    h3_gpu_tensor_free(out); h3_gpu_tensor_free(nt); h3_gpu_tensor_free(gt);
    h3_gpu_tensor_free(qt); h3_gpu_tensor_free(su); h3_gpu_tensor_free(pr);
    h3_gpu_tensor_free(tx); h3_gpu_tensor_free(in); h3_gpu_tensor_free(tr);
    h3_gpu_tensor_free(alpha_t); h3_gpu_tensor_free(alo);
    h3_gpu_tensor_free(dtb); h3_gpu_tensor_free(de);
    h3_gpu_tensor_free(mo); h3_gpu_tensor_free(mi); h3_gpu_free(gpu);
    if (!ok) {
        if (*error) fprintf(stderr, "%s\n", error);
        return 1;
    }
    puts("VDN frame mean, FP32 alpha, bidirectional scan and readout passed");
    return 0;
}
