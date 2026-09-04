#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

static float sepconv(const uint16_t *input, const uint16_t *spatial,
                     const uint16_t *temporal, unsigned frame, unsigned y,
                     unsigned x, unsigned channel, unsigned frames,
                     unsigned height, unsigned width, unsigned channels) {
    float result = 0.0f;
    for (int dt = -2; dt <= 2; dt++) {
        int st = (int)frame + dt;
        if (st < 0 || st >= (int)frames) continue;
        float spatial_sum = 0.0f;
        for (int dy = -2; dy <= 2; dy++) {
            int sy = (int)y + dy;
            if (sy < 0 || sy >= (int)height) continue;
            for (int dx = -2; dx <= 2; dx++) {
                int sx = (int)x + dx;
                if (sx < 0 || sx >= (int)width) continue;
                size_t row = ((size_t)st * height + (unsigned)sy) * width +
                             (unsigned)sx;
                size_t wi = (size_t)channel * 25 +
                            (unsigned)(dy + 2) * 5 + (unsigned)(dx + 2);
                spatial_sum += fp32(input[row * channels + channel]) *
                               fp32(spatial[wi]);
            }
        }
        result += spatial_sum *
                  fp32(temporal[(size_t)channel * 5 + (unsigned)(dt + 2)]);
    }
    return result;
}

int main(void) {
    enum { FRAMES = 3, HEIGHT = 2, WIDTH = 2, HEADS = 2, DIM = 4,
           CHANNELS = HEADS * DIM, ROWS = FRAMES * HEIGHT * WIDTH };
    uint16_t qraw[ROWS * CHANNELS], kraw[ROWS * CHANNELS];
    uint16_t vraw[ROWS * CHANNELS];
    uint16_t ksp[CHANNELS * 25], vsp[CHANNELS * 25];
    uint16_t ktm[CHANNELS * 5], vtm[CHANNELS * 5];
    for (size_t i = 0; i < ROWS * CHANNELS; i++) {
        qraw[i] = bf16(((float)((int)(i % 17) - 8)) * 0.08f);
        kraw[i] = bf16(sinf((float)i * 0.13f));
        vraw[i] = bf16(cosf((float)i * 0.09f));
    }
    for (size_t i = 0; i < CHANNELS * 25; i++) {
        ksp[i] = bf16(((float)((int)(i % 11) - 5)) * 0.035f);
        vsp[i] = bf16(((float)((int)(i % 7) - 3)) * 0.045f);
    }
    for (size_t i = 0; i < CHANNELS * 5; i++) {
        ktm[i] = bf16(((float)((int)(i % 5) - 2)) * 0.12f);
        vtm[i] = bf16(((float)((int)(i % 3) - 1)) * 0.15f);
    }
    char error[512];
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
#define FROM(name, host, count) h3_gpu_tensor *name = \
    h3_gpu_tensor_from_bf16(gpu, host, count)
    FROM(qr, qraw, ROWS * CHANNELS);
    FROM(kr, kraw, ROWS * CHANNELS);
    FROM(vr, vraw, ROWS * CHANNELS);
    FROM(ks, ksp, CHANNELS * 25);
    FROM(kt, ktm, CHANNELS * 5);
    FROM(vs, vsp, CHANNELS * 25);
    FROM(vt, vtm, CHANNELS * 5);
#undef FROM
    h3_gpu_tensor *q = h3_gpu_tensor_new_bf16(gpu, ROWS * CHANNELS);
    h3_gpu_tensor *k = h3_gpu_tensor_new_bf16(gpu, ROWS * CHANNELS);
    h3_gpu_tensor *v = h3_gpu_tensor_new_bf16(gpu, ROWS * CHANNELS);
    int ok = qr && kr && vr && ks && kt && vs && vt && q && k && v &&
        h3_gpu_begin(gpu) && h3_gpu_vdn_linear_features_bf16(
            gpu, q, k, v, qr, kr, vr, ks, kt, vs, vt, FRAMES, HEIGHT,
            WIDTH, HEADS, DIM, 1e-6f) && h3_gpu_submit(gpu);
    uint16_t qa[ROWS * CHANNELS], ka[ROWS * CHANNELS], va[ROWS * CHANNELS];
    ok = ok && h3_gpu_tensor_read_bf16(q, qa, ROWS * CHANNELS) &&
         h3_gpu_tensor_read_bf16(k, ka, ROWS * CHANNELS) &&
         h3_gpu_tensor_read_bf16(v, va, ROWS * CHANNELS);
    for (unsigned row = 0; row < ROWS && ok; row++) {
        unsigned frame = row / (HEIGHT * WIDTH);
        unsigned spatial = row % (HEIGHT * WIDTH);
        unsigned y = spatial / WIDTH;
        unsigned x = spatial % WIDTH;
        for (unsigned head = 0; head < HEADS; head++) {
            float qf[DIM], kf[DIM], vf[DIM];
            float qsum = 1e-6f, ksum = 1e-6f;
            for (unsigned d = 0; d < DIM; d++) {
                unsigned channel = head * DIM + d;
                size_t index = (size_t)row * CHANNELS + channel;
                float qx = fp32(qraw[index]);
                float kx = sepconv(kraw, ksp, ktm, frame, y, x, channel,
                                   FRAMES, HEIGHT, WIDTH, CHANNELS);
                float vx = sepconv(vraw, vsp, vtm, frame, y, x, channel,
                                   FRAMES, HEIGHT, WIDTH, CHANNELS);
                qf[d] = qx / (1.0f + expf(-qx));
                kf[d] = kx / (1.0f + expf(-kx));
                vf[d] = vx / (1.0f + expf(-vx));
                qsum += qf[d] * qf[d];
                ksum += kf[d] * kf[d];
            }
            for (unsigned d = 0; d < DIM; d++) {
                size_t index = (size_t)row * CHANNELS + head * DIM + d;
                if (fabsf(fp32(qa[index]) - qf[d] / sqrtf(qsum)) > 0.012f ||
                    fabsf(fp32(ka[index]) - kf[d] / sqrtf(ksum)) > 0.012f ||
                    fabsf(fp32(va[index]) - vf[d]) > 0.012f) {
                    fprintf(stderr, "feature mismatch at row=%u head=%u d=%u\n",
                            row, head, d);
                    ok = 0;
                    break;
                }
            }
        }
    }
    h3_gpu_tensor_free(v); h3_gpu_tensor_free(k); h3_gpu_tensor_free(q);
    h3_gpu_tensor_free(vt); h3_gpu_tensor_free(vs);
    h3_gpu_tensor_free(kt); h3_gpu_tensor_free(ks);
    h3_gpu_tensor_free(vr); h3_gpu_tensor_free(kr); h3_gpu_tensor_free(qr);
    h3_gpu_free(gpu);
    if (!ok) {
        if (*error) fprintf(stderr, "%s\n", error);
        return 1;
    }
    puts("VDN separable K/V short-conv and Q/K/V feature preparation passed");
    return 0;
}
