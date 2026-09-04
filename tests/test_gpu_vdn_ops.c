#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        ok = 0;                                                              \
        goto cleanup;                                                        \
    }                                                                        \
} while (0)

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

static int allowed(uint32_t query, uint32_t key, uint32_t video_start,
                   uint32_t frames, uint32_t per_frame, uint32_t radius,
                   uint32_t chunk) {
    uint32_t video_end = video_start + frames * per_frame;
    if (query < video_start || query >= video_end ||
        key < video_start || key >= video_end) return 1;
    uint32_t qf = (query - video_start) / per_frame;
    uint32_t kf = (key - video_start) / per_frame;
    if (qf == 0 || qf + 1 == frames || kf == 0 || kf + 1 == frames) return 1;
    uint32_t lower;
    uint32_t upper;
    if (chunk) {
        uint32_t qc = qf / chunk;
        lower = qc > radius ? (qc - radius) * chunk : 0;
        upper = (qc + radius + 1) * chunk - 1;
    } else {
        lower = qf > radius ? qf - radius : 0;
        upper = qf + radius;
    }
    if (upper >= frames) upper = frames - 1;
    return kf >= lower && kf <= upper;
}

int main(void) {
    enum { ROWS = 10, HEADS = 2, DIM = 4, INNER = HEADS * DIM };
    const uint32_t video_start = 2, frames = 4, per_frame = 2;
    int ok = 1;
    char error[512];
    h3_gpu *gpu = NULL;
    h3_gpu_tensor *qr = NULL, *kr = NULL, *v = NULL, *qw = NULL, *kw = NULL;
    h3_gpu_tensor *cosine = NULL, *sine = NULL, *q = NULL, *k = NULL;
    h3_gpu_tensor *attended = NULL, *logits = NULL, *gated = NULL;
    uint16_t qr_host[ROWS * INNER], kr_host[ROWS * INNER];
    uint16_t v_host[ROWS * INNER], norm_host[DIM];
    uint16_t cos_host[ROWS * 2], sin_host[ROWS * 2];
    uint16_t logits_host[ROWS * HEADS];
    for (size_t index = 0; index < ROWS * INNER; index++) {
        qr_host[index] = bf16(sinf((float)index * 0.17f) * 0.8f);
        kr_host[index] = bf16(cosf((float)index * 0.11f) * 0.7f);
        v_host[index] = bf16(((float)((int)(index % 13) - 6)) * 0.09f);
    }
    for (size_t index = 0; index < DIM; index++)
        norm_host[index] = bf16(0.8f + (float)index * 0.1f);
    for (size_t row = 0; row < ROWS; row++)
        for (size_t index = 0; index < 2; index++) {
            float angle = (float)(row * 2 + index) * 0.07f;
            cos_host[row * 2 + index] = bf16(cosf(angle));
            sin_host[row * 2 + index] = bf16(sinf(angle));
        }
    for (size_t index = 0; index < ROWS * HEADS; index++)
        logits_host[index] = bf16(((float)(index % 7) - 3.0f) * 0.2f);

    gpu = h3_gpu_create(NULL, error, sizeof(error));
    CHECK(gpu != NULL);
    qr = h3_gpu_tensor_from_bf16(gpu, qr_host, ROWS * INNER);
    kr = h3_gpu_tensor_from_bf16(gpu, kr_host, ROWS * INNER);
    v = h3_gpu_tensor_from_bf16(gpu, v_host, ROWS * INNER);
    qw = h3_gpu_tensor_from_bf16(gpu, norm_host, DIM);
    kw = h3_gpu_tensor_from_bf16(gpu, norm_host, DIM);
    cosine = h3_gpu_tensor_from_bf16(gpu, cos_host, ROWS * 2);
    sine = h3_gpu_tensor_from_bf16(gpu, sin_host, ROWS * 2);
    logits = h3_gpu_tensor_from_bf16(gpu, logits_host, ROWS * HEADS);
    q = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    k = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    attended = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    gated = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    CHECK(qr && kr && v && qw && kw && cosine && sine && logits && q && k &&
          attended && gated);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_vdn_qk_rope_bf16(gpu, q, k, qr, kr, qw, kw, cosine, sine,
                                  ROWS, HEADS, DIM, 2, 1e-5f));
    CHECK(h3_gpu_vdn_window_sdpa_bf16(
        gpu, attended, q, k, v, ROWS, HEADS, DIM, video_start, frames,
        per_frame, 0, 2, 1, 0.5f));
    CHECK(h3_gpu_vdn_softmax_gate_bf16(
        gpu, gated, attended, logits, ROWS, HEADS, DIM));
    CHECK(h3_gpu_submit(gpu));

    uint16_t q_actual[ROWS * INNER], k_actual[ROWS * INNER];
    uint16_t out_actual[ROWS * INNER], gated_actual[ROWS * INNER];
    CHECK(h3_gpu_tensor_read_bf16(q, q_actual, ROWS * INNER));
    CHECK(h3_gpu_tensor_read_bf16(k, k_actual, ROWS * INNER));
    CHECK(h3_gpu_tensor_read_bf16(attended, out_actual, ROWS * INNER));
    CHECK(h3_gpu_tensor_read_bf16(gated, gated_actual, ROWS * INNER));

    for (uint32_t row = 0; row < ROWS; row++)
        for (uint32_t head = 0; head < HEADS; head++) {
            size_t base = ((size_t)row * HEADS + head) * DIM;
            float qsum = 0.0f, ksum = 0.0f;
            for (uint32_t d = 0; d < DIM; d++) {
                qsum += fp32(qr_host[base + d]) * fp32(qr_host[base + d]);
                ksum += fp32(kr_host[base + d]) * fp32(kr_host[base + d]);
            }
            float qi = 1.0f / sqrtf(qsum / (float)DIM + 1e-5f);
            float ki = 1.0f / sqrtf(ksum / (float)DIM + 1e-5f);
            for (uint32_t d = 0; d < DIM; d++) {
                uint32_t rd = d % 2;
                uint32_t pair = d < 2 ? d + 2 : d - 2;
                float sign = d < 2 ? -1.0f : 1.0f;
                float c = fp32(cos_host[row * 2 + rd]);
                float s = fp32(sin_host[row * 2 + rd]);
                float q0 = fp32(qr_host[base + d]) * qi * fp32(norm_host[d]);
                float q1 = fp32(qr_host[base + pair]) * qi *
                           fp32(norm_host[pair]);
                float k0 = fp32(kr_host[base + d]) * ki * fp32(norm_host[d]);
                float k1 = fp32(kr_host[base + pair]) * ki *
                           fp32(norm_host[pair]);
                CHECK(fabsf(fp32(q_actual[base + d]) -
                             (q0 * c + sign * q1 * s)) < 0.02f);
                CHECK(fabsf(fp32(k_actual[base + d]) -
                             (k0 * c + sign * k1 * s)) < 0.02f);
            }
        }

    for (uint32_t qr_index = 0; qr_index < ROWS; qr_index++)
        for (uint32_t head = 0; head < HEADS; head++) {
            float scores[ROWS];
            float maximum = -INFINITY;
            for (uint32_t key_row = 0; key_row < ROWS; key_row++) {
                if (!allowed(qr_index, key_row, video_start, frames,
                             per_frame, 0, 2)) {
                    scores[key_row] = -INFINITY;
                    continue;
                }
                float score = 0.0f;
                for (uint32_t d = 0; d < DIM; d++) {
                    size_t qi = ((size_t)qr_index * HEADS + head) * DIM + d;
                    size_t ki = ((size_t)key_row * HEADS + head) * DIM + d;
                    score += fp32(q_actual[qi]) * fp32(k_actual[ki]);
                }
                scores[key_row] = score * 0.5f;
                if (scores[key_row] > maximum) maximum = scores[key_row];
            }
            float denominator = 0.0f;
            for (uint32_t key_row = 0; key_row < ROWS; key_row++)
                if (isfinite(scores[key_row]))
                    denominator += expf(scores[key_row] - maximum);
            for (uint32_t d = 0; d < DIM; d++) {
                float expected = 0.0f;
                for (uint32_t key_row = 0; key_row < ROWS; key_row++)
                    if (isfinite(scores[key_row])) {
                        size_t vi = ((size_t)key_row * HEADS + head) * DIM + d;
                        expected += expf(scores[key_row] - maximum) /
                                    denominator * fp32(v_host[vi]);
                    }
                size_t oi = ((size_t)qr_index * HEADS + head) * DIM + d;
                CHECK(fabsf(fp32(out_actual[oi]) - expected) < 0.015f);
                float gate = 1.0f / (1.0f + expf(-fp32(
                    logits_host[(size_t)qr_index * HEADS + head])));
                CHECK(fabsf(fp32(gated_actual[oi]) -
                             fp32(out_actual[oi]) * gate) < 0.01f);
            }
        }

cleanup:
    h3_gpu_tensor_free(gated); h3_gpu_tensor_free(logits);
    h3_gpu_tensor_free(attended); h3_gpu_tensor_free(k); h3_gpu_tensor_free(q);
    h3_gpu_tensor_free(sine); h3_gpu_tensor_free(cosine);
    h3_gpu_tensor_free(kw); h3_gpu_tensor_free(qw); h3_gpu_tensor_free(v);
    h3_gpu_tensor_free(kr); h3_gpu_tensor_free(qr);
    h3_gpu_free(gpu);
    if (!ok) return 1;
    puts("VDN QK/RoPE, chunk-window attention, anchors and softmax gate passed");
    return 0;
}
