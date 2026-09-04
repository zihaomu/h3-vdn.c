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

static int invert(const float *input, float *output, unsigned n) {
    float augmented[8][16];
    if (n > 8) return 0;
    for (unsigned row = 0; row < n; row++)
        for (unsigned column = 0; column < n * 2; column++)
            augmented[row][column] = column < n ?
                input[(size_t)row * n + column] :
                (column - n == row ? 1.0f : 0.0f);
    for (unsigned pivot = 0; pivot < n; pivot++) {
        unsigned best = pivot;
        for (unsigned row = pivot + 1; row < n; row++)
            if (fabsf(augmented[row][pivot]) >
                fabsf(augmented[best][pivot])) best = row;
        if (fabsf(augmented[best][pivot]) < 1e-8f) return 0;
        if (best != pivot)
            for (unsigned column = 0; column < n * 2; column++) {
                float temporary = augmented[pivot][column];
                augmented[pivot][column] = augmented[best][column];
                augmented[best][column] = temporary;
            }
        float divisor = augmented[pivot][pivot];
        for (unsigned column = 0; column < n * 2; column++)
            augmented[pivot][column] /= divisor;
        for (unsigned row = 0; row < n; row++) {
            if (row == pivot) continue;
            float factor = augmented[row][pivot];
            for (unsigned column = 0; column < n * 2; column++)
                augmented[row][column] -= factor * augmented[pivot][column];
        }
    }
    for (unsigned row = 0; row < n; row++)
        for (unsigned column = 0; column < n; column++)
            output[(size_t)row * n + column] = augmented[row][n + column];
    return 1;
}

int main(void) {
    enum { F = 2, S = 3, H = 2, D = 4, ROWS = F * S,
           FEATURES = ROWS * H * D, MATRICES = F * H,
           MATRIX_ELEMENTS = MATRICES * D * D };
    uint16_t key[FEATURES], value[FEATURES], beta[ROWS * H];
    float alpha[MATRICES * D];
    for (size_t i = 0; i < FEATURES; i++) {
        key[i] = bf16(((float)((int)(i % 9) - 4)) * 0.07f);
        value[i] = bf16(((float)((int)(i % 13) - 6)) * 0.05f);
    }
    for (size_t i = 0; i < ROWS * H; i++)
        beta[i] = bf16(((float)((int)(i % 5) - 2)) * 0.3f);
    for (size_t i = 0; i < MATRICES * D; i++)
        alpha[i] = 0.72f + (float)(i % D) * 0.04f;
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    if (!gpu) return 1;
    h3_gpu_tensor *kt = h3_gpu_tensor_from_bf16(gpu, key, FEATURES);
    h3_gpu_tensor *vt = h3_gpu_tensor_from_bf16(gpu, value, FEATURES);
    h3_gpu_tensor *bt = h3_gpu_tensor_from_bf16(gpu, beta, ROWS * H);
    h3_gpu_tensor *at = h3_gpu_tensor_new_f32(gpu, MATRIX_ELEMENTS);
    h3_gpu_tensor *bst = h3_gpu_tensor_new_f32(gpu, MATRIX_ELEMENTS);
    h3_gpu_tensor *alphat = h3_gpu_tensor_from_f32(gpu, alpha, MATRICES * D);
    h3_gpu_tensor *transition = h3_gpu_tensor_new_f32(gpu, MATRIX_ELEMENTS);
    h3_gpu_tensor *injection = h3_gpu_tensor_new_f32(gpu, MATRIX_ELEMENTS);
    int ok = kt && vt && bt && at && bst && alphat && transition && injection &&
        h3_gpu_begin(gpu) && h3_gpu_vdn_frame_stats_bf16(
            gpu, at, bst, kt, vt, bt, F, S, H, D) && h3_gpu_submit(gpu);
    float a[MATRIX_ELEMENTS], b[MATRIX_ELEMENTS];
    ok = ok && h3_gpu_tensor_read_f32(at, a, MATRIX_ELEMENTS) &&
         h3_gpu_tensor_read_f32(bst, b, MATRIX_ELEMENTS);
    for (unsigned frame = 0; frame < F && ok; frame++)
        for (unsigned head = 0; head < H && ok; head++)
            for (unsigned row = 0; row < D; row++)
                for (unsigned column = 0; column < D; column++) {
                    float ae = 0.0f, be = 0.0f;
                    for (unsigned token = 0; token < S; token++) {
                        size_t base = ((size_t)(frame * S + token) * H + head) * D;
                        float logit = fp32(beta[(size_t)(frame * S + token) * H + head]);
                        float weight = fp32(bf16(
                            1.0f / (1.0f + expf(-logit))));
                        ae += fp32(key[base + row]) * weight * fp32(key[base + column]);
                        float vb = fp32(bf16(fp32(value[base + row]) * weight));
                        be += vb * fp32(key[base + column]);
                    }
                    be = fp32(bf16(be));
                    size_t index = ((size_t)(frame * H + head) * D + row) * D + column;
                    if (fabsf(a[index] - ae) > 2e-6f ||
                        fabsf(b[index] - be) > 2e-6f) ok = 0;
                }
    if (ok)
        ok = h3_gpu_begin(gpu) && h3_gpu_vdn_solve_f32(
            gpu, transition, injection, at, bst, alphat, F, H, D) &&
            h3_gpu_submit(gpu);
    float actual_t[MATRIX_ELEMENTS], actual_i[MATRIX_ELEMENTS];
    ok = ok && h3_gpu_tensor_read_f32(transition, actual_t, MATRIX_ELEMENTS) &&
         h3_gpu_tensor_read_f32(injection, actual_i, MATRIX_ELEMENTS);
    for (unsigned matrix = 0; matrix < MATRICES && ok; matrix++) {
        float system[D * D], inverse[D * D];
        for (unsigned row = 0; row < D; row++)
            for (unsigned column = 0; column < D; column++) {
                size_t local = (size_t)row * D + column;
                system[local] = a[(size_t)matrix * D * D + local] +
                                (row == column ? 1.0f : 0.0f);
            }
        ok = invert(system, inverse, D);
        for (unsigned row = 0; row < D && ok; row++)
            for (unsigned column = 0; column < D; column++) {
                size_t index = (size_t)matrix * D * D + (size_t)row * D + column;
                float expected_t = alpha[(size_t)matrix * D + row] *
                                   inverse[(size_t)row * D + column];
                float expected_i = 0.0f;
                for (unsigned inner = 0; inner < D; inner++)
                    expected_i += b[(size_t)matrix * D * D +
                                    (size_t)row * D + inner] *
                                  inverse[(size_t)inner * D + column];
                if (fabsf(actual_t[index] - expected_t) > 2e-4f ||
                    fabsf(actual_i[index] - expected_i) > 2e-4f) {
                    fprintf(stderr, "solve mismatch matrix=%u row=%u col=%u: "
                            "t=%g/%g i=%g/%g\n", matrix, row, column,
                            actual_t[index], expected_t,
                            actual_i[index], expected_i);
                    ok = 0;
                    break;
                }
            }
    }
    h3_gpu_tensor_free(injection); h3_gpu_tensor_free(transition);
    h3_gpu_tensor_free(alphat); h3_gpu_tensor_free(bst);
    h3_gpu_tensor_free(at); h3_gpu_tensor_free(bt);
    h3_gpu_tensor_free(vt); h3_gpu_tensor_free(kt); h3_gpu_free(gpu);
    if (!ok) {
        if (*error) fprintf(stderr, "%s\n", error);
        return 1;
    }
    puts("VDN FP32 frame statistics and batched Cholesky solve passed");
    return 0;
}
