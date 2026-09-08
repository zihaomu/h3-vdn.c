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

static uint64_t fnv1a64(const void *data, size_t bytes) {
    const unsigned char *values = data;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < bytes; index++) {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int stress_production_cholesky(h3_gpu *gpu) {
    enum { F = 17, H = 56, D = 128, BATCHES = F * H };
    const size_t matrix_elements = (size_t)D * D;
    const size_t elements = (size_t)BATCHES * matrix_elements;
    const size_t alpha_elements = (size_t)BATCHES * D;
    unsigned iterations = 8;
    const char *value = getenv("H3_VDN_SOLVE_STRESS_ITERATIONS");
    if (value && *value) {
        char *end = NULL;
        unsigned long parsed = strtoul(value, &end, 10);
        if (!end || *end || !parsed || parsed > 1000) return 0;
        iterations = (unsigned)parsed;
    }
    float *a = malloc(elements * sizeof(*a));
    float *b = calloc(elements, sizeof(*b));
    float *alpha = malloc(alpha_elements * sizeof(*alpha));
    float *actual = malloc(elements * sizeof(*actual));
    if (!a || !b || !alpha || !actual) {
        free(actual); free(alpha); free(b); free(a);
        return 0;
    }
    for (unsigned batch = 0; batch < BATCHES; batch++)
        for (unsigned row = 0; row < D; row++) {
            float row_value =
                (float)((int)((row * 13u + batch * 7u) % 31u) - 15) *
                0.002f;
            for (unsigned column = 0; column < D; column++) {
                float column_value =
                    (float)((int)((column * 13u + batch * 7u) % 31u) - 15) *
                    0.002f;
                a[(size_t)batch * matrix_elements + (size_t)row * D +
                  column] = row_value * column_value +
                            (row == column ? 0.25f : 0.0f);
            }
        }
    for (size_t index = 0; index < alpha_elements; index++)
        alpha[index] = 0.72f + (float)(index % D) * 0.0005f;

    h3_gpu_tensor *at = h3_gpu_tensor_from_f32(gpu, a, elements);
    h3_gpu_tensor *bt = h3_gpu_tensor_from_f32(gpu, b, elements);
    h3_gpu_tensor *alphat =
        h3_gpu_tensor_from_f32(gpu, alpha, alpha_elements);
    h3_gpu_tensor *transition = h3_gpu_tensor_new_f32(gpu, elements);
    h3_gpu_tensor *injection = h3_gpu_tensor_new_f32(gpu, elements);
    h3_gpu_profile_stats before = {0}, after = {0};
    int ok = at && bt && alphat && transition && injection &&
             h3_gpu_get_profile_stats(gpu, &before);
    uint64_t reference_hash = 0;
    for (unsigned iteration = 0; iteration < iterations && ok; iteration++) {
        ok = h3_gpu_tensor_write_f32(at, a, elements) &&
             h3_gpu_begin(gpu) &&
             h3_gpu_vdn_solve_f32(gpu, transition, injection, at, bt,
                                  alphat, F, H, D) &&
             h3_gpu_submit(gpu) &&
             h3_gpu_tensor_read_f32(transition, actual, elements);
        if (ok) {
            uint64_t hash = fnv1a64(actual, elements * sizeof(*actual));
            if (!iteration) reference_hash = hash;
            else if (hash != reference_hash) {
                fprintf(stderr,
                    "production Cholesky stress changed at iteration %u: "
                    "%016llx != %016llx\n", iteration,
                    (unsigned long long)hash,
                    (unsigned long long)reference_hash);
                ok = 0;
            }
        }
    }
    ok = ok && h3_gpu_get_profile_stats(gpu, &after);
    uint64_t retries = after.solve_retries >= before.solve_retries ?
        after.solve_retries - before.solve_retries : 0;
    const char *fault_value = getenv("H3_TEST_VDN_CORRUPT_POTRF");
    if (ok && fault_value && !strcmp(fault_value, "1") &&
        retries < iterations) {
        fprintf(stderr,
                "production Cholesky fault injection was not retried: "
                "%llu retries for %u iterations\n",
                (unsigned long long)retries, iterations);
        ok = 0;
    }
    if (ok)
        printf("VDN production Cholesky stress passed: iterations=%u "
               "hash=%016llx retries=%llu\n", iterations,
               (unsigned long long)reference_hash,
               (unsigned long long)retries);
    h3_gpu_tensor_free(injection); h3_gpu_tensor_free(transition);
    h3_gpu_tensor_free(alphat); h3_gpu_tensor_free(bt);
    h3_gpu_tensor_free(at);
    free(actual); free(alpha); free(b); free(a);
    return ok;
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
    /* Keep the statistics producer and rocSOLVER consumer in the same command
     * group. Long H3 runs exposed this boundary; a separate submit here would
     * hide the ordering regression that this test is intended to catch. */
    for (unsigned iteration = 0; iteration < 64 && ok; iteration++)
        ok = h3_gpu_begin(gpu) && h3_gpu_vdn_frame_stats_bf16(
                 gpu, at, bst, kt, vt, bt, F, S, H, D) &&
             h3_gpu_vdn_solve_f32(
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
    h3_gpu_tensor_free(vt); h3_gpu_tensor_free(kt);
    if (ok) ok = stress_production_cholesky(gpu);
    h3_gpu_free(gpu);
    if (!ok) {
        if (*error) fprintf(stderr, "%s\n", error);
        return 1;
    }
    puts("VDN FP32 frame statistics and batched Cholesky solve passed");
    return 0;
}
