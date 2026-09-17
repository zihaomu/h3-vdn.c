#include "h3_gpu.h"
#include "h3_safetensors.h"
#include "h3_vdn_dit.h"
#include "h3_vdn_prompt.h"
#include "h3_vdn_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

typedef struct {
    const h3_st_header *header;
} refiner_observer_context;

static int observe_refiner_stage(
        h3_gpu *gpu, const char *stage, const h3_gpu_tensor *tensor,
        void *opaque, char *error, size_t error_size) {
    (void)gpu;
    refiner_observer_context *context = opaque;
    char name[128];
    int length = snprintf(name, sizeof(name), "refiner.%s", stage);
    if (length < 0 || (size_t)length >= sizeof(name)) {
        snprintf(error, error_size, "refiner oracle stage name is too long");
        return 0;
    }
    const h3_st_tensor *reference = h3_st_find(context->header, name);
    if (!reference) return 1; /* Fixture intentionally captures block 0 in detail. */
    size_t elements = h3_gpu_tensor_elements(tensor);
    if (reference->dtype != H3_DTYPE_BF16 ||
        h3_st_tensor_elements(reference) != elements) {
        snprintf(error, error_size, "%s has an incompatible oracle tensor", name);
        return 0;
    }
    uint16_t *actual = malloc(elements * sizeof(*actual));
    uint16_t *expected = malloc(elements * sizeof(*expected));
    if (!actual || !expected ||
        !h3_gpu_tensor_read_bf16(tensor, actual, elements) ||
        !h3_st_read_data(context->header, reference, expected,
                         elements * sizeof(*expected), error, error_size)) {
        if (!error[0]) snprintf(error, error_size, "cannot read %s", name);
        free(expected);
        free(actual);
        return 0;
    }
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double squared_actual = 0.0;
    double dot = 0.0;
    double max_abs = 0.0;
    size_t nonfinite = 0;
    size_t changed = 0;
    for (size_t index = 0; index < elements; index++) {
        double a = bf16_to_f32(actual[index]);
        double b = bf16_to_f32(expected[index]);
        if (!isfinite(a) || !isfinite(b)) {
            nonfinite++;
            continue;
        }
        double difference = a - b;
        double absolute = fabs(difference);
        if (absolute > max_abs) max_abs = absolute;
        squared_error += difference * difference;
        squared_reference += b * b;
        squared_actual += a * a;
        dot += a * b;
        changed += actual[index] != expected[index];
    }
    free(expected);
    free(actual);
    double relative_rmse = sqrt(squared_error / squared_reference);
    double cosine = dot / sqrt(squared_actual * squared_reference);
    printf("OpenVDN refiner stage %-32s max_abs=%-12.6g "
           "relative_rmse=%-12.6g cosine=%.12g changed=%zu/%zu\n",
           stage, max_abs, relative_rmse, cosine, changed, elements);
    fflush(stdout);
    if (nonfinite || !isfinite(relative_rmse) || !isfinite(cosine) ||
        relative_rmse > 0.02 || cosine < 0.999) {
        snprintf(error, error_size,
                 "first refiner divergence at %s "
                 "(relative_rmse=%.6g cosine=%.9g nonfinite=%zu)",
                 stage, relative_rmse, cosine, nonfinite);
        return 0;
    }
    return 1;
}

static int compare_oracle(const char *path, const h3_text_embedding *prompt,
                          const uint16_t *actual, size_t elements,
                          char *error, size_t error_size) {
    h3_st_header header;
    memset(&header, 0, sizeof(header));
    if (!h3_st_read_header(path, &header, error, error_size)) return 0;
    const h3_st_tensor *input = h3_st_find(&header, "input.prompt_embeds");
    const h3_st_tensor *final = h3_st_find(&header, "refiner.final");
    uint64_t rows = prompt->tokens;
    if (!input || input->dtype != H3_DTYPE_BF16 || input->ndim != 2 ||
        input->shape[0] != rows || input->shape[1] != 5120 ||
        !final || final->dtype != H3_DTYPE_BF16 || final->ndim != 3 ||
        final->shape[0] != 1 || final->shape[1] != rows ||
        final->shape[2] != 5376 || h3_st_tensor_elements(final) != elements) {
        snprintf(error, error_size, "invalid refiner oracle schema");
        h3_st_free_header(&header);
        return 0;
    }
    size_t prompt_bytes = prompt->tokens * prompt->width * sizeof(uint16_t);
    uint16_t *expected_input = malloc(prompt_bytes);
    uint16_t *expected = malloc(elements * sizeof(*expected));
    if (!expected_input || !expected ||
        !h3_st_read_data(&header, input, expected_input, prompt_bytes,
                         error, error_size) ||
        !h3_st_read_data(&header, final, expected,
                         elements * sizeof(*expected), error, error_size)) {
        if (!error[0]) snprintf(error, error_size, "cannot read refiner oracle");
        free(expected);
        free(expected_input);
        h3_st_free_header(&header);
        return 0;
    }
    if (memcmp(expected_input, prompt->values, prompt_bytes)) {
        snprintf(error, error_size, "oracle prompt bytes do not match test prompt");
        free(expected);
        free(expected_input);
        h3_st_free_header(&header);
        return 0;
    }
    free(expected_input);
    h3_st_free_header(&header);

    double squared_error = 0.0;
    double squared_reference = 0.0;
    double dot = 0.0;
    double squared_actual = 0.0;
    double max_abs = 0.0;
    size_t nonfinite = 0;
    size_t changed = 0;
    for (size_t index = 0; index < elements; index++) {
        double a = bf16_to_f32(actual[index]);
        double b = bf16_to_f32(expected[index]);
        if (!isfinite(a) || !isfinite(b)) {
            nonfinite++;
            continue;
        }
        double difference = a - b;
        double absolute = fabs(difference);
        if (absolute > max_abs) max_abs = absolute;
        squared_error += difference * difference;
        squared_reference += b * b;
        squared_actual += a * a;
        dot += a * b;
        changed += actual[index] != expected[index];
    }
    free(expected);
    double relative_rmse = sqrt(squared_error / squared_reference);
    double cosine = dot / sqrt(squared_actual * squared_reference);
    printf("OpenVDN refiner oracle: max_abs=%.9g relative_rmse=%.9g "
           "cosine=%.12g changed=%zu/%zu nonfinite=%zu\n",
           max_abs, relative_rmse, cosine, changed, elements, nonfinite);

    /* The bound is fixed from two known-correct upstream PyTorch backends:
     * native-vs-native_math final output gives relative_rmse=0.004386 and
     * cosine=0.99999038. Leave margin for rocBLAS ordering, but reject a
     * semantically diverged refiner rather than promoting a finite hash. */
    if (nonfinite || !isfinite(relative_rmse) || !isfinite(cosine) ||
        max_abs > 2.0 || relative_rmse > 0.01 || cosine < 0.9999) {
        snprintf(error, error_size,
                 "refiner exceeds upstream parity bounds "
                 "(max_abs<=2, relative_rmse<=0.01, cosine>=0.9999)");
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "usage: %s H3_BASE STAGE_DMD PROMPT [ORACLE]\n",
                argv[0]);
        return 2;
    }
    int status = 1;
    char error[512] = {0};
    h3_text_embedding prompt;
    h3_vdn_layout layout;
    h3_vdn_model_weights weights;
    memset(&prompt, 0, sizeof(prompt));
    memset(&layout, 0, sizeof(layout));
    memset(&weights, 0, sizeof(weights));
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    h3_vdn_weight_store *store = NULL;
    h3_gpu_tensor *refined = NULL;
    uint16_t *values = NULL;
    h3_st_header oracle_header;
    memset(&oracle_header, 0, sizeof(oracle_header));
    int oracle_open = 0;
    if (!gpu) goto failed;
    store = h3_vdn_weight_store_open(argv[1], argv[2], 1,
                                     error, sizeof(error));
    if (!store || !h3_vdn_prompt_load(argv[3], &prompt,
                                      error, sizeof(error)) ||
        !h3_vdn_model_weights_load(store, gpu, &weights,
                                   error, sizeof(error))) goto failed;
    if (!h3_vdn_layout_build(&prompt, 17, 2, 4, 93, &layout,
                             error, sizeof(error)) ||
        layout.text_rows != prompt.tokens ||
        layout.audio_start != prompt.tokens) {
        if (!error[0]) snprintf(error, sizeof(error),
                                "variable prompt layout mismatch");
        goto failed;
    }
    refiner_observer_context observer_context = {0};
    h3_vdn_refiner_observer observer = NULL;
    if (argc == 5) {
        if (!h3_st_read_header(argv[4], &oracle_header,
                               error, sizeof(error))) goto failed;
        oracle_open = 1;
        observer_context.header = &oracle_header;
        observer = observe_refiner_stage;
    }
    refined = h3_vdn_refine_prompt_observed(
        gpu, &weights, &prompt, observer, &observer_context,
        error, sizeof(error));
    if (!refined) goto failed;
    size_t elements = prompt.tokens * 5376;
    if (h3_gpu_tensor_elements(refined) != elements) {
        snprintf(error, sizeof(error), "refined prompt size mismatch");
        goto failed;
    }
    values = malloc(elements * sizeof(*values));
    if (!values || !h3_gpu_tensor_read_bf16(refined, values, elements)) {
        snprintf(error, sizeof(error), "cannot read refined prompt");
        goto failed;
    }
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t nonzero = 0;
    for (size_t index = 0; index < elements; index++) {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
        nonzero += values[index] != 0;
    }
    if (nonzero < elements / 2) {
        snprintf(error, sizeof(error), "refined prompt is unexpectedly sparse");
        goto failed;
    }
    if (argc == 5 && !compare_oracle(argv[4], &prompt, values, elements,
                                     error, sizeof(error))) goto failed;
    printf("VDN real prompt refinement passed: BF16[%zu,5376], "
           "hash=%016llx, nonzero=%zu/%zu\n",
           prompt.tokens, (unsigned long long)hash, nonzero, elements);
    status = 0;
    goto cleanup;
failed:
    fprintf(stderr, "VDN refiner smoke failed: %s\n", error);
cleanup:
    if (oracle_open) h3_st_free_header(&oracle_header);
    free(values);
    h3_gpu_tensor_free(refined);
    h3_vdn_model_weights_free(&weights);
    h3_vdn_prompt_free(&prompt);
    h3_vdn_layout_free(&layout);
    h3_vdn_weight_store_free(store);
    h3_gpu_free(gpu);
    return status;
}
