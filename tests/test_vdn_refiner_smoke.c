#include "h3_gpu.h"
#include "h3_vdn_dit.h"
#include "h3_vdn_prompt.h"
#include "h3_vdn_weights.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s H3_BASE STAGE_DMD PROMPT\n", argv[0]);
        return 2;
    }
    int status = 1;
    char error[512];
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
    refined = h3_vdn_refine_prompt(gpu, &weights, &prompt,
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
    printf("VDN real prompt refinement passed: BF16[%zu,5376], "
           "hash=%016llx, nonzero=%zu/%zu\n",
           prompt.tokens, (unsigned long long)hash, nonzero, elements);
    status = 0;
    goto cleanup;
failed:
    fprintf(stderr, "VDN refiner smoke failed: %s\n", error);
cleanup:
    free(values);
    h3_gpu_tensor_free(refined);
    h3_vdn_model_weights_free(&weights);
    h3_vdn_prompt_free(&prompt);
    h3_vdn_layout_free(&layout);
    h3_vdn_weight_store_free(store);
    h3_gpu_free(gpu);
    return status;
}
