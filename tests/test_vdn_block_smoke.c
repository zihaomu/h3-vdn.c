#include "h3_gpu.h"
#include "h3_vdn_dit.h"
#include "h3_vdn_prompt.h"
#include "h3_vdn_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TEXT_ROWS = 800, HIDDEN = 5376, VIDEO_PATCH = 96,
       FRAMES = 17, LATENT_H = 2, LATENT_W = 4,
       FRAME_H = LATENT_H / 2, FRAME_W = LATENT_W / 2,
       VIDEO_ROWS = FRAMES * FRAME_H * FRAME_W,
       SEQUENCE = TEXT_ROWS + VIDEO_ROWS, ROPE_HALF = 48 };

static uint64_t hash_bf16(const uint16_t *values, size_t count) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < count; index++) {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s H3_BASE STAGE_DMD PROMPT\n", argv[0]);
        return 2;
    }
    int status = 1;
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create(NULL, error, sizeof(error));
    h3_vdn_weight_store *store = NULL;
    h3_vdn_model_weights model;
    h3_vdn_block_weights block;
    h3_vdn_layout layout;
    h3_text_embedding prompt;
    memset(&model, 0, sizeof(model));
    memset(&block, 0, sizeof(block));
    memset(&prompt, 0, sizeof(prompt));
    memset(&layout, 0, sizeof(layout));
    h3_gpu_tensor *refined = NULL, *time = NULL, *modulation = NULL;
    h3_gpu_tensor *row_map = NULL, *rope_cos = NULL, *rope_sin = NULL;
    h3_gpu_tensor *video = NULL, *hidden = NULL;
    uint16_t *before = NULL, *after = NULL;
    unsigned block_count = 1;
    const char *block_count_text = getenv("VDN_SMOKE_BLOCKS");
    if (block_count_text && *block_count_text) {
        char *end = NULL;
        unsigned long parsed = strtoul(block_count_text, &end, 10);
        if (end == block_count_text || *end || parsed < 1 || parsed > 50) {
            fprintf(stderr, "VDN_SMOKE_BLOCKS must be in [1,50]\n");
            return 2;
        }
        block_count = (unsigned)parsed;
    }
    if (!gpu) goto failed;
    store = h3_vdn_weight_store_open(argv[1], argv[2], 1,
                                     error, sizeof(error));
    if (!store || !h3_vdn_prompt_load(argv[3], &prompt,
                                      error, sizeof(error)) ||
        !h3_vdn_model_weights_load(store, gpu, &model,
                                   error, sizeof(error))) goto failed;
    refined = h3_vdn_refine_prompt(gpu, &model, &prompt,
                                   error, sizeof(error));
    if (!refined) goto failed;
    if (!h3_vdn_layout_build(&prompt, FRAMES, LATENT_H, LATENT_W, 0,
                             &layout, error, sizeof(error)) ||
        layout.sequence != SEQUENCE || layout.text_rows != TEXT_ROWS ||
        layout.audio_rows != 0 || layout.video_start != TEXT_ROWS ||
        layout.video_rows != VIDEO_ROWS ||
        layout.frame_height != FRAME_H || layout.frame_width != FRAME_W) {
        if (!error[0]) snprintf(error, sizeof(error),
                                "unexpected formal VDN layout geometry");
        goto failed;
    }
    float timestep = 0.0f;
    time = h3_vdn_time_embedding(gpu, &model, &timestep, 1,
                                 error, sizeof(error));
    if (!time) goto failed;
    uint32_t map[SEQUENCE];
    float video_rows[VIDEO_ROWS * VIDEO_PATCH];
    for (size_t row = 0; row < SEQUENCE; row++)
        map[row] = layout.token_tags[row];
    for (size_t index = 0; index < VIDEO_ROWS * VIDEO_PATCH; index++)
        video_rows[index] = sinf((float)index * 0.019f) * 0.7f;
    row_map = h3_gpu_tensor_from_u32(gpu, map, SEQUENCE);
    rope_cos = h3_gpu_tensor_from_bf16(gpu, layout.rope_cos,
                                       SEQUENCE * ROPE_HALF);
    rope_sin = h3_gpu_tensor_from_bf16(gpu, layout.rope_sin,
                                       SEQUENCE * ROPE_HALF);
    video = h3_gpu_tensor_from_f32(gpu, video_rows,
                                   VIDEO_ROWS * VIDEO_PATCH);
    hidden = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * HIDDEN);
    if (!row_map || !rope_cos || !rope_sin || !video || !hidden ||
        !h3_gpu_begin(gpu) ||
        !h3_gpu_copy_bf16(gpu, hidden, 0, refined, 0,
                          (size_t)TEXT_ROWS * HIDDEN) ||
        !h3_gpu_patch_linear_bf16_offset(
            gpu, hidden, (size_t)TEXT_ROWS * HIDDEN, video, 0,
            model.video_in_weight, model.video_in_bias, VIDEO_ROWS,
            VIDEO_PATCH, HIDDEN) || !h3_gpu_submit(gpu)) {
        snprintf(error, sizeof(error), "cannot construct packed block input: %s",
                 h3_gpu_error(gpu));
        goto failed;
    }
    size_t elements = (size_t)SEQUENCE * HIDDEN;
    before = malloc(elements * sizeof(*before));
    after = malloc(elements * sizeof(*after));
    if (!before || !after || !h3_gpu_tensor_read_bf16(hidden, before, elements)) {
        snprintf(error, sizeof(error), "cannot read block input");
        goto failed;
    }
    for (unsigned layer = 0; layer < block_count; layer++) {
        if (!h3_vdn_block_weights_load(
                store, gpu, layer, &block, error, sizeof(error))) goto failed;
        modulation = h3_vdn_block_modulation(
            gpu, &block, time, 1, error, sizeof(error));
        if (!modulation || !h3_vdn_run_block(
                gpu, &block, hidden, modulation, row_map, rope_cos, rope_sin,
                SEQUENCE, TEXT_ROWS, TEXT_ROWS, FRAMES, FRAME_H, FRAME_W,
                1, 5, error, sizeof(error))) goto failed;
        h3_gpu_tensor_free(modulation);
        modulation = NULL;
        h3_vdn_block_weights_free(&block);
        fprintf(stderr, "VDN stack smoke: block %u/%u\n", layer + 1,
                block_count);
    }
    if (!h3_gpu_tensor_read_bf16(hidden, after, elements)) goto failed;
    size_t changed = 0;
    size_t invalid = 0;
    for (size_t index = 0; index < elements; index++) {
        changed += before[index] != after[index];
        invalid += (after[index] & UINT16_C(0x7f80)) == UINT16_C(0x7f80);
    }
    if (changed < elements / 2 || invalid) {
        snprintf(error, sizeof(error),
                 "block output changed=%zu/%zu invalid=%zu",
                 changed, elements, invalid);
        goto failed;
    }
    h3_gpu_stats stats;
    if (!h3_gpu_get_stats(gpu, &stats)) goto failed;
    printf("VDN real %u-block stack passed: BF16[%d,%d], hash=%016llx, "
           "changed=%zu/%zu, peak=%.3f GiB\n", block_count, SEQUENCE, HIDDEN,
           (unsigned long long)hash_bf16(after, elements),
           changed, elements,
           (double)stats.peak_live_bytes / (1024.0 * 1024.0 * 1024.0));
    status = 0;
    goto cleanup;
failed:
    fprintf(stderr, "VDN real block smoke failed: %s\n", error);
cleanup:
    free(after); free(before);
    h3_gpu_tensor_free(hidden); h3_gpu_tensor_free(video);
    h3_gpu_tensor_free(rope_sin); h3_gpu_tensor_free(rope_cos);
    h3_gpu_tensor_free(row_map); h3_gpu_tensor_free(modulation);
    h3_gpu_tensor_free(time); h3_gpu_tensor_free(refined);
    h3_vdn_block_weights_free(&block);
    h3_vdn_model_weights_free(&model);
    h3_vdn_prompt_free(&prompt);
    h3_vdn_layout_free(&layout);
    h3_vdn_weight_store_free(store);
    h3_gpu_free(gpu);
    return status;
}
