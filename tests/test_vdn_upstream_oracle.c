#include "h3_dit.h"
#include "h3_host.h"
#include "h3_safetensors.h"
#include "h3_vdn_dit.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEXT_ROWS = 800,
    LATENT_FRAMES = 17,
    LATENT_HEIGHT = 2,
    LATENT_WIDTH = 4,
    AUDIO_LATENTS = 3,
    NFE = 8,
    VIDEO_CHANNELS = 24,
    VIDEO_PATCH = 96,
    AUDIO_ROWS = AUDIO_LATENTS * 2,
    VIDEO_ROWS = LATENT_FRAMES * (LATENT_HEIGHT / 2) * (LATENT_WIDTH / 2),
    SEQUENCE = TEXT_ROWS + AUDIO_ROWS + VIDEO_ROWS
};

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_vdn_upstream_oracle.c: %s\n", message);
    exit(1);
}

static void *read_tensor(const h3_st_header *header, const char *name,
                         h3_dtype dtype, int ndim, const uint64_t *shape,
                         size_t *bytes) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor || tensor->dtype != dtype || tensor->ndim != ndim)
        die(name);
    for (int axis = 0; axis < ndim; axis++)
        if (tensor->shape[axis] != shape[axis]) die(name);
    uint64_t elements = h3_st_tensor_elements(tensor);
    size_t item_size = h3_dtype_size(dtype);
    if (!item_size || elements > SIZE_MAX / item_size) die(name);
    *bytes = (size_t)elements * item_size;
    void *data = malloc(*bytes ? *bytes : 1);
    char error[512] = {0};
    if (!data || !h3_st_read_data(header, tensor, data, *bytes,
                                  error, sizeof(error))) {
        fprintf(stderr, "cannot read %s: %s\n", name, error);
        free(data);
        exit(1);
    }
    return data;
}

static void require_close(double actual, double expected, double tolerance,
                          const char *label, size_t index) {
    if (!isfinite(actual) || !isfinite(expected) ||
        fabs(actual - expected) > tolerance) {
        fprintf(stderr, "FAIL %s[%zu]: %.17g != %.17g (tol %.3g)\n",
                label, index, actual, expected, tolerance);
        exit(1);
    }
}

static void test_prompt_and_layout(const h3_st_header *header) {
    const uint64_t prompt_shape[] = {TEXT_ROWS, 5120};
    const uint64_t text_tag_shape[] = {TEXT_ROWS};
    const uint64_t position_shape[] = {SEQUENCE, 3};
    const uint64_t sequence_shape[] = {SEQUENCE};
    const uint64_t text_index_shape[] = {TEXT_ROWS};
    const uint64_t audio_index_shape[] = {AUDIO_ROWS};
    const uint64_t video_index_shape[] = {VIDEO_ROWS};
    const uint64_t rope_shape[] = {SEQUENCE, 48};
    size_t bytes = 0;
    uint16_t *prompt = read_tensor(header, "prompt.embeds", H3_DTYPE_BF16,
                                   2, prompt_shape, &bytes);
    if (bytes != (size_t)TEXT_ROWS * 5120 * sizeof(*prompt))
        die("prompt byte count");
    int64_t *text_tags = read_tensor(
        header, "prompt.text_token_tags", H3_DTYPE_I64, 1,
        text_tag_shape, &bytes);
    double *positions = read_tensor(header, "layout.position_ids",
                                    H3_DTYPE_F64, 2, position_shape, &bytes);
    int64_t *tags = read_tensor(header, "layout.token_tags", H3_DTYPE_I64,
                                1, sequence_shape, &bytes);
    int64_t *text_indices = read_tensor(
        header, "layout.text_indices", H3_DTYPE_I64, 1,
        text_index_shape, &bytes);
    int64_t *audio_indices = read_tensor(
        header, "layout.audio_indices", H3_DTYPE_I64, 1,
        audio_index_shape, &bytes);
    int64_t *video_indices = read_tensor(
        header, "layout.video_indices", H3_DTYPE_I64, 1,
        video_index_shape, &bytes);
    uint16_t *rope_cos = read_tensor(
        header, "layout.rope_cos_half_bf16", H3_DTYPE_BF16, 2,
        rope_shape, &bytes);
    uint16_t *rope_sin = read_tensor(
        header, "layout.rope_sin_half_bf16", H3_DTYPE_BF16, 2,
        rope_shape, &bytes);

    uint8_t *native_text_tags = malloc(TEXT_ROWS);
    if (!native_text_tags) die("native text tags allocation");
    for (size_t row = 0; row < TEXT_ROWS; row++) {
        if (text_tags[row] < 0 || text_tags[row] > UINT8_MAX)
            die("text tag outside U8");
        native_text_tags[row] = (uint8_t)text_tags[row];
    }
    h3_text_embedding native_prompt = {
        .values = prompt,
        .tags = native_text_tags,
        .tokens = TEXT_ROWS,
        .width = 5120
    };
    h3_vdn_layout layout;
    char error[512] = {0};
    if (!h3_vdn_layout_build(
            &native_prompt, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH,
            AUDIO_LATENTS, &layout, error, sizeof(error))) {
        fprintf(stderr, "cannot build native layout: %s\n", error);
        exit(1);
    }
    if (layout.sequence != SEQUENCE || layout.packed.segment_count != 3 ||
        layout.packed.segments[0].kind != H3_SEG_TEXT ||
        layout.packed.segments[0].start != 0 ||
        layout.packed.segments[0].stop != TEXT_ROWS ||
        layout.packed.segments[1].kind != H3_SEG_AUDIO ||
        layout.packed.segments[1].start != TEXT_ROWS ||
        layout.packed.segments[1].stop != TEXT_ROWS + AUDIO_ROWS ||
        layout.packed.segments[2].kind != H3_SEG_VIDEO ||
        layout.packed.segments[2].start != TEXT_ROWS + AUDIO_ROWS ||
        layout.packed.segments[2].stop != SEQUENCE)
        die("native segment layout");

    for (size_t row = 0; row < SEQUENCE; row++) {
        require_close(layout.packed.positions[row].t, positions[row * 3], 2e-12,
                      "position.t", row);
        require_close(layout.packed.positions[row].h, positions[row * 3 + 1], 2e-12,
                      "position.h", row);
        require_close(layout.packed.positions[row].w, positions[row * 3 + 2], 2e-12,
                      "position.w", row);
        int64_t expected_tag = row < TEXT_ROWS ? text_tags[row] :
            (row < TEXT_ROWS + AUDIO_ROWS ? 2 : 0);
        if (tags[row] != expected_tag || layout.token_tags[row] != expected_tag)
            die("packed token tags");
    }
    size_t rope_elements = (size_t)SEQUENCE * 48;
    if (memcmp(layout.rope_cos, rope_cos,
               rope_elements * sizeof(*rope_cos))) {
        size_t mismatches = 0;
        for (size_t index = 0; index < rope_elements; index++) {
            if (layout.rope_cos[index] == rope_cos[index]) continue;
            if (!mismatches)
                fprintf(stderr,
                        "RoPE cos first mismatch row=%zu column=%zu: "
                        "native=0x%04x upstream=0x%04x\n",
                        index / 48, index % 48, layout.rope_cos[index],
                        rope_cos[index]);
            mismatches++;
        }
        fprintf(stderr, "RoPE cos mismatches: %zu/%zu\n",
                mismatches, rope_elements);
        die("BF16 RoPE cosine table");
    }
    if (memcmp(layout.rope_sin, rope_sin,
               rope_elements * sizeof(*rope_sin))) {
        size_t mismatches = 0;
        for (size_t index = 0; index < rope_elements; index++) {
            if (layout.rope_sin[index] == rope_sin[index]) continue;
            if (!mismatches)
                fprintf(stderr,
                        "RoPE sin first mismatch row=%zu column=%zu: "
                        "native=0x%04x upstream=0x%04x\n",
                        index / 48, index % 48, layout.rope_sin[index],
                        rope_sin[index]);
            mismatches++;
        }
        fprintf(stderr, "RoPE sin mismatches: %zu/%zu\n",
                mismatches, rope_elements);
        die("BF16 RoPE sine table");
    }
    for (size_t row = 0; row < TEXT_ROWS; row++)
        if (text_indices[row] != (int64_t)row) die("text indices");
    for (size_t row = 0; row < AUDIO_ROWS; row++)
        if (audio_indices[row] != (int64_t)(TEXT_ROWS + row))
            die("audio indices");
    for (size_t row = 0; row < VIDEO_ROWS; row++)
        if (video_indices[row] != (int64_t)(TEXT_ROWS + AUDIO_ROWS + row))
            die("video indices");

    h3_vdn_layout_free(&layout);
    free(native_text_tags);
    free(rope_sin);
    free(rope_cos);
    free(video_indices);
    free(audio_indices);
    free(text_indices);
    free(tags);
    free(positions);
    free(text_tags);
    free(prompt);
}

static void test_patchify(const h3_st_header *header) {
    const uint64_t latent_shape[] = {
        1, VIDEO_CHANNELS, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH
    };
    const uint64_t row_shape[] = {VIDEO_ROWS, VIDEO_PATCH};
    size_t bytes = 0;
    float *latent = read_tensor(header, "input.video_latents", H3_DTYPE_F32,
                                5, latent_shape, &bytes);
    float *expected = read_tensor(header, "input.video_rows", H3_DTYPE_F32,
                                  2, row_shape, &bytes);
    size_t elements = (size_t)VIDEO_ROWS * VIDEO_PATCH;
    float *actual = malloc(elements * sizeof(*actual));
    if (!actual || !h3_dit_patchify_video(
            latent, VIDEO_CHANNELS, LATENT_FRAMES, LATENT_HEIGHT,
            LATENT_WIDTH, actual, elements))
        die("native patchify");
    if (memcmp(actual, expected, elements * sizeof(*actual))) {
        for (size_t index = 0; index < elements; index++)
            if (memcmp(actual + index, expected + index, sizeof(*actual))) {
                fprintf(stderr, "patchify mismatch[%zu]: %.9g != %.9g\n",
                        index, actual[index], expected[index]);
                exit(1);
            }
        die("patchify mismatch");
    }
    free(actual);
    free(expected);
    free(latent);
}

static void test_schedule(const h3_st_header *header) {
    const uint64_t sigma_shape[] = {NFE + 1};
    const uint64_t step_shape[] = {NFE};
    size_t bytes = 0;
    float *video_sigmas = read_tensor(
        header, "schedule.video_sigmas", H3_DTYPE_F32, 1,
        sigma_shape, &bytes);
    float *audio_sigmas = read_tensor(
        header, "schedule.audio_sigmas", H3_DTYPE_F32, 1,
        sigma_shape, &bytes);
    float *video_timesteps = read_tensor(
        header, "schedule.video_timesteps", H3_DTYPE_F32, 1,
        step_shape, &bytes);
    float *audio_timesteps = read_tensor(
        header, "schedule.audio_timesteps", H3_DTYPE_F32, 1,
        step_shape, &bytes);
    float *video_scales = read_tensor(
        header, "schedule.video_euler_scales", H3_DTYPE_F32, 1,
        step_shape, &bytes);
    float *audio_scales = read_tensor(
        header, "schedule.audio_euler_scales", H3_DTYPE_F32, 1,
        step_shape, &bytes);

    h3_sigma_schedule schedule;
    if (!h3_serving_schedule_build(NFE, &schedule)) die("native schedule");
    for (size_t step = 0; step <= NFE; step++) {
        require_close(schedule.video[step], video_sigmas[step], 1e-7,
                      "video sigma", step);
        require_close(schedule.audio[step], audio_sigmas[step], 1e-7,
                      "audio sigma", step);
    }
    for (size_t step = 0; step < NFE; step++) {
        float video_t = 1.0f - schedule.video[step];
        float audio_t = 1.0f - schedule.audio[step];
        float video_scale = (1.0f - schedule.video[step + 1] /
                             schedule.video[step]) * (1.0f - video_t);
        float audio_scale = (1.0f - schedule.audio[step + 1] /
                             schedule.audio[step]) * (1.0f - audio_t);
        require_close(video_t, video_timesteps[step], 1e-7,
                      "video timestep", step);
        require_close(audio_t, audio_timesteps[step], 1e-7,
                      "audio timestep", step);
        require_close(video_scale, video_scales[step], 1e-7,
                      "video Euler scale", step);
        require_close(audio_scale, audio_scales[step], 1e-7,
                      "audio Euler scale", step);

        char unique_name[96], indices_name[96];
        snprintf(unique_name, sizeof(unique_name),
                 "schedule.nfe_%zu.unique_timesteps", step);
        snprintf(indices_name, sizeof(indices_name),
                 "schedule.nfe_%zu.timestep_indices", step);
        uint64_t unique_shape[] = {step ? 2 : 1};
        const uint64_t indices_shape[] = {SEQUENCE};
        float *unique = read_tensor(header, unique_name, H3_DTYPE_F32, 1,
                                    unique_shape, &bytes);
        int64_t *indices = read_tensor(header, indices_name, H3_DTYPE_I64, 1,
                                       indices_shape, &bytes);
        require_close(unique[0], video_t, 0.0, "unique video timestep", step);
        if (step) require_close(unique[1], audio_t, 0.0,
                                "unique audio timestep", step);
        for (size_t row = 0; row < SEQUENCE; row++) {
            int64_t expected = step && row >= TEXT_ROWS &&
                row < TEXT_ROWS + AUDIO_ROWS ? 1 : 0;
            if (indices[row] != expected) die("timestep row map");
        }
        free(indices);
        free(unique);
    }
    free(audio_scales);
    free(video_scales);
    free(audio_timesteps);
    free(video_timesteps);
    free(audio_sigmas);
    free(video_sigmas);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s ORACLE.safetensors\n", argv[0]);
        return 2;
    }
    h3_st_header header;
    char error[512] = {0};
    if (!h3_st_read_header(argv[1], &header, error, sizeof(error))) {
        fprintf(stderr, "cannot read oracle: %s\n", error);
        return 1;
    }
    test_prompt_and_layout(&header);
    test_patchify(&header);
    test_schedule(&header);
    h3_st_free_header(&header);
    puts("PASS: native prompt/layout/patchify/scheduler match OpenVDN upstream oracle");
    return 0;
}
