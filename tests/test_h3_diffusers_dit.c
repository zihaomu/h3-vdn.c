#include "h3_dit.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_h3_diffusers_dit.c: %s\n", message);
    exit(1);
}

static void *load(const h3_st_header *fixture, const char *name,
                  h3_dtype dtype, size_t *elements) {
    const h3_st_tensor *tensor = h3_st_find(fixture, name);
    if (!tensor || tensor->dtype != dtype) die(name);
    uint64_t count = h3_st_tensor_elements(tensor);
    if (!count || count > SIZE_MAX / h3_dtype_size(dtype))
        die("invalid fixture tensor size");
    void *result = malloc((size_t)count * h3_dtype_size(dtype));
    if (!result) die("out of memory reading fixture");
    char error[512];
    if (!h3_st_read_data(fixture, tensor, result,
                         (size_t)count * h3_dtype_size(dtype),
                         error, sizeof(error))) die(error);
    *elements = (size_t)count;
    return result;
}

static double compare(const char *label, const float *got, const float *want,
                      size_t count, double relative_l2_limit) {
    double maximum = 0.0, scale = 0.0, square_error = 0.0, square_value = 0.0;
    for (size_t index = 0; index < count; index++) {
        if (!isfinite(got[index]) || !isfinite(want[index]))
            die("non-finite parity tensor");
        double delta = (double)got[index] - want[index];
        maximum = fmax(maximum, fabs(delta));
        scale = fmax(scale, fabs((double)want[index]));
        square_error += delta * delta;
        square_value += (double)want[index] * want[index];
    }
    double relative_max = maximum / fmax(scale, 1e-12);
    double relative_l2 = sqrt(square_error / fmax(square_value, 1e-24));
    printf("%-24s rel-max %.8g abs %.8g rel-L2 %.8g\n",
           label, relative_max, maximum, relative_l2);
    if (relative_l2 >= relative_l2_limit)
        die("Diffusers parity relative-L2 gate failed");
    return relative_l2;
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static double compare_bf16(const char *label, const uint16_t *got,
                           const uint16_t *want, size_t count,
                           double relative_l2_limit) {
    double square_error = 0.0, square_value = 0.0, maximum = 0.0;
    for (size_t index = 0; index < count; index++) {
        double expected = bf16_to_f32(want[index]);
        double delta = (double)bf16_to_f32(got[index]) - expected;
        maximum = fmax(maximum, fabs(delta));
        square_error += delta * delta;
        square_value += expected * expected;
    }
    double relative_l2 = sqrt(square_error / fmax(square_value, 1e-24));
    printf("%-24s abs %.8g rel-L2 %.8g\n", label, maximum, relative_l2);
    if (relative_l2 >= relative_l2_limit)
        die("Diffusers BF16 parity relative-L2 gate failed");
    return relative_l2;
}

static void progress(const char *phase, int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 0 || completed == total || completed % 10 == 0)
        fprintf(stderr, "native Diffusers parity %-22s %d/%d\n",
                phase, completed, total);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s TRANSFORMER_DIR ORACLE STEPS\n", argv[0]);
        return 2;
    }
    int steps = atoi(argv[3]);
    if (steps < 2) die("STEPS must be at least two");
    setenv("H3_DISABLE_FUSED_MLP", "1", 1);
    setenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN", "1", 1);
    setenv("H3_DISABLE_FUSED_GATE_ADALN", "1", 1);
    setenv("H3_DISABLE_FUSED_FINAL_SLICE", "1", 1);
    setenv("H3_DISABLE_FUSED_PATCH_CAST", "1", 1);
    setenv("H3_DISABLE_FUSED_PATCH_PACK", "1", 1);
    setenv("H3_DISABLE_COOP_QKV", "1", 1);
    setenv("H3_DIT_F32_FINAL", "1", 1);
    setenv("H3_DIT_CAPTURE_BLOCKS", "1", 1);

    char error[512];
    h3_st_header fixture;
    if (!h3_st_read_header(argv[2], &fixture, error, sizeof(error))) die(error);
    const h3_st_tensor *prompt_tensor = h3_st_find(&fixture, "input.prompt");
    const h3_st_tensor *video_tensor = h3_st_find(&fixture, "input.video");
    const h3_st_tensor *audio_tensor = h3_st_find(&fixture, "input.audio");
    if (!prompt_tensor || prompt_tensor->dtype != H3_DTYPE_BF16 ||
        prompt_tensor->ndim != 2 || prompt_tensor->shape[1] != 5120 ||
        !video_tensor || video_tensor->dtype != H3_DTYPE_F32 ||
        video_tensor->ndim != 4 || video_tensor->shape[0] != 24 ||
        !audio_tensor || audio_tensor->dtype != H3_DTYPE_F32 ||
        audio_tensor->ndim != 3 || audio_tensor->shape[0] != 32 ||
        audio_tensor->shape[1] != 2)
        die("oracle geometry is incompatible");

    size_t prompt_count, video_count, audio_count, count;
    uint16_t *prompt = load(&fixture, "input.prompt", H3_DTYPE_BF16,
                            &prompt_count);
    float *video_initial = load(&fixture, "input.video", H3_DTYPE_F32,
                                &video_count);
    float *audio_initial = load(&fixture, "input.audio", H3_DTYPE_F32,
                                &audio_count);
    float *video_velocity_want = load(
        &fixture, "first.video_velocity", H3_DTYPE_F32, &count);
    if (count != video_count) die("video velocity size mismatch");
    float *audio_velocity_want = load(
        &fixture, "first.audio_velocity", H3_DTYPE_F32, &count);
    if (count != audio_count) die("audio velocity size mismatch");
    float *video_final_want = load(
        &fixture, "final.video", H3_DTYPE_F32, &count);
    if (count != video_count) die("final video size mismatch");
    float *audio_final_want = load(
        &fixture, "final.audio", H3_DTYPE_F32, &count);
    if (count != audio_count) die("final audio size mismatch");

    size_t tokens = (size_t)prompt_tensor->shape[0];
    uint8_t *tags = malloc(tokens);
    float *video_velocity = malloc(video_count * sizeof(*video_velocity));
    float *audio_velocity = malloc(audio_count * sizeof(*audio_velocity));
    float *video = malloc(video_count * sizeof(*video));
    float *audio = malloc(audio_count * sizeof(*audio));
    if (!tags || !video_velocity || !audio_velocity || !video || !audio)
        die("out of memory allocating parity state");
    memset(tags, 1, tokens);
    memcpy(video, video_initial, video_count * sizeof(*video));
    memcpy(audio, audio_initial, audio_count * sizeof(*audio));
    h3_text_embedding text = {tokens, 5120, prompt, {0}, tags};
    int latent_t = (int)video_tensor->shape[1];
    int latent_h = (int)video_tensor->shape[2];
    int latent_w = (int)video_tensor->shape[3];
    int audio_t = (int)audio_tensor->shape[2];
    h3_layout_spec spec = {(int)tokens, latent_t, latent_h, latent_w,
                           audio_t, 124, NULL, 0, NULL, 0};
    h3_layout layout;
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) die(error);
    h3_sigma_schedule sigmas;
    if (!h3_serving_schedule_build(steps, &sigmas))
        die("cannot build serving sigma schedule");

    h3_dit *dit = h3_dit_load_t2va(
        argv[1], "h3_shaders.metal", &text, &layout, &sigmas,
        50, 1, 0, 1, 1.0f,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        progress, NULL, error, sizeof(error));
    if (!dit) die(error);
    if (h3_dit_video_elements(dit) != video_count ||
        h3_dit_audio_elements(dit) != audio_count)
        die("native DiT geometry differs from oracle");

    size_t refined_count;
    uint16_t *refined_want = load(&fixture, "first.refiner", H3_DTYPE_BF16,
                                  &refined_count);
    uint16_t *refined = malloc(refined_count * sizeof(*refined));
    if (!refined || refined_count != tokens * 5376 ||
        !h3_dit_read_refined_text(dit, refined, refined_count))
        die("cannot read native refined text");
    compare_bf16("token refiner", refined, refined_want, refined_count, 0.01);

    if (!h3_dit_forward(dit, 0, video_initial, audio_initial,
                        video_velocity, audio_velocity,
                        error, sizeof(error))) die(error);
    uint16_t *block = malloc((size_t)layout.seq_len * 5376 * sizeof(*block));
    if (!block) die("out of memory reading block captures");
    for (unsigned index = 0; index < 50; index++) {
        char name[64], label[64];
        snprintf(name, sizeof(name), "first.block_%02u", index);
        snprintf(label, sizeof(label), "transformer block %02u", index);
        size_t block_count;
        uint16_t *block_want = load(&fixture, name, H3_DTYPE_BF16,
                                    &block_count);
        if (block_count != (size_t)layout.seq_len * 5376 ||
            !h3_dit_read_captured_block(dit, index, block, block_count))
            die("cannot read native block capture");
        compare_bf16(label, block, block_want, block_count, 0.15);
        free(block_want);
    }
    compare("first video velocity", video_velocity, video_velocity_want,
            video_count, 0.35);
    compare("first audio velocity", audio_velocity, audio_velocity_want,
            audio_count, 0.10);
    if (!h3_dit_denoise_euler(dit, video, audio, 1, progress, NULL,
                              error, sizeof(error))) die(error);
    compare("final video latent", video, video_final_want, video_count, 0.40);
    compare("final audio latent", audio, audio_final_want, audio_count, 0.15);

    h3_dit_free(dit);
    h3_layout_free(&layout);
    h3_st_free_header(&fixture);
    free(prompt); free(tags); free(video_initial); free(audio_initial);
    free(video_velocity_want); free(audio_velocity_want);
    free(video_final_want); free(audio_final_want);
    free(refined_want); free(refined);
    free(block);
    free(video_velocity); free(audio_velocity); free(video); free(audio);
    puts("ok: native 50-layer DiT is within the official BF16 oracle tolerances");
    return 0;
}
