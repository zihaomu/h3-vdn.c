#include "h3_safetensors.h"
#include "h3_video_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_video_vae.c: %s\n", message);
    exit(1);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 1 || completed == total || completed % 5 == 0)
        fprintf(stderr, "native video VAE load: %d/%d blocks\n",
                completed, total);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    const char *latent_path = argc > 2 ? argv[2] :
        "misc/fixtures/h3_real_dit_denoise20_bf16.safetensors";
    const char *frame_path = argc > 3 ? argv[3] :
        "misc/fixtures/h3_real_video_vae5_f32.safetensors";
    char error[512];
    h3_st_header latents, frames;
    if (!h3_st_read_header(latent_path, &latents, error, sizeof(error))) die(error);
    if (!h3_st_read_header(frame_path, &frames, error, sizeof(error))) die(error);
    enum { LATENT_COUNT = 24 * 2 * 2 * 2, FRAME_COUNT = 3 * 5 * 32 * 32 };
    const h3_st_tensor *latent_tensor = h3_st_find(
        &latents, "video.decode.normalized_latent");
    const h3_st_tensor *frame_tensor = h3_st_find(
        &frames, "video.decode.rgb_f32");
    int canonical_frames = frame_tensor != NULL;
    if (!latent_tensor)
        latent_tensor = h3_st_find(&latents, "x.video_after_20");
    if (!frame_tensor) frame_tensor = h3_st_find(&frames, "x.frames");
    if (!latent_tensor ||
        (latent_tensor->dtype != H3_DTYPE_BF16 &&
         latent_tensor->dtype != H3_DTYPE_F32) ||
        h3_st_tensor_elements(latent_tensor) != LATENT_COUNT ||
        !frame_tensor || frame_tensor->dtype != H3_DTYPE_F32 ||
        h3_st_tensor_elements(frame_tensor) != FRAME_COUNT)
        die("malformed visual decoder fixture");
    int latent_is_f32 = latent_tensor->dtype == H3_DTYPE_F32;
    uint16_t latent_bf[LATENT_COUNT];
    float latent[LATENT_COUNT];
    float *want = malloc(FRAME_COUNT * sizeof(*want));
    if (!want) die("out of memory loading expected frames");
    int latent_ok = latent_is_f32 ?
        h3_st_read_data(&latents, latent_tensor, latent, sizeof(latent),
                        error, sizeof(error)) :
        h3_st_read_data(&latents, latent_tensor, latent_bf, sizeof(latent_bf),
                        error, sizeof(error));
    if (!latent_ok || !h3_st_read_data(&frames, frame_tensor, want,
                         FRAME_COUNT * sizeof(*want), error, sizeof(error)))
        die(error);
    if (!latent_is_f32)
        for (size_t index = 0; index < LATENT_COUNT; index++)
            latent[index] = bf16_to_f32(latent_bf[index]);
    char weights[1024], modern_index[1024];
    snprintf(modern_index, sizeof(modern_index),
             "%s/diffusion_pytorch_model.safetensors.index.json", model_root);
    FILE *modern = fopen(modern_index, "rb");
    if (modern) {
        fclose(modern);
        snprintf(weights, sizeof(weights), "%s", model_root);
    } else {
        snprintf(weights, sizeof(weights), "%s/FL2VA/video_vae/source",
                 model_root);
    }
    h3_video_frames got;
    if (!h3_video_vae_decode(weights, "h3_shaders.metal", latent, 2, 2, 2,
                             progress, NULL, &got, error, sizeof(error)))
        die(error);
    if (got.frames != 5 || got.height != 32 || got.width != 32)
        die("native visual decoder returned the wrong shape");
    double maximum = 0.0, scale = 0.0, square_error = 0.0, square_value = 0.0;
    for (int frame = 0; frame < 5; frame++)
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 32; x++)
                for (int channel = 0; channel < 3; channel++) {
                    size_t native = (((size_t)frame * 32 + (size_t)y) * 32 +
                                     (size_t)x) * 3 + (size_t)channel;
                    size_t reference = canonical_frames ? native :
                        ((((size_t)channel * 5 + (size_t)frame) * 32 +
                          (size_t)y) * 32 + (size_t)x);
                    double delta = (double)got.rgb[native] - want[reference];
                    if (fabs(delta) > maximum) maximum = fabs(delta);
                    if (fabs(want[reference]) > scale)
                        scale = fabs(want[reference]);
                    square_error += delta * delta;
                    square_value += (double)want[reference] * want[reference];
                }
    double relative = maximum / (scale > 1e-12 ? scale : 1e-12);
    double l2 = sqrt(square_error / (square_value > 1e-24 ? square_value : 1e-24));
    printf("visual decoder frames: rel-max %.6g abs %.6g rel-L2 %.6g\n",
           relative, maximum, l2);
    printf("visual decoder: %.3f GiB allocated, %.3f GPU seconds, "
           "%llu linears, %llu SDPA, %llu submission\n",
           (double)got.gpu_stats.allocated_bytes / (1024.0 * 1024.0 * 1024.0),
           got.gpu_stats.gpu_seconds,
           (unsigned long long)got.gpu_stats.mps_linear_dispatches,
           (unsigned long long)got.gpu_stats.mps_sdpa_dispatches,
           (unsigned long long)got.gpu_stats.submissions);
    if (relative >= 0.05 || l2 >= 0.05)
        die("native visual decoder exceeds MLX parity bound");
    /* HIP records the post-quant and latent embedding linears explicitly;
       the Metal backend's historical counter folds those two operations. */
#ifdef H3_BACKEND_HIP
    const uint64_t expected_linears = 147;
#else
    const uint64_t expected_linears = 145;
#endif
    /* Modern Diffusers QKV and FFN tensors require one load-time reorder
       submission apiece per block; the legacy fused checkpoint does not. */
    if ((got.gpu_stats.submissions != 38 &&
         got.gpu_stats.submissions != 110) ||
        got.gpu_stats.mps_linear_dispatches != expected_linears ||
        got.gpu_stats.mps_sdpa_dispatches != 36)
        die("visual decoder did not batch/cache the expected hot path");
    h3_video_frames_free(&got);
    free(want);
    h3_st_free_header(&latents);
    h3_st_free_header(&frames);
    puts("ok: native visual decoder matches five upstream RGB frames");
    return 0;
}
