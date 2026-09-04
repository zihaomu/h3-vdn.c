#include "h3_video_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CHANNELS = 24, LATENT_T = 17, LATENT_H = 2, LATENT_W = 4 };

static uint64_t hash_f32(const float *values, size_t count) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < count; index++) {
        uint32_t bits;
        memcpy(&bits, &values[index], sizeof(bits));
        for (unsigned byte = 0; byte < 4; byte++) {
            hash ^= (bits >> (byte * 8)) & UINT32_C(0xff);
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 1 || completed == total || completed % 6 == 0)
        fprintf(stderr, "VDN video VAE weights: block %d/%d\n",
                completed, total);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s VAE_DIRECTORY\n", argv[0]);
        return 2;
    }
    size_t latent_elements =
        (size_t)CHANNELS * LATENT_T * LATENT_H * LATENT_W;
    float *latent = malloc(latent_elements * sizeof(*latent));
    if (!latent) return 1;
    for (size_t index = 0; index < latent_elements; index++)
        latent[index] = sinf((float)index * 0.013f) * 0.7f;
    h3_video_frames frames;
    char error[512] = {0};
    int ok = h3_video_vae_decode(
        argv[1], "unused-on-hip", latent, LATENT_T, LATENT_H, LATENT_W,
        progress, NULL, &frames, error, sizeof(error));
    free(latent);
    if (!ok) {
        fprintf(stderr, "VDN video VAE smoke failed: %s\n", error);
        return 1;
    }
    size_t elements = (size_t)frames.frames * frames.height * frames.width * 3;
    size_t invalid = 0, nonzero = 0;
    for (size_t index = 0; index < elements; index++) {
        invalid += !isfinite(frames.rgb[index]);
        nonzero += frames.rgb[index] != 0.0f;
        if (frames.rgb[index] < 0.0f || frames.rgb[index] > 1.0f)
            invalid++;
    }
    if (frames.frames != 56 || frames.height != 32 || frames.width != 64 ||
        invalid || nonzero < elements / 2) {
        fprintf(stderr, "VDN video VAE output invalid: [%d,%d,%d] "
                        "invalid=%zu nonzero=%zu\n",
                frames.frames, frames.height, frames.width, invalid, nonzero);
        h3_video_frames_free(&frames);
        return 1;
    }
    printf("VDN released video VAE passed: RGB F32[%d,%d,%d,3], "
           "hash=%016llx, peak=%.3f GiB\n",
           frames.frames, frames.height, frames.width,
           (unsigned long long)hash_f32(frames.rgb, elements),
           (double)frames.gpu_stats.peak_live_bytes /
               (1024.0 * 1024.0 * 1024.0));
    h3_video_frames_free(&frames);
    return 0;
}
