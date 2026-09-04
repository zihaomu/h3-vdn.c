#include "h3_video_vae.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    CHANNELS = 24,
    LATENT_T = 17,
    DEFAULT_LATENT_H = 2,
    DEFAULT_LATENT_W = 4
};

static double monotonic_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec / 1.0e9;
}

static int env_dimension(const char *name, int fallback, int *value) {
    const char *text = getenv(name);
    if (!text || !*text) {
        *value = fallback;
        return 1;
    }
    errno = 0;
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (errno || end == text || *end || parsed < 1 || parsed > INT_MAX)
        return 0;
    *value = (int)parsed;
    return 1;
}

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
    int latent_h, latent_w;
    if (!env_dimension("VDN_VAE_LATENT_H", DEFAULT_LATENT_H, &latent_h) ||
        !env_dimension("VDN_VAE_LATENT_W", DEFAULT_LATENT_W, &latent_w)) {
        fprintf(stderr, "invalid VDN video VAE latent geometry\n");
        return 2;
    }
    size_t latent_elements =
        (size_t)CHANNELS * LATENT_T * (size_t)latent_h * (size_t)latent_w;
    float *latent = malloc(latent_elements * sizeof(*latent));
    if (!latent) return 1;
    for (size_t index = 0; index < latent_elements; index++)
        latent[index] = sinf((float)index * 0.013f) * 0.7f;
    h3_video_frames frames;
    char error[512] = {0};
    double start = monotonic_seconds();
    int ok = h3_video_vae_decode(
        argv[1], "unused-on-hip", latent, LATENT_T, latent_h, latent_w,
        progress, NULL, &frames, error, sizeof(error));
    double seconds = monotonic_seconds() - start;
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
    if (frames.frames != 56 || frames.height != latent_h * 16 ||
        frames.width != latent_w * 16 ||
        invalid || nonzero < elements / 2) {
        fprintf(stderr, "VDN video VAE output invalid: [%d,%d,%d] "
                        "invalid=%zu nonzero=%zu\n",
                frames.frames, frames.height, frames.width, invalid, nonzero);
        h3_video_frames_free(&frames);
        return 1;
    }
    printf("VDN released video VAE passed: RGB F32[%d,%d,%d,3], "
           "wall=%.6fs, hash=%016llx, peak=%.3f GiB\n",
           frames.frames, frames.height, frames.width,
           seconds,
           (unsigned long long)hash_f32(frames.rgb, elements),
           (double)frames.gpu_stats.peak_live_bytes /
               (1024.0 * 1024.0 * 1024.0));
    h3_video_frames_free(&frames);
    return 0;
}
