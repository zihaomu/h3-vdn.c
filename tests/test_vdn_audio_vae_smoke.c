#include "h3_audio_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CHANNELS = 32, STEREO = 2, LATENT_T = 3 };

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
    fprintf(stderr, "VDN audio VAE: stage %d/%d\n", completed, total);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s AUDIO_VAE_DIRECTORY\n", argv[0]);
        return 2;
    }
    float latent[CHANNELS * STEREO * LATENT_T];
    for (size_t index = 0; index < sizeof(latent) / sizeof(*latent); index++)
        latent[index] = cosf((float)index * 0.037f) * 0.55f;
    h3_audio_waveform waveform;
    char error[512] = {0};
    if (!h3_audio_vae_decode(
            argv[1], "unused-on-hip", latent, LATENT_T, progress, NULL,
            &waveform, error, sizeof(error))) {
        fprintf(stderr, "VDN audio VAE smoke failed: %s\n", error);
        return 1;
    }
    size_t elements = (size_t)waveform.channels * waveform.samples;
    size_t invalid = 0, nonzero = 0;
    for (size_t index = 0; index < elements; index++) {
        invalid += !isfinite(waveform.pcm[index]);
        invalid += waveform.pcm[index] < -1.0f || waveform.pcm[index] > 1.0f;
        nonzero += waveform.pcm[index] != 0.0f;
    }
    if (waveform.channels != STEREO || waveform.samples != LATENT_T * 800 ||
        waveform.sample_rate != 32000 || invalid || nonzero < elements / 2) {
        fprintf(stderr, "VDN audio VAE output invalid: [%d,%d] rate=%d "
                        "invalid=%zu nonzero=%zu\n",
                waveform.channels, waveform.samples, waveform.sample_rate,
                invalid, nonzero);
        h3_audio_waveform_free(&waveform);
        return 1;
    }
    printf("VDN released audio VAE passed: PCM F32[%d,%d], hash=%016llx, "
           "peak=%.3f GiB\n", waveform.channels, waveform.samples,
           (unsigned long long)hash_f32(waveform.pcm, elements),
           (double)waveform.gpu_stats.peak_live_bytes /
               (1024.0 * 1024.0 * 1024.0));
    h3_audio_waveform_free(&waveform);
    return 0;
}
