#include "h3_vdn_pipeline.h"

#include "h3_internal.h"

#ifdef H3_BACKEND_HIP

#include "h3_audio_vae.h"
#include "h3_dit.h"
#include "h3_ffmpeg.h"
#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_vdn_dit.h"
#include "h3_vdn_prompt.h"
#include "h3_vdn_weights.h"
#include "h3_video_vae.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    H3_VDN_VIDEO_CHANNELS = 24,
    H3_VDN_AUDIO_CHANNELS = 32,
    H3_VDN_VIDEO_PATCH = 96,
    H3_VDN_AUDIO_WIDTH = 32
};

typedef struct {
    h3_ctx *ctx;
    const h3_params *params;
    int cancelled;
} h3_vdn_progress;

static double elapsed(const struct timespec *start,
                      const struct timespec *stop) {
    return (double)(stop->tv_sec - start->tv_sec) +
           (double)(stop->tv_nsec - start->tv_nsec) / 1.0e9;
}

static void emit(h3_vdn_progress *state, const char *phase,
                 int completed, int total) {
    if (!state || state->cancelled || !state->params->on_progress) return;
    if (state->params->on_progress(phase, completed, total,
                                   state->params->callback_opaque)) {
        state->cancelled = 1;
        h3_set_error(state->ctx, "generation cancelled during %s", phase);
    }
}

static void layer_progress(unsigned completed, unsigned total, void *opaque) {
    emit(opaque, "VDN block stream", (int)completed, (int)total);
}

static void nfe_progress(unsigned completed, unsigned total, void *opaque) {
    emit(opaque, "VDN denoise", (int)completed, (int)total);
}

static void video_progress(int completed, int total, void *opaque) {
    emit(opaque, "video VAE", completed, total);
}

static void audio_progress(int completed, int total, void *opaque) {
    emit(opaque, "audio VAE", completed, total);
}

static char *joined_path(const char *root, const char *suffix) {
    size_t a = strlen(root), b = strlen(suffix);
    if (a > SIZE_MAX - b - 1) return NULL;
    char *result = malloc(a + b + 1);
    if (result) snprintf(result, a + b + 1, "%s%s", root, suffix);
    return result;
}

static int json_string(FILE *file, const char *value) {
    if (fputc('"', file) == EOF) return 0;
    for (const unsigned char *cursor = (const unsigned char *)value;
         *cursor; cursor++) {
        switch (*cursor) {
        case '"': if (fputs("\\\"", file) == EOF) return 0; break;
        case '\\': if (fputs("\\\\", file) == EOF) return 0; break;
        case '\b': if (fputs("\\b", file) == EOF) return 0; break;
        case '\f': if (fputs("\\f", file) == EOF) return 0; break;
        case '\n': if (fputs("\\n", file) == EOF) return 0; break;
        case '\r': if (fputs("\\r", file) == EOF) return 0; break;
        case '\t': if (fputs("\\t", file) == EOF) return 0; break;
        default:
            if (*cursor < 0x20) {
                if (fprintf(file, "\\u%04x", *cursor) < 0) return 0;
            } else if (fputc(*cursor, file) == EOF) return 0;
        }
    }
    return fputc('"', file) != EOF;
}

static int write_record(h3_ctx *ctx, const h3_params *params,
                        const h3_temporal_shape *temporal,
                        int render_width, int render_height,
                        const h3_gpu_stats *dit,
                        const h3_video_frames *frames,
                        const h3_audio_waveform *waveform,
                        double dit_seconds, double video_seconds,
                        double audio_seconds, double mux_seconds,
                        double total_seconds,
                        char *error, size_t error_size) {
    if (!params->output_path || !*params->output_path) return 1;
    char *path = joined_path(params->output_path, ".inference.json");
    if (!path) {
        snprintf(error, error_size, "out of memory resolving inference record");
        return 0;
    }
    FILE *file = fopen(path, "wb");
    if (!file) {
        snprintf(error, error_size, "cannot create %s: %s", path,
                 strerror(errno));
        free(path);
        return 0;
    }
    uint64_t peak = dit->peak_live_bytes;
    if (frames->gpu_stats.peak_live_bytes > peak)
        peak = frames->gpu_stats.peak_live_bytes;
    if (waveform->gpu_stats.peak_live_bytes > peak)
        peak = waveform->gpu_stats.peak_live_bytes;
    int ok = fprintf(file,
        "{\n  \"engine_version\": \"%s\",\n  \"model_revision\": ",
        H3_VERSION) >= 0 &&
        json_string(file, ctx->model.vdn.model_revision) &&
        fputs(",\n  \"base_model\": ", file) != EOF &&
        json_string(file, ctx->model_dir) &&
        fputs(",\n  \"checkpoint\": ", file) != EOF &&
        json_string(file, ctx->vdn_checkpoint_dir) &&
        fputs(",\n  \"prompt_embeddings\": ", file) != EOF &&
        json_string(file, params->prompt_embeddings) &&
        fprintf(file,
        ",\n  \"backend\": \"%s\",\n  \"device\": ", ctx->device.backend) >= 0 &&
        json_string(file, ctx->device.name) &&
        fprintf(file,
        ",\n  \"device_index\": %d,\n"
        "  \"dtype\": \"BF16\",\n"
        "  \"seed\": %" PRIu64 ",\n"
        "  \"nfe\": %d,\n"
        "  \"video_shift\": %.9g,\n"
        "  \"audio_shift\": %.9g,\n"
        "  \"requested_shape\": {\"frames\": %d, \"width\": %d, \"height\": %d},\n"
        "  \"render_shape\": {\"frames\": %d, \"width\": %d, \"height\": %d},\n"
        "  \"latent_shape\": {\"video_t\": %d, \"audio_t\": %d, \"width\": %d, \"height\": %d},\n"
        "  \"output\": {\"frames\": %d, \"fps\": %d, \"sample_rate\": %d, \"channels\": %d},\n"
        "  \"peak_gpu_bytes\": %" PRIu64 ",\n"
        "  \"timing_seconds\": {\"dit\": %.6f, \"video_vae\": %.6f, \"audio_vae\": %.6f, \"mux\": %.6f, \"total\": %.6f}\n"
        "}\n",
        ctx->device.device_index, params->seed, params->steps,
        ctx->model.vdn.video_shift, ctx->model.vdn.audio_shift,
        params->frames, params->width, params->height,
        temporal->frame_count, render_width, render_height,
        temporal->video_t, temporal->audio_t,
        render_width / H3_VAE_SPATIAL_RATIO,
        render_height / H3_VAE_SPATIAL_RATIO,
        frames->frames, H3_FPS, waveform->sample_rate, waveform->channels,
        peak, dit_seconds, video_seconds, audio_seconds, mux_seconds,
        total_seconds) >= 0;
    if (fclose(file) != 0) ok = 0;
    if (!ok) snprintf(error, error_size, "cannot write inference record %s",
                      path);
    free(path);
    return ok;
}

h3_result *h3_vdn_generate_embedded(h3_ctx *ctx, const h3_params *params) {
    char detail[512] = {0};
    h3_result *result = NULL;
    h3_gpu *gpu = NULL;
    h3_vdn_weight_store *store = NULL;
    h3_vdn_model_weights model;
    h3_text_embedding prompt;
    h3_vdn_layout layout;
    h3_gpu_tensor *refined = NULL, *video = NULL, *audio = NULL;
    float *video_rows = NULL, *audio_rows = NULL;
    float *video_latent = NULL, *audio_latent = NULL;
    h3_video_frames frames;
    h3_audio_waveform waveform;
    uint8_t *rgb = NULL;
    char *video_path = NULL, *audio_path = NULL;
    h3_gpu_stats dit_stats;
    memset(&model, 0, sizeof(model));
    memset(&prompt, 0, sizeof(prompt));
    memset(&layout, 0, sizeof(layout));
    memset(&frames, 0, sizeof(frames));
    memset(&waveform, 0, sizeof(waveform));
    memset(&dit_stats, 0, sizeof(dit_stats));
    struct timespec total_start, dit_start, dit_stop, video_stop, audio_stop,
                    mux_stop;
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    if (!params->prompt_embeddings || !*params->prompt_embeddings) {
        h3_set_error(ctx, "VDN generation requires prompt_embeddings");
        return NULL;
    }
    if (params->reference_count || params->first_frame || params->last_frame) {
        h3_set_error(ctx,
            "OpenVDN does not define first/last-frame or ordered media "
            "conditioning; use MiniMax-H3 FL2VA/Ref2VA without "
            "--vdn-checkpoint");
        return NULL;
    }
    if (params->preview_denoise) {
        h3_set_error(ctx, "VDN denoising preview is not implemented");
        return NULL;
    }
    if (params->dit_layers != 50 || params->denoise_reuse != 1 ||
        params->core_reuse != 1 || params->token_reduction ||
        params->use_int8_row_fc2) {
        h3_set_error(ctx,
            "VDN correctness path requires layers=50, reuse=1, core_reuse=1, "
            "token_reduction=0, and int8_row_fc2=0");
        return NULL;
    }
    if (params->steps != ctx->model.vdn.num_steps) {
        h3_set_error(ctx, "VDN checkpoint requires exactly %d NFE, got %d",
                     ctx->model.vdn.num_steps, params->steps);
        return NULL;
    }
    h3_temporal_shape temporal = h3_temporal(params->frames);
    if (temporal.frame_count < 22) {
        h3_set_error(ctx,
            "VDN generation requires at least one trained 22-frame decoder chunk");
        return NULL;
    }
    int render_width = params->render_width ? params->render_width : params->width;
    int render_height = params->render_height ? params->render_height : params->height;
    int latent_w, latent_h;
    h3_latent_canvas(render_width, render_height, &latent_w, &latent_h);
    h3_vdn_progress progress = {ctx, params, 0};

    clock_gettime(CLOCK_MONOTONIC, &dit_start);
    gpu = h3_gpu_create(NULL, detail, sizeof(detail));
    if (gpu) h3_gpu_profile_set_label(gpu, "OpenVDN DiT");
    store = h3_vdn_weight_store_open(
        ctx->model_dir, ctx->vdn_checkpoint_dir,
        ctx->model.vdn.adapter_count > 1, detail, sizeof(detail));
    if (!gpu || !store ||
        !h3_vdn_prompt_load(params->prompt_embeddings, &prompt,
                            detail, sizeof(detail)) ||
        !h3_vdn_model_weights_load(store, gpu, &model,
                                   detail, sizeof(detail))) goto failed;
    refined = h3_vdn_refine_prompt(gpu, &model, &prompt,
                                   detail, sizeof(detail));
    if (!refined || !h3_vdn_layout_build(
            &prompt, (uint32_t)temporal.video_t, (uint32_t)latent_h,
            (uint32_t)latent_w, (uint32_t)temporal.audio_t, &layout,
            detail, sizeof(detail))) goto failed;
    size_t video_elements = (size_t)layout.video_rows * H3_VDN_VIDEO_PATCH;
    size_t audio_elements = (size_t)layout.audio_rows * H3_VDN_AUDIO_WIDTH;
    video_rows = malloc(video_elements * sizeof(*video_rows));
    audio_rows = malloc(audio_elements * sizeof(*audio_rows));
    if (!video_rows || !audio_rows) {
        snprintf(detail, sizeof(detail), "out of memory allocating VDN latents");
        goto failed;
    }
    h3_rng rng;
    h3_rng_seed(&rng, params->seed);
    h3_rng_fill_normal(&rng, video_rows, video_elements);
    h3_rng_fill_normal(&rng, audio_rows, audio_elements);
    video = h3_gpu_tensor_from_f32(gpu, video_rows, video_elements);
    audio = h3_gpu_tensor_from_f32(gpu, audio_rows, audio_elements);
    if (video && audio) h3_gpu_profile_mark(gpu, "setup");
    if (!video || !audio || !h3_vdn_denoise(
            gpu, store, &model, refined, &layout, video, audio,
            (unsigned)params->steps, 1, 5, layer_progress, nfe_progress,
            &progress, detail, sizeof(detail)) || progress.cancelled ||
        !h3_gpu_tensor_read_f32(video, video_rows, video_elements) ||
        !h3_gpu_tensor_read_f32(audio, audio_rows, audio_elements) ||
        !h3_gpu_get_stats(gpu, &dit_stats)) {
        if (!detail[0] && !progress.cancelled)
            snprintf(detail, sizeof(detail), "cannot read denoised VDN latents: %s",
                     h3_gpu_error(gpu));
        goto failed;
    }
    size_t video_latent_elements = (size_t)H3_VDN_VIDEO_CHANNELS *
        (size_t)temporal.video_t * (size_t)latent_h * (size_t)latent_w;
    size_t audio_latent_elements = (size_t)H3_VDN_AUDIO_CHANNELS * 2 *
                                   (size_t)temporal.audio_t;
    video_latent = malloc(video_latent_elements * sizeof(*video_latent));
    audio_latent = malloc(audio_latent_elements * sizeof(*audio_latent));
    if (!video_latent || !audio_latent ||
        !h3_dit_unpatchify_video(
            video_rows, H3_VDN_VIDEO_CHANNELS, temporal.video_t,
            latent_h, latent_w, video_latent, video_latent_elements) ||
        !h3_dit_unpack_audio(
            audio_rows, H3_VDN_AUDIO_CHANNELS, temporal.audio_t,
            audio_latent, audio_latent_elements)) {
        snprintf(detail, sizeof(detail), "cannot unpack denoised VDN latents");
        goto failed;
    }
    clock_gettime(CLOCK_MONOTONIC, &dit_stop);

    h3_gpu_tensor_free(audio); audio = NULL;
    h3_gpu_tensor_free(video); video = NULL;
    h3_gpu_tensor_free(refined); refined = NULL;
    h3_vdn_layout_free(&layout);
    h3_vdn_model_weights_free(&model);
    h3_vdn_prompt_free(&prompt);
    h3_vdn_weight_store_free(store); store = NULL;
    h3_gpu_free(gpu); gpu = NULL;

    video_path = joined_path(ctx->model_dir, "/vae");
    audio_path = joined_path(ctx->model_dir, "/audio_vae");
    if (!video_path || !audio_path) {
        snprintf(detail, sizeof(detail), "out of memory resolving VDN VAE paths");
        goto failed;
    }
    if (!h3_video_vae_decode(
            video_path, "unused-on-hip", video_latent, temporal.video_t,
            latent_h, latent_w, video_progress, &progress, &frames,
            detail, sizeof(detail)) || progress.cancelled) goto failed;
    clock_gettime(CLOCK_MONOTONIC, &video_stop);
    if (!h3_audio_vae_decode(
            audio_path, "unused-on-hip", audio_latent, temporal.audio_t,
            audio_progress, &progress, &waveform,
            detail, sizeof(detail)) || progress.cancelled) goto failed;
    clock_gettime(CLOCK_MONOTONIC, &audio_stop);

    size_t pixels = (size_t)frames.frames * (size_t)frames.height *
                    (size_t)frames.width;
    rgb = malloc(pixels * 3);
    if (!rgb) {
        snprintf(detail, sizeof(detail), "out of memory converting VDN RGB");
        goto failed;
    }
    for (size_t index = 0; index < pixels * 3; index++) {
        float value = frames.rgb[index];
        if (!isfinite(value)) {
            snprintf(detail, sizeof(detail), "non-finite decoded VDN pixel");
            goto failed;
        }
        value = fminf(1.0f, fmaxf(0.0f, value));
        rgb[index] = (uint8_t)lrintf(value * 255.0f);
    }
    int output_width = frames.width, output_height = frames.height;
    if (output_width != params->width || output_height != params->height) {
        uint8_t *resized = NULL;
        if (!h3_resize_rgb24_high_quality(
                rgb, frames.frames, output_width, output_height,
                params->width, params->height, &resized)) {
            snprintf(detail, sizeof(detail), "cannot resize decoded VDN frames");
            goto failed;
        }
        free(rgb);
        rgb = resized;
        output_width = params->width;
        output_height = params->height;
    }
    if (params->on_frame) {
        size_t frame_bytes = (size_t)output_width * (size_t)output_height * 3;
        for (int index = 0; index < frames.frames; index++) {
            h3_frame frame = {output_width, output_height, output_width * 3,
                              rgb + (size_t)index * frame_bytes,
                              index, frames.frames, -1, 0};
            if (params->on_frame(&frame, params->callback_opaque)) {
                snprintf(detail, sizeof(detail),
                         "generation cancelled while delivering frame %d", index);
                goto failed;
            }
        }
    }
    if (params->output_path && *params->output_path) {
        emit(&progress, "FFmpeg", 0, frames.frames);
        if (!h3_ffmpeg_write_av_rgb24_f32(
                params->output_path, rgb, frames.frames,
                output_width, output_height, H3_FPS,
                waveform.pcm, waveform.samples, waveform.channels,
                waveform.sample_rate, detail, sizeof(detail))) goto failed;
        emit(&progress, "FFmpeg", frames.frames, frames.frames);
    }
    clock_gettime(CLOCK_MONOTONIC, &mux_stop);

    result = calloc(1, sizeof(*result));
    if (!result) {
        snprintf(detail, sizeof(detail), "out of memory creating VDN result");
        goto failed;
    }
    result->width = output_width;
    result->height = output_height;
    result->frames = frames.frames;
    result->fps = H3_FPS;
    result->sample_rate = waveform.sample_rate;
    result->seed = params->seed;
    if (!write_record(
            ctx, params, &temporal, render_width, render_height, &dit_stats,
            &frames, &waveform, elapsed(&dit_start, &dit_stop),
            elapsed(&dit_stop, &video_stop), elapsed(&video_stop, &audio_stop),
            elapsed(&audio_stop, &mux_stop), elapsed(&total_start, &mux_stop),
            detail, sizeof(detail))) {
        free(result);
        result = NULL;
        goto failed;
    }
    goto cleanup;

failed:
    if (!ctx->error[0])
        h3_set_error(ctx, "%s", detail[0] ? detail : "VDN generation failed");
cleanup:
    free(video_path); free(audio_path);
    free(rgb);
    h3_audio_waveform_free(&waveform);
    h3_video_frames_free(&frames);
    free(audio_latent); free(video_latent);
    free(audio_rows); free(video_rows);
    h3_gpu_tensor_free(audio); h3_gpu_tensor_free(video);
    h3_gpu_tensor_free(refined);
    h3_vdn_layout_free(&layout);
    h3_vdn_model_weights_free(&model);
    h3_vdn_prompt_free(&prompt);
    h3_vdn_weight_store_free(store);
    h3_gpu_free(gpu);
    return result;
}

#else

h3_result *h3_vdn_generate_embedded(h3_ctx *ctx, const h3_params *params) {
    (void)params;
    h3_set_error(ctx, "OpenVDN execution currently requires BACKEND=hip");
    return NULL;
}

#endif
