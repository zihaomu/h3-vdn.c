#ifndef H3_VDN_DIT_H
#define H3_VDN_DIT_H

#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_text_encoder.h"
#include "h3_vdn_weights.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    h3_layout packed;
    uint8_t *token_tags;
    uint16_t *rope_cos;
    uint16_t *rope_sin;
    uint32_t sequence;
    uint32_t text_rows;
    uint32_t audio_start;
    uint32_t audio_rows;
    uint32_t video_start;
    uint32_t video_rows;
    uint32_t frames;
    uint32_t frame_height;
    uint32_t frame_width;
} h3_vdn_layout;

/* Build the exact Diffusers [text | audio | video] layout and the BF16 RoPE
 * tables consumed by the released transformer. */
int h3_vdn_layout_build(const h3_text_embedding *prompt,
                        uint32_t latent_frames, uint32_t latent_height,
                        uint32_t latent_width, uint32_t audio_latents,
                        h3_vdn_layout *layout,
                        char *error, size_t error_size);
void h3_vdn_layout_free(h3_vdn_layout *layout);

typedef struct {
    h3_gpu_tensor *video;
    h3_gpu_tensor *audio;
} h3_vdn_velocity;

typedef struct {
    double prepare_seconds;
    double input_projection_seconds;
    double timestep_seconds;
    double blocks_seconds;
    double output_head_seconds;
    double cleanup_seconds;
    double total_seconds;
} h3_vdn_forward_timing;

typedef struct {
    unsigned index;
    float video_timestep;
    float audio_timestep;
    double wall_seconds;
    double scheduler_seconds;
    double euler_seconds;
    h3_vdn_forward_timing forward;
    h3_gpu_stats gpu;
    h3_gpu_profile_stats profile;
} h3_vdn_nfe_timing;

typedef struct {
    unsigned count;
    h3_vdn_nfe_timing entries[H3_MAX_STEPS];
} h3_vdn_denoise_timing;

typedef void (*h3_vdn_layer_progress)(unsigned completed, unsigned total,
                                      void *opaque);

/* Execute one complete 50-layer VDN DiT model evaluation. Inputs and returned
 * velocity rows are F32. The caller owns the two returned tensors. */
int h3_vdn_forward(h3_gpu *gpu, h3_vdn_weight_store *store,
                   const h3_vdn_model_weights *weights,
                   const h3_gpu_tensor *refined_prompt,
                   const h3_vdn_layout *layout,
                   const h3_gpu_tensor *video_rows,
                   const h3_gpu_tensor *audio_rows,
                   float video_timestep, float audio_timestep,
                   uint32_t radius, uint32_t chunk,
                   h3_vdn_layer_progress progress, void *progress_opaque,
                   h3_vdn_velocity *velocity,
                   h3_vdn_forward_timing *timing,
                   char *error, size_t error_size);
void h3_vdn_velocity_free(h3_vdn_velocity *velocity);

typedef void (*h3_vdn_nfe_progress)(unsigned completed, unsigned total,
                                    void *opaque);

/* Run the released paired 12/3 shifted schedule and update the mutable F32
 * video/audio latent rows in place. */
int h3_vdn_denoise(h3_gpu *gpu, h3_vdn_weight_store *store,
                   const h3_vdn_model_weights *weights,
                   const h3_gpu_tensor *refined_prompt,
                   const h3_vdn_layout *layout,
                   h3_gpu_tensor *video_rows, h3_gpu_tensor *audio_rows,
                   unsigned evaluations, uint32_t radius, uint32_t chunk,
                   h3_vdn_layer_progress layer_progress,
                   h3_vdn_nfe_progress nfe_progress, void *progress_opaque,
                   h3_vdn_denoise_timing *timing,
                   char *error, size_t error_size);

/* Project and refine an official variable-length prompt. The returned BF16
 * tensor is [prompt->tokens,5376] and owned by the caller. */
h3_gpu_tensor *h3_vdn_refine_prompt(
    h3_gpu *gpu, const h3_vdn_model_weights *weights,
    const h3_text_embedding *prompt, char *error, size_t error_size);

h3_gpu_tensor *h3_vdn_time_embedding(
    h3_gpu *gpu, const h3_vdn_model_weights *weights,
    const float *timesteps, uint32_t rows,
    char *error, size_t error_size);

h3_gpu_tensor *h3_vdn_block_modulation(
    h3_gpu *gpu, const h3_vdn_block_weights *weights,
    const h3_gpu_tensor *time_embedding, uint32_t time_rows,
    char *error, size_t error_size);

int h3_vdn_run_block(
    h3_gpu *gpu, const h3_vdn_block_weights *weights,
    h3_gpu_tensor *hidden, const h3_gpu_tensor *modulation,
    const h3_gpu_tensor *row_map, const h3_gpu_tensor *rope_cos,
    const h3_gpu_tensor *rope_sin, uint32_t sequence,
    uint32_t text_rows, uint32_t video_start, uint32_t frames,
    uint32_t frame_height, uint32_t frame_width,
    uint32_t radius, uint32_t chunk,
    char *error, size_t error_size);

#endif
