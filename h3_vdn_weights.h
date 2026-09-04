#ifndef H3_VDN_WEIGHTS_H
#define H3_VDN_WEIGHTS_H

#include "h3_gpu.h"

#include <stddef.h>

typedef struct h3_vdn_weight_store h3_vdn_weight_store;

typedef struct {
    h3_gpu_tensor *alpha_a_log;
    h3_gpu_tensor *alpha_down;
    h3_gpu_tensor *alpha_dt_bias;
    h3_gpu_tensor *alpha_up;
    h3_gpu_tensor *beta;
    h3_gpu_tensor *norm;
    h3_gpu_tensor *gate_down;
    h3_gpu_tensor *gate_up_bias;
    h3_gpu_tensor *gate_up;
    h3_gpu_tensor *k_spatial;
    h3_gpu_tensor *k_temporal;
    h3_gpu_tensor *v_spatial;
    h3_gpu_tensor *v_temporal;
    h3_gpu_tensor *softmax_gate_bias;
    h3_gpu_tensor *softmax_gate_weight;
    h3_gpu_tensor *to_out;
} h3_vdn_linear_weights;

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *q;
    h3_gpu_tensor *k;
    h3_gpu_tensor *v;
    h3_gpu_tensor *q_norm;
    h3_gpu_tensor *k_norm;
    h3_gpu_tensor *out;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *fc2;
    h3_gpu_tensor *adaln_weight;
    h3_gpu_tensor *adaln_bias;
    h3_vdn_linear_weights linear;
    int borrowed;
} h3_vdn_block_weights;

typedef struct {
    uint64_t budget_bytes;
    uint64_t resident_bytes;
    uint64_t hits;
    uint64_t misses;
    unsigned resident_blocks;
    int admission_limited;
} h3_vdn_weight_cache_stats;

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *q;
    h3_gpu_tensor *k;
    h3_gpu_tensor *v;
    h3_gpu_tensor *q_norm;
    h3_gpu_tensor *k_norm;
    h3_gpu_tensor *out;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *fc2;
} h3_vdn_refiner_weights;

typedef struct {
    h3_gpu_tensor *video_in_weight;
    h3_gpu_tensor *video_in_bias;
    h3_gpu_tensor *audio_in_weight;
    h3_gpu_tensor *audio_in_bias;
    h3_gpu_tensor *context_weight;
    h3_gpu_tensor *context_bias;
    h3_gpu_tensor *time_linear1_weight;
    h3_gpu_tensor *time_linear1_bias;
    h3_gpu_tensor *time_linear2_weight;
    h3_gpu_tensor *time_linear2_bias;
    h3_vdn_refiner_weights refiner[2];
    h3_gpu_tensor *refiner_final_norm;
    h3_gpu_tensor *final_norm;
    h3_gpu_tensor *final_adaln_weight;
    h3_gpu_tensor *final_adaln_bias;
    h3_gpu_tensor *video_out_weight;
    h3_gpu_tensor *video_out_bias;
    h3_gpu_tensor *audio_out_weight;
    h3_gpu_tensor *audio_out_bias;
} h3_vdn_model_weights;

h3_vdn_weight_store *h3_vdn_weight_store_open(
    const char *base_model_dir, const char *checkpoint_dir, int use_turbo,
    char *error, size_t error_size);
void h3_vdn_weight_store_free(h3_vdn_weight_store *store);
int h3_vdn_weight_store_cache_stats(
    const h3_vdn_weight_store *store, h3_vdn_weight_cache_stats *stats);

int h3_vdn_block_weights_load(h3_vdn_weight_store *store, h3_gpu *gpu,
                              unsigned block, h3_vdn_block_weights *weights,
                              char *error, size_t error_size);
void h3_vdn_block_weights_free(h3_vdn_block_weights *weights);

int h3_vdn_model_weights_load(h3_vdn_weight_store *store, h3_gpu *gpu,
                              h3_vdn_model_weights *weights,
                              char *error, size_t error_size);
void h3_vdn_model_weights_free(h3_vdn_model_weights *weights);

#endif
