#ifndef H3_TESTS_VDN_REFERENCE_H
#define H3_TESTS_VDN_REFERENCE_H

#include <stddef.h>

typedef struct {
    int lo;
    int hi;
} h3_vdn_ref_window;

int h3_vdn_ref_window_bounds(size_t frames, int radius, int chunk,
                             h3_vdn_ref_window *bounds);
size_t h3_vdn_ref_prune_anchor_bounds(size_t frames,
                                      const h3_vdn_ref_window *bounds,
                                      h3_vdn_ref_window *inner_bounds);

void h3_vdn_ref_frame_statistics(size_t tokens, size_t key_dim,
                                 size_t value_dim, const float *key,
                                 const float *value, const float *beta,
                                 float *a, float *b);

int h3_vdn_ref_vdn_factor(size_t key_dim, size_t value_dim,
                          const float *alpha, const float *a, const float *b,
                          float *transition, float *injection);

int h3_vdn_ref_scan(size_t frames, size_t key_dim, size_t value_dim,
                    const float *transitions, const float *injections,
                    const float *text_state, float *prefix, float *suffix);

int h3_vdn_ref_gather(size_t frames, size_t key_dim, size_t value_dim,
                      const float *prefix, const float *suffix,
                      const float *alpha, const h3_vdn_ref_window *bounds,
                      const float *text_state, int bridge_alpha, float *output);

#endif
