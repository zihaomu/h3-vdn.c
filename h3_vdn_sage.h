#ifndef H3_VDN_SAGE_H
#define H3_VDN_SAGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { H3_VDN_SAGE_MAX_INTERVALS = 5 };

typedef enum {
    H3_VDN_SDPA_AUTO = 0,
    H3_VDN_SDPA_SCALAR,
    H3_VDN_SDPA_WAVE32,
    H3_VDN_SDPA_SAGE_I8_BF16,
    H3_VDN_SDPA_SAGE_I8_F16,
    H3_VDN_SDPA_SAGE_I8_FP8_E4M3
} h3_vdn_sdpa_mode;

typedef struct {
    uint32_t sequence;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t video_start;
    uint32_t frames;
    uint32_t tokens_per_frame;
    uint32_t radius;
    uint32_t chunk;
    int anchor_both;
} h3_vdn_sage_geometry;

typedef struct {
    uint32_t begin;
    uint32_t end;
} h3_vdn_key_interval;

typedef struct {
    uint32_t q_begin;
    uint32_t q_count;
    uint32_t interval_count;
    h3_vdn_key_interval allowed[H3_VDN_SAGE_MAX_INTERVALS];
} h3_vdn_q_task;

const char *h3_vdn_sdpa_mode_name(h3_vdn_sdpa_mode mode);
int h3_vdn_sdpa_mode_parse(const char *text, h3_vdn_sdpa_mode *mode);

int h3_vdn_sage_build_tasks(const h3_vdn_sage_geometry *geometry,
                            h3_vdn_q_task **tasks, size_t *task_count,
                            char *error, size_t error_size);
void h3_vdn_sage_free_tasks(h3_vdn_q_task *tasks);

int h3_vdn_sage_workspace_size(const h3_vdn_sage_geometry *geometry,
                               h3_vdn_sdpa_mode mode, size_t *bytes,
                               char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
