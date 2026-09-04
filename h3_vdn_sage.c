#include "h3_vdn_sage.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int checked_add(size_t left, size_t right, size_t *result) {
    if (left > SIZE_MAX - right) return 0;
    *result = left + right;
    return 1;
}

static int checked_mul(size_t left, size_t right, size_t *result) {
    if (left && right > SIZE_MAX / left) return 0;
    *result = left * right;
    return 1;
}

static int validate_geometry(const h3_vdn_sage_geometry *geometry,
                             uint32_t *video_end, char *error,
                             size_t error_size) {
    if (!geometry || !geometry->sequence || !geometry->heads ||
        !geometry->head_dim || !geometry->frames ||
        !geometry->tokens_per_frame ||
        (geometry->anchor_both != 0 && geometry->anchor_both != 1)) {
        set_error(error, error_size, "invalid VDN Sage geometry");
        return 0;
    }
    uint64_t video_rows = (uint64_t)geometry->frames *
                          geometry->tokens_per_frame;
    uint64_t end = (uint64_t)geometry->video_start + video_rows;
    if (geometry->video_start > geometry->sequence ||
        end > geometry->sequence || end > UINT32_MAX) {
        set_error(error, error_size, "VDN Sage video range exceeds sequence");
        return 0;
    }
    *video_end = (uint32_t)end;
    return 1;
}

const char *h3_vdn_sdpa_mode_name(h3_vdn_sdpa_mode mode) {
    switch (mode) {
        case H3_VDN_SDPA_AUTO: return "auto";
        case H3_VDN_SDPA_SCALAR: return "scalar";
        case H3_VDN_SDPA_WAVE32: return "wave32";
        case H3_VDN_SDPA_SAGE_I8_BF16: return "sage-i8-bf16";
        case H3_VDN_SDPA_SAGE_I8_F16: return "sage-i8-f16";
        case H3_VDN_SDPA_SAGE_I8_FP8_E4M3: return "sage-i8-fp8-e4m3";
    }
    return NULL;
}

int h3_vdn_sdpa_mode_parse(const char *text, h3_vdn_sdpa_mode *mode) {
    if (!text || !mode) return 0;
    for (int value = H3_VDN_SDPA_AUTO;
         value <= H3_VDN_SDPA_SAGE_I8_FP8_E4M3; value++) {
        h3_vdn_sdpa_mode candidate = (h3_vdn_sdpa_mode)value;
        const char *name = h3_vdn_sdpa_mode_name(candidate);
        if (name && strcmp(text, name) == 0) {
            *mode = candidate;
            return 1;
        }
    }
    return 0;
}

static int append_interval(h3_vdn_q_task *task, uint32_t begin,
                           uint32_t end) {
    if (begin >= end) return 1;
    if (task->interval_count) {
        h3_vdn_key_interval *last =
            &task->allowed[task->interval_count - 1];
        if (begin <= last->end) {
            if (end > last->end) last->end = end;
            return 1;
        }
    }
    if (task->interval_count == H3_VDN_SAGE_MAX_INTERVALS) return 0;
    task->allowed[task->interval_count].begin = begin;
    task->allowed[task->interval_count].end = end;
    task->interval_count++;
    return 1;
}

static int build_video_task(const h3_vdn_sage_geometry *geometry,
                            uint32_t video_end, uint32_t frame,
                            h3_vdn_q_task *task) {
    memset(task, 0, sizeof(*task));
    task->q_begin = geometry->video_start +
                    frame * geometry->tokens_per_frame;
    task->q_count = geometry->tokens_per_frame;
    if (geometry->anchor_both &&
        (frame == 0 || frame + 1 == geometry->frames))
        return append_interval(task, 0, geometry->sequence);

    uint32_t lower;
    uint32_t upper;
    if (geometry->chunk) {
        uint32_t query_chunk = frame / geometry->chunk;
        uint32_t lower_chunk = query_chunk > geometry->radius ?
            query_chunk - geometry->radius : 0;
        uint64_t upper_wide = ((uint64_t)query_chunk + geometry->radius + 1) *
                              geometry->chunk;
        lower = lower_chunk * geometry->chunk;
        upper = upper_wide >= geometry->frames ? geometry->frames - 1 :
                (uint32_t)(upper_wide - 1);
    } else {
        lower = frame > geometry->radius ? frame - geometry->radius : 0;
        uint64_t upper_wide = (uint64_t)frame + geometry->radius;
        upper = upper_wide >= geometry->frames ? geometry->frames - 1 :
                (uint32_t)upper_wide;
    }
    uint32_t window_begin = geometry->video_start +
                            lower * geometry->tokens_per_frame;
    uint32_t window_end = geometry->video_start +
                          (upper + 1) * geometry->tokens_per_frame;
    uint32_t first_anchor_end = geometry->video_start +
                                geometry->tokens_per_frame;
    uint32_t last_anchor_begin = video_end - geometry->tokens_per_frame;
    return append_interval(task, 0, geometry->video_start) &&
        (!geometry->anchor_both ||
         append_interval(task, geometry->video_start, first_anchor_end)) &&
        append_interval(task, window_begin, window_end) &&
        (!geometry->anchor_both ||
         append_interval(task, last_anchor_begin, video_end)) &&
        append_interval(task, video_end, geometry->sequence);
}

static int append_task(h3_vdn_q_task *tasks, size_t *count,
                       const h3_vdn_q_task *task) {
    if (!task->q_count || !task->interval_count) return 0;
    tasks[*count] = *task;
    (*count)++;
    return 1;
}

int h3_vdn_sage_build_tasks(const h3_vdn_sage_geometry *geometry,
                            h3_vdn_q_task **tasks_out, size_t *task_count,
                            char *error, size_t error_size) {
    if (!tasks_out || !task_count) {
        set_error(error, error_size, "VDN Sage task outputs are required");
        return 0;
    }
    *tasks_out = NULL;
    *task_count = 0;
    uint32_t video_end;
    if (!validate_geometry(geometry, &video_end, error, error_size)) return 0;
    size_t capacity;
    if (!checked_add((size_t)geometry->frames, 2, &capacity) ||
        !checked_mul(capacity, sizeof(h3_vdn_q_task), &capacity)) {
        set_error(error, error_size, "VDN Sage task allocation overflow");
        return 0;
    }
    h3_vdn_q_task *tasks = calloc(1, capacity);
    if (!tasks) {
        set_error(error, error_size, "out of memory building VDN Sage tasks");
        return 0;
    }
    size_t count = 0;
    h3_vdn_q_task task;
    if (geometry->video_start) {
        memset(&task, 0, sizeof(task));
        task.q_count = geometry->video_start;
        if (!append_interval(&task, 0, geometry->sequence) ||
            !append_task(tasks, &count, &task)) goto internal_error;
    }
    for (uint32_t frame = 0; frame < geometry->frames; frame++) {
        if (!build_video_task(geometry, video_end, frame, &task) ||
            !append_task(tasks, &count, &task)) goto internal_error;
    }
    if (video_end < geometry->sequence) {
        memset(&task, 0, sizeof(task));
        task.q_begin = video_end;
        task.q_count = geometry->sequence - video_end;
        if (!append_interval(&task, 0, geometry->sequence) ||
            !append_task(tasks, &count, &task)) goto internal_error;
    }
    *tasks_out = tasks;
    *task_count = count;
    return 1;

internal_error:
    free(tasks);
    set_error(error, error_size, "cannot represent VDN Sage mask tasks");
    return 0;
}

void h3_vdn_sage_free_tasks(h3_vdn_q_task *tasks) {
    free(tasks);
}

int h3_vdn_sage_workspace_size(const h3_vdn_sage_geometry *geometry,
                               h3_vdn_sdpa_mode mode, size_t *bytes,
                               char *error, size_t error_size) {
    if (!bytes) {
        set_error(error, error_size, "VDN Sage workspace output is required");
        return 0;
    }
    *bytes = 0;
    uint32_t video_end;
    if (!validate_geometry(geometry, &video_end, error, error_size)) return 0;
    (void)video_end;
    if (mode != H3_VDN_SDPA_SAGE_I8_BF16 &&
        mode != H3_VDN_SDPA_SAGE_I8_F16 &&
        mode != H3_VDN_SDPA_SAGE_I8_FP8_E4M3) {
        set_error(error, error_size, "workspace requested for non-Sage mode");
        return 0;
    }
    if (geometry->head_dim != 128) {
        set_error(error, error_size, "VDN Sage requires head_dim 128");
        return 0;
    }
    size_t elements;
    if (!checked_mul(geometry->sequence, geometry->heads, &elements) ||
        !checked_mul(elements, geometry->head_dim, &elements)) goto overflow;
    size_t total;
    if (!checked_mul(elements, 2, &total)) goto overflow;
    size_t q_groups = ((size_t)geometry->sequence + 31) / 32;
    size_t k_groups = ((size_t)geometry->sequence + 63) / 64;
    size_t scale_count;
    if (!checked_add(q_groups, k_groups, &scale_count) ||
        !checked_mul(scale_count, geometry->heads, &scale_count) ||
        !checked_mul(scale_count, sizeof(float), &scale_count) ||
        !checked_add(total, scale_count, &total)) goto overflow;
    if (mode == H3_VDN_SDPA_SAGE_I8_F16) {
        size_t value_bytes;
        if (!checked_mul(elements, sizeof(uint16_t), &value_bytes) ||
            !checked_add(total, value_bytes, &total)) goto overflow;
    } else if (mode == H3_VDN_SDPA_SAGE_I8_FP8_E4M3 &&
        (!checked_add(total, elements, &total) ||
         !checked_mul(geometry->heads, geometry->head_dim, &scale_count) ||
         !checked_mul(scale_count, sizeof(float), &scale_count) ||
         !checked_add(total, scale_count, &total))) goto overflow;
    size_t task_bytes;
    if (!checked_add((size_t)geometry->frames, 2, &task_bytes) ||
        !checked_mul(task_bytes, sizeof(h3_vdn_q_task), &task_bytes) ||
        !checked_add(total, task_bytes, &total)) goto overflow;
    *bytes = total;
    return 1;

overflow:
    set_error(error, error_size, "VDN Sage workspace size overflow");
    return 0;
}
