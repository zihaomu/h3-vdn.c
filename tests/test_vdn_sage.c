#include "h3_vdn_sage.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int checks;
static int failures;

#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        failures++; \
    } \
} while (0)

static int scalar_allowed(const h3_vdn_sage_geometry *geometry,
                          uint32_t query, uint32_t key) {
    uint32_t video_end = geometry->video_start +
        geometry->frames * geometry->tokens_per_frame;
    if (query < geometry->video_start || query >= video_end ||
        key < geometry->video_start || key >= video_end) return 1;
    uint32_t query_frame =
        (query - geometry->video_start) / geometry->tokens_per_frame;
    uint32_t key_frame =
        (key - geometry->video_start) / geometry->tokens_per_frame;
    if (geometry->anchor_both &&
        (query_frame == 0 || query_frame + 1 == geometry->frames ||
         key_frame == 0 || key_frame + 1 == geometry->frames)) return 1;
    uint32_t lower;
    uint32_t upper;
    if (geometry->chunk) {
        uint32_t query_chunk = query_frame / geometry->chunk;
        uint32_t lower_chunk = query_chunk > geometry->radius ?
            query_chunk - geometry->radius : 0;
        uint64_t upper_wide = ((uint64_t)query_chunk + geometry->radius + 1) *
                              geometry->chunk;
        lower = lower_chunk * geometry->chunk;
        upper = upper_wide >= geometry->frames ? geometry->frames - 1 :
                (uint32_t)(upper_wide - 1);
    } else {
        lower = query_frame > geometry->radius ?
            query_frame - geometry->radius : 0;
        uint64_t upper_wide = (uint64_t)query_frame + geometry->radius;
        upper = upper_wide >= geometry->frames ? geometry->frames - 1 :
                (uint32_t)upper_wide;
    }
    return key_frame >= lower && key_frame <= upper;
}

static int task_allowed(const h3_vdn_q_task *task, uint32_t key) {
    for (uint32_t index = 0; index < task->interval_count; index++)
        if (key >= task->allowed[index].begin &&
            key < task->allowed[index].end) return 1;
    return 0;
}

static void verify_geometry(const h3_vdn_sage_geometry *geometry) {
    char error[256] = {0};
    h3_vdn_q_task *tasks = NULL;
    size_t task_count = 0;
    CHECK(h3_vdn_sage_build_tasks(
        geometry, &tasks, &task_count, error, sizeof(error)));
    if (!tasks) return;
    uint32_t next_query = 0;
    for (size_t task_index = 0; task_index < task_count; task_index++) {
        const h3_vdn_q_task *task = &tasks[task_index];
        CHECK(task->q_begin == next_query);
        CHECK(task->q_count > 0);
        CHECK(task->interval_count > 0);
        CHECK(task->interval_count <= H3_VDN_SAGE_MAX_INTERVALS);
        uint32_t interval_end = 0;
        for (uint32_t interval = 0; interval < task->interval_count;
             interval++) {
            CHECK(task->allowed[interval].begin <
                  task->allowed[interval].end);
            CHECK(task->allowed[interval].end <= geometry->sequence);
            if (interval)
                CHECK(task->allowed[interval].begin > interval_end);
            interval_end = task->allowed[interval].end;
        }
        for (uint32_t offset = 0; offset < task->q_count; offset++) {
            uint32_t query = task->q_begin + offset;
            for (uint32_t key = 0; key < geometry->sequence; key++)
                CHECK(task_allowed(task, key) ==
                      scalar_allowed(geometry, query, key));
        }
        next_query += task->q_count;
    }
    CHECK(next_query == geometry->sequence);
    h3_vdn_sage_free_tasks(tasks);
}

static void test_modes(void) {
    for (int value = H3_VDN_SDPA_AUTO;
         value <= H3_VDN_SDPA_SAGE_I8_FP8_E4M3; value++) {
        h3_vdn_sdpa_mode expected = (h3_vdn_sdpa_mode)value;
        const char *name = h3_vdn_sdpa_mode_name(expected);
        h3_vdn_sdpa_mode parsed = H3_VDN_SDPA_AUTO;
        CHECK(name != NULL);
        CHECK(h3_vdn_sdpa_mode_parse(name, &parsed));
        CHECK(parsed == expected);
    }
    h3_vdn_sdpa_mode parsed;
    CHECK(!h3_vdn_sdpa_mode_parse("sage", &parsed));
    CHECK(!h3_vdn_sdpa_mode_parse(NULL, &parsed));
    CHECK(!h3_vdn_sdpa_mode_parse("auto", NULL));
    CHECK(h3_vdn_sdpa_mode_name((h3_vdn_sdpa_mode)99) == NULL);
}

static void test_workspace(void) {
    h3_vdn_sage_geometry geometry = {
        5338, 56, 128, 986, 17, 256, 1, 5, 1
    };
    char error[256] = {0};
    size_t bf16 = 0, f16 = 0, fp8 = 0;
    CHECK(h3_vdn_sage_workspace_size(
        &geometry, H3_VDN_SDPA_SAGE_I8_BF16, &bf16,
        error, sizeof(error)));
    CHECK(h3_vdn_sage_workspace_size(
        &geometry, H3_VDN_SDPA_SAGE_I8_F16, &f16,
        error, sizeof(error)));
    CHECK(h3_vdn_sage_workspace_size(
        &geometry, H3_VDN_SDPA_SAGE_I8_FP8_E4M3, &fp8,
        error, sizeof(error)));
    CHECK(bf16 > 72 * 1024 * 1024u);
    CHECK(bf16 < fp8);
    CHECK(fp8 < f16);
    CHECK(f16 < 160 * 1024 * 1024u);
    CHECK(!h3_vdn_sage_workspace_size(
        &geometry, H3_VDN_SDPA_WAVE32, &bf16, error, sizeof(error)));
    geometry.head_dim = 64;
    CHECK(!h3_vdn_sage_workspace_size(
        &geometry, H3_VDN_SDPA_SAGE_I8_BF16, &bf16,
        error, sizeof(error)));
}

static void test_invalid(void) {
    h3_vdn_sage_geometry geometry = {32, 1, 128, 4, 2, 8, 1, 0, 1};
    h3_vdn_q_task *tasks = NULL;
    size_t count = 0;
    char error[128] = {0};
    CHECK(!h3_vdn_sage_build_tasks(
        NULL, &tasks, &count, error, sizeof(error)));
    CHECK(!h3_vdn_sage_build_tasks(
        &geometry, NULL, &count, error, sizeof(error)));
    geometry.anchor_both = 2;
    CHECK(!h3_vdn_sage_build_tasks(
        &geometry, &tasks, &count, error, sizeof(error)));
    geometry.anchor_both = 1;
    geometry.video_start = 30;
    CHECK(!h3_vdn_sage_build_tasks(
        &geometry, &tasks, &count, error, sizeof(error)));
    geometry.sequence = UINT32_MAX;
    geometry.video_start = UINT32_MAX - 1;
    geometry.frames = UINT32_MAX;
    geometry.tokens_per_frame = UINT32_MAX;
    CHECK(!h3_vdn_sage_build_tasks(
        &geometry, &tasks, &count, error, sizeof(error)));
}

int main(void) {
    test_modes();
    h3_vdn_sage_geometry production = {
        5338, 56, 128, 986, 17, 256, 1, 5, 1
    };
    h3_vdn_sage_geometry suffix = {
        67, 7, 128, 5, 5, 7, 1, 2, 1
    };
    h3_vdn_sage_geometry no_anchor = {
        103, 3, 128, 7, 6, 13, 2, 0, 0
    };
    h3_vdn_sage_geometry short_chunk = {
        31, 1, 128, 3, 4, 5, 7, 9, 1
    };
    verify_geometry(&production);
    verify_geometry(&suffix);
    verify_geometry(&no_anchor);
    verify_geometry(&short_chunk);
    test_workspace();
    test_invalid();
    if (failures) {
        fprintf(stderr, "VDN Sage contract failed: %d/%d checks\n",
                failures, checks);
        return 1;
    }
    printf("VDN Sage mask/workspace contract passed: %d checks\n", checks);
    return 0;
}
