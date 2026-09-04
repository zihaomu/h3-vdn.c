#include "h3_vdn.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int check_common(const h3_vdn_info *info) {
    CHECK(info->enabled);
    CHECK(!strcmp(info->model_revision,
                  "18be6bcc4ee72585eee322ba28b5ccac2cf85ef0"));
    CHECK(info->checkpoint_format_version == 2);
    CHECK(info->model_spec_format_version == 2);
    CHECK(info->num_attention_heads == 56);
    CHECK(info->attention_head_dim == 128);
    CHECK(info->hidden_size == 5376);
    CHECK(info->num_layers == 50);
    CHECK(info->num_refiner_layers == 2);
    CHECK(info->ffn_dim == 14336);
    CHECK(info->in_channels == 24);
    CHECK(info->audio_in_channels == 32);
    CHECK(info->text_dim == 5120);
    CHECK(info->transform.version == 2);
    CHECK(info->transform.softmax_radius == 1);
    CHECK(info->transform.softmax_chunk == 5);
    CHECK(info->transform.anchor_both);
    CHECK(info->transform.enable_softmax_gate);
    CHECK(info->transform.linear_head_dim == 128);
    CHECK(info->transform.accumulator_f32);
    CHECK(info->transform.enable_text_state);
    CHECK(info->transform.short_conv_k && info->transform.short_conv_v);
    CHECK(!strcmp(info->transform.delta_rule, "vdn_solve"));
    CHECK(!strcmp(info->transform.bridge, "alpha"));
    CHECK(info->video_shift == 12.0);
    CHECK(info->audio_shift == 3.0);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s H3_BASE STAGE_B STAGE_DMD\n", argv[0]);
        return 2;
    }
    char error[512];
    h3_vdn_info info;

    CHECK(h3_vdn_inspect(argv[1], argv[2], 0, &info,
                         error, sizeof(error)));
    CHECK(check_common(&info) == 0);
    CHECK(!strcmp(info.checkpoint_name, "stage-b-step-2000"));
    CHECK(info.adapter_count == 1);
    CHECK(!strcmp(info.adapters[0].name, "default"));
    CHECK(info.adapters[0].rank == 64 && info.adapters[0].alpha == 64);
    CHECK(info.adapters[0].target_count == 8);
    CHECK(info.num_steps == 50);

    CHECK(h3_vdn_inspect(argv[1], argv[3], 0, &info,
                         error, sizeof(error)));
    CHECK(check_common(&info) == 0);
    CHECK(!strcmp(info.checkpoint_name, "stage-dmd-step-250"));
    CHECK(info.adapter_count == 2);
    CHECK(!strcmp(info.adapters[0].name, "default"));
    CHECK(!strcmp(info.adapters[1].name, "turbo"));
    CHECK(info.adapters[1].rank == 64 && info.adapters[1].alpha == 64);
    CHECK(info.adapters[1].exact_targets);
    CHECK(info.adapters[1].target_count == 363);
    CHECK(info.adapters[1].rank_pattern_count == 51);
    CHECK(info.adapters[1].alpha_pattern_count == 51);
    CHECK(info.num_steps == 8);

    int weights_present = info.weights_present;
    int strict = h3_vdn_inspect(argv[1], argv[3], 1, &info,
                                error, sizeof(error));
    CHECK(strict == weights_present);
    if (!strict) CHECK(error[0] != '\0');

    puts("VDN metadata tests passed");
    return 0;
}
