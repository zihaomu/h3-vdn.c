#include "h3_vdn.h"

#include "h3_json.h"
#include "h3_safetensors.h"

#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int h3_vdn_fail(char *error, size_t error_size,
                       const char *format, ...) {
    if (error && error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

static char *h3_vdn_path(const char *root, const char *relative) {
    if (!root || !relative) return NULL;
    size_t root_length = strlen(root);
    size_t relative_length = strlen(relative);
    if (root_length > SIZE_MAX - relative_length - 2) return NULL;
    char *path = malloc(root_length + relative_length + 2);
    if (path)
        snprintf(path, root_length + relative_length + 2,
                 "%s/%s", root, relative);
    return path;
}

static h3_json_value *h3_vdn_read_json(const char *root,
                                       const char *relative,
                                       char *error, size_t error_size) {
    char *path = h3_vdn_path(root, relative);
    if (!path) {
        h3_vdn_fail(error, error_size, "out of memory resolving %s", relative);
        return NULL;
    }
    h3_json_value *value = h3_json_parse_file(path, error, error_size);
    free(path);
    return value;
}

static const h3_json_value *h3_vdn_field(const h3_json_value *object,
                                         const char *key, const char *where,
                                         char *error, size_t error_size) {
    if (!object || h3_json_get_type(object) != H3_JSON_OBJECT) {
        h3_vdn_fail(error, error_size, "%s must be an object", where);
        return NULL;
    }
    const h3_json_value *value = h3_json_get(object, key);
    if (!value)
        h3_vdn_fail(error, error_size, "%s.%s is required", where, key);
    return value;
}

static int h3_vdn_int(const h3_json_value *object, const char *key,
                      const char *where, int *result,
                      char *error, size_t error_size) {
    const h3_json_value *value = h3_vdn_field(
        object, key, where, error, error_size);
    int64_t integer;
    if (!value) return 0;
    if (!h3_json_i64_value(value, &integer) || integer < INT32_MIN ||
        integer > INT32_MAX)
        return h3_vdn_fail(error, error_size,
                           "%s.%s must be a 32-bit integer", where, key);
    *result = (int)integer;
    return 1;
}

static int h3_vdn_number(const h3_json_value *object, const char *key,
                         const char *where, double *result,
                         char *error, size_t error_size) {
    const h3_json_value *value = h3_vdn_field(
        object, key, where, error, error_size);
    if (!value) return 0;
    if (!h3_json_f64_value(value, result))
        return h3_vdn_fail(error, error_size,
                           "%s.%s must be a number", where, key);
    return 1;
}

static int h3_vdn_boolean(const h3_json_value *object, const char *key,
                          const char *where, int *result,
                          char *error, size_t error_size) {
    const h3_json_value *value = h3_vdn_field(
        object, key, where, error, error_size);
    if (!value) return 0;
    if (!h3_json_boolean_value(value, result))
        return h3_vdn_fail(error, error_size,
                           "%s.%s must be a boolean", where, key);
    return 1;
}

static int h3_vdn_string(const h3_json_value *object, const char *key,
                         const char *where, char *result, size_t result_size,
                         char *error, size_t error_size) {
    const h3_json_value *value = h3_vdn_field(
        object, key, where, error, error_size);
    const char *string = h3_json_string_value(value);
    if (!value) return 0;
    if (!string)
        return h3_vdn_fail(error, error_size,
                           "%s.%s must be a string", where, key);
    if (strlen(string) >= result_size)
        return h3_vdn_fail(error, error_size,
                           "%s.%s is too long", where, key);
    snprintf(result, result_size, "%s", string);
    return 1;
}

static int h3_vdn_expect_int(const h3_json_value *object, const char *key,
                             const char *where, int expected, int *output,
                             char *error, size_t error_size) {
    int value;
    if (!h3_vdn_int(object, key, where, &value, error, error_size)) return 0;
    if (value != expected)
        return h3_vdn_fail(error, error_size,
                           "%s.%s is %d; expected %d",
                           where, key, value, expected);
    if (output) *output = value;
    return 1;
}

static int h3_vdn_expect_string(const h3_json_value *object, const char *key,
                                const char *where, const char *expected,
                                char *error, size_t error_size) {
    char value[128];
    if (!h3_vdn_string(object, key, where, value, sizeof(value),
                       error, error_size)) return 0;
    if (strcmp(value, expected))
        return h3_vdn_fail(error, error_size,
                           "%s.%s is %s; expected %s",
                           where, key, value, expected);
    return 1;
}

static int h3_vdn_parse_base(const h3_json_value *root, h3_vdn_info *info,
                             char *error, size_t error_size) {
    if (h3_json_get_type(root) != H3_JSON_OBJECT)
        return h3_vdn_fail(error, error_size,
                           "transformer/config.json must be an object");
    if (!h3_vdn_expect_string(root, "_class_name", "transformer",
                              "MiniMaxH3Transformer3DModel", error,
                              error_size) ||
        !h3_vdn_expect_int(root, "num_attention_heads", "transformer", 56,
                           &info->num_attention_heads, error, error_size) ||
        !h3_vdn_expect_int(root, "attention_head_dim", "transformer", 128,
                           &info->attention_head_dim, error, error_size) ||
        !h3_vdn_expect_int(root, "hidden_size", "transformer", 5376,
                           &info->hidden_size, error, error_size) ||
        !h3_vdn_expect_int(root, "num_layers", "transformer", 50,
                           &info->num_layers, error, error_size) ||
        !h3_vdn_expect_int(root, "num_refiner_layers", "transformer", 2,
                           &info->num_refiner_layers, error, error_size) ||
        !h3_vdn_expect_int(root, "ffn_dim", "transformer", 14336,
                           &info->ffn_dim, error, error_size) ||
        !h3_vdn_expect_int(root, "in_channels", "transformer", 24,
                           &info->in_channels, error, error_size) ||
        !h3_vdn_expect_int(root, "audio_in_channels", "transformer", 32,
                           &info->audio_in_channels, error, error_size) ||
        !h3_vdn_expect_int(root, "text_dim", "transformer", 5120,
                           &info->text_dim, error, error_size)) return 0;

    const h3_json_value *patch = h3_vdn_field(
        root, "patch_size", "transformer", error, error_size);
    if (!patch || h3_json_get_type(patch) != H3_JSON_ARRAY ||
        h3_json_size(patch) != 3)
        return h3_vdn_fail(error, error_size,
                           "transformer.patch_size must be [1,2,2]");
    static const int64_t expected[] = {1, 2, 2};
    for (size_t index = 0; index < 3; index++) {
        int64_t value;
        if (!h3_json_i64_value(h3_json_at(patch, index), &value) ||
            value != expected[index])
            return h3_vdn_fail(error, error_size,
                               "transformer.patch_size must be [1,2,2]");
    }
    return 1;
}

static int h3_vdn_target_present(const h3_json_value *targets,
                                 const char *wanted) {
    for (size_t index = 0; index < h3_json_size(targets); index++) {
        const char *target = h3_json_string_value(h3_json_at(targets, index));
        if (target && !strcmp(target, wanted)) return 1;
    }
    return 0;
}

static int h3_vdn_parse_transform(const h3_json_value *root,
                                  h3_vdn_transform_info *transform,
                                  char *error, size_t error_size) {
    memset(transform, 0, sizeof(*transform));
    if (!h3_vdn_expect_string(root, "type", "transform", "hybrid_attention",
                              error, error_size) ||
        !h3_vdn_expect_int(root, "version", "transform", 2,
                           &transform->version, error, error_size)) return 0;
    const h3_json_value *config = h3_vdn_field(
        root, "config", "transform", error, error_size);
    if (!config) return 0;
    char anchor[32];
    if (!h3_vdn_string(config, "anchor_frames", "transform.config",
                       anchor, sizeof(anchor), error, error_size) ||
        strcmp(anchor, "both"))
        return h3_vdn_fail(error, error_size,
                           "transform.config.anchor_frames must be both");
    transform->anchor_both = 1;
    if (!h3_vdn_boolean(config, "enable_softmax_gate", "transform.config",
                        &transform->enable_softmax_gate, error, error_size) ||
        !transform->enable_softmax_gate)
        return h3_vdn_fail(error, error_size,
                           "transform.config.enable_softmax_gate must be true");

    const h3_json_value *softmax = h3_vdn_field(
        config, "softmax_attention", "transform.config", error, error_size);
    if (!softmax ||
        !h3_vdn_expect_int(softmax, "radius", "softmax_attention", 1,
                           &transform->softmax_radius, error, error_size) ||
        !h3_vdn_expect_int(softmax, "chunk", "softmax_attention", 5,
                           &transform->softmax_chunk, error, error_size)) return 0;

    const h3_json_value *linear = h3_vdn_field(
        config, "linear_attention", "transform.config", error, error_size);
    if (!linear ||
        !h3_vdn_string(linear, "delta_rule", "linear_attention",
                       transform->delta_rule, sizeof(transform->delta_rule),
                       error, error_size) ||
        strcmp(transform->delta_rule, "vdn_solve") ||
        !h3_vdn_string(linear, "bridge", "linear_attention",
                       transform->bridge, sizeof(transform->bridge),
                       error, error_size) ||
        strcmp(transform->bridge, "alpha") ||
        !h3_vdn_expect_int(linear, "linear_head_dim", "linear_attention", 128,
                           &transform->linear_head_dim, error, error_size) ||
        !h3_vdn_boolean(linear, "a_fp32", "linear_attention",
                        &transform->accumulator_f32, error, error_size) ||
        !transform->accumulator_f32 ||
        !h3_vdn_boolean(linear, "enable_text_state", "linear_attention",
                        &transform->enable_text_state, error, error_size) ||
        !transform->enable_text_state)
        return h3_vdn_fail(error, error_size,
                           "unsupported VDN linear_attention configuration");

    const h3_json_value *short_conv = h3_vdn_field(
        linear, "short_conv", "linear_attention", error, error_size);
    const h3_json_value *targets = short_conv ? h3_vdn_field(
        short_conv, "targets", "linear_attention.short_conv", error,
        error_size) : NULL;
    if (!targets || h3_json_get_type(targets) != H3_JSON_ARRAY ||
        h3_json_size(targets) != 2 ||
        !h3_vdn_target_present(targets, "k") ||
        !h3_vdn_target_present(targets, "v"))
        return h3_vdn_fail(error, error_size,
                           "linear_attention.short_conv.targets must be [k,v]");
    transform->short_conv_k = 1;
    transform->short_conv_v = 1;
    return 1;
}

static int h3_vdn_same_transform(const h3_vdn_transform_info *left,
                                 const h3_vdn_transform_info *right) {
    return left->version == right->version &&
        left->softmax_radius == right->softmax_radius &&
        left->softmax_chunk == right->softmax_chunk &&
        left->enable_softmax_gate == right->enable_softmax_gate &&
        left->anchor_both == right->anchor_both &&
        left->linear_head_dim == right->linear_head_dim &&
        left->accumulator_f32 == right->accumulator_f32 &&
        left->enable_text_state == right->enable_text_state &&
        left->short_conv_k == right->short_conv_k &&
        left->short_conv_v == right->short_conv_v &&
        !strcmp(left->delta_rule, right->delta_rule) &&
        !strcmp(left->bridge, right->bridge);
}

static int h3_vdn_validate_targets(const h3_json_value *targets,
                                   h3_vdn_adapter_info *adapter,
                                   char *error, size_t error_size) {
    if (!targets || h3_json_get_type(targets) != H3_JSON_ARRAY ||
        !h3_json_size(targets))
        return h3_vdn_fail(error, error_size,
                           "adapter.config.targets must be a non-empty array");
    adapter->target_count = h3_json_size(targets);
    for (size_t index = 0; index < adapter->target_count; index++) {
        const char *target = h3_json_string_value(h3_json_at(targets, index));
        if (!target || !*target)
            return h3_vdn_fail(error, error_size,
                               "adapter target %zu must be a string", index);
        for (size_t prior = 0; prior < index; prior++) {
            const char *other = h3_json_string_value(h3_json_at(targets, prior));
            if (!strcmp(target, other))
                return h3_vdn_fail(error, error_size,
                                   "duplicate adapter target: %s", target);
        }
    }
    return 1;
}

static int h3_vdn_require_target(const h3_json_value *targets,
                                 const char *target,
                                 char *error, size_t error_size) {
    if (h3_vdn_target_present(targets, target)) return 1;
    return h3_vdn_fail(error, error_size,
                       "adapter target is missing: %s", target);
}

static int h3_vdn_require_pattern(const h3_json_value *pattern,
                                  const char *target, int expected,
                                  const char *label,
                                  char *error, size_t error_size) {
    int64_t value;
    const h3_json_value *entry = h3_json_get(pattern, target);
    if (!h3_json_i64_value(entry, &value) || value != expected)
        return h3_vdn_fail(error, error_size,
                           "%s.%s must be %d", label, target, expected);
    return 1;
}

static int h3_vdn_validate_adapter_schema(
        size_t index, const h3_json_value *targets,
        const h3_json_value *rank_pattern,
        const h3_json_value *alpha_pattern,
        const h3_vdn_adapter_info *adapter,
        char *error, size_t error_size) {
    if (adapter->rank != 64 || adapter->alpha != 64)
        return h3_vdn_fail(error, error_size,
                           "adapter %zu rank and alpha must both be 64", index);
    if (!index) {
        static const char *const expected[] = {
            "attn.orig.to_k", "attn.orig.to_out.0", "attn.orig.to_q",
            "attn.orig.to_v",
            "token_refiner.refiner_blocks.*.attn.to_k",
            "token_refiner.refiner_blocks.*.attn.to_out.0",
            "token_refiner.refiner_blocks.*.attn.to_q",
            "token_refiner.refiner_blocks.*.attn.to_v"
        };
        if (adapter->target_count != sizeof(expected) / sizeof(expected[0]) ||
            rank_pattern || alpha_pattern || adapter->exact_targets)
            return h3_vdn_fail(error, error_size,
                               "default adapter schema does not match the release");
        for (size_t item = 0; item < sizeof(expected) / sizeof(expected[0]);
             item++)
            if (!h3_vdn_require_target(targets, expected[item],
                                       error, error_size)) return 0;
        return 1;
    }

    if (!adapter->exact_targets || adapter->target_count != 363 ||
        !rank_pattern || !alpha_pattern ||
        adapter->rank_pattern_count != 51 ||
        adapter->alpha_pattern_count != 51)
        return h3_vdn_fail(error, error_size,
                           "turbo adapter schema does not match the release");
    if (!h3_vdn_require_target(targets, "norm_out.linear", error, error_size) ||
        !h3_vdn_require_pattern(rank_pattern, "norm_out.linear", 16,
                                "rank_pattern", error, error_size) ||
        !h3_vdn_require_pattern(alpha_pattern, "norm_out.linear", 16,
                                "alpha_pattern", error, error_size))
        return 0;
    char name[224];
    for (int block = 0; block < 50; block++) {
        snprintf(name, sizeof(name),
                 "transformer_blocks.%d.adaln_proj.linear", block);
        if (!h3_vdn_require_target(targets, name, error, error_size) ||
            !h3_vdn_require_pattern(rank_pattern, name, 16, "rank_pattern",
                                    error, error_size) ||
            !h3_vdn_require_pattern(alpha_pattern, name, 16, "alpha_pattern",
                                    error, error_size)) return 0;
        static const char *const attention[] = {
            "to_k", "to_out.0", "to_q", "to_v"
        };
        for (size_t item = 0;
             item < sizeof(attention) / sizeof(attention[0]); item++) {
            snprintf(name, sizeof(name),
                     "transformer_blocks.%d.attn.orig.%s", block,
                     attention[item]);
            if (!h3_vdn_require_target(targets, name, error, error_size))
                return 0;
        }
        snprintf(name, sizeof(name),
                 "transformer_blocks.%d.ff.net.0.proj", block);
        if (!h3_vdn_require_target(targets, name, error, error_size)) return 0;
        snprintf(name, sizeof(name),
                 "transformer_blocks.%d.ff.net.2", block);
        if (!h3_vdn_require_target(targets, name, error, error_size)) return 0;
    }
    for (int block = 0; block < 2; block++) {
        static const char *const attention[] = {
            "to_k", "to_out.0", "to_q", "to_v"
        };
        for (size_t item = 0;
             item < sizeof(attention) / sizeof(attention[0]); item++) {
            snprintf(name, sizeof(name),
                     "token_refiner.refiner_blocks.%d.attn.%s", block,
                     attention[item]);
            if (!h3_vdn_require_target(targets, name, error, error_size))
                return 0;
        }
        snprintf(name, sizeof(name),
                 "token_refiner.refiner_blocks.%d.ff.net.0.proj", block);
        if (!h3_vdn_require_target(targets, name, error, error_size)) return 0;
        snprintf(name, sizeof(name),
                 "token_refiner.refiner_blocks.%d.ff.net.2", block);
        if (!h3_vdn_require_target(targets, name, error, error_size)) return 0;
    }
    return 1;
}

static int h3_vdn_parse_adapter(const h3_json_value *root,
                                size_t index, h3_vdn_adapter_info *adapter,
                                char *error, size_t error_size) {
    memset(adapter, 0, sizeof(*adapter));
    if (!h3_vdn_expect_string(root, "type", "adapter", "lora",
                              error, error_size) ||
        !h3_vdn_expect_int(root, "version", "adapter", 1, NULL,
                           error, error_size)) return 0;
    const h3_json_value *config = h3_vdn_field(
        root, "config", "adapter", error, error_size);
    if (!config ||
        !h3_vdn_int(config, "rank", "adapter.config", &adapter->rank,
                    error, error_size) || adapter->rank < 1 ||
        !h3_vdn_int(config, "alpha", "adapter.config", &adapter->alpha,
                    error, error_size) || adapter->alpha < 1)
        return h3_vdn_fail(error, error_size,
                           "adapter %zu rank/alpha must be positive", index);
    const h3_json_value *name = h3_json_get(config, "name");
    if (name) {
        const char *string = h3_json_string_value(name);
        if (!string || strlen(string) >= sizeof(adapter->name))
            return h3_vdn_fail(error, error_size,
                               "adapter %zu has invalid name", index);
        snprintf(adapter->name, sizeof(adapter->name), "%s", string);
    } else {
        snprintf(adapter->name, sizeof(adapter->name), "default");
    }
    const h3_json_value *exact = h3_json_get(config, "exact_targets");
    if (exact && !h3_json_boolean_value(exact, &adapter->exact_targets))
        return h3_vdn_fail(error, error_size,
                           "adapter %zu exact_targets must be boolean", index);
    const h3_json_value *targets = h3_json_get(config, "targets");
    if (!h3_vdn_validate_targets(targets, adapter,
                                 error, error_size)) return 0;

    const h3_json_value *rank_pattern = h3_json_get(config, "rank_pattern");
    const h3_json_value *alpha_pattern = h3_json_get(config, "alpha_pattern");
    if (rank_pattern) {
        if (h3_json_get_type(rank_pattern) != H3_JSON_OBJECT)
            return h3_vdn_fail(error, error_size,
                               "adapter %zu rank_pattern must be an object", index);
        adapter->rank_pattern_count = h3_json_size(rank_pattern);
    }
    if (alpha_pattern) {
        if (h3_json_get_type(alpha_pattern) != H3_JSON_OBJECT)
            return h3_vdn_fail(error, error_size,
                               "adapter %zu alpha_pattern must be an object", index);
        adapter->alpha_pattern_count = h3_json_size(alpha_pattern);
    }
    return h3_vdn_validate_adapter_schema(
        index, targets, rank_pattern, alpha_pattern, adapter,
        error, error_size);
}

static int h3_vdn_json_equal(const h3_json_value *left,
                             const h3_json_value *right) {
    if (!left || !right || h3_json_get_type(left) != h3_json_get_type(right))
        return 0;
    size_t count = h3_json_size(left);
    if (count != h3_json_size(right)) return 0;
    switch (h3_json_get_type(left)) {
        case H3_JSON_NULL:
            return 1;
        case H3_JSON_BOOLEAN: {
            int left_value;
            int right_value;
            return h3_json_boolean_value(left, &left_value) &&
                   h3_json_boolean_value(right, &right_value) &&
                   left_value == right_value;
        }
        case H3_JSON_NUMBER: {
            double left_value;
            double right_value;
            return h3_json_f64_value(left, &left_value) &&
                   h3_json_f64_value(right, &right_value) &&
                   left_value == right_value;
        }
        case H3_JSON_STRING:
            return !strcmp(h3_json_string_value(left),
                           h3_json_string_value(right));
        case H3_JSON_ARRAY:
            for (size_t index = 0; index < count; index++)
                if (!h3_vdn_json_equal(h3_json_at(left, index),
                                       h3_json_at(right, index))) return 0;
            return 1;
        case H3_JSON_OBJECT:
            for (size_t index = 0; index < count; index++) {
                const char *key = h3_json_object_key(left, index);
                if (!key || !h3_vdn_json_equal(
                        h3_json_object_value(left, index),
                        h3_json_get(right, key))) return 0;
            }
            return 1;
    }
    return 0;
}

static int h3_vdn_parse_spec(const h3_json_value *root, h3_vdn_info *info,
                             char *error, size_t error_size) {
    if (!h3_vdn_expect_int(root, "format_version", "model_spec", 2,
                           &info->model_spec_format_version,
                           error, error_size)) return 0;
    const h3_json_value *base = h3_vdn_field(
        root, "base", "model_spec", error, error_size);
    if (!base ||
        !h3_vdn_expect_string(base, "library", "model_spec.base", "diffusers",
                              error, error_size) ||
        !h3_vdn_expect_string(base, "class_name", "model_spec.base",
                              "MiniMaxH3Transformer3DModel",
                              error, error_size)) return 0;

    const h3_json_value *transforms = h3_vdn_field(
        root, "transforms", "model_spec", error, error_size);
    if (!transforms || h3_json_get_type(transforms) != H3_JSON_ARRAY ||
        h3_json_size(transforms) != 1)
        return h3_vdn_fail(error, error_size,
                           "model_spec.transforms must contain one transform");
    if (!h3_vdn_parse_transform(h3_json_at(transforms, 0), &info->transform,
                                error, error_size)) return 0;

    const h3_json_value *adapters = h3_vdn_field(
        root, "adapters", "model_spec", error, error_size);
    if (!adapters || h3_json_get_type(adapters) != H3_JSON_ARRAY ||
        !h3_json_size(adapters) ||
        h3_json_size(adapters) > H3_VDN_MAX_ADAPTERS)
        return h3_vdn_fail(error, error_size,
                           "model_spec.adapters must contain one or two adapters");
    info->adapter_count = h3_json_size(adapters);
    for (size_t index = 0; index < info->adapter_count; index++) {
        if (!h3_vdn_parse_adapter(h3_json_at(adapters, index), index,
                                  &info->adapters[index], error, error_size))
            return 0;
    }
    if (strcmp(info->adapters[0].name, "default"))
        return h3_vdn_fail(error, error_size,
                           "first adapter must be default");
    if (info->adapter_count == 2 && strcmp(info->adapters[1].name, "turbo"))
        return h3_vdn_fail(error, error_size,
                           "second adapter must be turbo");
    return 1;
}

static int h3_vdn_parse_metadata(const h3_json_value *root,
                                 h3_vdn_info *info,
                                 char *error, size_t error_size) {
    if (!h3_vdn_expect_int(root, "checkpoint_format_version", "metadata", 2,
                           &info->checkpoint_format_version,
                           error, error_size) ||
        !h3_vdn_expect_string(root, "kind", "metadata", "weights",
                              error, error_size) ||
        !h3_vdn_expect_string(root, "weights_dtype", "metadata", "bfloat16",
                              error, error_size)) return 0;
    info->video_shift = 12.0;
    info->audio_shift = 3.0;
    info->num_steps = info->adapter_count == 2 ? 8 : 50;
    const h3_json_value *detail = h3_json_get(root, "metadata");
    if (info->adapter_count == 2) {
        if (!detail ||
            !h3_vdn_expect_int(detail, "turbo_num_steps", "metadata.metadata",
                               8, &info->num_steps, error, error_size) ||
            !h3_vdn_number(detail, "video_shift", "metadata.metadata",
                           &info->video_shift, error, error_size) ||
            !h3_vdn_number(detail, "audio_shift", "metadata.metadata",
                           &info->audio_shift, error, error_size) ||
            info->video_shift != 12.0 || info->audio_shift != 3.0)
            return h3_vdn_fail(error, error_size,
                               "unsupported DMD scheduler metadata");
    }
    return 1;
}

static int h3_vdn_validate_index(const char *directory,
                                 const h3_component_info *component,
                                 char *error, size_t error_size) {
    static const char index_name[] =
        "diffusion_pytorch_model.safetensors.index.json";
    char *path = h3_vdn_path(directory, index_name);
    if (!path)
        return h3_vdn_fail(error, error_size,
                           "out of memory resolving safetensors index");
    struct stat status;
    if (stat(path, &status)) {
        int stat_error = errno;
        if (stat_error != ENOENT) {
            int result = h3_vdn_fail(error, error_size,
                                     "cannot inspect %s: %s", path,
                                     strerror(stat_error));
            free(path);
            return result;
        }
        free(path);
        return 1; /* Single-file components have no index. */
    }
    if (!S_ISREG(status.st_mode)) {
        int result = h3_vdn_fail(error, error_size,
                                 "%s is not a regular file", path);
        free(path);
        return result;
    }
    h3_json_value *root = h3_json_parse_file(path, error, error_size);
    if (!root) {
        free(path);
        return 0;
    }
    const h3_json_value *weight_map = h3_json_get(root, "weight_map");
    const h3_json_value *metadata = h3_json_get(root, "metadata");
    const h3_json_value *total_value = metadata ?
        h3_json_get(metadata, "total_size") : NULL;
    int64_t total_size = 0;
    if (h3_json_get_type(root) != H3_JSON_OBJECT ||
        h3_json_get_type(weight_map) != H3_JSON_OBJECT ||
        !h3_json_size(weight_map) ||
        !h3_json_i64_value(total_value, &total_size) || total_size < 0) {
        h3_json_free(root);
        int result = h3_vdn_fail(error, error_size,
                                 "%s has an invalid weight_map/total_size",
                                 path);
        free(path);
        return result;
    }

    size_t tensor_count = h3_json_size(weight_map);
    const char **shards = calloc(tensor_count, sizeof(*shards));
    if (!shards) {
        h3_json_free(root);
        free(path);
        return h3_vdn_fail(error, error_size,
                           "out of memory validating safetensors index");
    }
    size_t shard_count = 0;
    int ok = 1;
    for (size_t index = 0; index < tensor_count && ok; index++) {
        const char *key = h3_json_object_key(weight_map, index);
        const char *shard = h3_json_string_value(
            h3_json_object_value(weight_map, index));
        if (!key || !*key || !shard || !*shard || strchr(shard, '/')) {
            h3_vdn_fail(error, error_size,
                        "%s weight_map entry %zu is invalid", path, index);
            ok = 0;
            break;
        }
        size_t found = 0;
        while (found < shard_count && strcmp(shards[found], shard)) found++;
        if (found == shard_count) shards[shard_count++] = shard;
    }
    for (size_t index = 0; index < shard_count && ok; index++) {
        char *shard_path = h3_vdn_path(directory, shards[index]);
        struct stat shard_status;
        if (!shard_path || stat(shard_path, &shard_status) ||
            !S_ISREG(shard_status.st_mode)) {
            h3_vdn_fail(error, error_size,
                        "%s references missing shard %s", path,
                        shards[index]);
            ok = 0;
        }
        free(shard_path);
    }
    if (ok && component->files != shard_count) {
        h3_vdn_fail(error, error_size,
                    "%s references %zu shards but directory contains %zu",
                    path, shard_count, component->files);
        ok = 0;
    }
    if (ok && component->tensors != tensor_count) {
        h3_vdn_fail(error, error_size,
                    "%s maps %zu tensors but shards contain %zu", path,
                    tensor_count, component->tensors);
        ok = 0;
    }
    if (ok && component->tensor_bytes != (uint64_t)total_size) {
        h3_vdn_fail(error, error_size,
                    "%s total_size is %lld but payload is %llu", path,
                    (long long)total_size,
                    (unsigned long long)component->tensor_bytes);
        ok = 0;
    }
    free(shards);
    h3_json_free(root);
    free(path);
    return ok;
}

static int h3_vdn_inventory(const char *root, const char *relative,
                            int require_weights, h3_component_info *component,
                            int *all_present,
                            char *error, size_t error_size) {
    char *path = h3_vdn_path(root, relative);
    if (!path)
        return h3_vdn_fail(error, error_size, "out of memory resolving %s",
                           relative);
    char detail[512];
    int ok = h3_st_inventory_dir(path, component, detail, sizeof(detail));
    if (ok)
        ok = h3_vdn_validate_index(path, component, detail, sizeof(detail));
    free(path);
    if (ok) return 1;
    *all_present = 0;
    memset(component, 0, sizeof(*component));
    if (require_weights)
        return h3_vdn_fail(error, error_size, "%s", detail);
    if (error && error_size) error[0] = '\0';
    return 1;
}

static int h3_vdn_expect_tensor(const h3_st_catalog *catalog,
                                const char *name, h3_dtype dtype, int ndim,
                                const uint64_t *shape,
                                char *error, size_t error_size) {
    const h3_st_tensor *tensor = h3_st_catalog_find(catalog, name);
    if (!tensor)
        return h3_vdn_fail(error, error_size,
                           "missing required tensor: %s", name);
    if (tensor->dtype != dtype || tensor->ndim != ndim)
        return h3_vdn_fail(error, error_size,
                           "%s has %s rank %d; expected %s rank %d", name,
                           h3_dtype_name(tensor->dtype), tensor->ndim,
                           h3_dtype_name(dtype), ndim);
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension])
            return h3_vdn_fail(
                error, error_size,
                "%s shape mismatch at dimension %d: %llu != %llu", name,
                dimension, (unsigned long long)tensor->shape[dimension],
                (unsigned long long)shape[dimension]);
    }
    return 1;
}

#define H3_VDN_EXPECT(catalog, name, dtype, ...)                               \
    h3_vdn_expect_tensor((catalog), (name), (dtype),                           \
        (int)(sizeof((uint64_t[]){__VA_ARGS__}) / sizeof(uint64_t)),           \
        (uint64_t[]){__VA_ARGS__}, error, error_size)

static int h3_vdn_validate_base_transformer(const char *directory,
                                             char *error,
                                             size_t error_size) {
    h3_st_catalog *catalog = h3_st_catalog_open(directory, error, error_size);
    if (!catalog) return 0;
    int ok = h3_st_catalog_tensor_count(catalog) == 638;
    if (!ok)
        h3_vdn_fail(error, error_size,
                    "base transformer contains %zu tensors; expected 638",
                    h3_st_catalog_tensor_count(catalog));
#define EXPECT(name, dtype, ...)                                               \
    do { if (ok && !H3_VDN_EXPECT(catalog, name, dtype, __VA_ARGS__))          \
        ok = 0; } while (0)
    EXPECT("proj_in.weight", H3_DTYPE_F32, 5376, 96);
    EXPECT("proj_in.bias", H3_DTYPE_F32, 5376);
    EXPECT("audio_proj_in.weight", H3_DTYPE_F32, 5376, 32);
    EXPECT("audio_proj_in.bias", H3_DTYPE_F32, 5376);
    EXPECT("context_embedder.weight", H3_DTYPE_BF16, 5376, 5120);
    EXPECT("context_embedder.bias", H3_DTYPE_BF16, 5376);
    EXPECT("time_embedder.linear_1.weight", H3_DTYPE_F32, 5376, 256);
    EXPECT("time_embedder.linear_1.bias", H3_DTYPE_F32, 5376);
    EXPECT("time_embedder.linear_2.weight", H3_DTYPE_F32, 2688, 5376);
    EXPECT("time_embedder.linear_2.bias", H3_DTYPE_F32, 2688);
    EXPECT("token_refiner.final_norm.weight", H3_DTYPE_BF16, 5376);
    EXPECT("norm_out.norm.weight", H3_DTYPE_BF16, 5376);
    EXPECT("norm_out.linear.weight", H3_DTYPE_BF16, 10752, 2688);
    EXPECT("norm_out.linear.bias", H3_DTYPE_BF16, 10752);
    EXPECT("proj_out.weight", H3_DTYPE_F32, 96, 5376);
    EXPECT("proj_out.bias", H3_DTYPE_F32, 96);
    EXPECT("audio_proj_out.weight", H3_DTYPE_F32, 32, 5376);
    EXPECT("audio_proj_out.bias", H3_DTYPE_F32, 32);
    char name[192];
    for (int block = 0; block < 50 && ok; block++) {
#define BLOCK_EXPECT(suffix, dtype, ...)                                       \
        do {                                                                   \
            snprintf(name, sizeof(name), "transformer_blocks.%d.%s",         \
                     block, suffix);                                           \
            if (!H3_VDN_EXPECT(catalog, name, dtype, __VA_ARGS__)) ok = 0;     \
        } while (0)
        BLOCK_EXPECT("norm1.weight", H3_DTYPE_BF16, 5376);
        BLOCK_EXPECT("norm2.weight", H3_DTYPE_BF16, 5376);
        BLOCK_EXPECT("attn.to_q.weight", H3_DTYPE_BF16, 7168, 5376);
        BLOCK_EXPECT("attn.to_k.weight", H3_DTYPE_BF16, 7168, 5376);
        BLOCK_EXPECT("attn.to_v.weight", H3_DTYPE_BF16, 7168, 5376);
        BLOCK_EXPECT("attn.norm_q.weight", H3_DTYPE_BF16, 128);
        BLOCK_EXPECT("attn.norm_k.weight", H3_DTYPE_BF16, 128);
        BLOCK_EXPECT("attn.to_out.0.weight", H3_DTYPE_BF16, 5376, 7168);
        BLOCK_EXPECT("ff.net.0.proj.weight", H3_DTYPE_BF16, 28672, 5376);
        BLOCK_EXPECT("ff.net.2.weight", H3_DTYPE_BF16, 5376, 14336);
        BLOCK_EXPECT("adaln_proj.linear.weight", H3_DTYPE_BF16, 96768, 2688);
        BLOCK_EXPECT("adaln_proj.linear.bias", H3_DTYPE_BF16, 96768);
#undef BLOCK_EXPECT
    }
    for (int block = 0; block < 2 && ok; block++) {
#define REFINER_EXPECT(suffix, dtype, ...)                                     \
        do {                                                                   \
            snprintf(name, sizeof(name),                                      \
                     "token_refiner.refiner_blocks.%d.%s", block, suffix);   \
            if (!H3_VDN_EXPECT(catalog, name, dtype, __VA_ARGS__)) ok = 0;     \
        } while (0)
        REFINER_EXPECT("norm1.weight", H3_DTYPE_BF16, 5376);
        REFINER_EXPECT("norm2.weight", H3_DTYPE_BF16, 5376);
        REFINER_EXPECT("attn.to_q.weight", H3_DTYPE_BF16, 7168, 5376);
        REFINER_EXPECT("attn.to_k.weight", H3_DTYPE_BF16, 7168, 5376);
        REFINER_EXPECT("attn.to_v.weight", H3_DTYPE_BF16, 7168, 5376);
        REFINER_EXPECT("attn.norm_q.weight", H3_DTYPE_BF16, 128);
        REFINER_EXPECT("attn.norm_k.weight", H3_DTYPE_BF16, 128);
        REFINER_EXPECT("attn.to_out.0.weight", H3_DTYPE_BF16, 5376, 7168);
        REFINER_EXPECT("ff.net.0.proj.weight", H3_DTYPE_BF16, 28672, 5376);
        REFINER_EXPECT("ff.net.2.weight", H3_DTYPE_BF16, 5376, 14336);
#undef REFINER_EXPECT
    }
#undef EXPECT
    h3_st_catalog_free(catalog);
    return ok;
}

static int h3_vdn_validate_linear_branch(const char *directory,
                                          char *error, size_t error_size) {
    h3_st_catalog *catalog = h3_st_catalog_open(directory, error, error_size);
    if (!catalog) return 0;
    int ok = h3_st_catalog_tensor_count(catalog) == 800;
    if (!ok)
        h3_vdn_fail(error, error_size,
                    "linear branch contains %zu tensors; expected 800",
                    h3_st_catalog_tensor_count(catalog));
    char name[224];
    for (int block = 0; block < 50 && ok; block++) {
#define LINEAR_EXPECT(suffix, ...)                                             \
        do {                                                                   \
            snprintf(name, sizeof(name),                                      \
                     "transformer_blocks.%d.attn.%s", block, suffix);        \
            if (!H3_VDN_EXPECT(catalog, name, H3_DTYPE_BF16, __VA_ARGS__))    \
                ok = 0;                                                        \
        } while (0)
        LINEAR_EXPECT("linear_attention.alpha.A_log", 56);
        LINEAR_EXPECT("linear_attention.alpha.down.weight", 128, 5376);
        LINEAR_EXPECT("linear_attention.alpha.dt_bias", 7168);
        LINEAR_EXPECT("linear_attention.alpha.up.weight", 7168, 128);
        LINEAR_EXPECT("linear_attention.beta_proj.weight", 56, 5376);
        LINEAR_EXPECT("linear_attention.norm.weight", 128);
        LINEAR_EXPECT("linear_attention.output_gate.down.weight", 128, 5376);
        LINEAR_EXPECT("linear_attention.output_gate.up.bias", 7168);
        LINEAR_EXPECT("linear_attention.output_gate.up.weight", 7168, 128);
        LINEAR_EXPECT("linear_attention.short_conv.k_sp.weight", 7168, 1, 5, 5);
        LINEAR_EXPECT("linear_attention.short_conv.k_tm.weight", 7168, 1, 5);
        LINEAR_EXPECT("linear_attention.short_conv.v_sp.weight", 7168, 1, 5, 5);
        LINEAR_EXPECT("linear_attention.short_conv.v_tm.weight", 7168, 1, 5);
        LINEAR_EXPECT("softmax_gate.up.bias", 56);
        LINEAR_EXPECT("softmax_gate.up.weight", 56, 5376);
        LINEAR_EXPECT("to_out_linear.weight", 5376, 7168);
#undef LINEAR_EXPECT
    }
    h3_st_catalog_free(catalog);
    return ok;
}

static int h3_vdn_expect_lora(const h3_st_catalog *catalog,
                              const char *target, const char *adapter,
                              uint64_t rank, uint64_t input,
                              uint64_t output,
                              char *error, size_t error_size) {
    char name[256];
    snprintf(name, sizeof(name), "%s.lora_A.%s.weight", target, adapter);
    if (!H3_VDN_EXPECT(catalog, name, H3_DTYPE_BF16, rank, input)) return 0;
    snprintf(name, sizeof(name), "%s.lora_B.%s.weight", target, adapter);
    return H3_VDN_EXPECT(catalog, name, H3_DTYPE_BF16, output, rank);
}

static int h3_vdn_validate_adapter(const char *directory,
                                   const char *adapter,
                                   char *error, size_t error_size) {
    h3_st_catalog *catalog = h3_st_catalog_open(directory, error, error_size);
    if (!catalog) return 0;
    size_t expected_count = !strcmp(adapter, "default") ? 416 : 726;
    int ok = h3_st_catalog_tensor_count(catalog) == expected_count;
    if (!ok)
        h3_vdn_fail(error, error_size,
                    "%s adapter contains %zu tensors; expected %zu", adapter,
                    h3_st_catalog_tensor_count(catalog), expected_count);
    char target[224];
    if (!strcmp(adapter, "default")) {
        for (int block = 0; block < 50 && ok; block++) {
            static const char *const names[] = {
                "to_q", "to_k", "to_v", "to_out.0"
            };
            for (size_t index = 0; index < 4 && ok; index++) {
                snprintf(target, sizeof(target),
                         "transformer_blocks.%d.attn.orig.%s", block,
                         names[index]);
                uint64_t input = index == 3 ? 7168 : 5376;
                uint64_t output = index == 3 ? 5376 : 7168;
                ok = h3_vdn_expect_lora(catalog, target, adapter, 64, input,
                                        output, error, error_size);
            }
        }
        for (int block = 0; block < 2 && ok; block++) {
            static const char *const names[] = {
                "to_q", "to_k", "to_v", "to_out.0"
            };
            for (size_t index = 0; index < 4 && ok; index++) {
                snprintf(target, sizeof(target),
                         "token_refiner.refiner_blocks.%d.attn.%s", block,
                         names[index]);
                uint64_t input = index == 3 ? 7168 : 5376;
                uint64_t output = index == 3 ? 5376 : 7168;
                ok = h3_vdn_expect_lora(catalog, target, adapter, 64, input,
                                        output, error, error_size);
            }
        }
    } else {
        ok = h3_vdn_expect_lora(catalog, "norm_out.linear", adapter, 16,
                                2688, 10752, error, error_size);
        for (int block = 0; block < 50 && ok; block++) {
            snprintf(target, sizeof(target),
                     "transformer_blocks.%d.adaln_proj.linear", block);
            ok = h3_vdn_expect_lora(catalog, target, adapter, 16, 2688,
                                    96768, error, error_size);
            static const char *const names[] = {
                "to_q", "to_k", "to_v", "to_out.0"
            };
            for (size_t index = 0; index < 4 && ok; index++) {
                snprintf(target, sizeof(target),
                         "transformer_blocks.%d.attn.orig.%s", block,
                         names[index]);
                uint64_t input = index == 3 ? 7168 : 5376;
                uint64_t output = index == 3 ? 5376 : 7168;
                ok = h3_vdn_expect_lora(catalog, target, adapter, 64, input,
                                        output, error, error_size);
            }
            if (ok) {
                snprintf(target, sizeof(target),
                         "transformer_blocks.%d.ff.net.0.proj", block);
                ok = h3_vdn_expect_lora(catalog, target, adapter, 64, 5376,
                                        28672, error, error_size);
            }
            if (ok) {
                snprintf(target, sizeof(target),
                         "transformer_blocks.%d.ff.net.2", block);
                ok = h3_vdn_expect_lora(catalog, target, adapter, 64, 14336,
                                        5376, error, error_size);
            }
        }
        for (int block = 0; block < 2 && ok; block++) {
            static const char *const names[] = {
                "to_q", "to_k", "to_v", "to_out.0"
            };
            for (size_t index = 0; index < 4 && ok; index++) {
                snprintf(target, sizeof(target),
                         "token_refiner.refiner_blocks.%d.attn.%s", block,
                         names[index]);
                uint64_t input = index == 3 ? 7168 : 5376;
                uint64_t output = index == 3 ? 5376 : 7168;
                ok = h3_vdn_expect_lora(catalog, target, adapter, 64, input,
                                        output, error, error_size);
            }
            if (ok) {
                snprintf(target, sizeof(target),
                         "token_refiner.refiner_blocks.%d.ff.net.0.proj", block);
                ok = h3_vdn_expect_lora(catalog, target, adapter, 64, 5376,
                                        28672, error, error_size);
            }
            if (ok) {
                snprintf(target, sizeof(target),
                         "token_refiner.refiner_blocks.%d.ff.net.2", block);
                ok = h3_vdn_expect_lora(catalog, target, adapter, 64, 14336,
                                        5376, error, error_size);
            }
        }
    }
    h3_st_catalog_free(catalog);
    return ok;
}

static int h3_vdn_validate_weight_schema(const char *base_model_dir,
                                         const char *checkpoint_dir,
                                         const h3_vdn_info *info,
                                         char *error, size_t error_size) {
    char *base = h3_vdn_path(base_model_dir, "transformer");
    char *linear = h3_vdn_path(checkpoint_dir, "linear_branch");
    char *default_adapter = h3_vdn_path(
        checkpoint_dir, "adapters/default");
    char *turbo_adapter = info->adapter_count == 2 ?
        h3_vdn_path(checkpoint_dir, "adapters/turbo") : NULL;
    if (!base || !linear || !default_adapter ||
        (info->adapter_count == 2 && !turbo_adapter)) {
        free(turbo_adapter); free(default_adapter); free(linear); free(base);
        return h3_vdn_fail(error, error_size,
                           "out of memory resolving VDN schema paths");
    }
    int ok = h3_vdn_validate_base_transformer(base, error, error_size) &&
             h3_vdn_validate_linear_branch(linear, error, error_size) &&
             h3_vdn_validate_adapter(default_adapter, "default",
                                     error, error_size) &&
             (info->adapter_count != 2 ||
              h3_vdn_validate_adapter(turbo_adapter, "turbo",
                                      error, error_size));
    free(turbo_adapter); free(default_adapter); free(linear); free(base);
    return ok;
}

#undef H3_VDN_EXPECT

static const char *h3_vdn_basename(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash && slash[1] ? slash + 1 : path;
}

static int h3_vdn_read_revision(const char *base_model_dir,
                                h3_vdn_info *info,
                                char *error, size_t error_size) {
    static const char marker_name[] = ".h3-vdn-revision";
    size_t length = strlen(base_model_dir);
    while (length > 1 && base_model_dir[length - 1] == '/') length--;
    char *base = malloc(length + 1);
    if (!base)
        return h3_vdn_fail(error, error_size,
                           "out of memory resolving model revision marker");
    memcpy(base, base_model_dir, length);
    base[length] = '\0';
    char *slash = strrchr(base, '/');
    const char *parent = ".";
    if (slash) {
        if (slash == base)
            slash[1] = '\0';
        else
            *slash = '\0';
        parent = base;
    }
    char *path = h3_vdn_path(parent, marker_name);
    free(base);
    if (!path)
        return h3_vdn_fail(error, error_size,
                           "out of memory resolving model revision marker");

    FILE *file = fopen(path, "rb");
    if (!file) {
        int open_error = errno;
        if (open_error == ENOENT) {
            free(path);
            return 1; /* Models obtained outside our downloader have no marker. */
        }
        int result = h3_vdn_fail(error, error_size,
                                 "cannot read %s: %s", path,
                                 strerror(open_error));
        free(path);
        return result;
    }

    char line[512];
    int found = 0;
    int ok = 1;
    while (fgets(line, sizeof(line), file)) {
        size_t line_length = strlen(line);
        if (line_length && line[line_length - 1] != '\n' && !feof(file)) {
            h3_vdn_fail(error, error_size, "%s contains an overlong line", path);
            ok = 0;
            break;
        }
        while (line_length &&
               (line[line_length - 1] == '\n' ||
                line[line_length - 1] == '\r'))
            line[--line_length] = '\0';
        static const char prefix[] = "revision=";
        if (strncmp(line, prefix, sizeof(prefix) - 1)) continue;
        const char *revision = line + sizeof(prefix) - 1;
        size_t revision_length = strlen(revision);
        if (found || revision_length != 40) {
            h3_vdn_fail(error, error_size,
                        "%s must contain one 40-character revision", path);
            ok = 0;
            break;
        }
        for (size_t index = 0; index < revision_length; index++) {
            if (!isxdigit((unsigned char)revision[index])) {
                h3_vdn_fail(error, error_size,
                            "%s contains a non-hex revision", path);
                ok = 0;
                break;
            }
        }
        if (!ok) break;
        snprintf(info->model_revision, sizeof(info->model_revision),
                 "%s", revision);
        found = 1;
    }
    if (ok && ferror(file))
        ok = h3_vdn_fail(error, error_size,
                         "failed reading %s", path);
    if (ok && !found)
        ok = h3_vdn_fail(error, error_size,
                         "%s does not contain revision=...", path);
    if (fclose(file) && ok)
        ok = h3_vdn_fail(error, error_size,
                         "failed closing %s", path);
    free(path);
    return ok;
}

int h3_vdn_inspect(const char *base_model_dir,
                   const char *checkpoint_dir,
                   int require_weights,
                   h3_vdn_info *info,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!base_model_dir || !*base_model_dir ||
        !checkpoint_dir || !*checkpoint_dir || !info)
        return h3_vdn_fail(error, error_size,
                           "base model, VDN checkpoint and output are required");
    memset(info, 0, sizeof(*info));
    const char *name = h3_vdn_basename(checkpoint_dir);
    if (strlen(name) >= sizeof(info->checkpoint_name))
        return h3_vdn_fail(error, error_size,
                           "VDN checkpoint directory name is too long");
    snprintf(info->checkpoint_name, sizeof(info->checkpoint_name), "%s", name);
    if (!h3_vdn_read_revision(base_model_dir, info, error, error_size)) {
        memset(info, 0, sizeof(*info));
        return 0;
    }

    h3_json_value *base = h3_vdn_read_json(
        base_model_dir, "transformer/config.json", error, error_size);
    h3_json_value *spec = h3_vdn_read_json(
        checkpoint_dir, "model_spec.json", error, error_size);
    h3_json_value *linear = h3_vdn_read_json(
        checkpoint_dir, "linear_branch/config.json", error, error_size);
    h3_json_value *metadata = h3_vdn_read_json(
        checkpoint_dir, "metadata.json", error, error_size);
    h3_json_value *default_adapter_config = h3_vdn_read_json(
        checkpoint_dir, "adapters/default/adapter_config.json",
        error, error_size);
    h3_json_value *turbo_adapter_config = NULL;
    if (!base || !spec || !linear || !metadata || !default_adapter_config)
        goto failed;

    h3_vdn_transform_info linear_transform;
    if (!h3_vdn_parse_base(base, info, error, error_size) ||
        !h3_vdn_parse_spec(spec, info, error, error_size)) goto failed;
    const h3_json_value *adapter_specs = h3_json_get(spec, "adapters");
    if (!h3_vdn_json_equal(h3_json_at(adapter_specs, 0),
                           default_adapter_config)) {
        h3_vdn_fail(error, error_size,
                    "default adapter_config.json does not match model_spec.json");
        goto failed;
    }
    if (info->adapter_count == 2) {
        turbo_adapter_config = h3_vdn_read_json(
            checkpoint_dir, "adapters/turbo/adapter_config.json",
            error, error_size);
        if (!turbo_adapter_config) goto failed;
        if (!h3_vdn_json_equal(h3_json_at(adapter_specs, 1),
                               turbo_adapter_config)) {
            h3_vdn_fail(error, error_size,
                        "turbo adapter_config.json does not match model_spec.json");
            goto failed;
        }
    }
    if (
        !h3_vdn_parse_transform(linear, &linear_transform,
                                error, error_size) ||
        !h3_vdn_same_transform(&info->transform, &linear_transform) ||
        !h3_vdn_parse_metadata(metadata, info, error, error_size)) {
        if (error && error_size && !error[0])
            h3_vdn_fail(error, error_size,
                        "linear_branch config does not match model_spec");
        goto failed;
    }

    int all_present = 1;
    if (!h3_vdn_inventory(base_model_dir, "transformer", require_weights,
                          &info->base_transformer, &all_present,
                          error, error_size) ||
        !h3_vdn_inventory(base_model_dir, "vae", require_weights,
                          &info->video_vae, &all_present,
                          error, error_size) ||
        !h3_vdn_inventory(base_model_dir, "audio_vae", require_weights,
                          &info->audio_vae, &all_present,
                          error, error_size) ||
        !h3_vdn_inventory(checkpoint_dir, "linear_branch", require_weights,
                          &info->linear_branch, &all_present,
                          error, error_size) ||
        !h3_vdn_inventory(checkpoint_dir, "adapters/default", require_weights,
                          &info->default_adapter, &all_present,
                          error, error_size)) goto failed;
    if (info->adapter_count == 2 &&
        !h3_vdn_inventory(checkpoint_dir, "adapters/turbo", require_weights,
                          &info->turbo_adapter, &all_present,
                          error, error_size)) goto failed;
    if (all_present && !h3_vdn_validate_weight_schema(
            base_model_dir, checkpoint_dir, info, error, error_size))
        goto failed;

    info->enabled = 1;
    info->weights_present = all_present;
    h3_json_free(turbo_adapter_config);
    h3_json_free(default_adapter_config);
    h3_json_free(metadata);
    h3_json_free(linear);
    h3_json_free(spec);
    h3_json_free(base);
    return 1;

failed:
    h3_json_free(turbo_adapter_config);
    h3_json_free(default_adapter_config);
    h3_json_free(metadata);
    h3_json_free(linear);
    h3_json_free(spec);
    h3_json_free(base);
    memset(info, 0, sizeof(*info));
    return 0;
}
