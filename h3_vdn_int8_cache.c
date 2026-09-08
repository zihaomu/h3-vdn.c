#include "h3_vdn_int8_cache.h"

#include "h3.h"
#include "h3_json.h"
#include "h3_safetensors.h"
#include "h3_sha256.h"
#include "h3_weights.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    CACHE_VERSION = 1,
    ADALN_OUT = 96768,
    ADALN_IN = 2688,
    FC1_OUT = 28672,
    FC1_IN = 5376,
    FC2_OUT = 5376,
    FC2_IN = 14336,
    TENSORS_PER_BLOCK = 6
};

static const char cache_format[] = "h3-vdn-int8-weight-cache";
static const char quantization[] = "symmetric-per-output-rne-i8-v1";

struct h3_vdn_int8_cache {
    char *directory;
    h3_weight_store *weights;
};

typedef struct {
    const char *field;
    h3_dtype dtype;
    uint64_t rows;
    uint64_t columns;
} tensor_schema;

static const tensor_schema schemas[TENSORS_PER_BLOCK] = {
    {"adaln.weight", H3_DTYPE_I8, ADALN_OUT, ADALN_IN},
    {"adaln.scales", H3_DTYPE_F32, ADALN_OUT, 0},
    {"fc1.weight", H3_DTYPE_I8, FC1_OUT, FC1_IN},
    {"fc1.scales", H3_DTYPE_F32, FC1_OUT, 0},
    {"fc2.weight", H3_DTYPE_I8, FC2_OUT, FC2_IN},
    {"fc2.scales", H3_DTYPE_F32, FC2_OUT, 0}
};

static int cache_fail(char *error, size_t error_size,
                      const char *format, ...) {
    if (error && error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

static int path_join(char *output, size_t output_size, const char *directory,
                     const char *name) {
    if (!output || !output_size || !directory || !*directory ||
        !name || !*name) return 0;
    int length = snprintf(output, output_size, "%s/%s", directory, name);
    return length >= 0 && (size_t)length < output_size;
}

static int block_filename(char output[32], unsigned block) {
    if (block >= H3_VDN_INT8_CACHE_BLOCKS) return 0;
    int length = snprintf(output, 32, "block-%02u.safetensors", block);
    return length > 0 && length < 32;
}

static int tensor_name(char output[96], unsigned block, const char *field) {
    if (block >= H3_VDN_INT8_CACHE_BLOCKS || !field) return 0;
    int length = snprintf(output, 96, "transformer_blocks.%u.%s",
                          block, field);
    return length > 0 && length < 96;
}

static int exact_string(const h3_json_value *object, const char *key,
                        const char *expected) {
    const char *value = h3_json_string_value(h3_json_get(object, key));
    return value && !strcmp(value, expected);
}

static int exact_i64(const h3_json_value *object, const char *key,
                     int64_t expected) {
    int64_t value;
    return h3_json_i64_value(h3_json_get(object, key), &value) &&
           value == expected;
}

static int hex_digit(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    return -1;
}

static int parse_digest(const char *text, uint8_t digest[32]) {
    if (!text || strlen(text) != 64) return 0;
    for (unsigned index = 0; index < 32; index++) {
        int high = hex_digit(text[index * 2]);
        int low = hex_digit(text[index * 2 + 1]);
        if (high < 0 || low < 0) return 0;
        digest[index] = (uint8_t)(high * 16 + low);
    }
    return 1;
}

int h3_vdn_int8_cache_read_manifest(
        const char *directory, int use_turbo,
        const uint8_t source_sha256[H3_VDN_INT8_CACHE_DIGEST_BYTES],
        h3_vdn_int8_cache_manifest *manifest,
        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!directory || !*directory || (use_turbo != 0 && use_turbo != 1) ||
        !source_sha256 || !manifest)
        return cache_fail(error, error_size,
                          "invalid VDN INT8 cache manifest arguments");
    memset(manifest, 0, sizeof(*manifest));
    char path[4096];
    if (!path_join(path, sizeof(path), directory, "manifest.json"))
        return cache_fail(error, error_size,
                          "VDN INT8 cache manifest path is too long");
    h3_json_value *root = h3_json_parse_file(path, error, error_size);
    if (!root) return 0;
    int ok = h3_json_get_type(root) == H3_JSON_OBJECT &&
        h3_json_size(root) == 8 &&
        exact_string(root, "format", cache_format) &&
        exact_i64(root, "version", CACHE_VERSION) &&
        exact_string(root, "quantization", quantization) &&
        exact_i64(root, "blocks", H3_VDN_INT8_CACHE_BLOCKS);
    int manifest_turbo = -1;
    const char *producer = h3_json_string_value(h3_json_get(root, "producer"));
    const char *source_text = h3_json_string_value(
        h3_json_get(root, "source_sha256"));
    uint8_t manifest_source[32];
    const h3_json_value *files = h3_json_get(root, "files");
    ok = ok && producer && *producer &&
        h3_json_boolean_value(h3_json_get(root, "use_turbo"),
                              &manifest_turbo) &&
        manifest_turbo == use_turbo &&
        parse_digest(source_text, manifest_source) &&
        !memcmp(source_sha256, manifest_source, sizeof(manifest_source)) &&
        h3_json_get_type(files) == H3_JSON_ARRAY &&
        h3_json_size(files) == H3_VDN_INT8_CACHE_BLOCKS;
    for (unsigned block = 0; ok && block < H3_VDN_INT8_CACHE_BLOCKS; block++) {
        const h3_json_value *entry = h3_json_at(files, block);
        char expected_path[32];
        int64_t entry_block = -1, entry_bytes = -1;
        uint8_t digest[32];
        const char *entry_path = entry ? h3_json_string_value(
            h3_json_get(entry, "path")) : NULL;
        const char *entry_sha = entry ? h3_json_string_value(
            h3_json_get(entry, "sha256")) : NULL;
        ok = entry && h3_json_get_type(entry) == H3_JSON_OBJECT &&
            h3_json_size(entry) == 4 && block_filename(expected_path, block) &&
            h3_json_i64_value(h3_json_get(entry, "block"), &entry_block) &&
            entry_block == (int64_t)block && entry_path &&
            !strcmp(entry_path, expected_path) &&
            h3_json_i64_value(h3_json_get(entry, "bytes"), &entry_bytes) &&
            entry_bytes > 0 && parse_digest(entry_sha, digest);
        if (ok) {
            manifest->block_bytes[block] = (uint64_t)entry_bytes;
            memcpy(manifest->block_sha256[block], digest, sizeof(digest));
        }
    }
    if (!ok)
        cache_fail(error, error_size,
                   "VDN INT8 cache manifest does not match v1/source/mode");
    h3_json_free(root);
    return ok;
}

static int validate_tensor(const h3_st_catalog *catalog, unsigned block,
                           const tensor_schema *schema, uint64_t data_begin,
                           char *error, size_t error_size) {
    char name[96];
    if (!tensor_name(name, block, schema->field))
        return cache_fail(error, error_size, "INT8 cache tensor name overflow");
    const h3_st_tensor *tensor = h3_st_catalog_find(catalog, name);
    int ndim = schema->columns ? 2 : 1;
    uint64_t elements = schema->rows * (schema->columns ? schema->columns : 1);
    uint64_t bytes = elements * h3_dtype_size(schema->dtype);
    if (!tensor || tensor->dtype != schema->dtype || tensor->ndim != ndim ||
        tensor->shape[0] != schema->rows ||
        (schema->columns && tensor->shape[1] != schema->columns) ||
        tensor->data_begin != data_begin ||
        tensor->data_end != data_begin + bytes)
        return cache_fail(error, error_size,
                          "VDN INT8 cache tensor schema mismatch: %s", name);
    return 1;
}

h3_vdn_int8_cache *h3_vdn_int8_cache_open(
        const char *directory, int use_turbo,
        const uint8_t source_sha256[H3_VDN_INT8_CACHE_DIGEST_BYTES],
        char *error, size_t error_size) {
    h3_vdn_int8_cache_manifest manifest;
    if (!h3_vdn_int8_cache_read_manifest(
            directory, use_turbo, source_sha256, &manifest,
            error, error_size)) return NULL;
    for (unsigned block = 0; block < H3_VDN_INT8_CACHE_BLOCKS; block++) {
        char filename[32], path[4096];
        struct stat status;
        uint8_t digest[32];
        if (!block_filename(filename, block) ||
            !path_join(path, sizeof(path), directory, filename) ||
            stat(path, &status) != 0 || !S_ISREG(status.st_mode) ||
            status.st_size < 0 ||
            (uint64_t)status.st_size != manifest.block_bytes[block]) {
            cache_fail(error, error_size,
                       "VDN INT8 cache block %u size/path mismatch", block);
            return NULL;
        }
        if (!h3_sha256_file(path, digest, error, error_size)) return NULL;
        if (memcmp(digest, manifest.block_sha256[block], sizeof(digest))) {
            cache_fail(error, error_size,
                       "VDN INT8 cache block %u SHA-256 mismatch", block);
            return NULL;
        }
    }
    h3_st_catalog *catalog = h3_st_catalog_open(
        directory, error, error_size);
    if (!catalog) return NULL;
    int ok = h3_st_catalog_file_count(catalog) == H3_VDN_INT8_CACHE_BLOCKS &&
             h3_st_catalog_tensor_count(catalog) ==
                 H3_VDN_INT8_CACHE_BLOCKS * TENSORS_PER_BLOCK;
    for (unsigned block = 0; ok && block < H3_VDN_INT8_CACHE_BLOCKS; block++) {
        uint64_t offset = 0;
        for (unsigned index = 0; ok && index < TENSORS_PER_BLOCK; index++) {
            ok = validate_tensor(catalog, block, &schemas[index], offset,
                                 error, error_size);
            uint64_t elements = schemas[index].rows *
                (schemas[index].columns ? schemas[index].columns : 1);
            offset += elements * h3_dtype_size(schemas[index].dtype);
        }
    }
    if (!ok && error && error_size && !error[0])
        cache_fail(error, error_size, "VDN INT8 cache tensor set mismatch");
    h3_st_catalog_free(catalog);
    if (!ok) return NULL;
    h3_vdn_int8_cache *cache = calloc(1, sizeof(*cache));
    if (!cache) {
        cache_fail(error, error_size, "out of memory opening VDN INT8 cache");
        return NULL;
    }
    cache->directory = strdup(directory);
    cache->weights = h3_weight_store_open(directory, error, error_size);
    if (!cache->directory || !cache->weights) {
        h3_vdn_int8_cache_free(cache);
        return NULL;
    }
    return cache;
}

void h3_vdn_int8_cache_free(h3_vdn_int8_cache *cache) {
    if (!cache) return;
    h3_weight_store_free(cache->weights);
    free(cache->directory);
    free(cache);
}

static h3_gpu_tensor *load_i8(const h3_vdn_int8_cache *cache, h3_gpu *gpu,
                              unsigned block, const char *field,
                              uint64_t rows, uint64_t columns,
                              char *error, size_t error_size) {
    char name[96];
    uint64_t shape[2] = {rows, columns};
    if (!tensor_name(name, block, field)) return NULL;
    return h3_weight_load_i8(cache->weights, gpu, name, 2, shape,
                             error, error_size);
}

static h3_gpu_tensor *load_scale(const h3_vdn_int8_cache *cache, h3_gpu *gpu,
                                 unsigned block, const char *field,
                                 uint64_t rows,
                                 char *error, size_t error_size) {
    char name[96];
    uint64_t shape[1] = {rows};
    if (!tensor_name(name, block, field)) return NULL;
    return h3_weight_load_f32(cache->weights, gpu, name, 1, shape,
                              error, error_size);
}

int h3_vdn_int8_cache_load_block(
        const h3_vdn_int8_cache *cache, h3_gpu *gpu, unsigned block,
        h3_vdn_int8_block *weights, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!cache || !gpu || !weights || block >= H3_VDN_INT8_CACHE_BLOCKS)
        return cache_fail(error, error_size,
                          "invalid VDN INT8 cache block load arguments");
    memset(weights, 0, sizeof(*weights));
    weights->adaln_weight = load_i8(cache, gpu, block, "adaln.weight",
                                    ADALN_OUT, ADALN_IN, error, error_size);
    weights->adaln_scales = weights->adaln_weight ?
        load_scale(cache, gpu, block, "adaln.scales", ADALN_OUT,
                   error, error_size) : NULL;
    weights->fc1_weight = weights->adaln_scales ?
        load_i8(cache, gpu, block, "fc1.weight", FC1_OUT, FC1_IN,
                error, error_size) : NULL;
    weights->fc1_scales = weights->fc1_weight ?
        load_scale(cache, gpu, block, "fc1.scales", FC1_OUT,
                   error, error_size) : NULL;
    weights->fc2_weight = weights->fc1_scales ?
        load_i8(cache, gpu, block, "fc2.weight", FC2_OUT, FC2_IN,
                error, error_size) : NULL;
    weights->fc2_scales = weights->fc2_weight ?
        load_scale(cache, gpu, block, "fc2.scales", FC2_OUT,
                   error, error_size) : NULL;
    if (!weights->fc2_scales) {
        h3_vdn_int8_block_free(weights);
        return 0;
    }
    return 1;
}

void h3_vdn_int8_block_free(h3_vdn_int8_block *weights) {
    if (!weights) return;
    h3_gpu_tensor_free(weights->fc2_scales);
    h3_gpu_tensor_free(weights->fc2_weight);
    h3_gpu_tensor_free(weights->fc1_scales);
    h3_gpu_tensor_free(weights->fc1_weight);
    h3_gpu_tensor_free(weights->adaln_scales);
    h3_gpu_tensor_free(weights->adaln_weight);
    memset(weights, 0, sizeof(*weights));
}

static int write_all(int descriptor, const void *opaque, size_t bytes,
                     char *error, size_t error_size) {
    const uint8_t *data = opaque;
    size_t completed = 0;
    while (completed < bytes) {
        ssize_t written = write(descriptor, data + completed,
                                bytes - completed);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0)
            return cache_fail(error, error_size, "cache write failed: %s",
                              written < 0 ? strerror(errno) : "short write");
        completed += (size_t)written;
    }
    return 1;
}

static int write_u64_le(int descriptor, uint64_t value,
                        char *error, size_t error_size) {
    uint8_t bytes[8];
    for (unsigned index = 0; index < 8; index++)
        bytes[index] = (uint8_t)(value >> (index * 8));
    return write_all(descriptor, bytes, sizeof(bytes), error, error_size);
}

static int read_and_write_tensor(int descriptor,
                                 const h3_gpu_tensor *tensor,
                                 char *error, size_t error_size) {
    size_t elements = h3_gpu_tensor_elements(tensor);
    h3_gpu_dtype dtype = h3_gpu_tensor_dtype(tensor);
    size_t item_size = dtype == H3_GPU_I8 ? sizeof(int8_t) : sizeof(float);
    if ((dtype != H3_GPU_I8 && dtype != H3_GPU_F32) ||
        elements > SIZE_MAX / item_size)
        return cache_fail(error, error_size,
                          "invalid tensor while writing VDN INT8 cache");
    size_t bytes = elements * item_size;
    void *host = malloc(bytes ? bytes : 1);
    if (!host)
        return cache_fail(error, error_size,
                          "out of memory staging VDN INT8 cache payload");
    int ok = dtype == H3_GPU_I8 ?
        h3_gpu_tensor_read_i8(tensor, host, elements) :
        h3_gpu_tensor_read_f32(tensor, host, elements);
    if (ok) ok = write_all(descriptor, host, bytes, error, error_size);
    free(host);
    if (!ok && error && error_size && !error[0])
        cache_fail(error, error_size, "cannot read cache tensor from GPU");
    return ok;
}

int h3_vdn_int8_cache_write_block(
        const char *directory, unsigned block,
        const h3_vdn_int8_block *weights, uint64_t *file_bytes,
        uint8_t file_sha256[H3_VDN_INT8_CACHE_DIGEST_BYTES],
        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    const h3_gpu_tensor *tensors[TENSORS_PER_BLOCK] = {
        weights ? weights->adaln_weight : NULL,
        weights ? weights->adaln_scales : NULL,
        weights ? weights->fc1_weight : NULL,
        weights ? weights->fc1_scales : NULL,
        weights ? weights->fc2_weight : NULL,
        weights ? weights->fc2_scales : NULL
    };
    if (!directory || !*directory || block >= H3_VDN_INT8_CACHE_BLOCKS ||
        !weights || !file_bytes || !file_sha256)
        return cache_fail(error, error_size,
                          "invalid VDN INT8 cache block write arguments");
    uint64_t offsets[TENSORS_PER_BLOCK + 1] = {0};
    for (unsigned index = 0; index < TENSORS_PER_BLOCK; index++) {
        h3_gpu_dtype expected = schemas[index].dtype == H3_DTYPE_I8 ?
            H3_GPU_I8 : H3_GPU_F32;
        uint64_t elements = schemas[index].rows *
            (schemas[index].columns ? schemas[index].columns : 1);
        if (!tensors[index] || h3_gpu_tensor_dtype(tensors[index]) != expected ||
            h3_gpu_tensor_elements(tensors[index]) != elements)
            return cache_fail(error, error_size,
                              "VDN INT8 cache block %u tensor %u mismatch",
                              block, index);
        offsets[index + 1] = offsets[index] + elements *
            h3_dtype_size(schemas[index].dtype);
    }
    char names[TENSORS_PER_BLOCK][96];
    for (unsigned index = 0; index < TENSORS_PER_BLOCK; index++)
        if (!tensor_name(names[index], block, schemas[index].field))
            return cache_fail(error, error_size, "cache tensor name overflow");
    char header[8192];
    int length = snprintf(header, sizeof(header),
        "{\"__metadata__\":{\"format\":\"%s\",\"version\":\"1\","
        "\"quantization\":\"%s\",\"block\":\"%u\"},"
        "\"%s\":{\"dtype\":\"I8\",\"shape\":[%u,%u],\"data_offsets\":[%llu,%llu]},"
        "\"%s\":{\"dtype\":\"F32\",\"shape\":[%u],\"data_offsets\":[%llu,%llu]},"
        "\"%s\":{\"dtype\":\"I8\",\"shape\":[%u,%u],\"data_offsets\":[%llu,%llu]},"
        "\"%s\":{\"dtype\":\"F32\",\"shape\":[%u],\"data_offsets\":[%llu,%llu]},"
        "\"%s\":{\"dtype\":\"I8\",\"shape\":[%u,%u],\"data_offsets\":[%llu,%llu]},"
        "\"%s\":{\"dtype\":\"F32\",\"shape\":[%u],\"data_offsets\":[%llu,%llu]}}",
        cache_format, quantization, block,
        names[0], ADALN_OUT, ADALN_IN,
        (unsigned long long)offsets[0], (unsigned long long)offsets[1],
        names[1], ADALN_OUT,
        (unsigned long long)offsets[1], (unsigned long long)offsets[2],
        names[2], FC1_OUT, FC1_IN,
        (unsigned long long)offsets[2], (unsigned long long)offsets[3],
        names[3], FC1_OUT,
        (unsigned long long)offsets[3], (unsigned long long)offsets[4],
        names[4], FC2_OUT, FC2_IN,
        (unsigned long long)offsets[4], (unsigned long long)offsets[5],
        names[5], FC2_OUT,
        (unsigned long long)offsets[5], (unsigned long long)offsets[6]);
    if (length < 0 || (size_t)length >= sizeof(header) - 8)
        return cache_fail(error, error_size,
                          "VDN INT8 cache safetensors header overflow");
    size_t header_bytes = (size_t)length;
    while (header_bytes % 8) header[header_bytes++] = ' ';
    char filename[32], temporary_name[48], path[4096], temporary_path[4096];
    if (!block_filename(filename, block) ||
        snprintf(temporary_name, sizeof(temporary_name), "%s.tmp", filename) < 0 ||
        !path_join(path, sizeof(path), directory, filename) ||
        !path_join(temporary_path, sizeof(temporary_path), directory,
                   temporary_name))
        return cache_fail(error, error_size, "cache block path overflow");
    struct stat existing;
    if (lstat(path, &existing) == 0 || errno != ENOENT)
        return cache_fail(error, error_size,
                          "refusing to overwrite cache block: %s", path);
    int descriptor = open(temporary_path,
                          O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor < 0)
        return cache_fail(error, error_size, "cannot create %s: %s",
                          temporary_path, strerror(errno));
    int ok = write_u64_le(descriptor, header_bytes, error, error_size) &&
             write_all(descriptor, header, header_bytes, error, error_size);
    for (unsigned index = 0; ok && index < TENSORS_PER_BLOCK; index++)
        ok = read_and_write_tensor(descriptor, tensors[index],
                                   error, error_size);
    if (ok && fsync(descriptor) != 0)
        ok = cache_fail(error, error_size, "cannot fsync %s: %s",
                        temporary_path, strerror(errno));
    if (close(descriptor) != 0 && ok)
        ok = cache_fail(error, error_size, "cannot close %s: %s",
                        temporary_path, strerror(errno));
    if (ok && link(temporary_path, path) != 0)
        ok = cache_fail(error, error_size, "cannot publish %s: %s",
                        path, strerror(errno));
    if (ok) unlink(temporary_path);
    if (!ok) {
        unlink(temporary_path);
        return 0;
    }
    struct stat status;
    if (stat(path, &status) != 0 || status.st_size < 0 ||
        !h3_sha256_file(path, file_sha256, error, error_size)) return 0;
    *file_bytes = (uint64_t)status.st_size;
    return 1;
}

int h3_vdn_int8_cache_write_manifest(
        const char *directory, int use_turbo,
        const uint8_t source_sha256[H3_VDN_INT8_CACHE_DIGEST_BYTES],
        const h3_vdn_int8_cache_manifest *manifest,
        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!directory || !*directory || (use_turbo != 0 && use_turbo != 1) ||
        !source_sha256 || !manifest)
        return cache_fail(error, error_size,
                          "invalid VDN INT8 cache manifest write arguments");
    char path[4096], temporary_path[4096];
    if (!path_join(path, sizeof(path), directory, "manifest.json") ||
        !path_join(temporary_path, sizeof(temporary_path), directory,
                   "manifest.json.tmp"))
        return cache_fail(error, error_size, "cache manifest path overflow");
    struct stat existing;
    if (lstat(path, &existing) == 0 || errno != ENOENT)
        return cache_fail(error, error_size,
                          "refusing to overwrite cache manifest: %s", path);
    FILE *stream = fopen(temporary_path, "wx");
    if (!stream)
        return cache_fail(error, error_size, "cannot create %s: %s",
                          temporary_path, strerror(errno));
    char source_hex[65];
    h3_sha256_hex(source_sha256, source_hex);
    int ok = fprintf(stream,
        "{\n  \"format\": \"%s\",\n  \"version\": %d,\n"
        "  \"quantization\": \"%s\",\n  \"producer\": \"h3-vdn.c-%s\",\n"
        "  \"use_turbo\": %s,\n  \"blocks\": %d,\n"
        "  \"source_sha256\": \"%s\",\n  \"files\": [\n",
        cache_format, CACHE_VERSION, quantization, H3_VERSION,
        use_turbo ? "true" : "false", H3_VDN_INT8_CACHE_BLOCKS,
        source_hex) > 0;
    for (unsigned block = 0; ok && block < H3_VDN_INT8_CACHE_BLOCKS; block++) {
        char filename[32], digest_hex[65];
        block_filename(filename, block);
        h3_sha256_hex(manifest->block_sha256[block], digest_hex);
        ok = manifest->block_bytes[block] > 0 &&
            fprintf(stream,
                "    {\"block\": %u, \"path\": \"%s\", \"bytes\": %llu, "
                "\"sha256\": \"%s\"}%s\n",
                block, filename,
                (unsigned long long)manifest->block_bytes[block], digest_hex,
                block + 1 == H3_VDN_INT8_CACHE_BLOCKS ? "" : ",") > 0;
    }
    if (ok) ok = fputs("  ]\n}\n", stream) >= 0;
    if (ok && fflush(stream) != 0) ok = 0;
    if (ok && fsync(fileno(stream)) != 0) ok = 0;
    if (fclose(stream) != 0) ok = 0;
    if (!ok) {
        unlink(temporary_path);
        return cache_fail(error, error_size,
                          "cannot write VDN INT8 cache manifest: %s",
                          strerror(errno));
    }
    if (link(temporary_path, path) != 0) {
        unlink(temporary_path);
        return cache_fail(error, error_size, "cannot publish %s: %s",
                          path, strerror(errno));
    }
    unlink(temporary_path);
    return 1;
}
