#include "h3_weights.h"

#include <dirent.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct h3_weight_store {
    h3_st_header *headers;
    size_t count;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int safetensors_name(const char *name) {
    static const char suffix[] = ".safetensors";
    size_t length = strlen(name);
    return length > sizeof(suffix) - 1 &&
           strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0;
}

static int compare_paths(const void *left, const void *right) {
    const char *const *a = left;
    const char *const *b = right;
    return strcmp(*a, *b);
}

static void free_paths(char **paths, size_t count) {
    if (!paths) return;
    for (size_t index = 0; index < count; index++) free(paths[index]);
    free(paths);
}

h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size) {
    if (!directory || !*directory) {
        fail(error, error_size, "weight directory is required");
        return NULL;
    }
    DIR *stream = opendir(directory);
    if (!stream) {
        fail(error, error_size, "cannot open weight directory: %s", directory);
        return NULL;
    }
    char **paths = NULL;
    size_t count = 0;
    size_t capacity = 0;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (!safetensors_name(entry->d_name)) continue;
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 8;
            char **grown = realloc(paths, next * sizeof(*grown));
            if (!grown) {
                fail(error, error_size, "out of memory listing weight shards");
                closedir(stream);
                free_paths(paths, count);
                return NULL;
            }
            paths = grown;
            capacity = next;
        }
        size_t length = strlen(directory) + strlen(entry->d_name) + 2;
        paths[count] = malloc(length);
        if (!paths[count]) {
            fail(error, error_size, "out of memory resolving a weight shard");
            closedir(stream);
            free_paths(paths, count);
            return NULL;
        }
        snprintf(paths[count], length, "%s/%s", directory, entry->d_name);
        count++;
    }
    closedir(stream);
    if (!count) {
        fail(error, error_size, "no safetensors shards in %s", directory);
        free(paths);
        return NULL;
    }
    qsort(paths, count, sizeof(*paths), compare_paths);
    h3_weight_store *store = calloc(1, sizeof(*store));
    if (!store) {
        fail(error, error_size, "out of memory creating weight store");
        free_paths(paths, count);
        return NULL;
    }
    store->headers = calloc(count, sizeof(*store->headers));
    if (!store->headers) {
        fail(error, error_size, "out of memory allocating weight headers");
        free(store);
        free_paths(paths, count);
        return NULL;
    }
    store->count = count;
    for (size_t index = 0; index < count; index++) {
        char detail[384];
        if (!h3_st_read_header(paths[index], &store->headers[index], detail,
                               sizeof(detail))) {
            fail(error, error_size, "%s", detail);
            free_paths(paths, count);
            h3_weight_store_free(store);
            return NULL;
        }
    }
    free_paths(paths, count);
    return store;
}

void h3_weight_store_free(h3_weight_store *store) {
    if (!store) return;
    for (size_t index = 0; index < store->count; index++) {
        h3_st_free_header(&store->headers[index]);
    }
    free(store->headers);
    free(store);
}

size_t h3_weight_store_shards(const h3_weight_store *store) {
    return store ? store->count : 0;
}

static const h3_st_tensor *find_exact(const h3_weight_store *store,
                                      const char *name,
                                      const h3_st_header **header) {
    for (size_t index = 0; index < store->count; index++) {
        const h3_st_tensor *tensor = h3_st_find(&store->headers[index], name);
        if (tensor) {
            if (header) *header = &store->headers[index];
            return tensor;
        }
    }
    return NULL;
}

static int replace_once(char *destination, size_t capacity,
                        const char *source, const char *old,
                        const char *replacement) {
    const char *match = strstr(source, old);
    if (!match) return snprintf(destination, capacity, "%s", source) >= 0 &&
                       strlen(source) < capacity;
    size_t prefix = (size_t)(match - source);
    size_t old_length = strlen(old);
    size_t replacement_length = strlen(replacement);
    size_t suffix = strlen(match + old_length);
    if (prefix + replacement_length + suffix + 1 > capacity) return 0;
    memcpy(destination, source, prefix);
    memcpy(destination + prefix, replacement, replacement_length);
    memcpy(destination + prefix + replacement_length, match + old_length,
           suffix + 1);
    return 1;
}

/* The first public MiniMax-H3 checkpoint used the original FL2VA module
 * names. The Diffusers release preserves the tensors but adopts standard
 * Diffusers names. Keep callers and the legacy checkpoint stable by resolving
 * the old names only after an exact lookup misses. Split Q/K/V projections
 * are handled by the DiT loader because they represent three payloads rather
 * than a simple alias. */
static int diffusers_alias(const char *name, char *alias, size_t capacity) {
    struct exact_alias { const char *old; const char *modern; };
    static const struct exact_alias exact[] = {
        {"condition_proj.", "context_embedder."},
        {"video_patch_proj.", "proj_in."},
        {"audio_patch_proj.", "audio_proj_in."},
        {"time_embedder.proj_in.", "time_embedder.linear_1."},
        {"time_embedder.proj_out.", "time_embedder.linear_2."},
        {"final_layer.adaln_proj.linear.", "norm_out.linear."},
        {"final_layer.norm.", "norm_out.norm."},
        {"final_layer.video_out.", "proj_out."},
        {"final_layer.audio_out.", "audio_proj_out."},
    };
    for (size_t index = 0; index < sizeof(exact) / sizeof(*exact); index++) {
        size_t length = strlen(exact[index].old);
        if (!strncmp(name, exact[index].old, length))
            return snprintf(alias, capacity, "%s%s", exact[index].modern,
                            name + length) >= 0 &&
                   strlen(exact[index].modern) + strlen(name + length) <
                       capacity;
    }

    char first[256];
    if (!strncmp(name, "token_refiner.blocks.", 21)) {
        if (snprintf(first, sizeof(first), "token_refiner.refiner_blocks.%s",
                     name + 21) < 0 ||
            strlen("token_refiner.refiner_blocks.") + strlen(name + 21) >=
                sizeof(first)) return 0;
    } else if (!strncmp(name, "blocks.", 7)) {
        if (snprintf(first, sizeof(first), "transformer_blocks.%s",
                     name + 7) < 0 ||
            strlen("transformer_blocks.") + strlen(name + 7) >= sizeof(first))
            return 0;
    } else {
        return 0;
    }

    struct suffix_alias { const char *old; const char *modern; };
    static const struct suffix_alias suffixes[] = {
        {".attn.q_norm.", ".attn.norm_q."},
        {".attn.k_norm.", ".attn.norm_k."},
        {".attn.out_proj.", ".attn.to_out.0."},
        {".mlp.fc1.", ".ff.net.0.proj."},
        {".mlp.fc2.", ".ff.net.2."},
    };
    for (size_t index = 0; index < sizeof(suffixes) / sizeof(*suffixes);
         index++) {
        if (strstr(first, suffixes[index].old))
            return replace_once(alias, capacity, first, suffixes[index].old,
                                suffixes[index].modern);
    }
    return snprintf(alias, capacity, "%s", first) >= 0 &&
           strlen(first) < capacity;
}

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header) {
    if (header) *header = NULL;
    if (!store || !name) return NULL;
    const h3_st_tensor *tensor = find_exact(store, name, header);
    if (tensor) return tensor;
    char alias[256];
    return diffusers_alias(name, alias, sizeof(alias)) ?
        find_exact(store, alias, header) : NULL;
}

static h3_gpu_tensor *load_tensor(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape, h3_dtype dtype,
                                  char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    if (tensor->dtype != dtype || tensor->ndim != ndim) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/%d",
             name, h3_dtype_name(tensor->dtype), tensor->ndim,
             h3_dtype_name(dtype), ndim);
        return NULL;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size, "weight %s shape mismatch at dimension %d",
                 name, dimension);
            return NULL;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return NULL;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return NULL;
    }
    h3_gpu_tensor *result = dtype == H3_DTYPE_BF16 ?
        h3_gpu_tensor_load_bf16(gpu, header->path, tensor->file_offset,
                                (size_t)elements) :
        dtype == H3_DTYPE_F32 ?
        h3_gpu_tensor_load_f32(gpu, header->path, tensor->file_offset,
                               (size_t)elements) :
        h3_gpu_tensor_load_i8(gpu, header->path, tensor->file_offset,
                              (size_t)elements);
    if (!result) {
        fail(error, error_size, "cannot load %s: %s", name, h3_gpu_error(gpu));
    }
    return result;
}

h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_BF16,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_F32,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_i8(const h3_weight_store *store, h3_gpu *gpu,
                                 const char *name, int ndim,
                                 const uint64_t *shape,
                                 char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_I8,
                       error, error_size);
}
