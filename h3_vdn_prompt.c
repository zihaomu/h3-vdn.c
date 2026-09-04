#include "h3_vdn_prompt.h"

#include "h3_safetensors.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VDN_PROMPT_WIDTH = 5120 };

static int prompt_fail(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return 0;
}

void h3_vdn_prompt_free(h3_text_embedding *embedding) {
    if (!embedding) return;
    free(embedding->values);
    free(embedding->tags);
    memset(embedding, 0, sizeof(*embedding));
}

int h3_vdn_prompt_load(const char *path, h3_text_embedding *embedding,
                       char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path || !embedding)
        return prompt_fail(error, error_size,
                           "invalid VDN prompt embedding arguments");
    memset(embedding, 0, sizeof(*embedding));
    h3_st_header header;
    if (!h3_st_read_header(path, &header, error, error_size)) return 0;
    const h3_st_tensor *values = h3_st_find(&header, "prompt_embeds");
    const h3_st_tensor *tags = h3_st_find(&header, "token_tags");
    int valid = header.tensor_count == 2 && values && tags &&
        values->dtype == H3_DTYPE_BF16 && values->ndim == 2 &&
        values->shape[0] > 0 && values->shape[0] <= UINT32_MAX &&
        values->shape[1] == VDN_PROMPT_WIDTH &&
        tags->dtype == H3_DTYPE_I64 && tags->ndim == 1 &&
        tags->shape[0] == values->shape[0];
    if (!valid) {
        h3_st_free_header(&header);
        return prompt_fail(error, error_size,
            "VDN prompt must contain only prompt_embeds BF16[L,5120] "
            "and token_tags I64[L], with 1 <= L <= UINT32_MAX");
    }
    size_t rows = (size_t)values->shape[0];
    if (rows > SIZE_MAX / VDN_PROMPT_WIDTH) {
        h3_st_free_header(&header);
        return prompt_fail(error, error_size,
                           "VDN prompt embedding size overflows host limits");
    }
    size_t value_count = rows * VDN_PROMPT_WIDTH;
    if (value_count > SIZE_MAX / sizeof(*embedding->values) ||
        rows > SIZE_MAX / sizeof(int64_t)) {
        h3_st_free_header(&header);
        return prompt_fail(error, error_size,
                           "VDN prompt storage size overflows host limits");
    }
    embedding->values = malloc(value_count * sizeof(*embedding->values));
    embedding->tags = malloc(rows * sizeof(*embedding->tags));
    int64_t *wide_tags = malloc(rows * sizeof(*wide_tags));
    if (!embedding->values || !embedding->tags || !wide_tags) {
        free(wide_tags);
        h3_st_free_header(&header);
        h3_vdn_prompt_free(embedding);
        return prompt_fail(error, error_size,
                           "out of memory loading VDN prompt embeddings");
    }
    int ok = h3_st_read_data(&header, values, embedding->values,
                             value_count * sizeof(*embedding->values),
                             error, error_size) &&
             h3_st_read_data(&header, tags, wide_tags,
                             rows * sizeof(*wide_tags),
                             error, error_size);
    if (ok) {
        for (size_t index = 0; index < rows; index++) {
            if (wide_tags[index] < 0 || wide_tags[index] > UINT8_MAX) {
                ok = prompt_fail(error, error_size,
                                 "VDN prompt token tag is outside U8 range");
                break;
            }
            embedding->tags[index] = (uint8_t)wide_tags[index];
        }
    }
    free(wide_tags);
    h3_st_free_header(&header);
    if (!ok) {
        h3_vdn_prompt_free(embedding);
        return 0;
    }
    embedding->tokens = rows;
    embedding->width = VDN_PROMPT_WIDTH;
    return 1;
}
