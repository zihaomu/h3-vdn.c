#include "h3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>

static void h3_tokenizer_unavailable(char *error, size_t error_size) {
    if (error && error_size) {
        snprintf(error, error_size,
                 "raw-prompt tokenizer is not implemented for the HIP backend; "
                 "use a pre-encoded VDN prompt during the MVP phase");
    }
}

h3_tokenizer *h3_tokenizer_load(const char *tokenizer_json,
                                char *error, size_t error_size) {
    (void)tokenizer_json;
    h3_tokenizer_unavailable(error, error_size);
    return NULL;
}

void h3_tokenizer_free(h3_tokenizer *tokenizer) {
    (void)tokenizer;
}

int h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size) {
    (void)tokenizer;
    (void)utf8;
    (void)pad_empty;
    if (ids) *ids = NULL;
    if (count) *count = 0;
    h3_tokenizer_unavailable(error, error_size);
    return 0;
}

void h3_tokenizer_ids_free(uint32_t *ids) {
    free(ids);
}

char *h3_tokenizer_decode(const h3_tokenizer *tokenizer,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size) {
    (void)tokenizer;
    (void)ids;
    (void)count;
    h3_tokenizer_unavailable(error, error_size);
    return NULL;
}
