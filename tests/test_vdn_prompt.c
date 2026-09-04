#include "h3_vdn_prompt.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        h3_vdn_prompt_free(&embedding);                                      \
        return 1;                                                            \
    }                                                                        \
} while (0)

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s PROMPT.safetensors\n", argv[0]);
        return 2;
    }
    h3_text_embedding embedding;
    memset(&embedding, 0, sizeof(embedding));
    char error[512];
    if (!h3_vdn_prompt_load(argv[1], &embedding, error, sizeof(error))) {
        fprintf(stderr, "cannot load VDN prompt: %s\n", error);
        return 1;
    }
    CHECK(embedding.tokens == 800);
    CHECK(embedding.width == 5120);
    CHECK(embedding.values != NULL);
    CHECK(embedding.tags != NULL);
    uint64_t value_hash = UINT64_C(1469598103934665603);
    uint64_t tag_hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < embedding.tokens * embedding.width; index++) {
        value_hash ^= embedding.values[index];
        value_hash *= UINT64_C(1099511628211);
    }
    for (size_t index = 0; index < embedding.tokens; index++) {
        tag_hash ^= embedding.tags[index];
        tag_hash *= UINT64_C(1099511628211);
    }
    CHECK(value_hash != UINT64_C(1469598103934665603));
    CHECK(tag_hash != UINT64_C(1469598103934665603));
    printf("VDN prompt loaded: BF16[%zu,%zu], value_hash=%016llx, "
           "tag_hash=%016llx\n", embedding.tokens, embedding.width,
           (unsigned long long)value_hash, (unsigned long long)tag_hash);
    h3_vdn_prompt_free(&embedding);
    return 0;
}
