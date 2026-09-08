#ifndef H3_SHA256_H
#define H3_SHA256_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
    void *accelerated;
    int failed;
} h3_sha256;

void h3_sha256_init(h3_sha256 *context);
void h3_sha256_update(h3_sha256 *context, const void *data, size_t bytes);
int h3_sha256_final(h3_sha256 *context, uint8_t digest[32]);
void h3_sha256_hex(const uint8_t digest[32], char output[65]);

int h3_sha256_update_file_range(h3_sha256 *context, const char *path,
                                uint64_t offset, uint64_t bytes,
                                char *error, size_t error_size);
int h3_sha256_file(const char *path, uint8_t digest[32],
                   char *error, size_t error_size);

#endif
