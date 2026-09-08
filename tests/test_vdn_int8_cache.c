#include "h3_vdn_int8_cache.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char directory[] = "/tmp/h3_vdn_int8_cache_XXXXXX";
    if (!mkdtemp(directory)) {
        perror("mkdtemp");
        return 1;
    }
    uint8_t source[H3_VDN_INT8_CACHE_DIGEST_BYTES];
    uint8_t wrong_source[H3_VDN_INT8_CACHE_DIGEST_BYTES];
    h3_vdn_int8_cache_manifest written;
    memset(&written, 0, sizeof(written));
    for (unsigned index = 0; index < sizeof(source); index++) {
        source[index] = (uint8_t)(index * 7 + 3);
        wrong_source[index] = source[index];
    }
    wrong_source[0] ^= 1;
    for (unsigned block = 0; block < H3_VDN_INT8_CACHE_BLOCKS; block++) {
        written.block_bytes[block] = UINT64_C(500000000) + block;
        for (unsigned index = 0; index < H3_VDN_INT8_CACHE_DIGEST_BYTES;
             index++)
            written.block_sha256[block][index] =
                (uint8_t)(block * 11 + index);
    }
    char error[512] = {0};
    int ok = h3_vdn_int8_cache_write_manifest(
        directory, 1, source, &written, error, sizeof(error));
    h3_vdn_int8_cache_manifest read;
    if (ok)
        ok = h3_vdn_int8_cache_read_manifest(
            directory, 1, source, &read, error, sizeof(error));
    if (ok && memcmp(&written, &read, sizeof(written))) {
        snprintf(error, sizeof(error), "manifest round-trip mismatch");
        ok = 0;
    }
    if (ok && h3_vdn_int8_cache_read_manifest(
            directory, 1, wrong_source, &read, error, sizeof(error))) {
        snprintf(error, sizeof(error), "wrong source digest was accepted");
        ok = 0;
    }
    char manifest_path[4096];
    int length = snprintf(manifest_path, sizeof(manifest_path),
                          "%s/manifest.json", directory);
    int descriptor = length > 0 && (size_t)length < sizeof(manifest_path) ?
        open(manifest_path, O_WRONLY | O_TRUNC | O_CLOEXEC) : -1;
    static const char malformed[] = "{}\n";
    if (ok && (descriptor < 0 ||
        write(descriptor, malformed, sizeof(malformed) - 1) !=
            (ssize_t)(sizeof(malformed) - 1))) {
        snprintf(error, sizeof(error), "cannot corrupt manifest fixture");
        ok = 0;
    }
    if (descriptor >= 0) close(descriptor);
    if (ok && h3_vdn_int8_cache_read_manifest(
            directory, 1, source, &read, error, sizeof(error))) {
        snprintf(error, sizeof(error), "malformed manifest was accepted");
        ok = 0;
    }
    unlink(manifest_path);
    rmdir(directory);
    if (!ok) {
        fprintf(stderr, "VDN INT8 cache manifest test failed: %s\n", error);
        return 1;
    }
    puts("PASS: VDN INT8 cache v1 manifest identity and strict parsing");
    return 0;
}
