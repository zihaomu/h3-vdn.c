#include "h3_sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int check(const void *data, size_t bytes, const char *expected,
                 const char *label) {
    h3_sha256 context;
    uint8_t digest[32];
    char actual[65];
    h3_sha256_init(&context);
    h3_sha256_update(&context, data, bytes);
    h3_sha256_final(&context, digest);
    h3_sha256_hex(digest, actual);
    if (strcmp(actual, expected)) {
        fprintf(stderr, "%s: got %s expected %s\n", label, actual, expected);
        return 0;
    }
    return 1;
}

int main(void) {
    int ok = check("", 0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "empty") &&
        check("abc", 3,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "abc") &&
        check("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 56,
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        "multi-block");
    char path[] = "/tmp/h3_sha256_XXXXXX";
    int descriptor = mkstemp(path);
    static const char contents[] = "xxabczz";
    if (descriptor < 0 ||
        write(descriptor, contents, sizeof(contents) - 1) !=
            (ssize_t)(sizeof(contents) - 1) || close(descriptor) != 0) {
        fprintf(stderr, "cannot create SHA-256 range fixture\n");
        if (descriptor >= 0) close(descriptor);
        unlink(path);
        return 1;
    }
    h3_sha256 context;
    uint8_t digest[32];
    char actual[65];
    char error[256] = {0};
    h3_sha256_init(&context);
    if (!h3_sha256_update_file_range(
            &context, path, 2, 3, error, sizeof(error))) {
        fprintf(stderr, "range hash failed: %s\n", error);
        ok = 0;
    } else {
        h3_sha256_final(&context, digest);
        h3_sha256_hex(digest, actual);
        if (strcmp(actual,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")) {
            fprintf(stderr, "range hash mismatch: %s\n", actual);
            ok = 0;
        }
    }
    unlink(path);
    if (!ok) return 1;
    puts("PASS: SHA-256 vectors and file-range hashing");
    return 0;
}
