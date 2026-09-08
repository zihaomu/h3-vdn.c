#include "h3_sha256.h"

#include <errno.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef void *(*evp_context_new_fn)(void);
typedef void (*evp_context_free_fn)(void *);
typedef const void *(*evp_sha256_fn)(void);
typedef int (*evp_digest_init_fn)(void *, const void *, void *);
typedef int (*evp_digest_update_fn)(void *, const void *, size_t);
typedef int (*evp_digest_final_fn)(void *, unsigned char *, unsigned int *);

typedef struct {
    void *library;
    void *context;
    evp_context_free_fn context_free;
    evp_digest_update_fn update;
    evp_digest_final_fn final;
} accelerated_sha256;

static int load_function(void *library, const char *name,
                         void *function, size_t function_size) {
    void *symbol = dlsym(library, name);
    if (!symbol || function_size != sizeof(symbol)) return 0;
    memcpy(function, &symbol, sizeof(symbol));
    return 1;
}

static accelerated_sha256 *accelerated_create(void) {
    const char *portable = getenv("H3_SHA256_PORTABLE");
    if (portable && *portable && strcmp(portable, "0")) return NULL;
    static const char *libraries[] = {
        "libcrypto.so.3", "libcrypto.so.1.1", "libcrypto.so"
    };
    void *library = NULL;
    for (size_t index = 0; index < sizeof(libraries) / sizeof(libraries[0]);
         index++) {
        library = dlopen(libraries[index], RTLD_NOW | RTLD_LOCAL);
        if (library) break;
    }
    if (!library) return NULL;
    accelerated_sha256 *accelerated = calloc(1, sizeof(*accelerated));
    evp_context_new_fn context_new = NULL;
    evp_sha256_fn sha256 = NULL;
    evp_digest_init_fn initialize = NULL;
    if (!accelerated ||
        !load_function(library, "EVP_MD_CTX_new", &context_new,
                       sizeof(context_new)) ||
        !load_function(library, "EVP_MD_CTX_free",
                       &accelerated->context_free,
                       sizeof(accelerated->context_free)) ||
        !load_function(library, "EVP_sha256", &sha256, sizeof(sha256)) ||
        !load_function(library, "EVP_DigestInit_ex", &initialize,
                       sizeof(initialize)) ||
        !load_function(library, "EVP_DigestUpdate", &accelerated->update,
                       sizeof(accelerated->update)) ||
        !load_function(library, "EVP_DigestFinal_ex", &accelerated->final,
                       sizeof(accelerated->final))) goto failed;
    accelerated->library = library;
    accelerated->context = context_new();
    if (!accelerated->context ||
        !initialize(accelerated->context, sha256(), NULL)) goto failed;
    return accelerated;

failed:
    if (accelerated) {
        if (accelerated->context && accelerated->context_free)
            accelerated->context_free(accelerated->context);
        free(accelerated);
    }
    dlclose(library);
    return NULL;
}

static void accelerated_free(accelerated_sha256 *accelerated) {
    if (!accelerated) return;
    accelerated->context_free(accelerated->context);
    dlclose(accelerated->library);
    free(accelerated);
}

static uint32_t rotate_right(uint32_t value, unsigned shift) {
    return (value >> shift) | (value << (32 - shift));
}

static uint32_t load_be32(const uint8_t *bytes) {
    return (uint32_t)bytes[0] << 24 | (uint32_t)bytes[1] << 16 |
           (uint32_t)bytes[2] << 8 | (uint32_t)bytes[3];
}

static void store_be32(uint8_t *bytes, uint32_t value) {
    bytes[0] = (uint8_t)(value >> 24);
    bytes[1] = (uint8_t)(value >> 16);
    bytes[2] = (uint8_t)(value >> 8);
    bytes[3] = (uint8_t)value;
}

static void transform(h3_sha256 *context, const uint8_t block[64]) {
    static const uint32_t constants[64] = {
        UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf),
        UINT32_C(0xe9b5dba5), UINT32_C(0x3956c25b), UINT32_C(0x59f111f1),
        UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5), UINT32_C(0xd807aa98),
        UINT32_C(0x12835b01), UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
        UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7),
        UINT32_C(0xc19bf174), UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786),
        UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc), UINT32_C(0x2de92c6f),
        UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
        UINT32_C(0x983e5152), UINT32_C(0xa831c66d), UINT32_C(0xb00327c8),
        UINT32_C(0xbf597fc7), UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147),
        UINT32_C(0x06ca6351), UINT32_C(0x14292967), UINT32_C(0x27b70a85),
        UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
        UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e),
        UINT32_C(0x92722c85), UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b),
        UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3), UINT32_C(0xd192e819),
        UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
        UINT32_C(0x19a4c116), UINT32_C(0x1e376c08), UINT32_C(0x2748774c),
        UINT32_C(0x34b0bcb5), UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a),
        UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3), UINT32_C(0x748f82ee),
        UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
        UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7),
        UINT32_C(0xc67178f2)
    };
    uint32_t words[64];
    for (unsigned index = 0; index < 16; index++)
        words[index] = load_be32(block + index * 4);
    for (unsigned index = 16; index < 64; index++) {
        uint32_t previous15 = words[index - 15];
        uint32_t previous2 = words[index - 2];
        uint32_t small0 = rotate_right(previous15, 7) ^
                          rotate_right(previous15, 18) ^ (previous15 >> 3);
        uint32_t small1 = rotate_right(previous2, 17) ^
                          rotate_right(previous2, 19) ^ (previous2 >> 10);
        words[index] = words[index - 16] + small0 + words[index - 7] + small1;
    }
    uint32_t a = context->state[0];
    uint32_t b = context->state[1];
    uint32_t c = context->state[2];
    uint32_t d = context->state[3];
    uint32_t e = context->state[4];
    uint32_t f = context->state[5];
    uint32_t g = context->state[6];
    uint32_t h = context->state[7];
    for (unsigned index = 0; index < 64; index++) {
        uint32_t big1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                        rotate_right(e, 25);
        uint32_t choose = (e & f) ^ (~e & g);
        uint32_t temporary1 = h + big1 + choose + constants[index] +
                              words[index];
        uint32_t big0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                        rotate_right(a, 22);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temporary2 = big0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

void h3_sha256_init(h3_sha256 *context) {
    if (!context) return;
    context->state[0] = UINT32_C(0x6a09e667);
    context->state[1] = UINT32_C(0xbb67ae85);
    context->state[2] = UINT32_C(0x3c6ef372);
    context->state[3] = UINT32_C(0xa54ff53a);
    context->state[4] = UINT32_C(0x510e527f);
    context->state[5] = UINT32_C(0x9b05688c);
    context->state[6] = UINT32_C(0x1f83d9ab);
    context->state[7] = UINT32_C(0x5be0cd19);
    context->bytes = 0;
    context->used = 0;
    context->failed = 0;
    context->accelerated = accelerated_create();
}

void h3_sha256_update(h3_sha256 *context, const void *opaque, size_t bytes) {
    if (!context || (!opaque && bytes)) return;
    if (context->accelerated) {
        accelerated_sha256 *accelerated = context->accelerated;
        if (!context->failed &&
            !accelerated->update(accelerated->context, opaque, bytes))
            context->failed = 1;
        return;
    }
    const uint8_t *data = opaque;
    context->bytes += bytes;
    while (bytes) {
        size_t room = sizeof(context->block) - context->used;
        size_t take = bytes < room ? bytes : room;
        memcpy(context->block + context->used, data, take);
        context->used += take;
        data += take;
        bytes -= take;
        if (context->used == sizeof(context->block)) {
            transform(context, context->block);
            context->used = 0;
        }
    }
}

int h3_sha256_final(h3_sha256 *context, uint8_t digest[32]) {
    if (!context || !digest) return 0;
    if (context->accelerated) {
        accelerated_sha256 *accelerated = context->accelerated;
        unsigned int bytes = 0;
        int ok = !context->failed &&
            accelerated->final(accelerated->context, digest, &bytes) &&
            bytes == 32;
        accelerated_free(accelerated);
        memset(context, 0, sizeof(*context));
        if (!ok) memset(digest, 0, 32);
        return ok;
    }
    uint64_t bit_count = context->bytes * UINT64_C(8);
    context->block[context->used++] = UINT8_C(0x80);
    if (context->used > 56) {
        memset(context->block + context->used, 0,
               sizeof(context->block) - context->used);
        transform(context, context->block);
        context->used = 0;
    }
    memset(context->block + context->used, 0, 56 - context->used);
    for (unsigned index = 0; index < 8; index++)
        context->block[63 - index] = (uint8_t)(bit_count >> (index * 8));
    transform(context, context->block);
    for (unsigned index = 0; index < 8; index++)
        store_be32(digest + index * 4, context->state[index]);
    memset(context, 0, sizeof(*context));
    return 1;
}

void h3_sha256_hex(const uint8_t digest[32], char output[65]) {
    static const char digits[] = "0123456789abcdef";
    if (!digest || !output) return;
    for (unsigned index = 0; index < 32; index++) {
        output[index * 2] = digits[digest[index] >> 4];
        output[index * 2 + 1] = digits[digest[index] & 15];
    }
    output[64] = '\0';
}

int h3_sha256_update_file_range(h3_sha256 *context, const char *path,
                                uint64_t offset, uint64_t bytes,
                                char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!context || !path || !*path ||
        offset > (uint64_t)INT64_MAX || bytes > (uint64_t)INT64_MAX - offset) {
        if (error && error_size)
            snprintf(error, error_size, "invalid SHA-256 file range");
        return 0;
    }
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        if (error && error_size)
            snprintf(error, error_size, "cannot open %s: %s", path,
                     strerror(errno));
        return 0;
    }
    uint8_t buffer[1024 * 1024];
    uint64_t completed = 0;
    while (completed < bytes) {
        size_t request = sizeof(buffer);
        if ((uint64_t)request > bytes - completed)
            request = (size_t)(bytes - completed);
        ssize_t got;
        do {
            got = pread(descriptor, buffer, request,
                        (off_t)(offset + completed));
        } while (got < 0 && errno == EINTR);
        if (got <= 0) {
            if (error && error_size)
                snprintf(error, error_size, "cannot hash %s: %s", path,
                         got < 0 ? strerror(errno) : "unexpected end of file");
            close(descriptor);
            return 0;
        }
        h3_sha256_update(context, buffer, (size_t)got);
        if (context->failed) {
            if (error && error_size)
                snprintf(error, error_size,
                         "accelerated SHA-256 update failed for %s", path);
            close(descriptor);
            return 0;
        }
        completed += (uint64_t)got;
    }
    close(descriptor);
    return 1;
}

int h3_sha256_file(const char *path, uint8_t digest[32],
                   char *error, size_t error_size) {
    struct stat status;
    if (!path || !digest || stat(path, &status) != 0 ||
        !S_ISREG(status.st_mode) || status.st_size < 0) {
        if (error && error_size)
            snprintf(error, error_size, "cannot stat %s: %s",
                     path ? path : "(null)", strerror(errno));
        return 0;
    }
    h3_sha256 context;
    h3_sha256_init(&context);
    if (!h3_sha256_update_file_range(
            &context, path, 0, (uint64_t)status.st_size,
            error, error_size)) return 0;
    if (!h3_sha256_final(&context, digest)) {
        if (error && error_size)
            snprintf(error, error_size, "SHA-256 finalization failed for %s",
                     path);
        return 0;
    }
    return 1;
}
