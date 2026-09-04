#include "h3_backend.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

int main(void) {
    char error[256] = {0};
    int count = h3_backend_device_count(error, sizeof(error));
    CHECK(count > 0);

    for (int index = 0; index < count; index++) {
        h3_device_info info;
        CHECK(h3_backend_probe(index, &info, error, sizeof(error)));
        CHECK(info.device_index == index);
        CHECK(info.backend[0] != '\0');
        CHECK(info.name[0] != '\0');
        CHECK(info.architecture[0] != '\0');
        CHECK(info.physical_memory > 0);
        printf("device %d: %s %s %s (%llu bytes)\n", index, info.backend,
               info.architecture, info.name,
               (unsigned long long)info.physical_memory);
    }

    h3_device_info invalid;
    error[0] = '\0';
    CHECK(!h3_backend_probe(count, &invalid, error, sizeof(error)));
    CHECK(strstr(error, "out of range") != NULL);
    puts("backend probe tests passed");
    return 0;
}
