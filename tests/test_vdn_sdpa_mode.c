#include "h3_vdn_sdpa_mode.h"

#include <stdio.h>

static int checks;
static int failures;

#define CHECK(condition) do { \
    checks++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        failures++; \
    } \
} while (0)

int main(void) {
    for (int value = H3_VDN_SDPA_AUTO;
         value <= H3_VDN_SDPA_SAGE_I8_FP8_E4M3; value++) {
        h3_vdn_sdpa_mode expected = (h3_vdn_sdpa_mode)value;
        const char *name = h3_vdn_sdpa_mode_name(expected);
        h3_vdn_sdpa_mode parsed = H3_VDN_SDPA_AUTO;
        CHECK(name != NULL);
        CHECK(h3_vdn_sdpa_mode_parse(name, &parsed));
        CHECK(parsed == expected);
    }
    h3_vdn_sdpa_mode parsed = H3_VDN_SDPA_AUTO;
    CHECK(!h3_vdn_sdpa_mode_parse("sage", &parsed));
    CHECK(!h3_vdn_sdpa_mode_parse(NULL, &parsed));
    CHECK(!h3_vdn_sdpa_mode_parse("auto", NULL));
    CHECK(h3_vdn_sdpa_mode_name((h3_vdn_sdpa_mode)99) == NULL);
    if (failures) {
        fprintf(stderr, "VDN SDPA mode contract failed: %d/%d checks\n",
                failures, checks);
        return 1;
    }
    printf("VDN SDPA mode contract passed: %d checks\n", checks);
    return 0;
}
