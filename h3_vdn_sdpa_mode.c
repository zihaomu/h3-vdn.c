#include "h3_vdn_sdpa_mode.h"

#include <stddef.h>
#include <string.h>

const char *h3_vdn_sdpa_mode_name(h3_vdn_sdpa_mode mode) {
    switch (mode) {
        case H3_VDN_SDPA_AUTO: return "auto";
        case H3_VDN_SDPA_SCALAR: return "scalar";
        case H3_VDN_SDPA_WAVE32: return "wave32";
        case H3_VDN_SDPA_SAGE_I8_BF16: return "sage-i8-bf16";
        case H3_VDN_SDPA_SAGE_I8_F16: return "sage-i8-f16";
        case H3_VDN_SDPA_SAGE_I8_FP8_E4M3: return "sage-i8-fp8-e4m3";
    }
    return NULL;
}

int h3_vdn_sdpa_mode_parse(const char *text, h3_vdn_sdpa_mode *mode) {
    if (!text || !mode) return 0;
    for (int value = H3_VDN_SDPA_AUTO;
         value <= H3_VDN_SDPA_SAGE_I8_FP8_E4M3; value++) {
        h3_vdn_sdpa_mode candidate = (h3_vdn_sdpa_mode)value;
        const char *name = h3_vdn_sdpa_mode_name(candidate);
        if (name && strcmp(text, name) == 0) {
            *mode = candidate;
            return 1;
        }
    }
    return 0;
}
