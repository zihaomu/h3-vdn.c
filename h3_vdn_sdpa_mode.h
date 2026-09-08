#ifndef H3_VDN_SDPA_MODE_H
#define H3_VDN_SDPA_MODE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    H3_VDN_SDPA_AUTO = 0,
    H3_VDN_SDPA_SCALAR,
    H3_VDN_SDPA_WAVE32,
    H3_VDN_SDPA_SAGE_I8_BF16,
    H3_VDN_SDPA_SAGE_I8_F16,
    H3_VDN_SDPA_SAGE_I8_FP8_E4M3
} h3_vdn_sdpa_mode;

const char *h3_vdn_sdpa_mode_name(h3_vdn_sdpa_mode mode);
int h3_vdn_sdpa_mode_parse(const char *text, h3_vdn_sdpa_mode *mode);

#ifdef __cplusplus
}
#endif

#endif
