#ifndef H3_VDN_PIPELINE_H
#define H3_VDN_PIPELINE_H

#include "h3.h"

/* Execute the OpenVDN pre-encoded-prompt inference path. */
h3_result *h3_vdn_generate_embedded(h3_ctx *ctx, const h3_params *params);

#endif
