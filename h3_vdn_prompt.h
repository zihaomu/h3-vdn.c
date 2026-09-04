#ifndef H3_VDN_PROMPT_H
#define H3_VDN_PROMPT_H

#include "h3_text_encoder.h"

#include <stddef.h>

int h3_vdn_prompt_load(const char *path, h3_text_embedding *embedding,
                       char *error, size_t error_size);
void h3_vdn_prompt_free(h3_text_embedding *embedding);

#endif
