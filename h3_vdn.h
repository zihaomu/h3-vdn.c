#ifndef H3_VDN_H
#define H3_VDN_H

#include "h3.h"

#include <stddef.h>

int h3_vdn_inspect(const char *base_model_dir,
                   const char *checkpoint_dir,
                   int require_weights,
                   h3_vdn_info *info,
                   char *error, size_t error_size);

#endif
