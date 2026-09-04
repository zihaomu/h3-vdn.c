#ifndef H3_BACKEND_H
#define H3_BACKEND_H

#include "h3.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int h3_backend_device_count(char *error, size_t error_size);
int h3_backend_probe(int device_index, h3_device_info *info,
                     char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
