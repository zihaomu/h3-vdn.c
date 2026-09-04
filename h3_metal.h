#ifndef H3_METAL_H
#define H3_METAL_H

#include "h3_backend.h"

/* Compatibility wrapper for older embedders; new code uses h3_probe_device(). */
static inline int h3_metal_probe(h3_device_info *info,
                                 char *error, size_t error_size) {
    return h3_backend_probe(0, info, error, error_size);
}

#endif
