#include "vdn_reference.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int h3_vdn_ref_matrix_size(size_t rows, size_t columns, size_t *size) {
    if (!rows || !columns || rows > SIZE_MAX / columns) return 0;
    *size = rows * columns;
    return *size <= SIZE_MAX / sizeof(float);
}

int h3_vdn_ref_window_bounds(size_t frames, int radius, int chunk,
                             h3_vdn_ref_window *bounds) {
    if (!frames || frames > (size_t)INT_MAX || radius < 0 || chunk < 0 ||
        !bounds)
        return 0;
    for (size_t frame = 0; frame < frames; frame++) {
        long long lo;
        long long hi;
        if (!chunk) {
            lo = (long long)frame - radius;
            hi = (long long)frame + radius;
        } else {
            long long group = (long long)frame / chunk;
            lo = (group - radius) * chunk;
            hi = (group + radius + 1) * chunk - 1;
        }
        if (lo < INT_MIN || lo > INT_MAX || hi < INT_MIN || hi > INT_MAX)
            return 0;
        bounds[frame].lo = (int)lo;
        bounds[frame].hi = (int)hi;
    }
    return 1;
}

size_t h3_vdn_ref_prune_anchor_bounds(size_t frames,
                                      const h3_vdn_ref_window *bounds,
                                      h3_vdn_ref_window *inner_bounds) {
    if (!bounds || !inner_bounds || frames <= 2) return 0;
    for (size_t frame = 1; frame + 1 < frames; frame++) {
        inner_bounds[frame - 1].lo = bounds[frame].lo - 1;
        inner_bounds[frame - 1].hi = bounds[frame].hi - 1;
    }
    return frames - 2;
}

void h3_vdn_ref_frame_statistics(size_t tokens, size_t key_dim,
                                 size_t value_dim, const float *key,
                                 const float *value, const float *beta,
                                 float *a, float *b) {
    for (size_t row = 0; row < key_dim; row++) {
        for (size_t column = 0; column < key_dim; column++) {
            double sum = 0.0;
            for (size_t token = 0; token < tokens; token++)
                sum += (double)key[token * key_dim + row] * beta[token] *
                       key[token * key_dim + column];
            a[row * key_dim + column] = (float)sum;
        }
    }
    for (size_t row = 0; row < key_dim; row++) {
        for (size_t column = row + 1; column < key_dim; column++) {
            float symmetric = 0.5f *
                (a[row * key_dim + column] + a[column * key_dim + row]);
            a[row * key_dim + column] = symmetric;
            a[column * key_dim + row] = symmetric;
        }
    }
    for (size_t row = 0; row < value_dim; row++) {
        for (size_t column = 0; column < key_dim; column++) {
            double sum = 0.0;
            for (size_t token = 0; token < tokens; token++)
                sum += (double)value[token * value_dim + row] * beta[token] *
                       key[token * key_dim + column];
            b[row * key_dim + column] = (float)sum;
        }
    }
}

int h3_vdn_ref_vdn_factor(size_t key_dim, size_t value_dim,
                          const float *alpha, const float *a, const float *b,
                          float *transition, float *injection) {
    size_t square;
    size_t rectangular;
    if (!alpha || !a || !b || !transition || !injection ||
        !h3_vdn_ref_matrix_size(key_dim, key_dim, &square) ||
        !h3_vdn_ref_matrix_size(value_dim, key_dim, &rectangular))
        return 0;
    if (square > SIZE_MAX / 3) return 0;
    float *storage = calloc(square * 3, sizeof(*storage));
    if (!storage) return 0;
    float *lower = storage;
    float *lower_inverse = storage + square;
    float *inverse = storage + square * 2;

    for (size_t row = 0; row < key_dim; row++) {
        for (size_t column = 0; column <= row; column++) {
            double sum = (double)a[row * key_dim + column] +
                         (row == column ? 1.0 : 0.0);
            for (size_t inner = 0; inner < column; inner++)
                sum -= (double)lower[row * key_dim + inner] *
                       lower[column * key_dim + inner];
            if (row == column) {
                if (!(sum > 0.0) || !isfinite(sum)) {
                    free(storage);
                    return 0;
                }
                lower[row * key_dim + column] = (float)sqrt(sum);
            } else {
                lower[row * key_dim + column] =
                    (float)(sum / lower[column * key_dim + column]);
            }
        }
    }

    for (size_t column = 0; column < key_dim; column++) {
        for (size_t row = 0; row < key_dim; row++) {
            double sum = row == column ? 1.0 : 0.0;
            for (size_t inner = 0; inner < row; inner++)
                sum -= (double)lower[row * key_dim + inner] *
                       lower_inverse[inner * key_dim + column];
            lower_inverse[row * key_dim + column] =
                (float)(sum / lower[row * key_dim + row]);
        }
    }
    for (size_t row = 0; row < key_dim; row++) {
        for (size_t column = 0; column < key_dim; column++) {
            double sum = 0.0;
            for (size_t inner = 0; inner < key_dim; inner++)
                sum += (double)lower_inverse[inner * key_dim + row] *
                       lower_inverse[inner * key_dim + column];
            inverse[row * key_dim + column] = (float)sum;
            transition[row * key_dim + column] = (float)(alpha[row] * sum);
        }
    }
    for (size_t row = 0; row < value_dim; row++) {
        for (size_t column = 0; column < key_dim; column++) {
            double sum = 0.0;
            for (size_t inner = 0; inner < key_dim; inner++)
                sum += (double)b[row * key_dim + inner] *
                       inverse[inner * key_dim + column];
            injection[row * key_dim + column] = (float)sum;
        }
    }
    free(storage);
    (void)rectangular;
    return 1;
}

static void h3_vdn_ref_step(size_t key_dim, size_t value_dim,
                            const float *state, const float *transition,
                            const float *injection, float *next) {
    for (size_t row = 0; row < value_dim; row++) {
        for (size_t column = 0; column < key_dim; column++) {
            double sum = injection[row * key_dim + column];
            for (size_t inner = 0; inner < key_dim; inner++)
                sum += (double)state[row * key_dim + inner] *
                       transition[inner * key_dim + column];
            next[row * key_dim + column] = (float)sum;
        }
    }
}

int h3_vdn_ref_scan(size_t frames, size_t key_dim, size_t value_dim,
                    const float *transitions, const float *injections,
                    const float *text_state, float *prefix, float *suffix) {
    size_t state_size;
    if (!frames || !transitions || !injections || !prefix || !suffix ||
        !h3_vdn_ref_matrix_size(value_dim, key_dim, &state_size) ||
        frames > SIZE_MAX / state_size)
        return 0;
    if (state_size > SIZE_MAX / 2) return 0;
    float *storage = calloc(state_size * 2, sizeof(*storage));
    if (!storage) return 0;
    float *state = storage;
    float *next = storage + state_size;
    if (text_state)
        memcpy(state, text_state, state_size * sizeof(*state));
    for (size_t frame = 0; frame < frames; frame++) {
        h3_vdn_ref_step(key_dim, value_dim, state,
                        transitions + frame * key_dim * key_dim,
                        injections + frame * state_size, next);
        memcpy(prefix + frame * state_size, next,
               state_size * sizeof(*next));
        float *swap = state;
        state = next;
        next = swap;
    }
    memset(state, 0, state_size * sizeof(*state));
    if (text_state)
        memcpy(state, text_state, state_size * sizeof(*state));
    for (size_t frame = frames; frame-- > 0;) {
        h3_vdn_ref_step(key_dim, value_dim, state,
                        transitions + frame * key_dim * key_dim,
                        injections + frame * state_size, next);
        memcpy(suffix + frame * state_size, next,
               state_size * sizeof(*next));
        float *swap = state;
        state = next;
        next = swap;
    }
    free(storage);
    return 1;
}

int h3_vdn_ref_gather(size_t frames, size_t key_dim, size_t value_dim,
                      const float *prefix, const float *suffix,
                      const float *alpha, const h3_vdn_ref_window *bounds,
                      const float *text_state, int bridge_alpha, float *output) {
    size_t state_size;
    if (!frames || frames > (size_t)INT_MAX || !prefix || !suffix || !alpha ||
        !bounds || !output ||
        !h3_vdn_ref_matrix_size(value_dim, key_dim, &state_size))
        return 0;
    for (size_t frame = 0; frame < frames; frame++) {
        long long last_before = (long long)bounds[frame].lo - 1;
        long long first_after = (long long)bounds[frame].hi + 1;
        int has_before = last_before >= 0;
        int has_after = first_after < (long long)frames;
        if (last_before >= (long long)frames || first_after < 0 ||
            bounds[frame].lo > (int)frame ||
            bounds[frame].hi < (int)frame)
            return 0;
        size_t before_index = has_before ? (size_t)last_before : 0;
        size_t after_index = has_after ? (size_t)first_after : frames - 1;
        int bridge_before = bounds[frame].lo > 0 ? bounds[frame].lo : 0;
        int bridge_after = first_after < (long long)frames ?
            (int)first_after : (int)frames;
        for (size_t column = 0; column < key_dim; column++) {
            double before_scale = 1.0;
            double after_scale = 1.0;
            if (bridge_alpha) {
                double before_log = 0.0;
                double after_log = 0.0;
                for (int index = bridge_before; index <= (int)frame; index++) {
                    double value = alpha[(size_t)index * key_dim + column];
                    before_log += log(value < 1e-12 ? 1e-12 : value);
                }
                for (int index = (int)frame; index < bridge_after; index++) {
                    double value = alpha[(size_t)index * key_dim + column];
                    after_log += log(value < 1e-12 ? 1e-12 : value);
                }
                before_scale = exp(before_log);
                after_scale = exp(after_log);
            }
            for (size_t row = 0; row < value_dim; row++) {
                size_t element = row * key_dim + column;
                float before = has_before ?
                    prefix[before_index * state_size + element] :
                    (text_state ? text_state[element] : 0.0f);
                float after = has_after ?
                    suffix[after_index * state_size + element] :
                    (text_state ? text_state[element] : 0.0f);
                output[frame * state_size + element] =
                    (float)(before * before_scale + after * after_scale);
            }
        }
    }
    return 1;
}
