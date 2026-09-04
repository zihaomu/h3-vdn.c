#include "vdn_reference.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1;                                                            \
    }                                                                        \
} while (0)

static int close_array(const float *actual, const float *expected,
                       size_t count, float tolerance, const char *label) {
    for (size_t index = 0; index < count; index++) {
        float error = fabsf(actual[index] - expected[index]);
        if (error > tolerance) {
            fprintf(stderr,
                    "FAIL %s[%zu]: %.9g != %.9g (error %.9g > %.9g)\n",
                    label, index, actual[index], expected[index], error,
                    tolerance);
            return 0;
        }
    }
    return 1;
}

static int test_windows(void) {
    h3_vdn_ref_window bounds[12];
    CHECK(h3_vdn_ref_window_bounds(12, 1, 5, bounds));
    for (size_t frame = 0; frame < 5; frame++)
        CHECK(bounds[frame].lo == -5 && bounds[frame].hi == 9);
    for (size_t frame = 5; frame < 10; frame++)
        CHECK(bounds[frame].lo == 0 && bounds[frame].hi == 14);
    for (size_t frame = 10; frame < 12; frame++)
        CHECK(bounds[frame].lo == 5 && bounds[frame].hi == 19);

    h3_vdn_ref_window inner[10];
    CHECK(h3_vdn_ref_prune_anchor_bounds(12, bounds, inner) == 10);
    CHECK(inner[0].lo == -6 && inner[0].hi == 8);
    CHECK(inner[3].lo == -6 && inner[3].hi == 8);
    CHECK(inner[4].lo == -1 && inner[4].hi == 13);
    CHECK(inner[8].lo == -1 && inner[8].hi == 13);
    CHECK(inner[9].lo == 4 && inner[9].hi == 18);
    CHECK(h3_vdn_ref_prune_anchor_bounds(2, bounds, inner) == 0);
    CHECK(!h3_vdn_ref_window_bounds(0, 1, 5, bounds));
    CHECK(!h3_vdn_ref_window_bounds(12, -1, 5, bounds));
    return 0;
}

static int test_statistics(void) {
    static const float key[] = {
        1.0f, 2.0f, -1.0f, 0.5f, 0.25f, -0.75f
    };
    static const float value[] = {
        0.5f, -1.0f, 2.0f, 0.25f, -0.5f, 1.5f
    };
    static const float beta[] = {0.2f, 0.6f, 0.8f};
    static const float expected_a[] = {0.85f, -0.05f, -0.05f, 1.4f};
    static const float expected_b[] = {-1.2f, 1.1f, -0.05f, -1.225f};
    float a[4];
    float b[4];
    h3_vdn_ref_frame_statistics(3, 2, 2, key, value, beta, a, b);
    CHECK(close_array(a, expected_a, 4, 1e-6f, "A"));
    CHECK(close_array(b, expected_b, 4, 1e-6f, "B"));
    CHECK(a[1] == a[2]);
    return 0;
}

static int test_solve_scan_gather(void) {
    static const float alphas[] = {
        0.8f, 0.9f, 0.75f, 0.86f, 0.7f, 0.82f, 0.65f, 0.78f
    };
    static const float matrices_a[] = {
        0.2f, 0.05f, 0.05f, 0.4f,
        0.3f, -0.05f, -0.05f, 0.45f,
        0.4f, 0.05f, 0.05f, 0.5f,
        0.5f, -0.05f, -0.05f, 0.55f
    };
    static const float matrices_b[] = {
        0.1f, -0.05f, 0.03f, 0.08f,
        0.2f, -0.1f, 0.06f, 0.16f,
        0.3f, -0.15f, 0.09f, 0.24f,
        0.4f, -0.2f, 0.12f, 0.32f
    };
    static const float expected_transitions[] = {
        0.667660209f, -0.0238450075f, -0.0268256334f, 0.643815201f,
        0.577689243f, 0.0199203187f, 0.0228419655f, 0.593891102f,
        0.500595948f, -0.0166865316f, -0.0195470799f, 0.547318236f,
        0.433799785f, 0.0139935414f, 0.0167922497f, 0.503767492f
    };
    static const float expected_injections[] = {
        0.084947839f, -0.0387481371f, 0.0226527571f, 0.0563338301f,
        0.151394422f, -0.0637450199f, 0.0504648074f, 0.112084993f,
        0.218116806f, -0.10727056f, 0.0586412396f, 0.158045292f,
        0.262648009f, -0.120559742f, 0.0869752422f, 0.209257266f
    };
    static const float expected_prefix[] = {
        0.221162444f, -0.107898659f, 0.0520119225f, 0.15171386f,
        0.27669297f, -0.123419447f, 0.0839769783f, 0.203222599f,
        0.359040675f, -0.17943732f, 0.0967073662f, 0.267871442f,
        0.41538662f, -0.20593018f, 0.133425041f, 0.345555469f
    };
    static const float expected_suffix[] = {
        0.340200934f, -0.16208255f, 0.0950012237f, 0.246490596f,
        0.375171242f, -0.177672803f, 0.120407498f, 0.299818773f,
        0.395475007f, -0.205097821f, 0.108718411f, 0.312461449f,
        0.347728741f, -0.168137783f, 0.111184069f, 0.285522067f
    };
    static const float expected_gather[] = {
        0.397285004f, -0.248745714f, 0.105231047f, 0.376845162f,
        0.302557589f, -0.195970764f, 0.0883716362f, 0.317450161f,
        0.207110283f, -0.140050134f, 0.0500562593f, 0.202928614f,
        0.255895301f, -0.156939078f, 0.0707095251f, 0.246981174f
    };
    static const float text_state[] = {0.2f, -0.1f, 0.05f, 0.15f};
    float transitions[16];
    float injections[16];
    for (size_t frame = 0; frame < 4; frame++) {
        CHECK(h3_vdn_ref_vdn_factor(
            2, 2, alphas + frame * 2, matrices_a + frame * 4,
            matrices_b + frame * 4, transitions + frame * 4,
            injections + frame * 4));
    }
    CHECK(close_array(transitions, expected_transitions, 16, 2e-6f,
                      "transition"));
    CHECK(close_array(injections, expected_injections, 16, 2e-6f,
                      "injection"));

    float prefix[16];
    float suffix[16];
    CHECK(h3_vdn_ref_scan(4, 2, 2, transitions, injections, text_state,
                          prefix, suffix));
    CHECK(close_array(prefix, expected_prefix, 16, 3e-6f, "prefix"));
    CHECK(close_array(suffix, expected_suffix, 16, 3e-6f, "suffix"));

    h3_vdn_ref_window bounds[4];
    CHECK(h3_vdn_ref_window_bounds(4, 1, 0, bounds));
    float gathered[16];
    CHECK(h3_vdn_ref_gather(4, 2, 2, prefix, suffix, alphas, bounds,
                            text_state, 1, gathered));
    CHECK(close_array(gathered, expected_gather, 16, 4e-6f, "gather"));

    float complement[16];
    CHECK(h3_vdn_ref_gather(4, 2, 2, prefix, suffix, alphas, bounds,
                            NULL, 0, complement));
    CHECK(close_array(complement, suffix + 8, 4, 0.0f, "left boundary"));
    CHECK(close_array(complement + 4, suffix + 12, 4, 0.0f,
                      "second frame"));
    CHECK(close_array(complement + 8, prefix, 4, 0.0f, "third frame"));
    CHECK(close_array(complement + 12, prefix + 4, 4, 0.0f,
                      "right boundary"));

    static const float invalid_a[] = {-2.0f, 0.0f, 0.0f, 0.0f};
    CHECK(!h3_vdn_ref_vdn_factor(2, 2, alphas, invalid_a, matrices_b,
                                 transitions, injections));
    return 0;
}

int main(void) {
    CHECK(test_windows() == 0);
    CHECK(test_statistics() == 0);
    CHECK(test_solve_scan_gather() == 0);
    puts("VDN CPU reference tests passed");
    return 0;
}
