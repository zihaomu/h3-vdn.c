#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int parse_positive(const char *text, uint64_t *value) {
    if (!text || !*text || !value) return 0;
    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || end == text || *end || !parsed) return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static int checked_product(uint64_t left, uint64_t right, uint64_t *output) {
    if (!output || (right && left > UINT64_MAX / right)) return 0;
    *output = left * right;
    return 1;
}

static int exact_file_size(FILE *file, uint64_t expected) {
    if (fseeko(file, 0, SEEK_END) != 0) return 0;
    off_t size = ftello(file);
    return size >= 0 && (uint64_t)size == expected &&
           fseeko(file, 0, SEEK_SET) == 0;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s BASE.f32 CANDIDATE.f32 FRAMES HEIGHT WIDTH\n",
                argv[0]);
        return 2;
    }
    uint64_t frames, height, width, pixels, frame_elements, total_elements;
    if (!parse_positive(argv[3], &frames) ||
        !parse_positive(argv[4], &height) ||
        !parse_positive(argv[5], &width) ||
        !checked_product(height, width, &pixels) ||
        !checked_product(pixels, 3, &frame_elements) ||
        !checked_product(frames, frame_elements, &total_elements) ||
        total_elements > UINT64_MAX / sizeof(float) ||
        frame_elements > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "invalid or overflowing F32 geometry\n");
        return 2;
    }
    FILE *base = fopen(argv[1], "rb");
    FILE *candidate = fopen(argv[2], "rb");
    uint64_t expected_bytes = total_elements * sizeof(float);
    if (!base || !candidate || !exact_file_size(base, expected_bytes) ||
        !exact_file_size(candidate, expected_bytes)) {
        fprintf(stderr, "cannot open exact-size F32 inputs (%" PRIu64
                        " bytes expected)\n", expected_bytes);
        if (base) fclose(base);
        if (candidate) fclose(candidate);
        return 1;
    }
    float *left = malloc((size_t)frame_elements * sizeof(*left));
    float *right = malloc((size_t)frame_elements * sizeof(*right));
    if (!left || !right) {
        fprintf(stderr, "out of memory allocating one F32 frame\n");
        free(left); free(right); fclose(base); fclose(candidate);
        return 1;
    }

    long double squared_error = 0.0L, base_squared = 0.0L;
    long double candidate_squared = 0.0L, dot = 0.0L;
    long double absolute_error = 0.0L;
    double maximum_error = 0.0;
    double minimum_frame_psnr = DBL_MAX, maximum_frame_psnr = -DBL_MAX;
    double minimum_frame_relative = DBL_MAX;
    double maximum_frame_relative = -DBL_MAX;
    int ok = 1;
    for (uint64_t frame = 0; frame < frames; frame++) {
        if (fread(left, sizeof(*left), (size_t)frame_elements, base) !=
                frame_elements ||
            fread(right, sizeof(*right), (size_t)frame_elements, candidate) !=
                frame_elements) {
            ok = 0;
            break;
        }
        long double frame_error = 0.0L, frame_base = 0.0L;
        for (uint64_t index = 0; index < frame_elements; index++) {
            double a = left[index], b = right[index], difference = b - a;
            double magnitude = fabs(difference);
            frame_error += (long double)difference * difference;
            frame_base += (long double)a * a;
            candidate_squared += (long double)b * b;
            dot += (long double)a * b;
            absolute_error += magnitude;
            if (magnitude > maximum_error) maximum_error = magnitude;
        }
        squared_error += frame_error;
        base_squared += frame_base;
        double frame_rmse = sqrt((double)(frame_error / frame_elements));
        double frame_psnr = frame_rmse > 0.0 ?
            20.0 * log10(1.0 / frame_rmse) : INFINITY;
        double frame_relative = frame_base > 0.0L ?
            100.0 * sqrt((double)(frame_error / frame_base)) : 0.0;
        if (frame_psnr < minimum_frame_psnr)
            minimum_frame_psnr = frame_psnr;
        if (frame_psnr > maximum_frame_psnr)
            maximum_frame_psnr = frame_psnr;
        if (frame_relative < minimum_frame_relative)
            minimum_frame_relative = frame_relative;
        if (frame_relative > maximum_frame_relative)
            maximum_frame_relative = frame_relative;
    }
    if (fclose(base) != 0 || fclose(candidate) != 0) ok = 0;
    if (!ok) {
        fprintf(stderr, "cannot read complete F32 inputs\n");
        free(left); free(right);
        return 1;
    }
    double rmse = sqrt((double)(squared_error / total_elements));
    double relative = base_squared > 0.0L ?
        100.0 * sqrt((double)(squared_error / base_squared)) : 0.0;
    double cosine = base_squared > 0.0L && candidate_squared > 0.0L ?
        (double)(dot / sqrtl(base_squared * candidate_squared)) : 0.0;
    double psnr = rmse > 0.0 ? 20.0 * log10(1.0 / rmse) : INFINITY;
    printf("elements=%" PRIu64 "\n", total_elements);
    printf("rmse=%.12g\n", rmse);
    printf("relative_rmse_percent=%.12g\n", relative);
    printf("cosine=%.15g\n", cosine);
    printf("psnr_db=%.9f\n", psnr);
    printf("mean_abs=%.12g\n",
           (double)(absolute_error / total_elements));
    printf("max_abs=%.12g\n", maximum_error);
    printf("frame_psnr_min=%.9f frame_psnr_max=%.9f\n",
           minimum_frame_psnr, maximum_frame_psnr);
    printf("frame_relative_rmse_min_percent=%.12g "
           "frame_relative_rmse_max_percent=%.12g\n",
           minimum_frame_relative, maximum_frame_relative);
    free(left); free(right);
    return 0;
}
