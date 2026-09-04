#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

constexpr unsigned kThreads = 256;

__global__ void fill_bf16(hip_bfloat16 *values, std::size_t elements,
                          float value) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (index < elements) values[index] = hip_bfloat16(value);
}

__global__ void fill_f32(float *values, std::size_t elements, float value) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (index < elements) values[index] = value;
}

__global__ void quantize_rows(const hip_bfloat16 *input, std::int8_t *output,
                              float *scales, std::uint32_t rows,
                              std::uint32_t columns) {
    std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float reductions[kThreads];
    float maximum = 0.0f;
    std::size_t offset = static_cast<std::size_t>(row) * columns;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x)
        maximum = fmaxf(maximum, fabsf(static_cast<float>(input[offset + column])));
    reductions[threadIdx.x] = maximum;
    __syncthreads();
    for (unsigned width = kThreads / 2; width; width >>= 1) {
        if (threadIdx.x < width)
            reductions[threadIdx.x] = fmaxf(
                reductions[threadIdx.x], reductions[threadIdx.x + width]);
        __syncthreads();
    }
    float scale = reductions[0] > 0.0f ? reductions[0] / 127.0f : 1.0f;
    if (threadIdx.x == 0) scales[row] = scale;
    float inverse = 1.0f / scale;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
        float quantized = nearbyintf(static_cast<float>(input[offset + column]) *
                                     inverse);
        quantized = fminf(127.0f, fmaxf(-127.0f, quantized));
        output[offset + column] = static_cast<std::int8_t>(quantized);
    }
}

__global__ void dequantize_output(const std::int32_t *input,
                                  hip_bfloat16 *output,
                                  const float *row_scales,
                                  const float *weight_scales,
                                  std::size_t elements,
                                  std::uint32_t columns) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (index >= elements) return;
    std::uint32_t row = static_cast<std::uint32_t>(index / columns);
    std::uint32_t column = static_cast<std::uint32_t>(index % columns);
    output[index] = hip_bfloat16(static_cast<float>(input[index]) *
                                row_scales[row] * weight_scales[column]);
}

bool checked_product(std::size_t left, std::size_t right,
                     std::size_t *result) {
    if (right && left > std::numeric_limits<std::size_t>::max() / right)
        return false;
    *result = left * right;
    return true;
}

bool parse_u32(const char *text, std::uint32_t *value) {
    char *end = nullptr;
    unsigned long parsed = std::strtoul(text, &end, 10);
    if (!text[0] || !end || *end || !parsed ||
        parsed > std::numeric_limits<std::uint32_t>::max()) return false;
    *value = static_cast<std::uint32_t>(parsed);
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s ROWS OUTPUT_DIM INPUT_DIM ITERATIONS\n",
                     argv[0]);
        return 2;
    }
    std::uint32_t rows, output_dim, input_dim, iterations;
    if (!parse_u32(argv[1], &rows) || !parse_u32(argv[2], &output_dim) ||
        !parse_u32(argv[3], &input_dim) || !parse_u32(argv[4], &iterations)) {
        std::fprintf(stderr, "all dimensions and iterations must be positive u32\n");
        return 2;
    }
    if (rows > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        output_dim > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        input_dim > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "rocBLAS dimensions exceed i32\n");
        return 2;
    }
    std::size_t input_elements, weight_elements, output_elements;
    if (!checked_product(rows, input_dim, &input_elements) ||
        !checked_product(output_dim, input_dim, &weight_elements) ||
        !checked_product(rows, output_dim, &output_elements)) {
        std::fprintf(stderr, "tensor size overflow\n");
        return 2;
    }
    std::size_t input_blocks_wide =
        (input_elements + kThreads - 1) / kThreads;
    std::size_t weight_blocks_wide =
        (weight_elements + kThreads - 1) / kThreads;
    std::size_t output_blocks_wide =
        (output_elements + kThreads - 1) / kThreads;
    if (input_blocks_wide > std::numeric_limits<std::uint32_t>::max() ||
        weight_blocks_wide > std::numeric_limits<std::uint32_t>::max() ||
        output_blocks_wide > std::numeric_limits<std::uint32_t>::max()) {
        std::fprintf(stderr, "kernel grid exceeds u32\n");
        return 2;
    }
    std::uint32_t input_blocks = static_cast<std::uint32_t>(input_blocks_wide);
    std::uint32_t weight_blocks = static_cast<std::uint32_t>(weight_blocks_wide);
    std::uint32_t output_blocks = static_cast<std::uint32_t>(output_blocks_wide);

    hipStream_t stream = nullptr;
    hipEvent_t begin = nullptr, end = nullptr;
    rocblas_handle handle = nullptr;
    hip_bfloat16 *input_bf16 = nullptr, *weight_bf16 = nullptr;
    hip_bfloat16 *output_bf16 = nullptr, *output_dequantized = nullptr;
    std::int8_t *input_i8 = nullptr, *weight_i8 = nullptr;
    std::int32_t *output_i32 = nullptr;
    float *input_scales = nullptr, *weight_scales = nullptr;
    int result = 1;
    float bf16_ms = 0.0f, int8_ms = 0.0f;
    const float bf16_alpha = 1.0f, bf16_beta = 0.0f;
    const std::int32_t i8_alpha = 1, i8_beta = 0;
    auto bf16_gemm = [&]() {
        return (rocblas_gemm_ex)(
            handle, rocblas_operation_transpose, rocblas_operation_none,
            static_cast<int>(output_dim), static_cast<int>(rows),
            static_cast<int>(input_dim), &bf16_alpha, weight_bf16,
            rocblas_datatype_bf16_r, static_cast<int>(input_dim), input_bf16,
            rocblas_datatype_bf16_r, static_cast<int>(input_dim), &bf16_beta,
            output_bf16, rocblas_datatype_bf16_r,
            static_cast<int>(output_dim), output_bf16,
            rocblas_datatype_bf16_r, static_cast<int>(output_dim),
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    };
    auto int8_pipeline = [&]() {
        quantize_rows<<<rows, kThreads, 0, stream>>>(
            input_bf16, input_i8, input_scales, rows, input_dim);
        rocblas_status status = (rocblas_gemm_ex)(
            handle, rocblas_operation_transpose, rocblas_operation_none,
            static_cast<int>(output_dim), static_cast<int>(rows),
            static_cast<int>(input_dim), &i8_alpha, weight_i8,
            rocblas_datatype_i8_r, static_cast<int>(input_dim), input_i8,
            rocblas_datatype_i8_r, static_cast<int>(input_dim), &i8_beta,
            output_i32, rocblas_datatype_i32_r,
            static_cast<int>(output_dim), output_i32,
            rocblas_datatype_i32_r, static_cast<int>(output_dim),
            rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0);
        dequantize_output<<<output_blocks, kThreads, 0, stream>>>(
            output_i32, output_dequantized, input_scales, weight_scales,
            output_elements, output_dim);
        return status;
    };

#define HIP(call) do {                                                        \
    hipError_t status = (call);                                               \
    if (status != hipSuccess) {                                               \
        std::fprintf(stderr, "%s failed: %s\n", #call,                      \
                     hipGetErrorString(status));                              \
        goto cleanup;                                                         \
    }                                                                         \
} while (0)
#define BLAS(call) do {                                                       \
    rocblas_status status = (call);                                           \
    if (status != rocblas_status_success) {                                   \
        std::fprintf(stderr, "%s failed: rocBLAS status %d\n", #call,       \
                     static_cast<int>(status));                               \
        goto cleanup;                                                         \
    }                                                                         \
} while (0)

    HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP(hipEventCreate(&begin));
    HIP(hipEventCreate(&end));
    BLAS(rocblas_create_handle(&handle));
    BLAS(rocblas_set_stream(handle, stream));
    HIP(hipMalloc(&input_bf16, input_elements * sizeof(*input_bf16)));
    HIP(hipMalloc(&weight_bf16, weight_elements * sizeof(*weight_bf16)));
    HIP(hipMalloc(&output_bf16, output_elements * sizeof(*output_bf16)));
    HIP(hipMalloc(&output_dequantized,
                  output_elements * sizeof(*output_dequantized)));
    HIP(hipMalloc(&input_i8, input_elements * sizeof(*input_i8)));
    HIP(hipMalloc(&weight_i8, weight_elements * sizeof(*weight_i8)));
    HIP(hipMalloc(&output_i32, output_elements * sizeof(*output_i32)));
    HIP(hipMalloc(&input_scales, rows * sizeof(*input_scales)));
    HIP(hipMalloc(&weight_scales, output_dim * sizeof(*weight_scales)));

    fill_bf16<<<input_blocks, kThreads, 0, stream>>>(
        input_bf16, input_elements, 0.125f);
    fill_bf16<<<weight_blocks, kThreads, 0, stream>>>(
        weight_bf16, weight_elements, 0.0078125f);
    HIP(hipMemsetAsync(weight_i8, 1, weight_elements, stream));
    fill_f32<<<(output_dim + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        weight_scales, output_dim, 0.0078125f);
    HIP(hipGetLastError());

    for (unsigned warmup = 0; warmup < 2; ++warmup) BLAS(bf16_gemm());
    HIP(hipStreamSynchronize(stream));
    HIP(hipEventRecord(begin, stream));
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
        BLAS(bf16_gemm());
    HIP(hipEventRecord(end, stream));
    HIP(hipEventSynchronize(end));
    HIP(hipEventElapsedTime(&bf16_ms, begin, end));
    bf16_ms /= static_cast<float>(iterations);

    for (unsigned warmup = 0; warmup < 2; ++warmup) BLAS(int8_pipeline());
    HIP(hipGetLastError());
    HIP(hipStreamSynchronize(stream));
    HIP(hipEventRecord(begin, stream));
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
        BLAS(int8_pipeline());
    HIP(hipGetLastError());
    HIP(hipEventRecord(end, stream));
    HIP(hipEventSynchronize(end));
    HIP(hipEventElapsedTime(&int8_ms, begin, end));
    int8_ms /= static_cast<float>(iterations);

    std::printf(
        "VDN INT8 GEMM probe: rows=%u output=%u input=%u iterations=%u "
        "bf16_ms=%.6f int8_quant_gemm_dequant_ms=%.6f speedup=%.6fx "
        "bf16_weight_gib=%.6f int8_weight_gib=%.6f\n",
        rows, output_dim, input_dim, iterations, bf16_ms, int8_ms,
        int8_ms > 0.0f ? bf16_ms / int8_ms : 0.0f,
        static_cast<double>(weight_elements * sizeof(*weight_bf16)) /
            (1024.0 * 1024.0 * 1024.0),
        static_cast<double>(weight_elements * sizeof(*weight_i8) +
                            output_dim * sizeof(*weight_scales)) /
            (1024.0 * 1024.0 * 1024.0));
    result = 0;

cleanup:
    (void)hipFree(weight_scales);
    (void)hipFree(input_scales);
    (void)hipFree(output_i32);
    (void)hipFree(weight_i8);
    (void)hipFree(input_i8);
    (void)hipFree(output_dequantized);
    (void)hipFree(output_bf16);
    (void)hipFree(weight_bf16);
    (void)hipFree(input_bf16);
    if (handle) (void)rocblas_destroy_handle(handle);
    if (end) (void)hipEventDestroy(end);
    if (begin) (void)hipEventDestroy(begin);
    if (stream) (void)hipStreamDestroy(stream);
    return result;

#undef BLAS
#undef HIP
}
