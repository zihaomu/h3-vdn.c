#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kGroups = 5;
constexpr float kE4m3Maximum = 448.0f;
constexpr size_t kMaximumWorkspace = 256u * 1024u * 1024u;

__global__ void fill_bf16(hip_bfloat16 *values, std::size_t elements,
                          float value) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (index < elements) values[index] = hip_bfloat16(value);
}

__global__ void quantize_rows_e4m3(const hip_bfloat16 *input,
                                   __hip_fp8_e4m3 *output, float *scales,
                                   std::uint32_t rows,
                                   std::uint32_t columns) {
    std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float reductions[kThreads];
    float maximum = 0.0f;
    std::size_t offset = static_cast<std::size_t>(row) * columns;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x)
        maximum = fmaxf(maximum,
                        fabsf(static_cast<float>(input[offset + column])));
    reductions[threadIdx.x] = maximum;
    __syncthreads();
    for (unsigned width = kThreads / 2; width; width >>= 1) {
        if (threadIdx.x < width)
            reductions[threadIdx.x] = fmaxf(
                reductions[threadIdx.x], reductions[threadIdx.x + width]);
        __syncthreads();
    }
    float scale = reductions[0] > 0.0f ?
        reductions[0] / kE4m3Maximum : 1.0f;
    if (threadIdx.x == 0) scales[row] = scale;
    float inverse = reductions[0] > 0.0f ? 1.0f / scale : 0.0f;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x)
        output[offset + column] = __hip_fp8_e4m3(
            static_cast<float>(input[offset + column]) * inverse);
}

__global__ void dequantize_output(const float *input,
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
    output[index] = hip_bfloat16(input[index] * row_scales[row] *
                                weight_scales[column]);
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

const char *hipblaslt_status_name(hipblasStatus_t status) {
    switch (status) {
        case HIPBLAS_STATUS_SUCCESS: return "success";
        case HIPBLAS_STATUS_NOT_INITIALIZED: return "not initialized";
        case HIPBLAS_STATUS_ALLOC_FAILED: return "allocation failed";
        case HIPBLAS_STATUS_INVALID_VALUE: return "invalid value";
        case HIPBLAS_STATUS_ARCH_MISMATCH: return "architecture mismatch";
        case HIPBLAS_STATUS_MAPPING_ERROR: return "mapping error";
        case HIPBLAS_STATUS_EXECUTION_FAILED: return "execution failed";
        case HIPBLAS_STATUS_INTERNAL_ERROR: return "internal error";
        case HIPBLAS_STATUS_NOT_SUPPORTED: return "not supported";
        default: return "unknown";
    }
}

float median_five(const float values[kGroups]) {
    float sorted[kGroups];
    for (unsigned index = 0; index < kGroups; index++)
        sorted[index] = values[index];
    for (unsigned left = 1; left < kGroups; left++) {
        float value = sorted[left];
        unsigned right = left;
        while (right && sorted[right - 1] > value) {
            sorted[right] = sorted[right - 1];
            right--;
        }
        sorted[right] = value;
    }
    return sorted[kGroups / 2];
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
        output_dim >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        input_dim >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "BLAS dimensions exceed i32\n");
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
    auto input_blocks = static_cast<std::uint32_t>(input_blocks_wide);
    auto weight_blocks = static_cast<std::uint32_t>(weight_blocks_wide);
    auto output_blocks = static_cast<std::uint32_t>(output_blocks_wide);

    hipStream_t stream = nullptr;
    hipEvent_t begin = nullptr, end = nullptr;
    rocblas_handle rocblas = nullptr;
    hipblasLtHandle_t hipblaslt = nullptr;
    hipblasLtMatmulDesc_t operation = nullptr;
    hipblasLtMatrixLayout_t weight_layout = nullptr, input_layout = nullptr;
    hipblasLtMatrixLayout_t output_layout = nullptr;
    hipblasLtMatmulPreference_t preference = nullptr;
    hip_bfloat16 *input_bf16 = nullptr, *weight_bf16 = nullptr;
    hip_bfloat16 *output_bf16 = nullptr, *output_dequantized = nullptr;
    __hip_fp8_e4m3 *input_fp8 = nullptr, *weight_fp8 = nullptr;
    float *output_f32 = nullptr, *input_scales = nullptr;
    float *weight_scales = nullptr;
    void *workspace = nullptr;
    int result = 1;
    float bf16_groups[kGroups] = {0.0f};
    float fp8_groups[kGroups] = {0.0f};
    float bf16_ms = 0.0f, fp8_ms = 0.0f;
    const float alpha = 1.0f, beta = 0.0f;
    hipblasLtMatmulHeuristicResult_t heuristics[32];
    int algorithm_count = 0;

#define HIP(call) do {                                                        \
    hipError_t status = (call);                                               \
    if (status != hipSuccess) {                                               \
        std::fprintf(stderr, "%s failed: %s\n", #call,                    \
                     hipGetErrorString(status));                              \
        goto cleanup;                                                         \
    }                                                                         \
} while (0)
#define ROCBLAS(call) do {                                                    \
    rocblas_status status = (call);                                           \
    if (status != rocblas_status_success) {                                   \
        std::fprintf(stderr, "%s failed: rocBLAS status %d\n", #call,     \
                     static_cast<int>(status));                               \
        goto cleanup;                                                         \
    }                                                                         \
} while (0)
#define HIPBLASLT(call) do {                                                  \
    hipblasStatus_t status = (call);                                          \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                   \
        std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                     hipblaslt_status_name(status), static_cast<int>(status));\
        goto cleanup;                                                         \
    }                                                                         \
} while (0)

    HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP(hipEventCreate(&begin));
    HIP(hipEventCreate(&end));
    ROCBLAS(rocblas_create_handle(&rocblas));
    ROCBLAS(rocblas_set_stream(rocblas, stream));
    HIPBLASLT(hipblasLtCreate(&hipblaslt));

    HIP(hipMalloc(&input_bf16, input_elements * sizeof(*input_bf16)));
    HIP(hipMalloc(&weight_bf16, weight_elements * sizeof(*weight_bf16)));
    HIP(hipMalloc(&output_bf16, output_elements * sizeof(*output_bf16)));
    HIP(hipMalloc(&output_dequantized,
                  output_elements * sizeof(*output_dequantized)));
    HIP(hipMalloc(&input_fp8, input_elements * sizeof(*input_fp8)));
    HIP(hipMalloc(&weight_fp8, weight_elements * sizeof(*weight_fp8)));
    HIP(hipMalloc(&output_f32, output_elements * sizeof(*output_f32)));
    HIP(hipMalloc(&input_scales, rows * sizeof(*input_scales)));
    HIP(hipMalloc(&weight_scales, output_dim * sizeof(*weight_scales)));
    HIP(hipMalloc(&workspace, kMaximumWorkspace));

    fill_bf16<<<input_blocks, kThreads, 0, stream>>>(
        input_bf16, input_elements, 0.125f);
    fill_bf16<<<weight_blocks, kThreads, 0, stream>>>(
        weight_bf16, weight_elements, 0.0078125f);
    quantize_rows_e4m3<<<output_dim, kThreads, 0, stream>>>(
        weight_bf16, weight_fp8, weight_scales, output_dim, input_dim);
    HIP(hipGetLastError());
    HIP(hipStreamSynchronize(stream));

    HIPBLASLT(hipblasLtMatmulDescCreate(
        &operation, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    HIPBLASLT(hipblasLtMatrixLayoutCreate(
        &weight_layout, HIP_R_8F_E4M3, input_dim, output_dim, input_dim));
    HIPBLASLT(hipblasLtMatrixLayoutCreate(
        &input_layout, HIP_R_8F_E4M3, input_dim, rows, input_dim));
    HIPBLASLT(hipblasLtMatrixLayoutCreate(
        &output_layout, HIP_R_32F, output_dim, rows, output_dim));
    {
        hipblasOperation_t transpose = HIPBLAS_OP_T;
        hipblasOperation_t identity = HIPBLAS_OP_N;
        HIPBLASLT(hipblasLtMatmulDescSetAttribute(
            operation, HIPBLASLT_MATMUL_DESC_TRANSA,
            &transpose, sizeof(transpose)));
        HIPBLASLT(hipblasLtMatmulDescSetAttribute(
            operation, HIPBLASLT_MATMUL_DESC_TRANSB,
            &identity, sizeof(identity)));
    }
    HIPBLASLT(hipblasLtMatmulPreferenceCreate(&preference));
    HIPBLASLT(hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &kMaximumWorkspace, sizeof(kMaximumWorkspace)));
    HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(
        hipblaslt, operation, weight_layout, input_layout,
        output_layout, output_layout, preference,
        static_cast<int>(sizeof(heuristics) / sizeof(heuristics[0])),
        heuristics, &algorithm_count));
    if (algorithm_count < 1) {
        std::fprintf(stderr,
                     "hipBLASLt returned no gfx1201 OCP E4M3 algorithm\n");
        goto cleanup;
    }

    {
        auto bf16_gemm = [&]() {
            return (rocblas_gemm_ex)(
                rocblas, rocblas_operation_transpose,
                rocblas_operation_none, static_cast<int>(output_dim),
                static_cast<int>(rows), static_cast<int>(input_dim), &alpha,
                weight_bf16, rocblas_datatype_bf16_r,
                static_cast<int>(input_dim), input_bf16,
                rocblas_datatype_bf16_r, static_cast<int>(input_dim), &beta,
                output_bf16, rocblas_datatype_bf16_r,
                static_cast<int>(output_dim), output_bf16,
                rocblas_datatype_bf16_r, static_cast<int>(output_dim),
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
        };
        auto fp8_pipeline = [&]() {
            quantize_rows_e4m3<<<rows, kThreads, 0, stream>>>(
                input_bf16, input_fp8, input_scales, rows, input_dim);
            hipblasStatus_t status = hipblasLtMatmul(
                hipblaslt, operation, &alpha, weight_fp8, weight_layout,
                input_fp8, input_layout, &beta, output_f32, output_layout,
                output_f32, output_layout, &heuristics[0].algo, workspace,
                kMaximumWorkspace, stream);
            if (status == HIPBLAS_STATUS_SUCCESS)
                dequantize_output<<<output_blocks, kThreads, 0, stream>>>(
                    output_f32, output_dequantized, input_scales,
                    weight_scales, output_elements, output_dim);
            return status;
        };

        for (unsigned warmup = 0; warmup < 2; ++warmup) {
            ROCBLAS(bf16_gemm());
            HIPBLASLT(fp8_pipeline());
        }
        HIP(hipStreamSynchronize(stream));
#define MEASURE_BF16(destination) do {                                        \
    HIP(hipEventRecord(begin, stream));                                       \
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)   \
        ROCBLAS(bf16_gemm());                                                 \
    HIP(hipEventRecord(end, stream));                                         \
    HIP(hipEventSynchronize(end));                                            \
    HIP(hipEventElapsedTime(&(destination), begin, end));                     \
    (destination) /= static_cast<float>(iterations);                          \
} while (0)
#define MEASURE_FP8(destination) do {                                         \
    HIP(hipEventRecord(begin, stream));                                       \
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)   \
        HIPBLASLT(fp8_pipeline());                                            \
    HIP(hipGetLastError());                                                   \
    HIP(hipEventRecord(end, stream));                                         \
    HIP(hipEventSynchronize(end));                                            \
    HIP(hipEventElapsedTime(&(destination), begin, end));                     \
    (destination) /= static_cast<float>(iterations);                          \
} while (0)
        for (unsigned group = 0; group < kGroups; group++) {
            if (group % 2 == 0) {
                MEASURE_BF16(bf16_groups[group]);
                MEASURE_FP8(fp8_groups[group]);
            } else {
                MEASURE_FP8(fp8_groups[group]);
                MEASURE_BF16(bf16_groups[group]);
            }
        }
#undef MEASURE_FP8
#undef MEASURE_BF16
        bf16_ms = median_five(bf16_groups);
        fp8_ms = median_five(fp8_groups);
    }

    {
        hip_bfloat16 bf16_value, fp8_value;
        HIP(hipMemcpy(&bf16_value, output_bf16, sizeof(bf16_value),
                      hipMemcpyDeviceToHost));
        HIP(hipMemcpy(&fp8_value, output_dequantized, sizeof(fp8_value),
                      hipMemcpyDeviceToHost));
        float expected = static_cast<float>(input_dim) * 0.125f * 0.0078125f;
        float bf16_actual = static_cast<float>(bf16_value);
        float fp8_actual = static_cast<float>(fp8_value);
        float tolerance = fmaxf(0.0078125f, fabsf(expected) * 0.01f);
        if (fabsf(bf16_actual - expected) > tolerance ||
            fabsf(fp8_actual - expected) > tolerance) {
            std::fprintf(stderr,
                         "output validation failed: expected=%g bf16=%g fp8=%g\n",
                         expected, bf16_actual, fp8_actual);
            goto cleanup;
        }
        std::printf(
            "VDN FP8 GEMM probe: rows=%u output=%u input=%u iterations=%u "
            "bf16_ms=%.6f fp8_quant_gemm_dequant_ms=%.6f speedup=%.6fx "
            "bf16_weight_gib=%.6f fp8_weight_gib=%.6f algorithms=%d "
            "workspace_bytes=%zu expected=%.6f bf16=%.6f fp8=%.6f\n",
            rows, output_dim, input_dim, iterations, bf16_ms, fp8_ms,
            fp8_ms > 0.0f ? bf16_ms / fp8_ms : 0.0f,
            static_cast<double>(weight_elements * sizeof(*weight_bf16)) /
                (1024.0 * 1024.0 * 1024.0),
            static_cast<double>(weight_elements * sizeof(*weight_fp8) +
                                output_dim * sizeof(*weight_scales)) /
                (1024.0 * 1024.0 * 1024.0),
            algorithm_count, heuristics[0].workspaceSize,
            expected, bf16_actual, fp8_actual);
        std::printf(
            "VDN FP8 GEMM groups: bf16_ms=[%.6f,%.6f,%.6f,%.6f,%.6f] "
            "fp8_ms=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
            bf16_groups[0], bf16_groups[1], bf16_groups[2],
            bf16_groups[3], bf16_groups[4], fp8_groups[0], fp8_groups[1],
            fp8_groups[2], fp8_groups[3], fp8_groups[4]);
    }
    result = 0;

cleanup:
    if (preference) (void)hipblasLtMatmulPreferenceDestroy(preference);
    if (output_layout) (void)hipblasLtMatrixLayoutDestroy(output_layout);
    if (input_layout) (void)hipblasLtMatrixLayoutDestroy(input_layout);
    if (weight_layout) (void)hipblasLtMatrixLayoutDestroy(weight_layout);
    if (operation) (void)hipblasLtMatmulDescDestroy(operation);
    if (hipblaslt) (void)hipblasLtDestroy(hipblaslt);
    if (rocblas) (void)rocblas_destroy_handle(rocblas);
    (void)hipFree(workspace);
    (void)hipFree(weight_scales);
    (void)hipFree(input_scales);
    (void)hipFree(output_f32);
    (void)hipFree(weight_fp8);
    (void)hipFree(input_fp8);
    (void)hipFree(output_dequantized);
    (void)hipFree(output_bf16);
    (void)hipFree(weight_bf16);
    (void)hipFree(input_bf16);
    if (end) (void)hipEventDestroy(end);
    if (begin) (void)hipEventDestroy(begin);
    if (stream) (void)hipStreamDestroy(stream);
    return result;

#undef HIPBLASLT
#undef ROCBLAS
#undef HIP
}
