#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr std::uint32_t kRows = 16;
constexpr std::uint32_t kOutputs = 16;
constexpr std::uint32_t kInputs = 128;
constexpr unsigned kThreads = 256;
constexpr float kE4m3Maximum = 448.0f;
constexpr size_t kWorkspaceBytes = 64u * 1024u * 1024u;

__global__ void quantize_rows(const hip_bfloat16 *input,
                              __hip_fp8_e4m3 *output, float *scales,
                              std::uint32_t rows, std::uint32_t columns) {
    std::uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float reductions[kThreads];
    std::size_t offset = static_cast<std::size_t>(row) * columns;
    float maximum = 0.0f;
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
    float inverse = reductions[0] > 0.0f ? 1.0f / scale : 0.0f;
    if (threadIdx.x == 0) scales[row] = scale;
    for (std::uint32_t column = threadIdx.x; column < columns;
         column += blockDim.x)
        output[offset + column] = __hip_fp8_e4m3(
            static_cast<float>(input[offset + column]) * inverse);
}

__global__ void dequantize(const float *input, hip_bfloat16 *output,
                           const float *row_scales,
                           const float *weight_scales,
                           std::size_t elements) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (index >= elements) return;
    std::uint32_t row = static_cast<std::uint32_t>(index / kOutputs);
    std::uint32_t column = static_cast<std::uint32_t>(index % kOutputs);
    output[index] = hip_bfloat16(input[index] * row_scales[row] *
                                weight_scales[column]);
}

void quantize_cpu(const std::vector<hip_bfloat16> &input,
                  std::vector<__hip_fp8_e4m3> *output,
                  std::vector<float> *scales,
                  std::uint32_t rows, std::uint32_t columns) {
    for (std::uint32_t row = 0; row < rows; row++) {
        std::size_t offset = static_cast<std::size_t>(row) * columns;
        float maximum = 0.0f;
        for (std::uint32_t column = 0; column < columns; column++)
            maximum = fmaxf(
                maximum, fabsf(static_cast<float>(input[offset + column])));
        float scale = maximum > 0.0f ? maximum / kE4m3Maximum : 1.0f;
        float inverse = maximum > 0.0f ? 1.0f / scale : 0.0f;
        (*scales)[row] = scale;
        for (std::uint32_t column = 0; column < columns; column++)
            (*output)[offset + column] = __hip_fp8_e4m3(
                static_cast<float>(input[offset + column]) * inverse);
    }
}

}  // namespace

int main() {
    static_assert(sizeof(__hip_fp8_e4m3) == 1,
                  "OCP E4M3 must occupy one byte");
    constexpr std::size_t input_elements =
        static_cast<std::size_t>(kRows) * kInputs;
    constexpr std::size_t weight_elements =
        static_cast<std::size_t>(kOutputs) * kInputs;
    constexpr std::size_t output_elements =
        static_cast<std::size_t>(kRows) * kOutputs;
    std::vector<hip_bfloat16> input(input_elements);
    std::vector<hip_bfloat16> weight(weight_elements);
    std::vector<__hip_fp8_e4m3> expected_input(input_elements);
    std::vector<__hip_fp8_e4m3> expected_weight(weight_elements);
    std::vector<__hip_fp8_e4m3> actual_input(input_elements);
    std::vector<__hip_fp8_e4m3> actual_weight(weight_elements);
    std::vector<float> expected_input_scales(kRows);
    std::vector<float> expected_weight_scales(kOutputs);
    std::vector<float> actual_input_scales(kRows);
    std::vector<float> actual_weight_scales(kOutputs);
    std::vector<hip_bfloat16> expected_output(output_elements);
    std::vector<hip_bfloat16> actual_output(output_elements);
    for (std::uint32_t row = 0; row < kRows; row++)
        for (std::uint32_t column = 0; column < kInputs; column++) {
            int centered = static_cast<int>((row * 37 + column * 13 + 5) %
                                            97) - 48;
            input[static_cast<std::size_t>(row) * kInputs + column] =
                hip_bfloat16(row + 1 == kRows ? 0.0f :
                             static_cast<float>(centered) / 32.0f);
        }
    for (std::uint32_t row = 0; row < kOutputs; row++)
        for (std::uint32_t column = 0; column < kInputs; column++) {
            int centered = static_cast<int>((row * 29 + column * 17 + 11) %
                                            127) - 63;
            weight[static_cast<std::size_t>(row) * kInputs + column] =
                hip_bfloat16(row + 1 == kOutputs ? 0.0f :
                             static_cast<float>(centered) / 64.0f);
        }
    quantize_cpu(input, &expected_input, &expected_input_scales,
                 kRows, kInputs);
    quantize_cpu(weight, &expected_weight, &expected_weight_scales,
                 kOutputs, kInputs);
    for (std::uint32_t row = 0; row < kRows; row++)
        for (std::uint32_t output = 0; output < kOutputs; output++) {
            float sum = 0.0f;
            for (std::uint32_t input_index = 0; input_index < kInputs;
                 input_index++)
                sum += static_cast<float>(expected_input[
                           static_cast<std::size_t>(row) * kInputs +
                           input_index]) *
                       static_cast<float>(expected_weight[
                           static_cast<std::size_t>(output) * kInputs +
                           input_index]);
            expected_output[static_cast<std::size_t>(row) * kOutputs +
                            output] = hip_bfloat16(
                sum * expected_input_scales[row] *
                expected_weight_scales[output]);
        }

    hipStream_t stream = nullptr;
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t operation = nullptr;
    hipblasLtMatrixLayout_t input_layout = nullptr, weight_layout = nullptr;
    hipblasLtMatrixLayout_t output_layout = nullptr;
    hipblasLtMatmulPreference_t preference = nullptr;
    hip_bfloat16 *input_gpu = nullptr, *weight_gpu = nullptr;
    hip_bfloat16 *output_gpu = nullptr;
    __hip_fp8_e4m3 *input_fp8_gpu = nullptr, *weight_fp8_gpu = nullptr;
    float *input_scales_gpu = nullptr, *weight_scales_gpu = nullptr;
    float *output_f32_gpu = nullptr;
    void *workspace = nullptr;
    int result = 1;
    hipblasLtMatmulHeuristicResult_t algorithms[16];
    int algorithm_count = 0;
    const float alpha = 1.0f, beta = 0.0f;

#define HIP(call) do {                                                        \
    hipError_t status = (call);                                               \
    if (status != hipSuccess) {                                               \
        std::fprintf(stderr, "%s failed: %s\n", #call,                    \
                     hipGetErrorString(status));                              \
        goto cleanup;                                                         \
    }                                                                         \
} while (0)
#define BLAS(call) do {                                                       \
    hipblasStatus_t status = (call);                                          \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                   \
        std::fprintf(stderr, "%s failed: hipBLASLt status %d\n", #call,   \
                     static_cast<int>(status));                               \
        goto cleanup;                                                         \
    }                                                                         \
} while (0)

    HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    BLAS(hipblasLtCreate(&handle));
    HIP(hipMalloc(&input_gpu, input_elements * sizeof(*input_gpu)));
    HIP(hipMalloc(&weight_gpu, weight_elements * sizeof(*weight_gpu)));
    HIP(hipMalloc(&input_fp8_gpu,
                  input_elements * sizeof(*input_fp8_gpu)));
    HIP(hipMalloc(&weight_fp8_gpu,
                  weight_elements * sizeof(*weight_fp8_gpu)));
    HIP(hipMalloc(&input_scales_gpu,
                  kRows * sizeof(*input_scales_gpu)));
    HIP(hipMalloc(&weight_scales_gpu,
                  kOutputs * sizeof(*weight_scales_gpu)));
    HIP(hipMalloc(&output_f32_gpu,
                  output_elements * sizeof(*output_f32_gpu)));
    HIP(hipMalloc(&output_gpu, output_elements * sizeof(*output_gpu)));
    HIP(hipMalloc(&workspace, kWorkspaceBytes));
    HIP(hipMemcpyAsync(input_gpu, input.data(),
                       input_elements * sizeof(*input_gpu),
                       hipMemcpyHostToDevice, stream));
    HIP(hipMemcpyAsync(weight_gpu, weight.data(),
                       weight_elements * sizeof(*weight_gpu),
                       hipMemcpyHostToDevice, stream));
    quantize_rows<<<kRows, kThreads, 0, stream>>>(
        input_gpu, input_fp8_gpu, input_scales_gpu, kRows, kInputs);
    quantize_rows<<<kOutputs, kThreads, 0, stream>>>(
        weight_gpu, weight_fp8_gpu, weight_scales_gpu, kOutputs, kInputs);
    HIP(hipGetLastError());

    BLAS(hipblasLtMatmulDescCreate(
        &operation, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    BLAS(hipblasLtMatrixLayoutCreate(
        &weight_layout, HIP_R_8F_E4M3, kInputs, kOutputs, kInputs));
    BLAS(hipblasLtMatrixLayoutCreate(
        &input_layout, HIP_R_8F_E4M3, kInputs, kRows, kInputs));
    BLAS(hipblasLtMatrixLayoutCreate(
        &output_layout, HIP_R_32F, kOutputs, kRows, kOutputs));
    {
        hipblasOperation_t transpose = HIPBLAS_OP_T;
        hipblasOperation_t identity = HIPBLAS_OP_N;
        BLAS(hipblasLtMatmulDescSetAttribute(
            operation, HIPBLASLT_MATMUL_DESC_TRANSA,
            &transpose, sizeof(transpose)));
        BLAS(hipblasLtMatmulDescSetAttribute(
            operation, HIPBLASLT_MATMUL_DESC_TRANSB,
            &identity, sizeof(identity)));
    }
    BLAS(hipblasLtMatmulPreferenceCreate(&preference));
    BLAS(hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &kWorkspaceBytes, sizeof(kWorkspaceBytes)));
    BLAS(hipblasLtMatmulAlgoGetHeuristic(
        handle, operation, weight_layout, input_layout,
        output_layout, output_layout, preference,
        static_cast<int>(sizeof(algorithms) / sizeof(algorithms[0])),
        algorithms, &algorithm_count));
    if (algorithm_count < 1) {
        std::fprintf(stderr, "no gfx1201 OCP E4M3 algorithm\n");
        goto cleanup;
    }
    BLAS(hipblasLtMatmul(
        handle, operation, &alpha, weight_fp8_gpu, weight_layout,
        input_fp8_gpu, input_layout, &beta, output_f32_gpu, output_layout,
        output_f32_gpu, output_layout, &algorithms[0].algo,
        workspace, kWorkspaceBytes, stream));
    dequantize<<<1, kThreads, 0, stream>>>(
        output_f32_gpu, output_gpu, input_scales_gpu, weight_scales_gpu,
        output_elements);
    HIP(hipGetLastError());
    HIP(hipMemcpyAsync(actual_input.data(), input_fp8_gpu,
                       input_elements * sizeof(*input_fp8_gpu),
                       hipMemcpyDeviceToHost, stream));
    HIP(hipMemcpyAsync(actual_weight.data(), weight_fp8_gpu,
                       weight_elements * sizeof(*weight_fp8_gpu),
                       hipMemcpyDeviceToHost, stream));
    HIP(hipMemcpyAsync(actual_input_scales.data(), input_scales_gpu,
                       kRows * sizeof(*input_scales_gpu),
                       hipMemcpyDeviceToHost, stream));
    HIP(hipMemcpyAsync(actual_weight_scales.data(), weight_scales_gpu,
                       kOutputs * sizeof(*weight_scales_gpu),
                       hipMemcpyDeviceToHost, stream));
    HIP(hipMemcpyAsync(actual_output.data(), output_gpu,
                       output_elements * sizeof(*output_gpu),
                       hipMemcpyDeviceToHost, stream));
    HIP(hipStreamSynchronize(stream));

    {
        std::size_t input_mismatches = 0, weight_mismatches = 0;
        for (std::size_t index = 0; index < input_elements; index++)
            input_mismatches += expected_input[index].__x !=
                                actual_input[index].__x;
        for (std::size_t index = 0; index < weight_elements; index++)
            weight_mismatches += expected_weight[index].__x !=
                                 actual_weight[index].__x;
        std::size_t scale_mismatches =
            std::memcmp(expected_input_scales.data(),
                        actual_input_scales.data(),
                        kRows * sizeof(float)) != 0 ||
            std::memcmp(expected_weight_scales.data(),
                        actual_weight_scales.data(),
                        kOutputs * sizeof(float)) != 0;
        float maximum = 0.0f;
        double squared_error = 0.0, squared_reference = 0.0;
        std::size_t output_mismatches = 0, invalid = 0;
        for (std::size_t index = 0; index < output_elements; index++) {
            float expected = static_cast<float>(expected_output[index]);
            float actual = static_cast<float>(actual_output[index]);
            float difference = actual - expected;
            maximum = fmaxf(maximum, fabsf(difference));
            squared_error += static_cast<double>(difference) * difference;
            squared_reference += static_cast<double>(expected) * expected;
            output_mismatches += expected_output[index].data !=
                                 actual_output[index].data;
            invalid += !std::isfinite(actual);
        }
        double relative_rmse = squared_reference > 0.0 ?
            std::sqrt(squared_error / squared_reference) : 0.0;
        std::printf(
            "VDN FP8 oracle: input_mismatches=%zu weight_mismatches=%zu "
            "scale_mismatches=%zu output_bf16_mismatches=%zu "
            "max_abs=%.9g relative_rmse=%.9g invalid=%zu algorithms=%d\n",
            input_mismatches, weight_mismatches, scale_mismatches,
            output_mismatches, maximum, relative_rmse, invalid,
            algorithm_count);
        if (input_mismatches || weight_mismatches || scale_mismatches ||
            invalid || relative_rmse > 0.005) goto cleanup;
    }
    result = 0;

cleanup:
    if (preference) (void)hipblasLtMatmulPreferenceDestroy(preference);
    if (output_layout) (void)hipblasLtMatrixLayoutDestroy(output_layout);
    if (input_layout) (void)hipblasLtMatrixLayoutDestroy(input_layout);
    if (weight_layout) (void)hipblasLtMatrixLayoutDestroy(weight_layout);
    if (operation) (void)hipblasLtMatmulDescDestroy(operation);
    if (handle) (void)hipblasLtDestroy(handle);
    (void)hipFree(workspace);
    (void)hipFree(output_gpu);
    (void)hipFree(output_f32_gpu);
    (void)hipFree(weight_scales_gpu);
    (void)hipFree(input_scales_gpu);
    (void)hipFree(weight_fp8_gpu);
    (void)hipFree(input_fp8_gpu);
    (void)hipFree(weight_gpu);
    (void)hipFree(input_gpu);
    if (stream) (void)hipStreamDestroy(stream);
    return result;

#undef BLAS
#undef HIP
}
