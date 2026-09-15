#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kTile = 16;
constexpr unsigned kWaves = kThreads / 32;
constexpr unsigned kRowsPerBlock = kTile * kWaves;
[[maybe_unused]] constexpr unsigned kDepthChunk = 256;
constexpr unsigned kCrossedGroups = 3;
constexpr std::size_t kLtWorkspaceBytes = 64u * 1024u * 1024u;
using f32x8 = float __attribute__((ext_vector_type(8)));
using u16x8 = std::uint16_t __attribute__((ext_vector_type(8)));

struct metrics {
    double squared_error;
    double squared_reference;
    double squared_candidate;
    double dot;
    unsigned long long nonfinite;
    unsigned long long deterministic_mismatches;
    unsigned maximum_absolute_bits;
};

[[maybe_unused]] __device__ __forceinline__ int fragment_k(int lane,
                                                            int element) {
    const int reg = element >> 1;
    const int half = element & 1;
    return ((reg >> 1) << 3) + (((lane >> 4) & 1) << 2) +
           ((reg & 1) << 1) + half;
}

__global__ void fill_input(hip_bfloat16 *values, std::size_t elements) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (index >= elements) return;
    const int centered = static_cast<int>((index * 29u + 7u) % 257u) - 128;
    values[index] = hip_bfloat16(static_cast<float>(centered) / 256.0f);
}

__global__ void fill_weight(hip_bfloat16 *values, std::size_t elements,
                            std::uint32_t columns) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (index >= elements) return;
    const std::uint32_t row = static_cast<std::uint32_t>(index / columns);
    const std::uint32_t column = static_cast<std::uint32_t>(index % columns);
    const int centered = static_cast<int>(
        (static_cast<std::uint64_t>(row) * 17u +
         static_cast<std::uint64_t>(column) * 43u + 11u) % 509u) - 254;
    const float envelope = 1.0f + static_cast<float>((column / 16u) % 11u);
    values[index] = hip_bfloat16(
        static_cast<float>(centered) * envelope / 32768.0f);
}

__global__ void quantize_weight_blocks(
    const hip_bfloat16 *__restrict__ input,
    std::int8_t *__restrict__ output,
    float *__restrict__ scales,
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint32_t group_size,
    std::uint32_t groups) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t group = blockIdx.y;
    if (row >= rows || group >= groups) return;
    const std::uint32_t begin = group * group_size;
    const std::uint32_t end = min(begin + group_size, columns);
    __shared__ float reduction[kThreads];
    float maximum = 0.0f;
    const std::size_t row_offset = static_cast<std::size_t>(row) * columns;
    for (std::uint32_t column = begin + threadIdx.x; column < end;
         column += blockDim.x)
        maximum = fmaxf(maximum,
                        fabsf(static_cast<float>(input[row_offset + column])));
    reduction[threadIdx.x] = maximum;
    __syncthreads();
    for (unsigned width = kThreads / 2; width; width >>= 1) {
        if (threadIdx.x < width)
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + width]);
        __syncthreads();
    }
    const float scale = reduction[0] > 0.0f ? reduction[0] / 127.0f : 1.0f;
    const float inverse = reduction[0] > 0.0f ? 1.0f / scale : 0.0f;
    if (threadIdx.x == 0)
        scales[static_cast<std::size_t>(row) * groups + group] = scale;
    for (std::uint32_t column = begin + threadIdx.x; column < end;
         column += blockDim.x) {
        float value = nearbyintf(
            static_cast<float>(input[row_offset + column]) * inverse);
        value = fminf(127.0f, fmaxf(-127.0f, value));
        output[row_offset + column] = static_cast<std::int8_t>(value);
    }
}

__global__ __launch_bounds__(kThreads) void block_weight_gemm(
    const hip_bfloat16 *__restrict__ input,
    const std::int8_t *__restrict__ weight,
    const float *__restrict__ scales,
    hip_bfloat16 *__restrict__ output,
    std::uint32_t rows,
    std::uint32_t output_dim,
    std::uint32_t input_dim,
    std::uint32_t group_size,
    std::uint32_t groups) {
#if defined(__gfx12__) || defined(__gfx1200__) || defined(__gfx1201__)
    __shared__ std::uint16_t weight_tile[kTile * kDepthChunk];
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int wave = static_cast<int>(threadIdx.x >> 5u);
    const std::uint32_t output_tile = blockIdx.x * kTile;
    const std::uint32_t row_tile = blockIdx.y * kRowsPerBlock +
                                   static_cast<std::uint32_t>(wave) * kTile;
    f32x8 accumulator = {0, 0, 0, 0, 0, 0, 0, 0};
    for (std::uint32_t depth_tile = 0; depth_tile < input_dim;
         depth_tile += kDepthChunk) {
        for (std::uint32_t local = threadIdx.x;
             local < kTile * kDepthChunk; local += blockDim.x) {
            const std::uint32_t output_row =
                output_tile + local / kDepthChunk;
            const std::uint32_t depth = depth_tile + local % kDepthChunk;
            std::uint16_t dequantized = 0;
            if (output_row < output_dim && depth < input_dim) {
                const float scale = scales[
                    static_cast<std::size_t>(output_row) * groups +
                    depth / group_size];
                const float value = static_cast<float>(weight[
                    static_cast<std::size_t>(output_row) * input_dim +
                    depth]) * scale;
                dequantized = hip_bfloat16(value).data;
            }
            weight_tile[local] = dequantized;
        }
        __syncthreads();

        for (std::uint32_t inner = 0;
             inner < kDepthChunk && depth_tile + inner < input_dim;
             inner += kTile) {
            u16x8 weight_fragment;
            u16x8 input_fragment;
            const std::uint32_t weight_row =
                static_cast<std::uint32_t>(lane & 15);
            const std::uint32_t input_row = row_tile +
                static_cast<std::uint32_t>(lane & 15);
#pragma unroll
            for (int element = 0; element < 8; ++element) {
                const std::uint32_t depth_offset =
                    static_cast<std::uint32_t>(fragment_k(lane, element));
                weight_fragment[element] = weight_tile[
                    weight_row * kDepthChunk + inner + depth_offset];
                const std::uint32_t input_depth =
                    depth_tile + inner + depth_offset;
                input_fragment[element] =
                    input_row < rows && input_depth < input_dim
                        ? input[static_cast<std::size_t>(input_row) *
                                    input_dim + input_depth]
                              .data
                        : 0;
            }
            accumulator =
                __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
                    weight_fragment, input_fragment, accumulator);
        }
        __syncthreads();
    }

    const std::uint32_t row = row_tile +
                              static_cast<std::uint32_t>(lane & 15);
#pragma unroll
    for (int element = 0; element < 8; ++element) {
        const std::uint32_t column = output_tile +
            static_cast<std::uint32_t>(((lane >> 4) << 3) + element);
        if (row < rows && column < output_dim)
            output[static_cast<std::size_t>(row) * output_dim + column] =
                hip_bfloat16(accumulator[element]);
    }
#else
    (void)input;
    (void)weight;
    (void)scales;
    (void)output;
    (void)rows;
    (void)output_dim;
    (void)input_dim;
    (void)group_size;
    (void)groups;
#endif
}

__global__ void compare_outputs(
    const hip_bfloat16 *__restrict__ reference,
    const hip_bfloat16 *__restrict__ candidate,
    const hip_bfloat16 *__restrict__ repeat,
    std::size_t elements,
    metrics *__restrict__ result) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double squared_candidate = 0.0;
    double dot = 0.0;
    unsigned long long nonfinite = 0;
    unsigned long long mismatches = 0;
    float maximum_absolute = 0.0f;
    for (; index < elements; index += stride) {
        const float expected = static_cast<float>(reference[index]);
        const float actual = static_cast<float>(candidate[index]);
        const float difference = actual - expected;
        squared_error += static_cast<double>(difference) * difference;
        squared_reference += static_cast<double>(expected) * expected;
        squared_candidate += static_cast<double>(actual) * actual;
        dot += static_cast<double>(expected) * actual;
        maximum_absolute = fmaxf(maximum_absolute, fabsf(difference));
        nonfinite += !isfinite(actual);
        mismatches += candidate[index].data != repeat[index].data;
    }
    atomicAdd(&result->squared_error, squared_error);
    atomicAdd(&result->squared_reference, squared_reference);
    atomicAdd(&result->squared_candidate, squared_candidate);
    atomicAdd(&result->dot, dot);
    atomicAdd(&result->nonfinite, nonfinite);
    atomicAdd(&result->deterministic_mismatches, mismatches);
    atomicMax(&result->maximum_absolute_bits,
              __builtin_bit_cast(unsigned, maximum_absolute));
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
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (!text[0] || !end || *end || !parsed ||
        parsed > std::numeric_limits<std::uint32_t>::max()) return false;
    *value = static_cast<std::uint32_t>(parsed);
    return true;
}

float median(float values[kCrossedGroups]) {
    std::sort(values, values + kCrossedGroups);
    return values[kCrossedGroups / 2];
}

int query_native_block_scaled(hipblasLtHandle_t handle,
                              hipDataType input_type) {
    hipblasLtMatmulDesc_t operation = nullptr;
    hipblasLtMatrixLayout_t weight_layout = nullptr;
    hipblasLtMatrixLayout_t input_layout = nullptr;
    hipblasLtMatrixLayout_t output_layout = nullptr;
    hipblasLtMatmulPreference_t preference = nullptr;
    hipblasLtMatmulHeuristicResult_t algorithms[32];
    int count = 0;
    hipblasStatus_t status = hipblasLtMatmulDescCreate(
        &operation, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    if (status == HIPBLAS_STATUS_SUCCESS)
        status = hipblasLtMatrixLayoutCreate(
            &weight_layout, HIP_R_8F_E4M3, 128, 16, 128);
    if (status == HIPBLAS_STATUS_SUCCESS)
        status = hipblasLtMatrixLayoutCreate(
            &input_layout, input_type, 128, 16, 128);
    if (status == HIPBLAS_STATUS_SUCCESS)
        status = hipblasLtMatrixLayoutCreate(
            &output_layout, HIP_R_16BF, 16, 16, 16);
    hipblasOperation_t transpose = HIPBLAS_OP_T;
    hipblasOperation_t identity = HIPBLAS_OP_N;
    hipblasLtMatmulMatrixScale_t block_scale =
        HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    if (status == HIPBLAS_STATUS_SUCCESS)
        status = hipblasLtMatmulDescSetAttribute(
            operation, HIPBLASLT_MATMUL_DESC_TRANSA,
            &transpose, sizeof(transpose));
    if (status == HIPBLAS_STATUS_SUCCESS)
        status = hipblasLtMatmulDescSetAttribute(
            operation, HIPBLASLT_MATMUL_DESC_TRANSB,
            &identity, sizeof(identity));
    if (status == HIPBLAS_STATUS_SUCCESS)
        status = hipblasLtMatmulDescSetAttribute(
            operation, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
            &block_scale, sizeof(block_scale));
    if (input_type == HIP_R_8F_E4M3 && status == HIPBLAS_STATUS_SUCCESS)
        status = hipblasLtMatmulDescSetAttribute(
            operation, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
            &block_scale, sizeof(block_scale));
    if (status == HIPBLAS_STATUS_SUCCESS)
        status = hipblasLtMatmulPreferenceCreate(&preference);
    if (status == HIPBLAS_STATUS_SUCCESS)
        status = hipblasLtMatmulPreferenceSetAttribute(
            preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &kLtWorkspaceBytes, sizeof(kLtWorkspaceBytes));
    if (status == HIPBLAS_STATUS_SUCCESS)
        status = hipblasLtMatmulAlgoGetHeuristic(
            handle, operation, weight_layout, input_layout,
            output_layout, output_layout, preference,
            static_cast<int>(sizeof(algorithms) / sizeof(algorithms[0])),
            algorithms, &count);
    if (status != HIPBLAS_STATUS_SUCCESS) count = -static_cast<int>(status);
    if (preference) (void)hipblasLtMatmulPreferenceDestroy(preference);
    if (output_layout) (void)hipblasLtMatrixLayoutDestroy(output_layout);
    if (input_layout) (void)hipblasLtMatrixLayoutDestroy(input_layout);
    if (weight_layout) (void)hipblasLtMatrixLayoutDestroy(weight_layout);
    if (operation) (void)hipblasLtMatmulDescDestroy(operation);
    return count;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 6) {
        std::fprintf(stderr,
            "usage: %s ROWS OUTPUT_DIM INPUT_DIM GROUP_SIZE ITERATIONS\n",
            argv[0]);
        return 2;
    }
    std::uint32_t rows, output_dim, input_dim, group_size, iterations;
    if (!parse_u32(argv[1], &rows) || !parse_u32(argv[2], &output_dim) ||
        !parse_u32(argv[3], &input_dim) || !parse_u32(argv[4], &group_size) ||
        !parse_u32(argv[5], &iterations) || group_size % kTile != 0) {
        std::fprintf(stderr,
            "dimensions/iterations must be positive u32 and group size a multiple of 16\n");
        return 2;
    }
    if (rows > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        output_dim >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        input_dim >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
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
    const std::uint32_t groups = (input_dim + group_size - 1) / group_size;
    std::size_t scale_elements;
    if (!checked_product(output_dim, groups, &scale_elements)) {
        std::fprintf(stderr, "scale tensor size overflow\n");
        return 2;
    }
    const std::size_t input_blocks_wide =
        (input_elements + kThreads - 1) / kThreads;
    const std::size_t weight_blocks_wide =
        (weight_elements + kThreads - 1) / kThreads;
    if (input_blocks_wide > std::numeric_limits<std::uint32_t>::max() ||
        weight_blocks_wide > std::numeric_limits<std::uint32_t>::max()) {
        std::fprintf(stderr, "initialization grid exceeds u32\n");
        return 2;
    }

    hipStream_t stream = nullptr;
    hipEvent_t begin = nullptr, end = nullptr;
    rocblas_handle blas = nullptr;
    hipblasLtHandle_t lt = nullptr;
    hip_bfloat16 *input = nullptr, *weight = nullptr;
    hip_bfloat16 *reference = nullptr, *candidate = nullptr, *repeat = nullptr;
    std::uint8_t *candidate_storage = nullptr, *repeat_storage = nullptr;
    std::uint8_t *quantized_storage = nullptr, *scale_storage = nullptr;
    std::int8_t *quantized = nullptr;
    float *scales = nullptr;
    metrics *device_metrics = nullptr;
    int result = 1;
    float baseline_groups[kCrossedGroups] = {0.0f};
    float candidate_groups[kCrossedGroups] = {0.0f};
    metrics host_metrics = {};
    constexpr std::size_t guard_bytes = 64;
    std::uint8_t guards[8][guard_bytes] = {};
    std::size_t quant_mismatches = 0, scale_mismatches = 0;
    const bool verify_quant = weight_elements <= (1u << 20);
    std::vector<hip_bfloat16> host_weight(
        verify_quant ? weight_elements : 0);
    std::vector<std::int8_t> host_quantized(
        verify_quant ? weight_elements : 0);
    std::vector<float> host_scales(verify_quant ? scale_elements : 0);
    const float alpha = 1.0f, beta = 0.0f;
    const dim3 candidate_grid(
        (output_dim + kTile - 1) / kTile,
        (rows + kRowsPerBlock - 1) / kRowsPerBlock);
    const auto baseline = [&]() {
        return (rocblas_gemm_ex)(
            blas, rocblas_operation_transpose, rocblas_operation_none,
            static_cast<int>(output_dim), static_cast<int>(rows),
            static_cast<int>(input_dim), &alpha, weight,
            rocblas_datatype_bf16_r, static_cast<int>(input_dim), input,
            rocblas_datatype_bf16_r, static_cast<int>(input_dim), &beta,
            reference, rocblas_datatype_bf16_r,
            static_cast<int>(output_dim), reference,
            rocblas_datatype_bf16_r, static_cast<int>(output_dim),
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    };
    const auto run_candidate = [&](hip_bfloat16 *target) {
        block_weight_gemm<<<candidate_grid, kThreads, 0, stream>>>(
            input, quantized, scales, target, rows, output_dim, input_dim,
            group_size, groups);
    };

#define HIP(call) do {                                                        \
    const hipError_t status = (call);                                         \
    if (status != hipSuccess) {                                               \
        std::fprintf(stderr, "%s failed: %s\n", #call,                    \
                     hipGetErrorString(status));                              \
        goto cleanup;                                                         \
    }                                                                         \
} while (0)
#define BLAS(call) do {                                                       \
    const rocblas_status status = (call);                                     \
    if (status != rocblas_status_success) {                                   \
        std::fprintf(stderr, "%s failed: rocBLAS status %d\n", #call,     \
                     static_cast<int>(status));                               \
        goto cleanup;                                                         \
    }                                                                         \
} while (0)

    HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP(hipEventCreate(&begin));
    HIP(hipEventCreate(&end));
    BLAS(rocblas_create_handle(&blas));
    BLAS(rocblas_set_stream(blas, stream));
    if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "hipblasLtCreate failed\n");
        goto cleanup;
    }
    HIP(hipMalloc(&input, input_elements * sizeof(*input)));
    HIP(hipMalloc(&weight, weight_elements * sizeof(*weight)));
    HIP(hipMalloc(&reference, output_elements * sizeof(*reference)));
    HIP(hipMalloc(&candidate_storage,
                  output_elements * sizeof(*candidate) + 2 * guard_bytes));
    HIP(hipMalloc(&repeat_storage,
                  output_elements * sizeof(*repeat) + 2 * guard_bytes));
    HIP(hipMalloc(&quantized_storage,
                  weight_elements * sizeof(*quantized) + 2 * guard_bytes));
    HIP(hipMalloc(&scale_storage,
                  scale_elements * sizeof(*scales) + 2 * guard_bytes));
    candidate = reinterpret_cast<hip_bfloat16 *>(
        candidate_storage + guard_bytes);
    repeat = reinterpret_cast<hip_bfloat16 *>(repeat_storage + guard_bytes);
    quantized = reinterpret_cast<std::int8_t *>(
        quantized_storage + guard_bytes);
    scales = reinterpret_cast<float *>(scale_storage + guard_bytes);
    HIP(hipMemsetAsync(candidate_storage, 0xa5,
                       output_elements * sizeof(*candidate) +
                           2 * guard_bytes, stream));
    HIP(hipMemsetAsync(repeat_storage, 0xa5,
                       output_elements * sizeof(*repeat) +
                           2 * guard_bytes, stream));
    HIP(hipMemsetAsync(quantized_storage, 0xa5,
                       weight_elements * sizeof(*quantized) +
                           2 * guard_bytes, stream));
    HIP(hipMemsetAsync(scale_storage, 0xa5,
                       scale_elements * sizeof(*scales) +
                           2 * guard_bytes, stream));
    HIP(hipMalloc(&device_metrics, sizeof(*device_metrics)));
    fill_input<<<static_cast<std::uint32_t>(input_blocks_wide), kThreads,
                 0, stream>>>(input, input_elements);
    fill_weight<<<static_cast<std::uint32_t>(weight_blocks_wide), kThreads,
                  0, stream>>>(weight, weight_elements, input_dim);
    quantize_weight_blocks<<<dim3(output_dim, groups), kThreads, 0, stream>>>(
        weight, quantized, scales, output_dim, input_dim, group_size, groups);
    HIP(hipGetLastError());
    HIP(hipStreamSynchronize(stream));

    if (verify_quant) {
        HIP(hipMemcpy(host_weight.data(), weight,
                      weight_elements * sizeof(*weight),
                      hipMemcpyDeviceToHost));
        HIP(hipMemcpy(host_quantized.data(), quantized,
                      weight_elements * sizeof(*quantized),
                      hipMemcpyDeviceToHost));
        HIP(hipMemcpy(host_scales.data(), scales,
                      scale_elements * sizeof(*scales),
                      hipMemcpyDeviceToHost));
        for (std::uint32_t row = 0; row < output_dim; ++row) {
            for (std::uint32_t group = 0; group < groups; ++group) {
                const std::uint32_t first = group * group_size;
                const std::uint32_t last =
                    std::min(first + group_size, input_dim);
                float maximum = 0.0f;
                for (std::uint32_t column = first; column < last; ++column)
                    maximum = std::fmax(maximum, std::fabs(static_cast<float>(
                        host_weight[static_cast<std::size_t>(row) * input_dim +
                                    column])));
                const float expected_scale =
                    maximum > 0.0f ? maximum / 127.0f : 1.0f;
                scale_mismatches += expected_scale != host_scales[
                    static_cast<std::size_t>(row) * groups + group];
                const float inverse = maximum > 0.0f ?
                    1.0f / expected_scale : 0.0f;
                for (std::uint32_t column = first; column < last; ++column) {
                    float value = std::nearbyint(static_cast<float>(host_weight[
                        static_cast<std::size_t>(row) * input_dim + column]) *
                        inverse);
                    value = std::fmin(127.0f, std::fmax(-127.0f, value));
                    quant_mismatches += static_cast<std::int8_t>(value) !=
                        host_quantized[
                            static_cast<std::size_t>(row) * input_dim + column];
                }
            }
        }
        if (quant_mismatches || scale_mismatches) {
            std::fprintf(stderr,
                "block quant oracle failed: bytes=%zu scales=%zu\n",
                quant_mismatches, scale_mismatches);
            goto cleanup;
        }
    }

    if (std::getenv("H3_BLOCK_WEIGHT_QUERY_NATIVE"))
        std::printf(
            "VDN block-weight native capability: mixed_bf16_fp8=%d "
            "fp8_fp8=%d (positive=heuristic-count, negative=-status)\n",
            query_native_block_scaled(lt, HIP_R_16BF),
            query_native_block_scaled(lt, HIP_R_8F_E4M3));

    for (unsigned crossed = 0; crossed < kCrossedGroups; ++crossed) {
        if ((crossed & 1u) == 0u) {
            BLAS(baseline());
            HIP(hipStreamSynchronize(stream));
            HIP(hipEventRecord(begin, stream));
            for (std::uint32_t iteration = 0; iteration < iterations;
                 ++iteration) BLAS(baseline());
            HIP(hipEventRecord(end, stream));
            HIP(hipEventSynchronize(end));
            HIP(hipEventElapsedTime(&baseline_groups[crossed], begin, end));
            baseline_groups[crossed] /= static_cast<float>(iterations);
            run_candidate(candidate);
            HIP(hipStreamSynchronize(stream));
            HIP(hipEventRecord(begin, stream));
            for (std::uint32_t iteration = 0; iteration < iterations;
                 ++iteration) run_candidate(candidate);
            HIP(hipGetLastError());
            HIP(hipEventRecord(end, stream));
            HIP(hipEventSynchronize(end));
            HIP(hipEventElapsedTime(&candidate_groups[crossed], begin, end));
            candidate_groups[crossed] /= static_cast<float>(iterations);
        } else {
            run_candidate(candidate);
            HIP(hipStreamSynchronize(stream));
            HIP(hipEventRecord(begin, stream));
            for (std::uint32_t iteration = 0; iteration < iterations;
                 ++iteration) run_candidate(candidate);
            HIP(hipGetLastError());
            HIP(hipEventRecord(end, stream));
            HIP(hipEventSynchronize(end));
            HIP(hipEventElapsedTime(&candidate_groups[crossed], begin, end));
            candidate_groups[crossed] /= static_cast<float>(iterations);
            BLAS(baseline());
            HIP(hipStreamSynchronize(stream));
            HIP(hipEventRecord(begin, stream));
            for (std::uint32_t iteration = 0; iteration < iterations;
                 ++iteration) BLAS(baseline());
            HIP(hipEventRecord(end, stream));
            HIP(hipEventSynchronize(end));
            HIP(hipEventElapsedTime(&baseline_groups[crossed], begin, end));
            baseline_groups[crossed] /= static_cast<float>(iterations);
        }
    }
    run_candidate(repeat);
    HIP(hipMemsetAsync(device_metrics, 0, sizeof(*device_metrics), stream));
    compare_outputs<<<4096, kThreads, 0, stream>>>(
        reference, candidate, repeat, output_elements, device_metrics);
    HIP(hipGetLastError());
    HIP(hipMemcpyAsync(&host_metrics, device_metrics, sizeof(host_metrics),
                       hipMemcpyDeviceToHost, stream));
    HIP(hipMemcpyAsync(guards[0], candidate_storage, guard_bytes,
                       hipMemcpyDeviceToHost, stream));
    HIP(hipMemcpyAsync(guards[1], candidate_storage + guard_bytes +
                           output_elements * sizeof(*candidate), guard_bytes,
                       hipMemcpyDeviceToHost, stream));
    HIP(hipMemcpyAsync(guards[2], repeat_storage, guard_bytes,
                       hipMemcpyDeviceToHost, stream));
    HIP(hipMemcpyAsync(guards[3], repeat_storage + guard_bytes +
                           output_elements * sizeof(*repeat), guard_bytes,
                       hipMemcpyDeviceToHost, stream));
    HIP(hipMemcpyAsync(guards[4], quantized_storage, guard_bytes,
                       hipMemcpyDeviceToHost, stream));
    HIP(hipMemcpyAsync(guards[5], quantized_storage + guard_bytes +
                           weight_elements * sizeof(*quantized), guard_bytes,
                       hipMemcpyDeviceToHost, stream));
    HIP(hipMemcpyAsync(guards[6], scale_storage, guard_bytes,
                       hipMemcpyDeviceToHost, stream));
    HIP(hipMemcpyAsync(guards[7], scale_storage + guard_bytes +
                           scale_elements * sizeof(*scales), guard_bytes,
                       hipMemcpyDeviceToHost, stream));
    HIP(hipStreamSynchronize(stream));

    for (const auto &guard : guards)
        for (std::uint8_t value : guard)
            if (value != 0xa5u) {
                std::fprintf(stderr, "block-weight canary changed\n");
                goto cleanup;
            }

    {
        const float baseline_median = median(baseline_groups);
        const float candidate_median = median(candidate_groups);
        const double relative_rmse = host_metrics.squared_reference > 0.0
            ? std::sqrt(host_metrics.squared_error /
                        host_metrics.squared_reference)
            : std::numeric_limits<double>::infinity();
        const double cosine = host_metrics.squared_reference > 0.0 &&
                                      host_metrics.squared_candidate > 0.0
            ? host_metrics.dot /
                  std::sqrt(host_metrics.squared_reference *
                            host_metrics.squared_candidate)
            : 0.0;
        const float maximum_absolute =
            __builtin_bit_cast(float, host_metrics.maximum_absolute_bits);
        std::printf(
            "VDN block-weight GEMM: M=%u N=%u K=%u group=%u groups=%u "
            "iterations=%u bf16_ms=[%.6f,%.6f,%.6f] "
            "candidate_ms=[%.6f,%.6f,%.6f] medians=%.6f/%.6f "
            "speedup=%.6fx rel_rmse=%.9g cosine=%.12g max_abs=%g "
            "nonfinite=%llu deterministic_mismatches=%llu "
            "quant_oracle=%s quant_mismatches=%zu scale_mismatches=%zu "
            "canary=pass "
            "bf16_weight_gib=%.6f quantized_weight_gib=%.6f\n",
            rows, output_dim, input_dim, group_size, groups, iterations,
            baseline_groups[0], baseline_groups[1], baseline_groups[2],
            candidate_groups[0], candidate_groups[1], candidate_groups[2],
            baseline_median, candidate_median,
            candidate_median > 0.0f ? baseline_median / candidate_median : 0.0f,
            relative_rmse, cosine, static_cast<double>(maximum_absolute),
            host_metrics.nonfinite, host_metrics.deterministic_mismatches,
            verify_quant ? "pass" : "not-run-large-shape",
            quant_mismatches, scale_mismatches,
            static_cast<double>(weight_elements * sizeof(*weight)) /
                (1024.0 * 1024.0 * 1024.0),
            static_cast<double>(weight_elements * sizeof(*quantized) +
                                scale_elements * sizeof(*scales)) /
                (1024.0 * 1024.0 * 1024.0));
        if (host_metrics.nonfinite ||
            host_metrics.deterministic_mismatches) goto cleanup;
    }
    result = 0;

cleanup:
    (void)hipFree(device_metrics);
    (void)hipFree(scale_storage);
    (void)hipFree(quantized_storage);
    (void)hipFree(repeat_storage);
    (void)hipFree(candidate_storage);
    (void)hipFree(reference);
    (void)hipFree(weight);
    (void)hipFree(input);
    if (lt) (void)hipblasLtDestroy(lt);
    if (blas) (void)rocblas_destroy_handle(blas);
    if (end) (void)hipEventDestroy(end);
    if (begin) (void)hipEventDestroy(begin);
    if (stream) (void)hipStreamDestroy(stream);
    return result;

#undef BLAS
#undef HIP
}
