#define ROCBLAS_BETA_FEATURES_API
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>

#ifdef rocblas_gemm_ex
#undef rocblas_gemm_ex
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr unsigned kThreads = 256;
constexpr size_t kMaximumWorkspace = 256ULL * 1024ULL * 1024ULL;
constexpr int kMaximumAlgorithms = 32;

struct result {
    int32_t solution;
    float milliseconds;
    unsigned long long mismatches;
};

__global__ void fill(float *values, size_t elements, uint32_t salt) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float position = static_cast<float>((index + salt) % 1048573u);
    values[index] = sinf(position * 0.01317f) * 0.117f +
                    cosf(position * 0.00731f) * 0.031f;
}

__global__ void compare_bits(const float *reference, const float *candidate,
                             size_t elements,
                             unsigned long long *mismatches) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t left = __float_as_uint(reference[index]);
    uint32_t right = __float_as_uint(candidate[index]);
    if (left != right) atomicAdd(mismatches, 1ULL);
}

bool parse_u32(const char *text, uint32_t *value) {
    if (!text || !*text || !value) return false;
    errno = 0;
    char *end = nullptr;
    unsigned long parsed = std::strtoul(text, &end, 10);
    if (errno || end == text || *end || !parsed ||
        parsed > std::numeric_limits<uint32_t>::max()) return false;
    *value = static_cast<uint32_t>(parsed);
    return true;
}

bool product(size_t left, size_t right, size_t *output) {
    if (!output || (right && left > SIZE_MAX / right)) return false;
    *output = left * right;
    return true;
}

const char *rocblas_name(rocblas_status status) {
    const char *name = rocblas_status_to_string(status);
    return name ? name : "unknown";
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s M N K ITERATIONS\n", argv[0]);
        return 2;
    }
    uint32_t rows = 0, output_dim = 0, input_dim = 0, iterations = 0;
    if (!parse_u32(argv[1], &rows) || !parse_u32(argv[2], &output_dim) ||
        !parse_u32(argv[3], &input_dim) ||
        !parse_u32(argv[4], &iterations) ||
        rows > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        output_dim >
            static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        input_dim > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "dimensions and iterations must be positive i32\n");
        return 2;
    }
    size_t input_elements = 0, weight_elements = 0, output_elements = 0;
    if (!product(rows, input_dim, &input_elements) ||
        !product(output_dim, input_dim, &weight_elements) ||
        !product(rows, output_dim, &output_elements)) {
        std::fprintf(stderr, "tensor size overflow\n");
        return 2;
    }
    auto blocks = [](size_t elements) {
        return (elements + kThreads - 1) / kThreads;
    };
    if (blocks(input_elements) > std::numeric_limits<unsigned>::max() ||
        blocks(weight_elements) > std::numeric_limits<unsigned>::max() ||
        blocks(output_elements) > std::numeric_limits<unsigned>::max()) {
        std::fprintf(stderr, "kernel grid exceeds u32\n");
        return 2;
    }

    hipStream_t stream = nullptr;
    hipEvent_t begin = nullptr, end = nullptr;
    rocblas_handle handle = nullptr;
    hipblasLtHandle_t lt_handle = nullptr;
    hipblasLtMatmulDesc_t lt_operation = nullptr;
    hipblasLtMatrixLayout_t lt_weight_layout = nullptr;
    hipblasLtMatrixLayout_t lt_input_layout = nullptr;
    hipblasLtMatrixLayout_t lt_output_layout = nullptr;
    hipblasLtMatmulPreference_t lt_preference = nullptr;
    float *input = nullptr, *weight = nullptr;
    float *reference = nullptr, *candidate = nullptr;
    unsigned long long *mismatches = nullptr;
    void *workspace = nullptr;
    int exit_code = 1;
    const float alpha = 1.0f, beta = 0.0f;
    float default_ms = 0.0f;
    rocblas_int solution_count = 0;
    std::vector<rocblas_int> solutions;
    std::vector<result> results;
    hipblasLtMatmulHeuristicResult_t
        lt_heuristics[kMaximumAlgorithms] = {};
    int lt_algorithm_count = 0;
    char rocblas_version[128] = {0};

    auto gemm = [&](float *output, rocblas_gemm_algo algorithm,
                    int32_t solution) {
        return rocblas_gemm_ex(
            handle, rocblas_operation_transpose, rocblas_operation_none,
            static_cast<int>(output_dim), static_cast<int>(rows),
            static_cast<int>(input_dim), &alpha, weight,
            rocblas_datatype_f32_r, static_cast<int>(input_dim), input,
            rocblas_datatype_f32_r, static_cast<int>(input_dim), &beta,
            output, rocblas_datatype_f32_r, static_cast<int>(output_dim),
            output, rocblas_datatype_f32_r, static_cast<int>(output_dim),
            rocblas_datatype_f32_r, algorithm, solution, 0);
    };
    auto measure = [&](float *output, rocblas_gemm_algo algorithm,
                       int32_t solution, float *milliseconds) {
        if (gemm(output, algorithm, solution) != rocblas_status_success)
            return false;
        if (hipStreamSynchronize(stream) != hipSuccess) return false;
        if (hipEventRecord(begin, stream) != hipSuccess) return false;
        for (uint32_t iteration = 0; iteration < iterations; iteration++)
            if (gemm(output, algorithm, solution) != rocblas_status_success)
                return false;
        if (hipEventRecord(end, stream) != hipSuccess ||
            hipEventSynchronize(end) != hipSuccess ||
            hipEventElapsedTime(milliseconds, begin, end) != hipSuccess)
            return false;
        *milliseconds /= static_cast<float>(iterations);
        return true;
    };
    auto lt_matmul = [&](const hipblasLtMatmulAlgo_t *algorithm) {
        return hipblasLtMatmul(
            lt_handle, lt_operation, &alpha, weight, lt_weight_layout,
            input, lt_input_layout, &beta, candidate, lt_output_layout,
            candidate, lt_output_layout, algorithm, workspace,
            kMaximumWorkspace, stream);
    };
    auto lt_measure = [&](const hipblasLtMatmulAlgo_t *algorithm,
                          float *milliseconds) {
        if (lt_matmul(algorithm) != HIPBLAS_STATUS_SUCCESS ||
            hipStreamSynchronize(stream) != hipSuccess ||
            hipEventRecord(begin, stream) != hipSuccess) return false;
        for (uint32_t iteration = 0; iteration < iterations; iteration++)
            if (lt_matmul(algorithm) != HIPBLAS_STATUS_SUCCESS) return false;
        if (hipEventRecord(end, stream) != hipSuccess ||
            hipEventSynchronize(end) != hipSuccess ||
            hipEventElapsedTime(milliseconds, begin, end) != hipSuccess)
            return false;
        *milliseconds /= static_cast<float>(iterations);
        return true;
    };

#define HIP(call) do {                                                        \
    hipError_t status = (call);                                               \
    if (status != hipSuccess) {                                               \
        std::fprintf(stderr, "%s failed: %s\n", #call,                  \
                     hipGetErrorString(status));                              \
        goto cleanup;                                                         \
    }                                                                         \
} while (0)
#define ROCBLAS(call) do {                                                    \
    rocblas_status status = (call);                                           \
    if (status != rocblas_status_success) {                                   \
        std::fprintf(stderr, "%s failed: %s\n", #call,                  \
                     rocblas_name(status));                                   \
        goto cleanup;                                                         \
    }                                                                         \
} while (0)
#define HIPBLASLT(call) do {                                                  \
    hipblasStatus_t status = (call);                                          \
    if (status != HIPBLAS_STATUS_SUCCESS) {                                   \
        std::fprintf(stderr, "%s failed: hipBLASLt status %d\n", #call, \
                     static_cast<int>(status));                               \
        goto cleanup;                                                         \
    }                                                                         \
} while (0)

    HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP(hipEventCreate(&begin));
    HIP(hipEventCreate(&end));
    ROCBLAS(rocblas_create_handle(&handle));
    ROCBLAS(rocblas_set_stream(handle, stream));
    ROCBLAS(rocblas_get_version_string(rocblas_version,
                                       sizeof(rocblas_version)));
    std::printf("rocblas_version=%s\n", rocblas_version);
    HIPBLASLT(hipblasLtCreate(&lt_handle));
    HIP(hipMalloc(&input, input_elements * sizeof(*input)));
    HIP(hipMalloc(&weight, weight_elements * sizeof(*weight)));
    HIP(hipMalloc(&reference, output_elements * sizeof(*reference)));
    HIP(hipMalloc(&candidate, output_elements * sizeof(*candidate)));
    HIP(hipMalloc(&mismatches, sizeof(*mismatches)));
    HIP(hipMalloc(&workspace, kMaximumWorkspace));
    fill<<<static_cast<unsigned>(blocks(input_elements)), kThreads, 0, stream>>>(
        input, input_elements, 17u);
    fill<<<static_cast<unsigned>(blocks(weight_elements)), kThreads, 0, stream>>>(
        weight, weight_elements, 29u);
    HIP(hipGetLastError());

    ROCBLAS(gemm(reference, rocblas_gemm_algo_standard, 0));
    HIP(hipStreamSynchronize(stream));
    if (!measure(reference, rocblas_gemm_algo_standard, 0, &default_ms)) {
        std::fprintf(stderr, "cannot measure default GEMM\n");
        goto cleanup;
    }

    ROCBLAS(rocblas_gemm_ex_get_solutions(
        handle, rocblas_operation_transpose, rocblas_operation_none,
        static_cast<int>(output_dim), static_cast<int>(rows),
        static_cast<int>(input_dim), &alpha, weight, rocblas_datatype_f32_r,
        static_cast<int>(input_dim), input, rocblas_datatype_f32_r,
        static_cast<int>(input_dim), &beta, candidate, rocblas_datatype_f32_r,
        static_cast<int>(output_dim), candidate, rocblas_datatype_f32_r,
        static_cast<int>(output_dim), rocblas_datatype_f32_r,
        rocblas_gemm_algo_solution_index, 0, nullptr, &solution_count));
    if (solution_count < 1) {
        std::fprintf(stderr, "rocBLAS returned no F32 solutions\n");
        goto cleanup;
    }
    solutions.resize(static_cast<size_t>(solution_count));
    ROCBLAS(rocblas_gemm_ex_get_solutions(
        handle, rocblas_operation_transpose, rocblas_operation_none,
        static_cast<int>(output_dim), static_cast<int>(rows),
        static_cast<int>(input_dim), &alpha, weight, rocblas_datatype_f32_r,
        static_cast<int>(input_dim), input, rocblas_datatype_f32_r,
        static_cast<int>(input_dim), &beta, candidate, rocblas_datatype_f32_r,
        static_cast<int>(output_dim), candidate, rocblas_datatype_f32_r,
        static_cast<int>(output_dim), rocblas_datatype_f32_r,
        rocblas_gemm_algo_solution_index, 0, solutions.data(),
        &solution_count));

    results.reserve(static_cast<size_t>(solution_count));
    for (rocblas_int index = 0; index < solution_count; index++) {
        int32_t solution = solutions[static_cast<size_t>(index)];
        float milliseconds = 0.0f;
        if (!measure(candidate, rocblas_gemm_algo_solution_index, solution,
                     &milliseconds)) continue;
        HIP(hipMemsetAsync(mismatches, 0, sizeof(*mismatches), stream));
        compare_bits<<<static_cast<unsigned>(blocks(output_elements)),
                       kThreads, 0, stream>>>(
            reference, candidate, output_elements, mismatches);
        HIP(hipGetLastError());
        unsigned long long host_mismatches = 0;
        HIP(hipMemcpyAsync(&host_mismatches, mismatches,
                           sizeof(host_mismatches), hipMemcpyDeviceToHost,
                           stream));
        HIP(hipStreamSynchronize(stream));
        results.push_back({solution, milliseconds, host_mismatches});
    }
    std::sort(results.begin(), results.end(),
              [](const result &left, const result &right) {
                  return left.milliseconds < right.milliseconds;
              });
    std::printf("shape M=%u N=%u K=%u iterations=%u solutions=%d "
                "default_ms=%.6f\n", rows, output_dim, input_dim,
                iterations, solution_count, default_ms);
    for (const result &value : results)
        std::printf("solution=%d ms=%.6f speedup=%.6f exact=%d "
                    "mismatches=%llu\n", value.solution,
                    value.milliseconds, default_ms / value.milliseconds,
                    value.mismatches == 0, value.mismatches);

    HIPBLASLT(hipblasLtMatmulDescCreate(
        &lt_operation, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    HIPBLASLT(hipblasLtMatrixLayoutCreate(
        &lt_weight_layout, HIP_R_32F, input_dim, output_dim, input_dim));
    HIPBLASLT(hipblasLtMatrixLayoutCreate(
        &lt_input_layout, HIP_R_32F, input_dim, rows, input_dim));
    HIPBLASLT(hipblasLtMatrixLayoutCreate(
        &lt_output_layout, HIP_R_32F, output_dim, rows, output_dim));
    {
        hipblasOperation_t transpose = HIPBLAS_OP_T;
        hipblasOperation_t identity = HIPBLAS_OP_N;
        HIPBLASLT(hipblasLtMatmulDescSetAttribute(
            lt_operation, HIPBLASLT_MATMUL_DESC_TRANSA,
            &transpose, sizeof(transpose)));
        HIPBLASLT(hipblasLtMatmulDescSetAttribute(
            lt_operation, HIPBLASLT_MATMUL_DESC_TRANSB,
            &identity, sizeof(identity)));
    }
    HIPBLASLT(hipblasLtMatmulPreferenceCreate(&lt_preference));
    HIPBLASLT(hipblasLtMatmulPreferenceSetAttribute(
        lt_preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &kMaximumWorkspace, sizeof(kMaximumWorkspace)));
    HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(
        lt_handle, lt_operation, lt_weight_layout, lt_input_layout,
        lt_output_layout, lt_output_layout, lt_preference,
        kMaximumAlgorithms, lt_heuristics, &lt_algorithm_count));
    std::printf("hipblaslt_algorithms=%d max_workspace=%zu\n",
                lt_algorithm_count, kMaximumWorkspace);
    for (int index = 0; index < lt_algorithm_count; index++) {
        float milliseconds = 0.0f;
        if (!lt_measure(&lt_heuristics[index].algo, &milliseconds)) continue;
        HIP(hipMemsetAsync(mismatches, 0, sizeof(*mismatches), stream));
        compare_bits<<<static_cast<unsigned>(blocks(output_elements)),
                       kThreads, 0, stream>>>(
            reference, candidate, output_elements, mismatches);
        HIP(hipGetLastError());
        unsigned long long host_mismatches = 0;
        HIP(hipMemcpyAsync(&host_mismatches, mismatches,
                           sizeof(host_mismatches), hipMemcpyDeviceToHost,
                           stream));
        HIP(hipStreamSynchronize(stream));
        std::printf("hipblaslt_rank=%d workspace=%zu ms=%.6f speedup=%.6f "
                    "exact=%d mismatches=%llu\n", index,
                    lt_heuristics[index].workspaceSize, milliseconds,
                    default_ms / milliseconds, host_mismatches == 0,
                    host_mismatches);
    }
    exit_code = 0;

cleanup:
    if (lt_preference)
        (void)hipblasLtMatmulPreferenceDestroy(lt_preference);
    if (lt_output_layout)
        (void)hipblasLtMatrixLayoutDestroy(lt_output_layout);
    if (lt_input_layout)
        (void)hipblasLtMatrixLayoutDestroy(lt_input_layout);
    if (lt_weight_layout)
        (void)hipblasLtMatrixLayoutDestroy(lt_weight_layout);
    if (lt_operation) (void)hipblasLtMatmulDescDestroy(lt_operation);
    if (lt_handle) (void)hipblasLtDestroy(lt_handle);
    if (workspace) (void)hipFree(workspace);
    if (mismatches) (void)hipFree(mismatches);
    if (candidate) (void)hipFree(candidate);
    if (reference) (void)hipFree(reference);
    if (weight) (void)hipFree(weight);
    if (input) (void)hipFree(input);
    if (handle) (void)rocblas_destroy_handle(handle);
    if (end) (void)hipEventDestroy(end);
    if (begin) (void)hipEventDestroy(begin);
    if (stream) (void)hipStreamDestroy(stream);
#undef HIPBLASLT
#undef ROCBLAS
#undef HIP
    return exit_code;
}

#pragma clang diagnostic pop
