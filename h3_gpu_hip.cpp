#include "h3_gpu.h"
#include "h3_vdn_sage.h"
#include "h3_vdn_sage_gfx12.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <rocwmma/rocwmma.hpp>

/* ROCm 7 keeps a variadic compatibility macro for the removed workspace
 * arguments. Use the current function declaration directly under C++17. */
#ifdef rocblas_gemm_ex
#undef rocblas_gemm_ex
#endif

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <pthread.h>
#include <unistd.h>

struct h3_gpu_staging {
    void *data;
    h3_gpu_staging *next;
};

struct h3_gpu {
    int device;
    int warp_size;
    char gcn_arch_name[64];
    hipStream_t stream;
    rocblas_handle blas;
    char error[512];
    h3_gpu_stats stats;
    int recording;
    char profile_label[128];
    double profile_start_wall;
    double profile_mark_wall;
    double command_start_wall;
    h3_gpu_stats profile_start_stats;
    h3_gpu_stats profile_mark_stats;
    hipEvent_t *profile_events;
    uint8_t *profile_categories;
    size_t profile_capacity;
    size_t profile_count;
    int profile_event_open;
    double profile_linear_ms;
    double profile_lora_ms;
    double profile_sdpa_ms;
    double profile_solve_ms;
    double profile_scan_ms;
    h3_gpu_profile_stats profile_totals;
    pthread_mutex_t staging_lock;
    int staging_lock_initialized;
    int staging_cache_enabled;
    h3_gpu_staging *staging_ready;
    size_t staging_ready_count;
    uint64_t staging_hits;
    uint64_t staging_misses;
    hipEvent_t staging_copy_start[2];
    hipEvent_t staging_copy_end[2];
    int staging_events_initialized;
    void *sage_workspace;
    size_t sage_workspace_bytes;
    h3_vdn_sage_geometry sage_geometry;
    size_t sage_tasks_offset;
    uint32_t sage_task_count;
    int sage_tasks_valid;
};

struct h3_gpu_tensor {
    h3_gpu *owner;
    void *data;
    size_t elements;
    size_t bytes;
    h3_gpu_dtype dtype;
};

static void h3_gpu_set_error(h3_gpu *gpu, const char *format, ...) {
    if (!gpu) return;
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(gpu->error, sizeof(gpu->error), format, arguments);
    va_end(arguments);
}

static int h3_gpu_check(h3_gpu *gpu, hipError_t status,
                        const char *operation) {
    if (status == hipSuccess) return 1;
    h3_gpu_set_error(gpu, "%s: %s", operation, hipGetErrorString(status));
    return 0;
}

static int h3_gpu_check_blas(h3_gpu *gpu, rocblas_status status,
                             const char *operation) {
    if (status == rocblas_status_success) return 1;
    h3_gpu_set_error(gpu, "%s: %s", operation,
                     rocblas_status_to_string(status));
    return 0;
}

static size_t h3_gpu_dtype_size(h3_gpu_dtype dtype) {
    switch (dtype) {
        case H3_GPU_F32: return sizeof(float);
        case H3_GPU_BF16: return sizeof(uint16_t);
        case H3_GPU_I8: return sizeof(int8_t);
        case H3_GPU_U32: return sizeof(uint32_t);
    }
    return 0;
}

static void *h3_gpu_acquire_staging(h3_gpu *gpu, size_t bytes) {
    if (gpu->staging_cache_enabled) {
        (void)pthread_mutex_lock(&gpu->staging_lock);
        h3_gpu_staging *item = gpu->staging_ready;
        if (item) {
            gpu->staging_ready = item->next;
            gpu->staging_ready_count--;
            gpu->staging_hits++;
            (void)pthread_mutex_unlock(&gpu->staging_lock);
            void *data = item->data;
            std::free(item);
            return data;
        }
        gpu->staging_misses++;
        (void)pthread_mutex_unlock(&gpu->staging_lock);
    }
    void *data = nullptr;
    if (!h3_gpu_check(gpu, hipHostMalloc(&data, bytes),
                      "hipHostMalloc weight staging")) return nullptr;
    return data;
}

static void h3_gpu_release_staging(h3_gpu *gpu, void *data) {
    constexpr size_t max_ready = 32;
    if (!data) return;
    if (gpu->staging_cache_enabled) {
        h3_gpu_staging *item = static_cast<h3_gpu_staging *>(
            std::malloc(sizeof(*item)));
        if (item) {
            item->data = data;
            (void)pthread_mutex_lock(&gpu->staging_lock);
            if (gpu->staging_ready_count < max_ready) {
                item->next = gpu->staging_ready;
                gpu->staging_ready = item;
                gpu->staging_ready_count++;
                (void)pthread_mutex_unlock(&gpu->staging_lock);
                return;
            }
            (void)pthread_mutex_unlock(&gpu->staging_lock);
            std::free(item);
        }
    }
    (void)hipHostFree(data);
}

static void h3_gpu_purge_staging(h3_gpu *gpu) {
    if (!gpu || !gpu->staging_lock_initialized) return;
    (void)pthread_mutex_lock(&gpu->staging_lock);
    h3_gpu_staging *item = gpu->staging_ready;
    gpu->staging_ready = nullptr;
    gpu->staging_ready_count = 0;
    (void)pthread_mutex_unlock(&gpu->staging_lock);
    while (item) {
        h3_gpu_staging *next = item->next;
        (void)hipHostFree(item->data);
        std::free(item);
        item = next;
    }
}

static int h3_gpu_init_staging_events(h3_gpu *gpu) {
    if (gpu->staging_events_initialized) return 1;
    for (unsigned slot = 0; slot < 2; slot++) {
        if (!h3_gpu_check(gpu, hipEventCreate(&gpu->staging_copy_start[slot]),
                          "hipEventCreate staging start") ||
            !h3_gpu_check(gpu, hipEventCreate(&gpu->staging_copy_end[slot]),
                          "hipEventCreate staging end")) {
            for (unsigned cleanup = 0; cleanup < 2; cleanup++) {
                if (gpu->staging_copy_start[cleanup])
                    (void)hipEventDestroy(gpu->staging_copy_start[cleanup]);
                if (gpu->staging_copy_end[cleanup])
                    (void)hipEventDestroy(gpu->staging_copy_end[cleanup]);
                gpu->staging_copy_start[cleanup] = nullptr;
                gpu->staging_copy_end[cleanup] = nullptr;
            }
            return 0;
        }
    }
    gpu->staging_events_initialized = 1;
    return 1;
}

static void h3_gpu_destroy_staging_events(h3_gpu *gpu) {
    if (!gpu) return;
    for (unsigned slot = 0; slot < 2; slot++) {
        if (gpu->staging_copy_start[slot])
            (void)hipEventDestroy(gpu->staging_copy_start[slot]);
        if (gpu->staging_copy_end[slot])
            (void)hipEventDestroy(gpu->staging_copy_end[slot]);
    }
    gpu->staging_events_initialized = 0;
}

static int h3_gpu_finish_staging_copy(h3_gpu *gpu, unsigned slot,
                                      size_t *pending_bytes, int profile) {
    if (!*pending_bytes) return 1;
    if (!h3_gpu_check(gpu, hipEventSynchronize(gpu->staging_copy_end[slot]),
                      "hipEventSynchronize weight upload")) return 0;
    if (profile) {
        float milliseconds = 0.0f;
        if (!h3_gpu_check(
                gpu, hipEventElapsedTime(&milliseconds,
                                         gpu->staging_copy_start[slot],
                                         gpu->staging_copy_end[slot]),
                "hipEventElapsedTime weight upload")) return 0;
        gpu->profile_totals.weight_upload_seconds +=
            static_cast<double>(milliseconds) / 1000.0;
        gpu->profile_totals.weight_upload_bytes +=
            static_cast<uint64_t>(*pending_bytes);
    }
    *pending_bytes = 0;
    return 1;
}

enum h3_gpu_profile_category {
    H3_HIP_PROFILE_LINEAR = 0,
    H3_HIP_PROFILE_LORA = 1,
    H3_HIP_PROFILE_SDPA = 2,
    H3_HIP_PROFILE_SOLVE = 3,
    H3_HIP_PROFILE_SCAN = 4
};

static double h3_gpu_now(void) {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static int h3_gpu_profile_enabled(void) {
    const char *value = std::getenv("H3_PROFILE");
    return value && *value && std::strcmp(value, "0");
}

static int h3_gpu_profile_init_events(h3_gpu *gpu) {
    constexpr size_t capacity = 512;
    if (gpu->profile_events) return 1;
    hipEvent_t *events = static_cast<hipEvent_t *>(
        std::calloc(capacity * 2, sizeof(*events)));
    uint8_t *categories = static_cast<uint8_t *>(
        std::calloc(capacity, sizeof(*categories)));
    if (!events || !categories) {
        std::free(events);
        std::free(categories);
        return 0;
    }
    size_t created = 0;
    for (; created < capacity * 2; created++) {
        if (hipEventCreateWithFlags(&events[created], hipEventDefault) !=
            hipSuccess) break;
    }
    if (created != capacity * 2) {
        for (size_t index = 0; index < created; index++)
            (void)hipEventDestroy(events[index]);
        std::free(events);
        std::free(categories);
        return 0;
    }
    gpu->profile_events = events;
    gpu->profile_categories = categories;
    gpu->profile_capacity = capacity;
    return 1;
}

static void h3_gpu_profile_begin_op(h3_gpu *gpu) {
    if (!gpu || !h3_gpu_profile_enabled() || gpu->profile_event_open ||
        (gpu->profile_events &&
         gpu->profile_count >= gpu->profile_capacity))
        return;
    if (!h3_gpu_profile_init_events(gpu) ||
        gpu->profile_count >= gpu->profile_capacity) return;
    if (hipEventRecord(gpu->profile_events[gpu->profile_count * 2],
                       gpu->stream) == hipSuccess)
        gpu->profile_event_open = 1;
}

static void h3_gpu_profile_end_op(h3_gpu *gpu,
                                  h3_gpu_profile_category category) {
    if (!gpu || !gpu->profile_event_open ||
        gpu->profile_count >= gpu->profile_capacity) return;
    size_t index = gpu->profile_count;
    if (hipEventRecord(gpu->profile_events[index * 2 + 1], gpu->stream) ==
        hipSuccess) {
        gpu->profile_categories[index] = static_cast<uint8_t>(category);
        gpu->profile_count++;
    }
    gpu->profile_event_open = 0;
}

struct h3_gpu_profile_scope {
    h3_gpu *gpu;
    h3_gpu_profile_category category;
    h3_gpu_profile_scope(h3_gpu *value, h3_gpu_profile_category kind)
        : gpu(value), category(kind) { h3_gpu_profile_begin_op(gpu); }
    ~h3_gpu_profile_scope() { h3_gpu_profile_end_op(gpu, category); }
};

static void h3_gpu_profile_flush_ops(h3_gpu *gpu) {
    if (!gpu || !gpu->profile_count) return;
    for (size_t index = 0; index < gpu->profile_count; index++) {
        float milliseconds = 0.0f;
        if (hipEventElapsedTime(&milliseconds, gpu->profile_events[index * 2],
                                gpu->profile_events[index * 2 + 1]) !=
            hipSuccess) continue;
        switch (gpu->profile_categories[index]) {
        case H3_HIP_PROFILE_LINEAR:
            gpu->profile_linear_ms += milliseconds;
            gpu->profile_totals.linear_seconds += milliseconds / 1000.0;
            gpu->profile_totals.linear_calls++;
            break;
        case H3_HIP_PROFILE_LORA:
            gpu->profile_lora_ms += milliseconds;
            gpu->profile_totals.lora_seconds += milliseconds / 1000.0;
            gpu->profile_totals.lora_calls++;
            break;
        case H3_HIP_PROFILE_SDPA:
            gpu->profile_sdpa_ms += milliseconds;
            gpu->profile_totals.sdpa_seconds += milliseconds / 1000.0;
            gpu->profile_totals.sdpa_calls++;
            break;
        case H3_HIP_PROFILE_SOLVE:
            gpu->profile_solve_ms += milliseconds;
            gpu->profile_totals.solve_seconds += milliseconds / 1000.0;
            gpu->profile_totals.solve_calls++;
            break;
        case H3_HIP_PROFILE_SCAN:
            gpu->profile_scan_ms += milliseconds;
            gpu->profile_totals.scan_seconds += milliseconds / 1000.0;
            gpu->profile_totals.scan_calls++;
            break;
        }
    }
    gpu->profile_count = 0;
}

static void h3_gpu_profile_destroy_events(h3_gpu *gpu) {
    if (!gpu || !gpu->profile_events) return;
    for (size_t index = 0; index < gpu->profile_capacity * 2; index++)
        (void)hipEventDestroy(gpu->profile_events[index]);
    std::free(gpu->profile_events);
    std::free(gpu->profile_categories);
    gpu->profile_events = nullptr;
    gpu->profile_categories = nullptr;
    gpu->profile_capacity = 0;
    gpu->profile_count = 0;
}

static uint64_t h3_gpu_counter_delta(uint64_t value, uint64_t start) {
    return value >= start ? value - start : 0;
}

static void h3_gpu_profile_emit(h3_gpu *gpu, const char *phase,
                                const h3_gpu_stats &start,
                                double wall_start) {
    if (!gpu || !phase || !h3_gpu_profile_enabled()) return;
    const h3_gpu_stats &value = gpu->stats;
    std::fprintf(stderr,
        "h3 profile: %-20s %-14s wall=%8.3fs encode=%8.3fs wait=%8.3fs "
        "peak=%7.3fGiB alloc=%7.3fGiB submissions=%llu linear=%llu "
        "attention=%llu direct=%llu\n",
        gpu->profile_label[0] ? gpu->profile_label : "HIP context", phase,
        h3_gpu_now() - wall_start,
        value.command_encode_seconds - start.command_encode_seconds,
        value.command_wait_seconds - start.command_wait_seconds,
        static_cast<double>(value.peak_live_bytes) /
            (1024.0 * 1024.0 * 1024.0),
        static_cast<double>(h3_gpu_counter_delta(value.allocated_bytes,
                                                  start.allocated_bytes)) /
            (1024.0 * 1024.0 * 1024.0),
        static_cast<unsigned long long>(h3_gpu_counter_delta(
            value.submissions, start.submissions)),
        static_cast<unsigned long long>(h3_gpu_counter_delta(
            value.mps_linear_dispatches, start.mps_linear_dispatches)),
        static_cast<unsigned long long>(h3_gpu_counter_delta(
            value.mps_sdpa_dispatches, start.mps_sdpa_dispatches)),
        static_cast<unsigned long long>(h3_gpu_counter_delta(
            value.direct_dispatches, start.direct_dispatches)));
}

static void h3_gpu_profile_emit_ops(h3_gpu *gpu) {
    if (!gpu || !h3_gpu_profile_enabled()) return;
    double total = gpu->profile_linear_ms + gpu->profile_lora_ms +
                   gpu->profile_sdpa_ms +
                   gpu->profile_solve_ms + gpu->profile_scan_ms;
    if (total <= 0.0) return;
    std::fprintf(stderr,
        "h3 profile: %-20s %-14s measured=%8.3fs linear=%8.3fs "
        "lora=%8.3fs sdpa=%8.3fs solve=%8.3fs scan=%8.3fs\n",
        gpu->profile_label[0] ? gpu->profile_label : "HIP context",
        "gpu-op-classes", total / 1000.0, gpu->profile_linear_ms / 1000.0,
        gpu->profile_lora_ms / 1000.0, gpu->profile_sdpa_ms / 1000.0,
        gpu->profile_solve_ms / 1000.0,
        gpu->profile_scan_ms / 1000.0);
}

static void h3_gpu_profile_emit_load(h3_gpu *gpu) {
    if (!gpu || !h3_gpu_profile_enabled() ||
        !gpu->profile_totals.weight_read_bytes) return;
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    std::fprintf(stderr,
        "h3 profile: %-20s %-14s read=%8.3fs (%7.3fGiB, %6.2fGiB/s) "
        "upload=%8.3fs (%7.3fGiB, %6.2fGiB/s) staging-hit=%llu/%llu\n",
        gpu->profile_label[0] ? gpu->profile_label : "HIP context",
        "weight-load", gpu->profile_totals.weight_read_seconds,
        static_cast<double>(gpu->profile_totals.weight_read_bytes) / gib,
        gpu->profile_totals.weight_read_seconds > 0.0 ?
            static_cast<double>(gpu->profile_totals.weight_read_bytes) / gib /
                gpu->profile_totals.weight_read_seconds : 0.0,
        gpu->profile_totals.weight_upload_seconds,
        static_cast<double>(gpu->profile_totals.weight_upload_bytes) / gib,
        gpu->profile_totals.weight_upload_seconds > 0.0 ?
            static_cast<double>(gpu->profile_totals.weight_upload_bytes) / gib /
                gpu->profile_totals.weight_upload_seconds : 0.0,
        static_cast<unsigned long long>(gpu->staging_hits),
        static_cast<unsigned long long>(gpu->staging_hits +
                                        gpu->staging_misses));
}

static h3_gpu_tensor *h3_gpu_tensor_new(h3_gpu *gpu, const void *values,
                                        size_t elements,
                                        h3_gpu_dtype dtype) {
    size_t item_size = h3_gpu_dtype_size(dtype);
    if (!gpu || !item_size || elements > SIZE_MAX / item_size) return nullptr;
    size_t bytes = elements * item_size;

    h3_gpu_tensor *tensor =
        static_cast<h3_gpu_tensor *>(std::calloc(1, sizeof(*tensor)));
    if (!tensor) {
        h3_gpu_set_error(gpu, "out of memory creating HIP tensor metadata");
        return nullptr;
    }
    tensor->owner = gpu;
    tensor->elements = elements;
    tensor->bytes = bytes;
    tensor->dtype = dtype;

    if (!h3_gpu_check(gpu, hipSetDevice(gpu->device), "hipSetDevice") ||
        !h3_gpu_check(gpu, hipMalloc(&tensor->data, bytes ? bytes : 1),
                      "hipMalloc")) {
        std::free(tensor);
        return nullptr;
    }
    if (values && bytes &&
        !h3_gpu_check(gpu,
                      hipMemcpyAsync(tensor->data, values, bytes,
                                     hipMemcpyHostToDevice, gpu->stream),
                      "hipMemcpyAsync host-to-device")) {
        (void)hipFree(tensor->data);
        std::free(tensor);
        return nullptr;
    }

    gpu->stats.allocated_bytes += bytes;
    gpu->stats.live_bytes += bytes;
    if (gpu->stats.live_bytes > gpu->stats.peak_live_bytes)
        gpu->stats.peak_live_bytes = gpu->stats.live_bytes;
    gpu->stats.tensor_allocations++;
    return tensor;
}

static int h3_gpu_ensure_sage_workspace(h3_gpu *gpu, size_t bytes) {
    if (gpu->sage_workspace_bytes >= bytes) return 1;
    void *replacement = nullptr;
    if (!h3_gpu_check(gpu, hipMalloc(&replacement, bytes ? bytes : 1),
                      "hipMalloc VDN Sage workspace")) return 0;
    if (gpu->sage_workspace) {
        if (!h3_gpu_check(gpu, hipFree(gpu->sage_workspace),
                          "hipFree old VDN Sage workspace")) {
            (void)hipFree(replacement);
            return 0;
        }
        gpu->stats.live_bytes -= gpu->sage_workspace_bytes;
    }
    gpu->sage_workspace = replacement;
    gpu->sage_workspace_bytes = bytes;
    gpu->sage_tasks_valid = 0;
    gpu->stats.allocated_bytes += bytes;
    gpu->stats.live_bytes += bytes;
    if (gpu->stats.live_bytes > gpu->stats.peak_live_bytes)
        gpu->stats.peak_live_bytes = gpu->stats.live_bytes;
    return 1;
}

static int h3_gpu_same_sage_geometry(
        const h3_vdn_sage_geometry *left,
        const h3_vdn_sage_geometry *right) {
    return left->sequence == right->sequence &&
        left->heads == right->heads &&
        left->head_dim == right->head_dim &&
        left->video_start == right->video_start &&
        left->frames == right->frames &&
        left->tokens_per_frame == right->tokens_per_frame &&
        left->radius == right->radius && left->chunk == right->chunk &&
        left->anchor_both == right->anchor_both;
}

static int h3_gpu_prepare_sage_tasks(
        h3_gpu *gpu, const h3_vdn_sage_geometry *geometry,
        size_t tasks_offset, h3_vdn_q_task **device_tasks,
        uint32_t *task_count) {
    if (!gpu || !geometry || !device_tasks || !task_count) return 0;
    if (gpu->sage_tasks_valid && gpu->sage_tasks_offset == tasks_offset &&
        h3_gpu_same_sage_geometry(&gpu->sage_geometry, geometry)) {
        *device_tasks = reinterpret_cast<h3_vdn_q_task *>(
            static_cast<uint8_t *>(gpu->sage_workspace) + tasks_offset);
        *task_count = gpu->sage_task_count;
        return 1;
    }

    h3_vdn_q_task *host_tasks = nullptr;
    size_t host_task_count = 0;
    char task_error[256] = {};
    if (!h3_vdn_sage_build_tasks(
            geometry, &host_tasks, &host_task_count, task_error,
            sizeof(task_error))) {
        h3_gpu_set_error(gpu, "cannot build VDN Sage tasks: %s",
                         task_error[0] ? task_error : "unknown error");
        return 0;
    }
    if (host_task_count > UINT32_MAX ||
        host_task_count > (SIZE_MAX - tasks_offset) /
                              sizeof(h3_vdn_q_task)) {
        h3_vdn_sage_free_tasks(host_tasks);
        h3_gpu_set_error(gpu, "VDN Sage task workspace size overflow");
        return 0;
    }
    const size_t task_bytes = host_task_count * sizeof(h3_vdn_q_task);
    const size_t required_bytes = tasks_offset + task_bytes;
    if (!h3_gpu_ensure_sage_workspace(gpu, required_bytes)) {
        h3_vdn_sage_free_tasks(host_tasks);
        return 0;
    }
    h3_vdn_q_task *destination = reinterpret_cast<h3_vdn_q_task *>(
        static_cast<uint8_t *>(gpu->sage_workspace) + tasks_offset);
    const hipError_t copy_status = hipMemcpyAsync(
        destination, host_tasks, task_bytes, hipMemcpyHostToDevice,
        gpu->stream);
    h3_vdn_sage_free_tasks(host_tasks);
    if (!h3_gpu_check(gpu, copy_status,
                      "hipMemcpyAsync VDN Sage task metadata")) return 0;

    gpu->sage_geometry = *geometry;
    gpu->sage_tasks_offset = tasks_offset;
    gpu->sage_task_count = static_cast<uint32_t>(host_task_count);
    gpu->sage_tasks_valid = 1;
    *device_tasks = destination;
    *task_count = gpu->sage_task_count;
    return 1;
}

static int h3_gpu_valid_range(const h3_gpu_tensor *tensor,
                              size_t offset, size_t elements,
                              h3_gpu_dtype dtype) {
    return tensor && tensor->dtype == dtype && offset <= tensor->elements &&
           elements <= tensor->elements - offset;
}

static int h3_gpu_copy_to_host(const h3_gpu_tensor *tensor, size_t offset,
                               void *values, size_t elements,
                               h3_gpu_dtype dtype) {
    if (!values || !h3_gpu_valid_range(tensor, offset, elements, dtype))
        return 0;
    h3_gpu *gpu = tensor->owner;
    size_t item_size = h3_gpu_dtype_size(dtype);
    const unsigned char *source =
        static_cast<const unsigned char *>(tensor->data) + offset * item_size;
    if (!h3_gpu_check(gpu, hipSetDevice(gpu->device), "hipSetDevice") ||
        !h3_gpu_check(gpu,
                      hipMemcpyAsync(values, source, elements * item_size,
                                     hipMemcpyDeviceToHost, gpu->stream),
                      "hipMemcpyAsync device-to-host") ||
        !h3_gpu_check(gpu, hipStreamSynchronize(gpu->stream),
                      "hipStreamSynchronize"))
        return 0;
    return 1;
}

static int h3_gpu_copy_from_host(h3_gpu_tensor *tensor, size_t offset,
                                 const void *values, size_t elements,
                                 h3_gpu_dtype dtype) {
    if (!values || !h3_gpu_valid_range(tensor, offset, elements, dtype))
        return 0;
    h3_gpu *gpu = tensor->owner;
    size_t item_size = h3_gpu_dtype_size(dtype);
    unsigned char *destination =
        static_cast<unsigned char *>(tensor->data) + offset * item_size;
    if (!h3_gpu_check(gpu, hipSetDevice(gpu->device), "hipSetDevice") ||
        !h3_gpu_check(gpu,
                      hipMemcpyAsync(destination, values, elements * item_size,
                                     hipMemcpyHostToDevice, gpu->stream),
                      "hipMemcpyAsync host-to-device"))
        return 0;
    return 1;
}

static int h3_gpu_read_file(h3_gpu_tensor *tensor, const char *path,
                            uint64_t file_offset, size_t elements,
                            h3_gpu_dtype dtype, char *error,
                            size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tensor || !path || !*path || tensor->dtype != dtype ||
        tensor->elements != elements ||
        file_offset > static_cast<uint64_t>(
            std::numeric_limits<off_t>::max())) {
        if (error && error_size)
            std::snprintf(error, error_size, "invalid HIP tensor file read");
        return 0;
    }

    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        if (error && error_size)
            std::snprintf(error, error_size, "cannot open %s: %s",
                          path, std::strerror(errno));
        return 0;
    }

    constexpr size_t chunk_capacity = 8 * 1024 * 1024;
    h3_gpu *gpu = tensor->owner;
    void *staging = h3_gpu_acquire_staging(gpu, chunk_capacity);
    if (!staging) {
        close(descriptor);
        if (error && error_size)
            std::snprintf(error, error_size, "%s", gpu->error);
        return 0;
    }

    size_t completed = 0;
    size_t bytes = tensor->bytes;
    int profile = h3_gpu_profile_enabled();
    const char *serial_value = std::getenv("H3_HIP_SERIAL_STAGING");
    int serial = !gpu->staging_cache_enabled ||
        (serial_value && *serial_value && std::strcmp(serial_value, "0"));
    if (!serial && bytes > chunk_capacity) {
        void *staging_pair[2] = {
            staging, h3_gpu_acquire_staging(gpu, chunk_capacity)
        };
        size_t pending_bytes[2] = {0, 0};
        int ok = staging_pair[1] && h3_gpu_init_staging_events(gpu);
        unsigned slot = 0;
        while (ok && completed < bytes) {
            ok = h3_gpu_finish_staging_copy(
                gpu, slot, &pending_bytes[slot], profile);
            if (!ok) break;
            size_t request = bytes - completed;
            if (request > chunk_capacity) request = chunk_capacity;
            ssize_t got;
            double read_start = profile ? h3_gpu_now() : 0.0;
            do {
                got = pread(descriptor, staging_pair[slot], request,
                            static_cast<off_t>(file_offset + completed));
            } while (got < 0 && errno == EINTR);
            if (profile && got > 0) {
                gpu->profile_totals.weight_read_seconds +=
                    h3_gpu_now() - read_start;
                gpu->profile_totals.weight_read_bytes +=
                    static_cast<uint64_t>(got);
            }
            if (got <= 0) {
                if (error && error_size)
                    std::snprintf(error, error_size, "cannot read %s: %s",
                                  path, got < 0 ? std::strerror(errno) :
                                                 "unexpected end of file");
                ok = 0;
                break;
            }
            unsigned char *destination =
                static_cast<unsigned char *>(tensor->data) + completed;
            ok = h3_gpu_check(
                     gpu, hipEventRecord(gpu->staging_copy_start[slot],
                                         gpu->stream),
                     "hipEventRecord weight upload start") &&
                 h3_gpu_check(
                     gpu, hipMemcpyAsync(destination, staging_pair[slot],
                                         static_cast<size_t>(got),
                                         hipMemcpyHostToDevice, gpu->stream),
                     "hipMemcpyAsync file-to-device") &&
                 h3_gpu_check(
                     gpu, hipEventRecord(gpu->staging_copy_end[slot],
                                         gpu->stream),
                     "hipEventRecord weight upload end");
            if (!ok) break;
            pending_bytes[slot] = static_cast<size_t>(got);
            completed += static_cast<size_t>(got);
            slot ^= 1;
        }
        for (unsigned finish = 0; finish < 2; finish++)
            if (!h3_gpu_finish_staging_copy(
                    gpu, finish, &pending_bytes[finish], profile)) ok = 0;
        if (!ok) (void)hipStreamSynchronize(gpu->stream);
        h3_gpu_release_staging(gpu, staging_pair[1]);
        h3_gpu_release_staging(gpu, staging_pair[0]);
        close(descriptor);
        if (!ok && error && error_size && !error[0])
            std::snprintf(error, error_size, "%s", gpu->error);
        return ok;
    }
    while (completed < bytes) {
        size_t request = bytes - completed;
        if (request > chunk_capacity) request = chunk_capacity;
        ssize_t got;
        double read_start = profile ? h3_gpu_now() : 0.0;
        do {
            got = pread(descriptor, staging, request,
                        static_cast<off_t>(file_offset + completed));
        } while (got < 0 && errno == EINTR);
        if (profile && got > 0) {
            gpu->profile_totals.weight_read_seconds +=
                h3_gpu_now() - read_start;
            gpu->profile_totals.weight_read_bytes +=
                static_cast<uint64_t>(got);
        }
        if (got <= 0) {
            if (error && error_size) {
                std::snprintf(error, error_size, "cannot read %s: %s", path,
                              got < 0 ? std::strerror(errno)
                                      : "unexpected end of file");
            }
            h3_gpu_release_staging(gpu, staging);
            close(descriptor);
            return 0;
        }
        unsigned char *destination =
            static_cast<unsigned char *>(tensor->data) + completed;
        double upload_start = profile ? h3_gpu_now() : 0.0;
        if (!h3_gpu_check(gpu,
                          hipMemcpyAsync(destination, staging,
                                         static_cast<size_t>(got),
                                         hipMemcpyHostToDevice, gpu->stream),
                          "hipMemcpyAsync file-to-device") ||
            !h3_gpu_check(gpu, hipStreamSynchronize(gpu->stream),
                          "hipStreamSynchronize weight upload")) {
            if (error && error_size)
                std::snprintf(error, error_size, "%s", gpu->error);
            h3_gpu_release_staging(gpu, staging);
            close(descriptor);
            return 0;
        }
        if (profile) {
            gpu->profile_totals.weight_upload_seconds +=
                h3_gpu_now() - upload_start;
            gpu->profile_totals.weight_upload_bytes +=
                static_cast<uint64_t>(got);
        }
        completed += static_cast<size_t>(got);
    }

    h3_gpu_release_staging(gpu, staging);
    close(descriptor);
    return 1;
}

static h3_gpu_tensor *h3_gpu_tensor_load(h3_gpu *gpu, const char *path,
                                         uint64_t file_offset,
                                         size_t elements,
                                         h3_gpu_dtype dtype) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new(gpu, nullptr, elements, dtype);
    if (!tensor) return nullptr;
    char detail[512];
    if (!h3_gpu_read_file(tensor, path, file_offset, elements, dtype,
                          detail, sizeof(detail))) {
        h3_gpu_set_error(gpu, "%s", detail);
        h3_gpu_tensor_free(tensor);
        return nullptr;
    }
    return tensor;
}

static int h3_gpu_copy_device(h3_gpu *gpu, h3_gpu_tensor *destination,
                              size_t destination_offset,
                              const h3_gpu_tensor *source,
                              size_t source_offset, size_t elements,
                              h3_gpu_dtype dtype) {
    if (!gpu || !h3_gpu_valid_range(destination, destination_offset,
                                     elements, dtype) ||
        !h3_gpu_valid_range(source, source_offset, elements, dtype) ||
        source->owner->device != destination->owner->device) {
        if (gpu) h3_gpu_set_error(gpu, "invalid HIP device copy");
        return 0;
    }
    size_t item_size = h3_gpu_dtype_size(dtype);
    unsigned char *destination_bytes =
        static_cast<unsigned char *>(destination->data) +
        destination_offset * item_size;
    const unsigned char *source_bytes =
        static_cast<const unsigned char *>(source->data) +
        source_offset * item_size;
    if (!h3_gpu_check(gpu,
                      hipMemcpyAsync(destination_bytes, source_bytes,
                                     elements * item_size,
                                     hipMemcpyDeviceToDevice, gpu->stream),
                      "hipMemcpyAsync device-to-device"))
        return 0;
    gpu->stats.blit_copies++;
    return 1;
}

static int h3_gpu_require_tensor(h3_gpu *gpu,
                                 const h3_gpu_tensor *tensor,
                                 size_t elements, h3_gpu_dtype dtype,
                                 const char *label) {
    if (!gpu || !tensor || tensor->owner != gpu || tensor->dtype != dtype ||
        tensor->elements < elements) {
        if (gpu)
            h3_gpu_set_error(gpu, "invalid %s tensor (need %zu elements, %s)",
                             label, elements,
                             dtype == H3_GPU_F32 ? "F32" :
                             dtype == H3_GPU_BF16 ? "BF16" :
                             dtype == H3_GPU_I8 ? "I8" : "U32");
        return 0;
    }
    return 1;
}

static int h3_gpu_count_2d(h3_gpu *gpu, uint32_t rows, uint32_t columns,
                           size_t *count, const char *label) {
    if (!rows || !columns || static_cast<size_t>(rows) >
            SIZE_MAX / static_cast<size_t>(columns)) {
        h3_gpu_set_error(gpu, "invalid %s dimensions (%u x %u)", label,
                         rows, columns);
        return 0;
    }
    *count = static_cast<size_t>(rows) * columns;
    return 1;
}

static int h3_gpu_require_compute(h3_gpu *gpu, const char *operation) {
    if (!gpu || !gpu->recording) {
        if (gpu)
            h3_gpu_set_error(gpu, "%s requires h3_gpu_begin", operation);
        return 0;
    }
    if (!h3_gpu_check(gpu, hipSetDevice(gpu->device), "hipSetDevice"))
        return 0;
    return 1;
}

static int h3_gpu_kernel_enqueued(h3_gpu *gpu, const char *operation) {
    if (!h3_gpu_check(gpu, hipGetLastError(), operation)) return 0;
    gpu->stats.direct_dispatches++;
    return 1;
}

constexpr uint32_t H3_HIP_THREADS = 256;

static dim3 h3_gpu_grid_1d(uint32_t elements) {
    return dim3((elements + H3_HIP_THREADS - 1) / H3_HIP_THREADS);
}

__global__ static void h3_hip_silu_f32_kernel(const float *input,
                                               float *output,
                                               uint32_t elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        float value = input[index];
        output[index] = value / (1.0f + expf(-value));
    }
}

__global__ static void h3_hip_cast_f32_to_bf16_kernel(
        const float *input, hip_bfloat16 *output, uint32_t elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = hip_bfloat16(input[index]);
}

__global__ static void h3_hip_cast_bf16_to_f32_kernel(
        const hip_bfloat16 *input, float *output, uint32_t elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = static_cast<float>(input[index]);
}

template <typename T>
__device__ static float h3_hip_load(T value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ static T h3_hip_store(float value) {
    return static_cast<T>(value);
}

template <typename T>
__global__ static void h3_hip_silu_kernel(const T *input, T *output,
                                          uint32_t elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        float value = h3_hip_load(input[index]);
        output[index] = h3_hip_store<T>(value / (1.0f + expf(-value)));
    }
}

template <typename T>
__global__ static void h3_hip_binary_kernel(const T *left, const T *right,
                                            T *output, uint32_t elements,
                                            int subtract) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        float a = h3_hip_load(left[index]);
        float b = h3_hip_load(right[index]);
        output[index] = h3_hip_store<T>(subtract ? a - b : a + b);
    }
}

__global__ static void h3_hip_add_scaled_f32_kernel(
        const float *left, const float *right, float *output,
        float left_scale, float right_scale, uint32_t elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
        output[index] = left[index] * left_scale + right[index] * right_scale;
}

__global__ static void h3_hip_scale_add_f32_kernel(
        const float *residual, const float *branch, const float *scale,
        float *output, uint32_t elements, uint32_t width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
        output[index] = residual[index] + branch[index] * scale[index % width];
}

__global__ static void h3_hip_clip_f32_kernel(
        const float *input, float *output, uint32_t elements,
        float minimum, float maximum) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
        output[index] = fminf(maximum, fmaxf(minimum, input[index]));
}

template <typename T>
__global__ static void h3_hip_rms_norm_kernel(
        const T *input, const T *weight, T *output, uint32_t rows,
        uint32_t width, float epsilon) {
    uint32_t row = blockIdx.x;
    uint32_t lane = threadIdx.x;
    if (row >= rows) return;
    extern __shared__ float reduction[];
    const T *values = input + static_cast<size_t>(row) * width;
    float sum = 0.0f;
    for (uint32_t column = lane; column < width; column += blockDim.x) {
        float value = h3_hip_load(values[column]);
        sum = fmaf(value, value, sum);
    }
    reduction[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reduction[0] / static_cast<float>(width) + epsilon);
    for (uint32_t column = lane; column < width; column += blockDim.x) {
        float normalized = h3_hip_load(values[column]) * inverse;
        output[static_cast<size_t>(row) * width + column] =
            h3_hip_store<T>(normalized * h3_hip_load(weight[column]));
    }
}

template <typename T>
__global__ static void h3_hip_layer_norm_kernel(
        const T *input, const T *weight, const T *bias, T *output,
        uint32_t rows, uint32_t width, float epsilon) {
    uint32_t row = blockIdx.x;
    uint32_t lane = threadIdx.x;
    if (row >= rows) return;
    extern __shared__ float reduction[];
    const T *values = input + static_cast<size_t>(row) * width;
    float sum = 0.0f;
    for (uint32_t column = lane; column < width; column += blockDim.x)
        sum += h3_hip_load(values[column]);
    reduction[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    float mean = reduction[0] / static_cast<float>(width);
    float square_sum = 0.0f;
    for (uint32_t column = lane; column < width; column += blockDim.x) {
        float centered = h3_hip_load(values[column]) - mean;
        square_sum = fmaf(centered, centered, square_sum);
    }
    reduction[lane] = square_sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reduction[0] / static_cast<float>(width) + epsilon);
    for (uint32_t column = lane; column < width; column += blockDim.x) {
        float normalized = (h3_hip_load(values[column]) - mean) * inverse;
        output[static_cast<size_t>(row) * width + column] = h3_hip_store<T>(
            fmaf(normalized, h3_hip_load(weight[column]),
                 h3_hip_load(bias[column])));
    }
}

__global__ static void h3_hip_bias_f32_kernel(float *output,
                                               const float *bias,
                                               uint32_t elements,
                                               uint32_t width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] += bias[index % width];
}

__global__ static void h3_hip_bias_bf16_kernel(hip_bfloat16 *output,
                                                const hip_bfloat16 *bias,
                                                uint32_t elements,
                                                uint32_t width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
        output[index] = hip_bfloat16(static_cast<float>(output[index]) +
                                     static_cast<float>(bias[index % width]));
}

__global__ static void h3_hip_gelu_bf16_kernel(
        const hip_bfloat16 *input, hip_bfloat16 *output, uint32_t elements,
        int approximate) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float value = static_cast<float>(input[index]);
    float activated;
    if (approximate) {
        float inner = 0.7978845608028654f *
                      (value + 0.044715f * value * value * value);
        activated = inner <= -10.0f ? 0.0f :
                    inner >= 10.0f ? value :
                    0.5f * value * (1.0f + tanhf(inner));
    } else {
        activated = value <= -10.0f ? 0.0f :
                    value >= 10.0f ? value :
                    0.5f * value *
                        (1.0f + erff(value * 0.7071067811865475f));
    }
    output[index] = hip_bfloat16(activated);
}

__global__ static void h3_hip_swiglu_bf16_kernel(
        const hip_bfloat16 *fused, hip_bfloat16 *output, uint32_t rows,
        uint32_t width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t elements = rows * width;
    if (index >= elements) return;
    uint32_t row = index / width;
    uint32_t column = index % width;
    size_t base = static_cast<size_t>(row) * width * 2;
    float gate = static_cast<float>(fused[base + column]);
    float up = static_cast<float>(fused[base + width + column]);
    output[index] = hip_bfloat16(gate / (1.0f + expf(-gate)) * up);
}

__global__ static void h3_hip_silu_mul_bf16_kernel(
        const hip_bfloat16 *gate, const hip_bfloat16 *up,
        hip_bfloat16 *output, uint32_t elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        float value = static_cast<float>(gate[index]);
        output[index] = hip_bfloat16(value / (1.0f + expf(-value)) *
                                     static_cast<float>(up[index]));
    }
}

__global__ static void h3_hip_euler_bf16_kernel(
        float *sample, size_t sample_offset, const hip_bfloat16 *last,
        const hip_bfloat16 *previous, uint32_t elements, float delta,
        float ratio) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        float last_value = static_cast<float>(last[index]);
        float velocity = fmaf(ratio,
                              last_value - static_cast<float>(previous[index]),
                              last_value);
        size_t sample_index = sample_offset + index;
        sample[sample_index] = fmaf(delta, velocity, sample[sample_index]);
    }
}

__global__ static void h3_hip_euler_f32_kernel(
        float *sample, const float *velocity, uint32_t elements,
        float velocity_scale) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
        sample[index] = fmaf(velocity_scale, velocity[index], sample[index]);
}

__global__ static void h3_hip_weight_norm_f32_kernel(
        float *output, const float *vector, const float *magnitude,
        uint32_t outer, uint32_t inner) {
    uint32_t row = blockIdx.x;
    uint32_t lane = threadIdx.x;
    if (row >= outer) return;
    extern __shared__ float reduction[];
    size_t base = static_cast<size_t>(row) * inner;
    float sum = 0.0f;
    for (uint32_t column = lane; column < inner; column += blockDim.x)
        sum = fmaf(vector[base + column], vector[base + column], sum);
    reduction[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    float scale = magnitude[row] / sqrtf(reduction[0]);
    for (uint32_t column = lane; column < inner; column += blockDim.x)
        output[base + column] = vector[base + column] * scale;
}

__global__ static void h3_hip_conv1d_f32_kernel(
        float *output, const float *input, const float *weight,
        const float *bias, uint32_t output_elements, uint32_t length,
        uint32_t output_length, uint32_t input_channels,
        uint32_t output_channels, uint32_t kernel, uint32_t stride,
        uint32_t padding, uint32_t dilation, int has_bias) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;
    uint32_t output_channel = index % output_channels;
    uint32_t position = (index / output_channels) % output_length;
    uint32_t batch = index / (output_channels * output_length);
    float sum = has_bias ? bias[output_channel] : 0.0f;
    int64_t start = static_cast<int64_t>(position) * stride - padding;
    for (uint32_t input_channel = 0; input_channel < input_channels;
         input_channel++)
        for (uint32_t tap = 0; tap < kernel; tap++) {
            int64_t source_position = start +
                static_cast<int64_t>(tap) * dilation;
            if (source_position < 0 || source_position >= length) continue;
            size_t source =
                (static_cast<size_t>(batch) * length +
                 static_cast<size_t>(source_position)) * input_channels +
                input_channel;
            size_t coefficient =
                (static_cast<size_t>(output_channel) * input_channels +
                 input_channel) * kernel + tap;
            sum = fmaf(input[source], weight[coefficient], sum);
        }
    output[index] = sum;
}

__global__ static void h3_hip_conv_transpose1d_f32_kernel(
        float *output, const float *input, const float *weight,
        const float *bias, uint32_t output_elements, uint32_t length,
        uint32_t output_length, uint32_t input_channels,
        uint32_t output_channels, uint32_t kernel, uint32_t stride,
        uint32_t padding, int has_bias) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_elements) return;
    uint32_t output_channel = index % output_channels;
    uint32_t position = (index / output_channels) % output_length;
    uint32_t batch = index / (output_channels * output_length);
    float sum = has_bias ? bias[output_channel] : 0.0f;
    for (uint32_t input_channel = 0; input_channel < input_channels;
         input_channel++)
        for (uint32_t tap = 0; tap < kernel; tap++) {
            int64_t numerator = static_cast<int64_t>(position) + padding - tap;
            if (numerator < 0 || numerator % stride) continue;
            int64_t source_position = numerator / stride;
            if (source_position >= length) continue;
            size_t source =
                (static_cast<size_t>(batch) * length +
                 static_cast<size_t>(source_position)) * input_channels +
                input_channel;
            size_t coefficient =
                (static_cast<size_t>(input_channel) * output_channels +
                 output_channel) * kernel + tap;
            sum = fmaf(input[source], weight[coefficient], sum);
        }
    output[index] = sum;
}

__device__ static float h3_hip_alias_upsample(
        const float *input, const float *filter, uint32_t batch,
        uint32_t channel, int64_t up_position, uint32_t length,
        uint32_t channels) {
    /* ratio=2, kernel=12, replicate-pad 5 before transposed convolution,
     * then crop 15 samples from each side. */
    int64_t raw_position = up_position + 15;
    float sum = 0.0f;
    for (int64_t tap = 0; tap < 12; tap++) {
        int64_t numerator = raw_position - tap;
        if (numerator < 0 || numerator % 2) continue;
        int64_t padded_position = numerator / 2;
        if (padded_position >= static_cast<int64_t>(length) + 10) continue;
        int64_t source_position = padded_position - 5;
        if (source_position < 0) source_position = 0;
        if (source_position >= length) source_position = length - 1;
        size_t source =
            (static_cast<size_t>(batch) * length +
             static_cast<size_t>(source_position)) * channels + channel;
        sum = fmaf(input[source], filter[tap], sum);
    }
    return 2.0f * sum;
}

__global__ static void h3_hip_alias_free_snake_f32_kernel(
        float *output, const float *input, const float *alpha_log,
        const float *beta_log, const float *upsample_filter,
        const float *downsample_filter, uint32_t elements,
        uint32_t length, uint32_t channels) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t channel = index % channels;
    uint32_t position = (index / channels) % length;
    uint32_t batch = index / (channels * length);
    float alpha = expf(alpha_log[channel]);
    float inverse_beta = 1.0f / (expf(beta_log[channel]) + 1e-9f);
    float sum = 0.0f;
    for (int64_t tap = 0; tap < 12; tap++) {
        int64_t up_position = static_cast<int64_t>(position) * 2 + tap - 5;
        if (up_position < 0) up_position = 0;
        int64_t up_length = static_cast<int64_t>(length) * 2;
        if (up_position >= up_length) up_position = up_length - 1;
        float value = h3_hip_alias_upsample(
            input, upsample_filter, batch, channel, up_position, length,
            channels);
        float sine = sinf(alpha * value);
        float activated = value + inverse_beta * sine * sine;
        sum = fmaf(activated, downsample_filter[tap], sum);
    }
    output[index] = sum;
}

__global__ static void h3_hip_snake1d_f32_kernel(
        float *output, const float *input, const float *alpha,
        uint32_t elements, uint32_t channels) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t channel = index % channels;
    float a = alpha[channel];
    float sine = sinf(a * input[index]);
    output[index] = input[index] + sine * sine / (a + 1e-9f);
}

__global__ static void h3_hip_swiglu_f32_kernel(
        const float *fused, float *output, uint32_t rows, uint32_t width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t elements = rows * width;
    if (index >= elements) return;
    uint32_t row = index / width;
    uint32_t column = index % width;
    size_t base = static_cast<size_t>(row) * width * 2;
    float gate = fused[base + column];
    output[index] = gate / (1.0f + expf(-gate)) *
                    fused[base + width + column];
}

__global__ static void h3_hip_patch_linear_bf16_kernel(
        const float *input, size_t input_offset, const float *weight,
        const float *bias, hip_bfloat16 *output, size_t output_offset,
        const uint32_t *row_map, uint32_t rows, uint32_t input_dim,
        uint32_t output_dim, int has_bias, int mapped) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= rows || column >= output_dim) return;
    const float *values = input + input_offset +
                          static_cast<size_t>(row) * input_dim;
    const float *weights = weight + static_cast<size_t>(column) * input_dim;
    float sum = has_bias ? bias[column] : 0.0f;
    for (uint32_t k = 0; k < input_dim; k++)
        sum = fmaf(values[k], weights[k], sum);
    size_t output_row = mapped ? row_map[row] : row;
    output[output_offset + output_row * output_dim + column] =
        hip_bfloat16(sum);
}

__global__ static void h3_hip_adaln_bf16_kernel(
        const hip_bfloat16 *input, size_t input_offset,
        const hip_bfloat16 *weight, const hip_bfloat16 *modulation,
        const uint32_t *row_map, hip_bfloat16 *output, uint32_t rows,
        uint32_t width, uint32_t slots, uint32_t shift_slot,
        uint32_t scale_slot, float epsilon) {
    uint32_t row = blockIdx.x;
    uint32_t lane = threadIdx.x;
    if (row >= rows) return;
    extern __shared__ float reduction[];
    const hip_bfloat16 *values = input + input_offset +
        static_cast<size_t>(row) * width;
    float sum = 0.0f;
    for (uint32_t column = lane; column < width; column += blockDim.x) {
        float value = static_cast<float>(values[column]);
        sum = fmaf(value, value, sum);
    }
    reduction[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reduction[0] / static_cast<float>(width) + epsilon);
    size_t base = static_cast<size_t>(row_map[row]) * slots * width;
    for (uint32_t column = lane; column < width; column += blockDim.x) {
        float normalized = static_cast<float>(values[column]) * inverse *
                           static_cast<float>(weight[column]);
        float shift = static_cast<float>(
            modulation[base + static_cast<size_t>(shift_slot) * width + column]);
        float scale = static_cast<float>(
            modulation[base + static_cast<size_t>(scale_slot) * width + column]);
        output[static_cast<size_t>(row) * width + column] =
            hip_bfloat16(normalized * (1.0f + scale) + shift);
    }
}

__global__ static void h3_hip_gate_bf16_kernel(
        const hip_bfloat16 *residual, const hip_bfloat16 *branch,
        const hip_bfloat16 *modulation, const uint32_t *row_map,
        hip_bfloat16 *output, uint32_t elements, uint32_t width,
        uint32_t slots, uint32_t gate_slot) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t row = index / width;
    uint32_t column = index % width;
    size_t base = static_cast<size_t>(row_map[row]) * slots * width;
    float gate = static_cast<float>(
        modulation[base + static_cast<size_t>(gate_slot) * width + column]);
    output[index] = hip_bfloat16(static_cast<float>(residual[index]) +
                                 static_cast<float>(branch[index]) * gate);
}

__global__ static void h3_hip_gate_adaln_bf16_kernel(
        const hip_bfloat16 *residual, const hip_bfloat16 *branch,
        const hip_bfloat16 *norm_weight,
        const hip_bfloat16 *gate_modulation,
        const hip_bfloat16 *norm_modulation, const uint32_t *row_map,
        hip_bfloat16 *gated_residual, hip_bfloat16 *output, uint32_t rows,
        uint32_t width, uint32_t slots, uint32_t gate_slot,
        uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
    uint32_t row = blockIdx.x;
    uint32_t lane = threadIdx.x;
    if (row >= rows) return;
    extern __shared__ float reduction[];
    size_t modulation_base = static_cast<size_t>(row_map[row]) * slots * width;
    size_t row_base = static_cast<size_t>(row) * width;
    float sum = 0.0f;
    for (uint32_t column = lane; column < width; column += blockDim.x) {
        size_t index = row_base + column;
        float gate = static_cast<float>(gate_modulation[
            modulation_base + static_cast<size_t>(gate_slot) * width + column]);
        hip_bfloat16 rounded(static_cast<float>(residual[index]) +
                             static_cast<float>(branch[index]) * gate);
        gated_residual[index] = rounded;
        float value = static_cast<float>(rounded);
        sum = fmaf(value, value, sum);
    }
    reduction[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reduction[0] / static_cast<float>(width) + epsilon);
    for (uint32_t column = lane; column < width; column += blockDim.x) {
        size_t index = row_base + column;
        float normalized = static_cast<float>(gated_residual[index]) * inverse *
                           static_cast<float>(norm_weight[column]);
        float shift = static_cast<float>(norm_modulation[
            modulation_base + static_cast<size_t>(shift_slot) * width + column]);
        float scale = static_cast<float>(norm_modulation[
            modulation_base + static_cast<size_t>(scale_slot) * width + column]);
        output[index] = hip_bfloat16(normalized * (1.0f + scale) + shift);
    }
}

__global__ static void h3_hip_qkv_rope_bf16_kernel(
        const hip_bfloat16 *qkv, const hip_bfloat16 *q_weight,
        const hip_bfloat16 *k_weight, const hip_bfloat16 *rope_cos,
        const hip_bfloat16 *rope_sin, hip_bfloat16 *query,
        hip_bfloat16 *key, hip_bfloat16 *value, uint32_t sequence,
        uint32_t heads, uint32_t head_dim, uint32_t rope_half,
        int grouped, float epsilon) {
    uint32_t head = blockIdx.x;
    uint32_t row = blockIdx.y;
    uint32_t lane = threadIdx.x;
    if (head >= heads || row >= sequence) return;
    extern __shared__ float reduction[];
    float *q_reduction = reduction;
    float *k_reduction = reduction + blockDim.x;
    size_t inner = static_cast<size_t>(heads) * head_dim;
    size_t row_base = static_cast<size_t>(row) * inner * 3;
    size_t q_base = row_base + static_cast<size_t>(head) * head_dim;
    size_t k_base = q_base + inner;
    size_t v_base = q_base + inner * 2;
    if (grouped) {
        q_base = row_base + static_cast<size_t>(head) * head_dim * 3;
        k_base = q_base + head_dim;
        v_base = k_base + head_dim;
    }
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint32_t dimension = lane; dimension < head_dim;
         dimension += blockDim.x) {
        float q = static_cast<float>(qkv[q_base + dimension]);
        float k = static_cast<float>(qkv[k_base + dimension]);
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    q_reduction[lane] = q_sum;
    k_reduction[lane] = k_sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) {
            q_reduction[lane] += q_reduction[lane + stride];
            k_reduction[lane] += k_reduction[lane + stride];
        }
        __syncthreads();
    }
    float q_inverse = rsqrtf(q_reduction[0] /
                             static_cast<float>(head_dim) + epsilon);
    float k_inverse = rsqrtf(k_reduction[0] /
                             static_cast<float>(head_dim) + epsilon);
    for (uint32_t dimension = lane; dimension < head_dim;
         dimension += blockDim.x) {
        float q0 = static_cast<float>(qkv[q_base + dimension]) * q_inverse *
                   static_cast<float>(q_weight[dimension]);
        float k0 = static_cast<float>(qkv[k_base + dimension]) * k_inverse *
                   static_cast<float>(k_weight[dimension]);
        if (dimension < rope_half * 2) {
            uint32_t rope_dimension = dimension % rope_half;
            uint32_t pair = dimension < rope_half ? dimension + rope_half :
                                                    dimension - rope_half;
            float q1 = static_cast<float>(qkv[q_base + pair]) * q_inverse *
                       static_cast<float>(q_weight[pair]);
            float k1 = static_cast<float>(qkv[k_base + pair]) * k_inverse *
                       static_cast<float>(k_weight[pair]);
            float cosine = static_cast<float>(
                rope_cos[static_cast<size_t>(row) * rope_half + rope_dimension]);
            float sine = static_cast<float>(
                rope_sin[static_cast<size_t>(row) * rope_half + rope_dimension]);
            float sign = dimension < rope_half ? -1.0f : 1.0f;
            q0 = fmaf(sign * q1, sine, q0 * cosine);
            k0 = fmaf(sign * k1, sine, k0 * cosine);
        }
        size_t output_index =
            (static_cast<size_t>(row) * heads + head) * head_dim + dimension;
        query[output_index] = hip_bfloat16(q0);
        key[output_index] = hip_bfloat16(k0);
        value[output_index] = qkv[v_base + dimension];
    }
}

__global__ static void h3_hip_video_qkv_rope_f32_kernel(
        const float *qkv, const float *rope_cos, const float *rope_sin,
        float *query, float *key, float *value, uint32_t sequence,
        uint32_t heads, uint32_t head_dim, uint32_t rope_half,
        float epsilon) {
    uint32_t head = blockIdx.x;
    uint32_t row = blockIdx.y;
    uint32_t lane = threadIdx.x;
    if (head >= heads || row >= sequence) return;
    extern __shared__ float reduction[];
    float *q_reduction = reduction;
    float *k_reduction = reduction + blockDim.x;
    size_t input_base =
        (static_cast<size_t>(row) * heads + head) * head_dim * 3;
    size_t q_base = input_base;
    size_t k_base = input_base + head_dim;
    size_t v_base = input_base + head_dim * 2;
    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint32_t dimension = lane; dimension < head_dim;
         dimension += blockDim.x) {
        float q = qkv[q_base + dimension];
        float k = qkv[k_base + dimension];
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    q_reduction[lane] = q_sum;
    k_reduction[lane] = k_sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) {
            q_reduction[lane] += q_reduction[lane + stride];
            k_reduction[lane] += k_reduction[lane + stride];
        }
        __syncthreads();
    }
    float q_inverse = rsqrtf(q_reduction[0] /
                             static_cast<float>(head_dim) + epsilon);
    float k_inverse = rsqrtf(k_reduction[0] /
                             static_cast<float>(head_dim) + epsilon);
    for (uint32_t dimension = lane; dimension < head_dim;
         dimension += blockDim.x) {
        float q0 = qkv[q_base + dimension] * q_inverse;
        float k0 = qkv[k_base + dimension] * k_inverse;
        if (dimension < rope_half * 2) {
            uint32_t rope_dimension = dimension % rope_half;
            uint32_t pair = dimension < rope_half ?
                            dimension + rope_half : dimension - rope_half;
            float q1 = qkv[q_base + pair] * q_inverse;
            float k1 = qkv[k_base + pair] * k_inverse;
            float cosine = rope_cos[
                static_cast<size_t>(row) * rope_half + rope_dimension];
            float sine = rope_sin[
                static_cast<size_t>(row) * rope_half + rope_dimension];
            float sign = dimension < rope_half ? -1.0f : 1.0f;
            q0 = fmaf(sign * q1, sine, q0 * cosine);
            k0 = fmaf(sign * k1, sine, k0 * cosine);
        }
        size_t output_index =
            (static_cast<size_t>(row) * heads + head) * head_dim + dimension;
        query[output_index] = q0;
        key[output_index] = k0;
        value[output_index] = qkv[v_base + dimension];
    }
}

__global__ static void h3_hip_vdn_qk_rope_bf16_kernel(
        const hip_bfloat16 *query_raw, const hip_bfloat16 *key_raw,
        const hip_bfloat16 *q_weight, const hip_bfloat16 *k_weight,
        const hip_bfloat16 *rope_cos, const hip_bfloat16 *rope_sin,
        hip_bfloat16 *query, hip_bfloat16 *key, uint32_t sequence,
        uint32_t heads, uint32_t head_dim, uint32_t rope_half,
        float epsilon) {
    uint32_t head = blockIdx.x;
    uint32_t row = blockIdx.y;
    uint32_t lane = threadIdx.x;
    if (head >= heads || row >= sequence) return;
    extern __shared__ float reduction[];
    float *q_reduction = reduction;
    float *k_reduction = reduction + blockDim.x;
    size_t inner = static_cast<size_t>(heads) * head_dim;
    size_t base = static_cast<size_t>(row) * inner +
                  static_cast<size_t>(head) * head_dim;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint32_t dimension = lane; dimension < head_dim;
         dimension += blockDim.x) {
        float q = static_cast<float>(query_raw[base + dimension]);
        float k = static_cast<float>(key_raw[base + dimension]);
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    q_reduction[lane] = q_sum;
    k_reduction[lane] = k_sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) {
            q_reduction[lane] += q_reduction[lane + stride];
            k_reduction[lane] += k_reduction[lane + stride];
        }
        __syncthreads();
    }
    float q_inverse = rsqrtf(q_reduction[0] /
                             static_cast<float>(head_dim) + epsilon);
    float k_inverse = rsqrtf(k_reduction[0] /
                             static_cast<float>(head_dim) + epsilon);
    for (uint32_t dimension = lane; dimension < head_dim;
         dimension += blockDim.x) {
        float q0 = static_cast<float>(query_raw[base + dimension]) *
                   q_inverse * static_cast<float>(q_weight[dimension]);
        float k0 = static_cast<float>(key_raw[base + dimension]) *
                   k_inverse * static_cast<float>(k_weight[dimension]);
        if (dimension < rope_half * 2) {
            uint32_t rope_dimension = dimension % rope_half;
            uint32_t pair = dimension < rope_half ? dimension + rope_half :
                                                    dimension - rope_half;
            float q1 = static_cast<float>(query_raw[base + pair]) *
                       q_inverse * static_cast<float>(q_weight[pair]);
            float k1 = static_cast<float>(key_raw[base + pair]) *
                       k_inverse * static_cast<float>(k_weight[pair]);
            float cosine = static_cast<float>(rope_cos[
                static_cast<size_t>(row) * rope_half + rope_dimension]);
            float sine = static_cast<float>(rope_sin[
                static_cast<size_t>(row) * rope_half + rope_dimension]);
            float sign = dimension < rope_half ? -1.0f : 1.0f;
            q0 = fmaf(sign * q1, sine, q0 * cosine);
            k0 = fmaf(sign * k1, sine, k0 * cosine);
        }
        query[base + dimension] = hip_bfloat16(q0);
        key[base + dimension] = hip_bfloat16(k0);
    }
}

__device__ static int h3_hip_vdn_key_allowed(
        uint32_t query_row, uint32_t key_row, uint32_t video_start,
        uint32_t video_end, uint32_t frames, uint32_t tokens_per_frame,
        uint32_t radius, uint32_t chunk, int anchor_both) {
    if (query_row < video_start || query_row >= video_end ||
        key_row < video_start || key_row >= video_end) return 1;
    uint32_t query_frame = (query_row - video_start) / tokens_per_frame;
    uint32_t key_frame = (key_row - video_start) / tokens_per_frame;
    if (anchor_both &&
        (query_frame == 0 || query_frame + 1 == frames ||
         key_frame == 0 || key_frame + 1 == frames)) return 1;
    uint32_t lower;
    uint32_t upper;
    if (chunk) {
        uint32_t query_chunk = query_frame / chunk;
        uint32_t lower_chunk = query_chunk > radius ?
            query_chunk - radius : 0;
        uint64_t upper_wide =
            ((uint64_t)query_chunk + radius + 1) * chunk;
        lower = lower_chunk * chunk;
        upper = upper_wide >= frames ? frames - 1 :
                static_cast<uint32_t>(upper_wide - 1);
    } else {
        lower = query_frame > radius ? query_frame - radius : 0;
        uint64_t upper_wide = (uint64_t)query_frame + radius;
        upper = upper_wide >= frames ? frames - 1 :
                static_cast<uint32_t>(upper_wide);
    }
    return key_frame >= lower && key_frame <= upper;
}

template <uint32_t group_rows>
__global__ static void h3_hip_vdn_sage_quant_bf16_i8_kernel(
        const hip_bfloat16 *input, int8_t *output, float *scales,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        uint32_t groups) {
    uint32_t group = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    if (group >= groups || head >= heads) return;
    uint32_t row_begin = group * group_rows;
    uint32_t group_elements = group_rows * head_dim;
    float local_amax = 0.0f;
    for (uint32_t local = lane; local < group_elements;
         local += blockDim.x) {
        uint32_t row = row_begin + local / head_dim;
        uint32_t dimension = local % head_dim;
        if (row < sequence) {
            size_t index = (static_cast<size_t>(row) * heads + head) *
                           head_dim + dimension;
            local_amax = fmaxf(local_amax,
                               fabsf(static_cast<float>(input[index])));
        }
    }
    extern __shared__ float reduction[];
    reduction[lane] = local_amax;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride)
            reduction[lane] = fmaxf(reduction[lane],
                                    reduction[lane + stride]);
        __syncthreads();
    }
    float amax = reduction[0];
    float quant_scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    float inverse_scale = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (lane == 0)
        scales[static_cast<size_t>(head) * groups + group] = quant_scale;
    for (uint32_t local = lane; local < group_elements;
         local += blockDim.x) {
        uint32_t row = row_begin + local / head_dim;
        uint32_t dimension = local % head_dim;
        if (row < sequence) {
            size_t index = (static_cast<size_t>(row) * heads + head) *
                           head_dim + dimension;
            int32_t rounded = __float2int_rn(
                static_cast<float>(input[index]) * inverse_scale);
            if (rounded > 127) rounded = 127;
            if (rounded < -127) rounded = -127;
            output[index] = static_cast<int8_t>(rounded);
        }
    }
}

__global__ static void h3_hip_vdn_sage_wmma_qk_tile_i8_kernel(
        const int8_t *query, const int8_t *key,
        const float *query_scales, const float *key_scales, float *scores,
        uint32_t sequence, uint32_t heads, uint32_t query_start,
        uint32_t key_start, uint32_t head, uint32_t query_groups,
        uint32_t key_groups, float scale) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr uint32_t tile = 16;
    constexpr uint32_t head_dim = 128;
    constexpr uint32_t depth_tile = 16;
    using query_fragment = rocwmma::fragment<
        rocwmma::matrix_a, tile, tile, depth_tile, int8_t,
        rocwmma::row_major>;
    using key_fragment = rocwmma::fragment<
        rocwmma::matrix_b, tile, tile, depth_tile, int8_t,
        rocwmma::col_major>;
    using score_fragment = rocwmma::fragment<
        rocwmma::accumulator, tile, tile, depth_tile, int32_t>;
    uint32_t lane = threadIdx.x;
    size_t inner = static_cast<size_t>(heads) * head_dim;
    size_t query_base = static_cast<size_t>(query_start) * inner +
                        static_cast<size_t>(head) * head_dim;
    size_t key_base = static_cast<size_t>(key_start) * inner +
                      static_cast<size_t>(head) * head_dim;
    score_fragment accumulator;
    rocwmma::fill_fragment(accumulator, 0);
#pragma unroll
    for (uint32_t depth = 0; depth < head_dim; depth += depth_tile) {
        query_fragment query_values;
        key_fragment key_values;
        rocwmma::load_matrix_sync(
            query_values, query + query_base + depth,
            static_cast<uint32_t>(inner));
        rocwmma::load_matrix_sync(
            key_values, key + key_base + depth,
            static_cast<uint32_t>(inner));
        rocwmma::mma_sync(
            accumulator, query_values, key_values, accumulator);
    }
    __shared__ int32_t integer_scores[tile * tile];
    rocwmma::store_matrix_sync(
        integer_scores, accumulator, tile, rocwmma::mem_row_major);
    __syncwarp();
    for (uint32_t index = lane; index < tile * tile; index += 32) {
        uint32_t query_row = query_start + index / tile;
        uint32_t key_row = key_start + index % tile;
        float query_scale = query_scales[
            static_cast<size_t>(head) * query_groups + query_row / 32];
        float key_scale = key_scales[
            static_cast<size_t>(head) * key_groups + key_row / 64];
        scores[index] = static_cast<float>(integer_scores[index]) *
                        query_scale * key_scale * scale;
    }
    (void)sequence;
#else
    /* The host entry point rejects non-gfx12 devices. Keeping a device-side
     * fallback lets a single HIP source compile into multi-architecture
     * bundles without asking rocWMMA for an unavailable I8 instruction. */
    (void)query; (void)key; (void)query_scales; (void)key_scales;
    (void)scores; (void)sequence; (void)heads; (void)query_start;
    (void)key_start; (void)head; (void)query_groups; (void)key_groups;
    (void)scale;
#endif
}

#if defined(__gfx1200__) || defined(__gfx1201__)
__device__ __forceinline__ static void h3_hip_vdn_sage_qk_wmma_16x16(
        const int8_t *query, uint32_t query_stride,
        const int8_t *key, uint32_t key_stride, int32_t *scores) {
    constexpr uint32_t tile = 16;
    constexpr uint32_t depth_tile = 16;
    using query_fragment = rocwmma::fragment<
        rocwmma::matrix_a, tile, tile, depth_tile, int8_t,
        rocwmma::row_major>;
    using key_fragment = rocwmma::fragment<
        rocwmma::matrix_b, tile, tile, depth_tile, int8_t,
        rocwmma::col_major>;
    using score_fragment = rocwmma::fragment<
        rocwmma::accumulator, tile, tile, depth_tile, int32_t>;
    score_fragment accumulator;
    rocwmma::fill_fragment(accumulator, 0);
#pragma unroll
    for (uint32_t depth = 0; depth < 128; depth += depth_tile) {
        query_fragment query_values;
        key_fragment key_values;
        rocwmma::load_matrix_sync(
            query_values, query + depth, query_stride);
        rocwmma::load_matrix_sync(key_values, key + depth, key_stride);
        rocwmma::mma_sync(
            accumulator, query_values, key_values, accumulator);
    }
    rocwmma::store_matrix_sync(
        scores, accumulator, tile, rocwmma::mem_row_major);
    __syncwarp();
}
#endif

/* Experimental two-pass FlashAttention-style gfx12 kernel. One 256-thread
 * block owns 16 queries for one head. Wave 0 computes I8 QK tiles; all eight
 * waves compute one 16-column BF16 P*V tile each. The first pass obtains the
 * exact softmax normalizer without an SxS score buffer, while the second pass
 * recomputes QK and immediately consumes BF16 probabilities. */
[[maybe_unused]] __global__ static void
h3_hip_vdn_sage_attention_i8_bf16_kernel(
        const int8_t *query, const int8_t *key,
        const float *query_scales, const float *key_scales,
        const hip_bfloat16 *value, hip_bfloat16 *output,
        uint32_t sequence, uint32_t heads, uint32_t video_start,
        uint32_t video_end, uint32_t frames, uint32_t tokens_per_frame,
        uint32_t radius, uint32_t chunk, int anchor_both, float scale,
        uint32_t query_groups, uint32_t key_groups) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    constexpr uint32_t tile = 16;
    constexpr uint32_t head_dim = 128;
    constexpr uint32_t waves = head_dim / tile;
    uint32_t thread = threadIdx.x;
    uint32_t lane = thread & 31;
    uint32_t wave = thread >> 5;
    uint32_t query_start = blockIdx.x * tile;
    uint32_t head = blockIdx.y;
    if (head >= heads || query_start >= sequence) return;
    size_t inner_wide = static_cast<size_t>(heads) * head_dim;
    uint32_t inner = heads * head_dim;

    __shared__ int32_t integer_scores[tile * tile];
    __shared__ hip_bfloat16 probabilities[tile * tile];
    __shared__ int8_t query_tail[tile * head_dim];
    __shared__ int8_t key_tail[tile * head_dim];
    __shared__ hip_bfloat16 value_tail[tile * head_dim];
    __shared__ float output_tiles[waves * tile * tile];
    __shared__ float row_maximum[tile];
    __shared__ float row_sum[tile];

    int query_is_tail = sequence - query_start < tile;
    const int8_t *query_tile = query +
        static_cast<size_t>(query_start) * inner_wide +
        static_cast<size_t>(head) * head_dim;
    uint32_t query_stride = inner;
    if (query_is_tail && wave == 0) {
        for (uint32_t index = lane; index < tile * head_dim; index += 32) {
            uint32_t row = index / head_dim;
            uint32_t dimension = index % head_dim;
            query_tail[index] = query_start + row < sequence ?
                query[(static_cast<size_t>(query_start + row) * heads +
                       head) * head_dim + dimension] : 0;
        }
        __syncwarp();
    }
    if (query_is_tail) {
        query_tile = query_tail;
        query_stride = head_dim;
    }

    float running_maximum = -INFINITY;
    float running_sum = 0.0f;
    for (uint32_t key_start = 0; key_start < sequence; key_start += tile) {
        int key_is_tail = sequence - key_start < tile;
        const int8_t *key_tile = key +
            static_cast<size_t>(key_start) * inner_wide +
            static_cast<size_t>(head) * head_dim;
        uint32_t key_stride = inner;
        if (key_is_tail && wave == 0) {
            for (uint32_t index = lane; index < tile * head_dim; index += 32) {
                uint32_t row = index / head_dim;
                uint32_t dimension = index % head_dim;
                key_tail[index] = key_start + row < sequence ?
                    key[(static_cast<size_t>(key_start + row) * heads +
                         head) * head_dim + dimension] : 0;
            }
            __syncwarp();
            key_tile = key_tail;
            key_stride = head_dim;
        }
        if (wave == 0) {
            h3_hip_vdn_sage_qk_wmma_16x16(
                query_tile, query_stride, key_tile, key_stride,
                integer_scores);
            if (lane < tile) {
                uint32_t query_row = query_start + lane;
                if (query_row < sequence) {
                    float query_scale = query_scales[
                        static_cast<size_t>(head) * query_groups +
                        query_row / 32];
#pragma unroll
                    for (uint32_t column = 0; column < tile; column++) {
                        uint32_t key_row = key_start + column;
                        if (key_row >= sequence ||
                            !h3_hip_vdn_key_allowed(
                                query_row, key_row, video_start, video_end,
                                frames, tokens_per_frame, radius, chunk,
                                anchor_both)) continue;
                        float key_scale = key_scales[
                            static_cast<size_t>(head) * key_groups +
                            key_row / 64];
                        float score =
                            static_cast<float>(integer_scores[lane * tile +
                                                              column]) *
                            query_scale * key_scale * scale;
                        float next_maximum = fmaxf(running_maximum, score);
                        float old_scale = running_sum == 0.0f ? 0.0f :
                            expf(running_maximum - next_maximum);
                        float new_scale = expf(score - next_maximum);
                        running_sum = running_sum * old_scale + new_scale;
                        running_maximum = next_maximum;
                    }
                }
            }
        }
    }
    if (wave == 0 && lane < tile) {
        row_maximum[lane] = running_maximum;
        row_sum[lane] = running_sum;
    }
    __syncthreads();

    using probability_fragment = rocwmma::fragment<
        rocwmma::matrix_a, tile, tile, tile, rocwmma::bfloat16_t,
        rocwmma::row_major>;
    using value_fragment = rocwmma::fragment<
        rocwmma::matrix_b, tile, tile, tile, rocwmma::bfloat16_t,
        rocwmma::row_major>;
    using output_fragment = rocwmma::fragment<
        rocwmma::accumulator, tile, tile, tile, float>;
    output_fragment accumulator;
    rocwmma::fill_fragment(accumulator, 0.0f);

    for (uint32_t key_start = 0; key_start < sequence; key_start += tile) {
        int key_is_tail = sequence - key_start < tile;
        const int8_t *key_tile = key +
            static_cast<size_t>(key_start) * inner_wide +
            static_cast<size_t>(head) * head_dim;
        const hip_bfloat16 *value_tile = value +
            static_cast<size_t>(key_start) * inner_wide +
            static_cast<size_t>(head) * head_dim;
        uint32_t key_stride = inner;
        uint32_t value_stride = inner;
        if (key_is_tail) {
            for (uint32_t index = thread; index < tile * head_dim;
                 index += blockDim.x) {
                uint32_t row = index / head_dim;
                uint32_t dimension = index % head_dim;
                if (key_start + row < sequence) {
                    size_t source =
                        (static_cast<size_t>(key_start + row) * heads +
                         head) * head_dim + dimension;
                    key_tail[index] = key[source];
                    value_tail[index] = value[source];
                } else {
                    key_tail[index] = 0;
                    value_tail[index] = hip_bfloat16(0.0f);
                }
            }
            __syncthreads();
            key_tile = key_tail;
            value_tile = value_tail;
            key_stride = head_dim;
            value_stride = head_dim;
        }
        if (wave == 0)
            h3_hip_vdn_sage_qk_wmma_16x16(
                query_tile, query_stride, key_tile, key_stride,
                integer_scores);
        __syncthreads();
        uint32_t query_local = thread / tile;
        uint32_t key_local = thread % tile;
        uint32_t query_row = query_start + query_local;
        uint32_t key_row = key_start + key_local;
        float probability = 0.0f;
        if (query_row < sequence && key_row < sequence &&
            h3_hip_vdn_key_allowed(
                query_row, key_row, video_start, video_end, frames,
                tokens_per_frame, radius, chunk, anchor_both)) {
            float query_scale = query_scales[
                static_cast<size_t>(head) * query_groups + query_row / 32];
            float key_scale = key_scales[
                static_cast<size_t>(head) * key_groups + key_row / 64];
            float score = static_cast<float>(integer_scores[thread]) *
                          query_scale * key_scale * scale;
            probability = expf(score - row_maximum[query_local]) /
                          row_sum[query_local];
        }
        probabilities[thread] = hip_bfloat16(probability);
        __syncthreads();
        probability_fragment probability_values;
        value_fragment value_values;
        rocwmma::load_matrix_sync(
            probability_values, probabilities, tile);
        rocwmma::load_matrix_sync(
            value_values, value_tile + wave * tile, value_stride);
        rocwmma::mma_sync(
            accumulator, probability_values, value_values, accumulator);
        __syncthreads();
    }
    rocwmma::store_matrix_sync(
        output_tiles + wave * tile * tile, accumulator, tile,
        rocwmma::mem_row_major);
    __syncthreads();
    for (uint32_t index = thread; index < tile * head_dim;
         index += blockDim.x) {
        uint32_t query_local = index / head_dim;
        uint32_t dimension = index % head_dim;
        uint32_t query_row = query_start + query_local;
        if (query_row < sequence) {
            uint32_t output_wave = dimension / tile;
            uint32_t output_column = dimension % tile;
            float result = output_tiles[
                output_wave * tile * tile + query_local * tile +
                output_column];
            output[(static_cast<size_t>(query_row) * heads + head) *
                   head_dim + dimension] = hip_bfloat16(result);
        }
    }
#else
    (void)query; (void)key; (void)query_scales; (void)key_scales;
    (void)value; (void)output; (void)sequence; (void)heads;
    (void)video_start; (void)video_end; (void)frames;
    (void)tokens_per_frame; (void)radius; (void)chunk;
    (void)anchor_both; (void)scale; (void)query_groups; (void)key_groups;
#endif
}

__global__ static void h3_hip_vdn_window_sdpa_bf16_scalar_kernel(
        const hip_bfloat16 *query, const hip_bfloat16 *key,
        const hip_bfloat16 *value, hip_bfloat16 *output,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        uint32_t video_start, uint32_t video_end, uint32_t frames,
        uint32_t tokens_per_frame, uint32_t radius, uint32_t chunk,
        int anchor_both, float scale) {
    uint32_t query_row = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    if (query_row >= sequence || head >= heads) return;
    extern __shared__ float shared[];
    float *dot = shared;
    float *coefficients = shared + blockDim.x;
    size_t inner = static_cast<size_t>(heads) * head_dim;
    size_t query_base = static_cast<size_t>(query_row) * inner +
                        static_cast<size_t>(head) * head_dim;
    float accumulator = 0.0f;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (uint32_t key_row = 0; key_row < sequence; key_row++) {
        if (!h3_hip_vdn_key_allowed(query_row, key_row, video_start,
                video_end, frames, tokens_per_frame, radius, chunk,
                anchor_both)) continue;
        size_t key_base = static_cast<size_t>(key_row) * inner +
                          static_cast<size_t>(head) * head_dim;
        float partial = 0.0f;
        for (uint32_t dimension = lane; dimension < head_dim;
             dimension += blockDim.x)
            partial = fmaf(static_cast<float>(query[query_base + dimension]),
                           static_cast<float>(key[key_base + dimension]),
                           partial);
        dot[lane] = partial;
        __syncthreads();
        for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
            if (lane < stride) dot[lane] += dot[lane + stride];
            __syncthreads();
        }
        if (lane == 0) {
            float score = dot[0] * scale;
            float next_max = fmaxf(running_max, score);
            float old_scale = running_sum == 0.0f ? 0.0f :
                              expf(running_max - next_max);
            float new_scale = expf(score - next_max);
            running_sum = running_sum * old_scale + new_scale;
            running_max = next_max;
            coefficients[0] = old_scale;
            coefficients[1] = new_scale;
            coefficients[2] = running_sum;
        }
        __syncthreads();
        if (lane < head_dim)
            accumulator = accumulator * coefficients[0] +
                coefficients[1] * static_cast<float>(value[key_base + lane]);
        __syncthreads();
    }
    if (lane < head_dim)
        output[query_base + lane] = hip_bfloat16(accumulator / coefficients[2]);
}

/* gfx12 executes wave32. One wave handles a query/head pair and keeps four
 * output dimensions per lane for the production D=128 shape. This removes the
 * two block-wide barriers at every reduction step in the scalar oracle. */
template <bool fixed_d128, bool cache_query = false,
          bool jump_mask_gaps = false, bool distributed_softmax = false>
__global__ static void h3_hip_vdn_window_sdpa_bf16_wave32_kernel(
        const hip_bfloat16 *query, const hip_bfloat16 *key,
        const hip_bfloat16 *value, hip_bfloat16 *output,
        uint32_t sequence, uint32_t heads, uint32_t head_dim,
        uint32_t video_start, uint32_t video_end, uint32_t frames,
        uint32_t tokens_per_frame, uint32_t radius, uint32_t chunk,
        int anchor_both, float scale) {
    uint32_t query_row = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    if (query_row >= sequence || head >= heads || lane >= 32) return;
    size_t inner = static_cast<size_t>(heads) * head_dim;
    size_t query_base = static_cast<size_t>(query_row) * inner +
                        static_cast<size_t>(head) * head_dim;
    constexpr uint32_t accumulator_count = fixed_d128 ? 4 : 8;
    float accumulators[accumulator_count];
#pragma unroll
    for (uint32_t slot = 0; slot < accumulator_count; slot++)
        accumulators[slot] = 0.0f;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float cached_query[4];
    if constexpr (fixed_d128 && cache_query) {
#pragma unroll
        for (uint32_t slot = 0; slot < 4; slot++)
            cached_query[slot] = static_cast<float>(
                query[query_base + lane + slot * 32]);
    }
    int restrict_video_keys =
        query_row >= video_start && query_row < video_end;
    uint32_t window_start = video_start;
    uint32_t window_end = video_end;
    if (restrict_video_keys) {
        uint32_t query_frame =
            (query_row - video_start) / tokens_per_frame;
        if (anchor_both &&
            (query_frame == 0 || query_frame + 1 == frames)) {
            restrict_video_keys = 0;
        } else {
            uint32_t lower;
            uint32_t upper;
            if (chunk) {
                uint32_t query_chunk = query_frame / chunk;
                uint32_t lower_chunk = query_chunk > radius ?
                    query_chunk - radius : 0;
                uint64_t upper_wide =
                    (static_cast<uint64_t>(query_chunk) + radius + 1) *
                    chunk;
                lower = lower_chunk * chunk;
                upper = upper_wide >= frames ? frames - 1 :
                    static_cast<uint32_t>(upper_wide - 1);
            } else {
                lower = query_frame > radius ? query_frame - radius : 0;
                uint64_t upper_wide =
                    static_cast<uint64_t>(query_frame) + radius;
                upper = upper_wide >= frames ? frames - 1 :
                    static_cast<uint32_t>(upper_wide);
            }
            window_start = video_start + lower * tokens_per_frame;
            window_end = video_start + (upper + 1) * tokens_per_frame;
        }
    }
    for (uint32_t key_row = 0; key_row < sequence; key_row++) {
        if constexpr (jump_mask_gaps) {
            /* With both endpoint frames anchored, allowed video keys are the
             * ordered union of first anchor, window, and last anchor. Jump
             * over the two possible gaps without changing reduction order. */
            if (restrict_video_keys) {
                uint32_t interior_start = video_start + tokens_per_frame;
                uint32_t interior_end = video_end - tokens_per_frame;
                if (key_row == interior_start &&
                    window_start > interior_start)
                    key_row = window_start;
                else if (key_row == window_end && window_end < interior_end)
                    key_row = interior_end;
            }
        } else {
            if (restrict_video_keys && key_row >= video_start &&
                key_row < video_end &&
                (key_row < window_start || key_row >= window_end) &&
                (!anchor_both ||
                 (key_row >= video_start + tokens_per_frame &&
                  key_row < video_end - tokens_per_frame))) continue;
        }
        size_t key_base = static_cast<size_t>(key_row) * inner +
                          static_cast<size_t>(head) * head_dim;
        float partial = 0.0f;
        if constexpr (fixed_d128) {
            float product0 = fmaf(
                cache_query ? cached_query[0] :
                    static_cast<float>(query[query_base + lane]),
                static_cast<float>(key[key_base + lane]), 0.0f);
            float product32 = fmaf(
                cache_query ? cached_query[1] :
                    static_cast<float>(query[query_base + lane + 32]),
                static_cast<float>(key[key_base + lane + 32]), 0.0f);
            float product64 = fmaf(
                cache_query ? cached_query[2] :
                    static_cast<float>(query[query_base + lane + 64]),
                static_cast<float>(key[key_base + lane + 64]), 0.0f);
            float product96 = fmaf(
                cache_query ? cached_query[3] :
                    static_cast<float>(query[query_base + lane + 96]),
                static_cast<float>(key[key_base + lane + 96]), 0.0f);
            /* Match the scalar 256-thread reduction tree exactly: its first
             * non-zero steps pair d+64, then d+32/d+96. */
            partial = (product0 + product64) + (product32 + product96);
        } else if (head_dim == 128) {
            float product0 = fmaf(
                static_cast<float>(query[query_base + lane]),
                static_cast<float>(key[key_base + lane]), 0.0f);
            float product32 = fmaf(
                static_cast<float>(query[query_base + lane + 32]),
                static_cast<float>(key[key_base + lane + 32]), 0.0f);
            float product64 = fmaf(
                static_cast<float>(query[query_base + lane + 64]),
                static_cast<float>(key[key_base + lane + 64]), 0.0f);
            float product96 = fmaf(
                static_cast<float>(query[query_base + lane + 96]),
                static_cast<float>(key[key_base + lane + 96]), 0.0f);
            partial = (product0 + product64) + (product32 + product96);
        } else {
            for (uint32_t dimension = lane; dimension < head_dim;
                 dimension += 32)
                partial = fmaf(
                    static_cast<float>(query[query_base + dimension]),
                    static_cast<float>(key[key_base + dimension]), partial);
        }
#pragma unroll
        for (uint32_t offset = 16; offset; offset >>= 1)
            partial += __shfl_down(partial, offset, 32);
        float old_scale = 0.0f;
        float new_scale = 0.0f;
        if constexpr (distributed_softmax) {
            partial = __shfl(partial, 0, 32);
            float score = partial * scale;
            float next_max = fmaxf(running_max, score);
            old_scale = running_sum == 0.0f ? 0.0f :
                        expf(running_max - next_max);
            new_scale = expf(score - next_max);
            running_sum = running_sum * old_scale + new_scale;
            running_max = next_max;
        } else {
            if (lane == 0) {
                float score = partial * scale;
                float next_max = fmaxf(running_max, score);
                old_scale = running_sum == 0.0f ? 0.0f :
                            expf(running_max - next_max);
                new_scale = expf(score - next_max);
                running_sum = running_sum * old_scale + new_scale;
                running_max = next_max;
            }
            old_scale = __shfl(old_scale, 0, 32);
            new_scale = __shfl(new_scale, 0, 32);
        }
        if constexpr (fixed_d128) {
#pragma unroll
            for (uint32_t slot = 0; slot < 4; slot++) {
                uint32_t dimension = lane + slot * 32;
                accumulators[slot] = accumulators[slot] * old_scale +
                    new_scale *
                    static_cast<float>(value[key_base + dimension]);
            }
        } else {
            uint32_t slot = 0;
            for (uint32_t dimension = lane; dimension < head_dim;
                 dimension += 32, slot++)
                accumulators[slot] = accumulators[slot] * old_scale +
                    new_scale *
                    static_cast<float>(value[key_base + dimension]);
        }
    }
    float denominator = __shfl(running_sum, 0, 32);
    if constexpr (fixed_d128) {
#pragma unroll
        for (uint32_t slot = 0; slot < 4; slot++) {
            uint32_t dimension = lane + slot * 32;
            output[query_base + dimension] =
                hip_bfloat16(accumulators[slot] / denominator);
        }
    } else {
        uint32_t slot = 0;
        for (uint32_t dimension = lane; dimension < head_dim;
             dimension += 32, slot++)
            output[query_base + dimension] =
                hip_bfloat16(accumulators[slot] / denominator);
    }
}

template <bool fixed_d128, bool cache_query = false,
          bool jump_mask_gaps = false, bool distributed_softmax = false>
static void h3_hip_launch_vdn_window_sdpa_bf16_wave32(
        hipStream_t stream, const hip_bfloat16 *query,
        const hip_bfloat16 *key, const hip_bfloat16 *value,
        hip_bfloat16 *output, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, uint32_t video_start, uint32_t video_end,
        uint32_t frames, uint32_t tokens_per_frame, uint32_t radius,
        uint32_t chunk, int anchor_both, float scale) {
    hipLaunchKernelGGL(
        (h3_hip_vdn_window_sdpa_bf16_wave32_kernel<
            fixed_d128, cache_query, jump_mask_gaps, distributed_softmax>),
        dim3(sequence, heads), dim3(32), 0, stream, query, key, value, output,
        sequence, heads, head_dim, video_start, video_end, frames,
        tokens_per_frame, radius, chunk, anchor_both, scale);
}

__global__ static void h3_hip_vdn_softmax_gate_bf16_kernel(
        const hip_bfloat16 *attended, const hip_bfloat16 *gate_logits,
        hip_bfloat16 *output, uint32_t elements, uint32_t head_dim) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    uint32_t token_head = index / head_dim;
    float logit = static_cast<float>(gate_logits[token_head]);
    float gate = 1.0f / (1.0f + expf(-logit));
    output[index] = hip_bfloat16(
        static_cast<float>(attended[index]) * gate);
}

__device__ static float h3_hip_vdn_sepconv5(
        const hip_bfloat16 *input, const hip_bfloat16 *spatial,
        const hip_bfloat16 *temporal, uint32_t frame, uint32_t y,
        uint32_t x, uint32_t channel, uint32_t frames,
        uint32_t frame_height, uint32_t frame_width, uint32_t channels) {
    float result = 0.0f;
    for (int dt = -2; dt <= 2; dt++) {
        int source_frame = static_cast<int>(frame) + dt;
        if (source_frame < 0 || source_frame >= static_cast<int>(frames))
            continue;
        float spatial_sum = 0.0f;
        for (int dy = -2; dy <= 2; dy++) {
            int source_y = static_cast<int>(y) + dy;
            if (source_y < 0 ||
                source_y >= static_cast<int>(frame_height)) continue;
            for (int dx = -2; dx <= 2; dx++) {
                int source_x = static_cast<int>(x) + dx;
                if (source_x < 0 ||
                    source_x >= static_cast<int>(frame_width)) continue;
                size_t source_row =
                    (static_cast<size_t>(source_frame) * frame_height +
                     static_cast<uint32_t>(source_y)) * frame_width +
                    static_cast<uint32_t>(source_x);
                size_t source_index = source_row * channels + channel;
                size_t weight_index = static_cast<size_t>(channel) * 25 +
                    static_cast<uint32_t>(dy + 2) * 5 +
                    static_cast<uint32_t>(dx + 2);
                spatial_sum = fmaf(static_cast<float>(input[source_index]),
                                   static_cast<float>(spatial[weight_index]),
                                   spatial_sum);
            }
        }
        result = fmaf(spatial_sum,
                      static_cast<float>(temporal[
                          static_cast<size_t>(channel) * 5 +
                          static_cast<uint32_t>(dt + 2)]), result);
    }
    return result;
}

__global__ static void h3_hip_vdn_linear_features_bf16_kernel(
        const hip_bfloat16 *query_raw, const hip_bfloat16 *key_raw,
        const hip_bfloat16 *value_raw, const hip_bfloat16 *k_spatial,
        const hip_bfloat16 *k_temporal, const hip_bfloat16 *v_spatial,
        const hip_bfloat16 *v_temporal, hip_bfloat16 *query,
        hip_bfloat16 *key, hip_bfloat16 *value, uint32_t frames,
        uint32_t frame_height, uint32_t frame_width, uint32_t heads,
        uint32_t head_dim, float epsilon) {
    uint32_t token = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    uint32_t tokens_per_frame = frame_height * frame_width;
    uint32_t token_count = frames * tokens_per_frame;
    if (token >= token_count || head >= heads) return;
    uint32_t frame = token / tokens_per_frame;
    uint32_t spatial_token = token % tokens_per_frame;
    uint32_t y = spatial_token / frame_width;
    uint32_t x = spatial_token % frame_width;
    uint32_t channels = heads * head_dim;
    size_t base = static_cast<size_t>(token) * channels +
                  static_cast<size_t>(head) * head_dim;
    extern __shared__ float reductions[];
    float *q_reduction = reductions;
    float *k_reduction = reductions + blockDim.x;
    float q_feature = 0.0f;
    float k_feature = 0.0f;
    float v_feature = 0.0f;
    if (lane < head_dim) {
        uint32_t channel = head * head_dim + lane;
        float q_raw = static_cast<float>(query_raw[base + lane]);
        float k_raw = h3_hip_vdn_sepconv5(
            key_raw, k_spatial, k_temporal, frame, y, x, channel,
            frames, frame_height, frame_width, channels);
        float v_raw = h3_hip_vdn_sepconv5(
            value_raw, v_spatial, v_temporal, frame, y, x, channel,
            frames, frame_height, frame_width, channels);
        q_feature = q_raw / (1.0f + expf(-q_raw));
        k_feature = k_raw / (1.0f + expf(-k_raw));
        v_feature = v_raw / (1.0f + expf(-v_raw));
    }
    q_reduction[lane] = q_feature * q_feature;
    k_reduction[lane] = k_feature * k_feature;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) {
            q_reduction[lane] += q_reduction[lane + stride];
            k_reduction[lane] += k_reduction[lane + stride];
        }
        __syncthreads();
    }
    if (lane < head_dim) {
        float q_inverse = rsqrtf(q_reduction[0] + epsilon);
        float k_inverse = rsqrtf(k_reduction[0] + epsilon);
        query[base + lane] = hip_bfloat16(q_feature * q_inverse);
        key[base + lane] = hip_bfloat16(k_feature * k_inverse);
        value[base + lane] = hip_bfloat16(v_feature);
    }
}

__global__ static void h3_hip_vdn_text_features_bf16_kernel(
        const hip_bfloat16 *key_raw, const hip_bfloat16 *value_raw,
        hip_bfloat16 *key, hip_bfloat16 *value, uint32_t rows,
        uint32_t heads, uint32_t head_dim, float epsilon) {
    uint32_t row = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    if (row >= rows || head >= heads) return;
    size_t base = (static_cast<size_t>(row) * heads + head) * head_dim;
    extern __shared__ float reduction[];
    float feature = 0.0f;
    if (lane < head_dim) {
        float raw = static_cast<float>(key_raw[base + lane]);
        feature = raw / (1.0f + expf(-raw));
        float v = static_cast<float>(value_raw[base + lane]);
        value[base + lane] = hip_bfloat16(v / (1.0f + expf(-v)));
    }
    reduction[lane] = feature * feature;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    if (lane < head_dim)
        key[base + lane] = hip_bfloat16(
            feature * rsqrtf(reduction[0] + epsilon));
}

__global__ static void h3_hip_vdn_frame_stats_bf16_kernel(
        const hip_bfloat16 *key, const hip_bfloat16 *value,
        const hip_bfloat16 *beta_logits, float *a, float *b,
        uint32_t frames, uint32_t tokens_per_frame, uint32_t heads,
        uint32_t head_dim, uint64_t matrix_elements) {
    uint64_t matrix_index =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t matrices = static_cast<uint64_t>(frames) * heads;
    if (matrix_index >= matrices * matrix_elements) return;
    uint32_t matrix = static_cast<uint32_t>(matrix_index / matrix_elements);
    uint32_t entry = static_cast<uint32_t>(matrix_index % matrix_elements);
    uint32_t frame = matrix / heads;
    uint32_t head = matrix % heads;
    uint32_t row = entry / head_dim;
    uint32_t column = entry % head_dim;
    float a_sum = 0.0f;
    float b_sum = 0.0f;
    for (uint32_t token = 0; token < tokens_per_frame; token++) {
        uint32_t token_row = frame * tokens_per_frame + token;
        size_t base = (static_cast<size_t>(token_row) * heads + head) *
                      head_dim;
        float logit = static_cast<float>(
            beta_logits[static_cast<size_t>(token_row) * heads + head]);
        hip_bfloat16 beta_rounded(1.0f / (1.0f + expf(-logit)));
        float beta = static_cast<float>(beta_rounded);
        float kr = static_cast<float>(key[base + row]);
        float kc = static_cast<float>(key[base + column]);
        float vr = static_cast<float>(value[base + row]);
        float vb = static_cast<float>(hip_bfloat16(vr * beta));
        a_sum = fmaf(kr * beta, kc, a_sum);
        b_sum = fmaf(vb, kc, b_sum);
    }
    a[matrix_index] = a_sum;
    b[matrix_index] = static_cast<float>(hip_bfloat16(b_sum));
}

__global__ static void h3_hip_vdn_add_identity_f32_kernel(
        float *matrices, uint32_t batches, uint32_t dimension) {
    uint32_t batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batches) return;
    size_t stride = static_cast<size_t>(dimension) * dimension;
    float *matrix = matrices + static_cast<size_t>(batch) * stride;
    for (uint32_t index = 0; index < dimension; index++)
        matrix[static_cast<size_t>(index) * dimension + index] += 1.0f;
}

__global__ static void h3_hip_vdn_symmetrize_inverse_f32_kernel(
        float *matrices, uint64_t elements, uint32_t dimension) {
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (index >= elements) return;
    uint32_t entry = static_cast<uint32_t>(index %
        (static_cast<uint64_t>(dimension) * dimension));
    uint32_t row = entry / dimension;
    uint32_t column = entry % dimension;
    if (row > column) {
        uint64_t matrix_base = index - entry;
        matrices[index] = matrices[matrix_base +
            static_cast<uint64_t>(column) * dimension + row];
    }
}

__global__ static void h3_hip_vdn_transition_f32_kernel(
        const float *inverse, const float *alpha, float *transition,
        uint64_t elements, uint32_t dimension) {
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (index >= elements) return;
    uint64_t matrix_elements = static_cast<uint64_t>(dimension) * dimension;
    uint64_t batch = index / matrix_elements;
    uint32_t row = static_cast<uint32_t>(index % matrix_elements) / dimension;
    transition[index] = inverse[index] *
        alpha[batch * dimension + row];
}

__global__ static void h3_hip_vdn_frame_mean_bf16_kernel(
        const hip_bfloat16 *input, size_t input_offset, float *output,
        uint32_t tokens_per_frame, uint32_t width,
        uint64_t elements) {
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (index >= elements) return;
    uint32_t frame = static_cast<uint32_t>(index / width);
    uint32_t column = static_cast<uint32_t>(index % width);
    float sum = 0.0f;
    for (uint32_t token = 0; token < tokens_per_frame; token++) {
        size_t source = input_offset +
            (static_cast<size_t>(frame) * tokens_per_frame + token) * width +
            column;
        sum += static_cast<float>(input[source]);
    }
    output[index] = sum / static_cast<float>(tokens_per_frame);
}

__global__ static void h3_hip_vdn_alpha_f32_kernel(
        const float *delta, const hip_bfloat16 *dt_bias,
        const hip_bfloat16 *a_log, float *output, uint64_t elements,
        uint32_t heads, uint32_t head_dim) {
    uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (index >= elements) return;
    uint32_t channel = static_cast<uint32_t>(index %
        (static_cast<uint64_t>(heads) * head_dim));
    uint32_t head = channel / head_dim;
    float x = delta[index] + static_cast<float>(dt_bias[channel]);
    float softplus = x > 20.0f ? x : log1pf(expf(x));
    float scale = expf(static_cast<float>(a_log[head]));
    output[index] = expf(-scale * softplus);
}

__global__ static void h3_hip_vdn_readout_bf16_kernel(
        const hip_bfloat16 *query, const float *prefix,
        const float *suffix, const float *alpha, const float *text_state,
        float text_state_scale, const hip_bfloat16 *norm_weight,
        const hip_bfloat16 *gate_logits, hip_bfloat16 *output,
        uint32_t frames, uint32_t tokens_per_frame, uint32_t heads,
        uint32_t head_dim, uint32_t radius, uint32_t chunk, float epsilon) {
    uint32_t token = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    uint32_t token_count = frames * tokens_per_frame;
    if (token >= token_count || head >= heads) return;
    uint32_t frame = token / tokens_per_frame;
    uint32_t original_frame = frame + 1;
    int lower;
    int upper;
    if (chunk) {
        int original_chunk = static_cast<int>(original_frame / chunk);
        lower = (original_chunk - static_cast<int>(radius)) *
                static_cast<int>(chunk) - 1;
        upper = (original_chunk + static_cast<int>(radius) + 1) *
                static_cast<int>(chunk) - 2;
    } else {
        lower = static_cast<int>(original_frame) -
                static_cast<int>(radius) - 1;
        upper = static_cast<int>(original_frame) +
                static_cast<int>(radius) - 1;
    }
    int before_index = lower - 1;
    int after_index = upper + 1;
    int has_before = before_index >= 0;
    int has_after = after_index < static_cast<int>(frames);
    int bridge_before = lower > 0 ? lower : 0;
    int bridge_after = after_index < static_cast<int>(frames) ?
                       after_index : static_cast<int>(frames);
    size_t matrix_elements = static_cast<size_t>(head_dim) * head_dim;
    size_t head_matrix = static_cast<size_t>(head) * matrix_elements;
    const float *before = has_before ?
        prefix + (static_cast<size_t>(before_index) * heads * matrix_elements +
                  head_matrix) : text_state + head_matrix;
    const float *after = has_after ?
        suffix + (static_cast<size_t>(after_index) * heads * matrix_elements +
                  head_matrix) : text_state + head_matrix;
    extern __shared__ float shared[];
    float *before_bridge = shared;
    float *after_bridge = shared + head_dim;
    float *readout_reduction = shared + head_dim * 2;
    if (lane < head_dim) {
        float before_log = 0.0f;
        for (int index = bridge_before; index <= static_cast<int>(frame); index++)
            before_log += logf(fmaxf(alpha[
                (static_cast<size_t>(index) * heads + head) * head_dim + lane],
                1e-12f));
        float after_log = 0.0f;
        for (int index = static_cast<int>(frame); index < bridge_after; index++)
            after_log += logf(fmaxf(alpha[
                (static_cast<size_t>(index) * heads + head) * head_dim + lane],
                1e-12f));
        before_bridge[lane] = expf(before_log);
        after_bridge[lane] = expf(after_log);
    }
    __syncthreads();
    float readout = 0.0f;
    if (lane < head_dim) {
        size_t query_base = (static_cast<size_t>(token) * heads + head) *
                            head_dim;
        for (uint32_t key_channel = 0; key_channel < head_dim; key_channel++) {
            float before_value = before[
                static_cast<size_t>(lane) * head_dim + key_channel];
            float after_value = after[
                static_cast<size_t>(lane) * head_dim + key_channel];
            if (!has_before) before_value *= text_state_scale;
            if (!has_after) after_value *= text_state_scale;
            float state = before_value * before_bridge[key_channel] +
                          after_value * after_bridge[key_channel];
            readout = fmaf(static_cast<float>(
                              query[query_base + key_channel]),
                          state, readout);
        }
    }
    readout_reduction[lane] = readout * readout;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (lane < stride)
            readout_reduction[lane] += readout_reduction[lane + stride];
        __syncthreads();
    }
    if (lane < head_dim) {
        size_t output_index = (static_cast<size_t>(token) * heads + head) *
                              head_dim + lane;
        float inverse = rsqrtf(readout_reduction[0] /
                               static_cast<float>(head_dim) + epsilon);
        float logit = static_cast<float>(gate_logits[output_index]);
        float gate = 1.0f / (1.0f + expf(-logit));
        output[output_index] = hip_bfloat16(
            readout * inverse * static_cast<float>(norm_weight[lane]) * gate);
    }
}

template <typename T>
__global__ static void h3_hip_sdpa_kernel(
        const T *query, const T *key, const T *value, T *output,
        uint32_t batch, uint32_t sequence, uint32_t heads,
        uint32_t head_dim, float scale, int causal) {
    uint32_t query_row = blockIdx.x;
    uint32_t head_batch = blockIdx.y;
    uint32_t lane = threadIdx.x;
    uint32_t head = head_batch % heads;
    uint32_t batch_index = head_batch / heads;
    if (query_row >= sequence || batch_index >= batch) return;
    extern __shared__ float shared[];
    float *scores = shared;
    float *sdpa_reduce = shared + sequence;
    size_t batch_base = static_cast<size_t>(batch_index) * sequence * heads *
                        head_dim;
    size_t q_base = batch_base +
        (static_cast<size_t>(query_row) * heads + head) * head_dim;
    uint32_t key_count = causal ? query_row + 1 : sequence;
    for (uint32_t key_row = 0; key_row < key_count; key_row++) {
        size_t k_base = batch_base +
            (static_cast<size_t>(key_row) * heads + head) * head_dim;
        float partial = 0.0f;
        for (uint32_t dimension = lane; dimension < head_dim;
             dimension += blockDim.x)
            partial = fmaf(h3_hip_load(query[q_base + dimension]),
                           h3_hip_load(key[k_base + dimension]), partial);
        sdpa_reduce[lane] = partial;
        __syncthreads();
        for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
            if (lane < stride)
                sdpa_reduce[lane] += sdpa_reduce[lane + stride];
            __syncthreads();
        }
        if (lane == 0) scores[key_row] = sdpa_reduce[0] * scale;
        __syncthreads();
    }
    if (lane == 0) {
        float maximum = -INFINITY;
        for (uint32_t key_row = 0; key_row < key_count; key_row++)
            maximum = fmaxf(maximum, scores[key_row]);
        float denominator = 0.0f;
        for (uint32_t key_row = 0; key_row < key_count; key_row++) {
            scores[key_row] = expf(scores[key_row] - maximum);
            denominator += scores[key_row];
        }
        sdpa_reduce[0] = 1.0f / denominator;
    }
    __syncthreads();
    float inverse = sdpa_reduce[0];
    for (uint32_t dimension = lane; dimension < head_dim;
         dimension += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t key_row = 0; key_row < key_count; key_row++) {
            size_t v_index = batch_base +
                (static_cast<size_t>(key_row) * heads + head) * head_dim +
                dimension;
            sum = fmaf(scores[key_row] * inverse,
                       h3_hip_load(value[v_index]), sum);
        }
        output[q_base + dimension] = h3_hip_store<T>(sum);
    }
}

/* Exact specialization for the video VAE shape. The scalar
 * D=64 kernel starts with 256 threads, but only lanes 0..63 contribute. Its
 * 128/64 reduction steps add zero before d and d+32 are paired. One wave can
 * therefore reproduce the same tree as (p[d] + p[d+32]), followed by the
 * original 16/8/4/2/1 reductions. Each lane owns those same two PV output
 * dimensions and retains the original key/FMA order. */
__global__ static void h3_hip_sdpa_f32_d64_wave32_kernel(
        const float *query, const float *key, const float *value,
        float *output, uint32_t batch, uint32_t sequence, uint32_t heads,
        float scale, int causal) {
    uint32_t query_row = blockIdx.x;
    uint32_t head_batch = blockIdx.y;
    uint32_t lane = threadIdx.x;
    uint32_t head = head_batch % heads;
    uint32_t batch_index = head_batch / heads;
    if (query_row >= sequence || batch_index >= batch || lane >= 32) return;
    extern __shared__ float scores[];
    constexpr uint32_t head_dim = 64;
    size_t batch_base = static_cast<size_t>(batch_index) * sequence * heads *
                        head_dim;
    size_t query_base = batch_base +
        (static_cast<size_t>(query_row) * heads + head) * head_dim;
    uint32_t key_count = causal ? query_row + 1 : sequence;
    for (uint32_t key_row = 0; key_row < key_count; key_row++) {
        size_t key_base = batch_base +
            (static_cast<size_t>(key_row) * heads + head) * head_dim;
        float product0 = fmaf(query[query_base + lane],
                              key[key_base + lane], 0.0f);
        float product32 = fmaf(query[query_base + lane + 32],
                               key[key_base + lane + 32], 0.0f);
        float partial = product0 + product32;
#pragma unroll
        for (uint32_t offset = 16; offset; offset >>= 1)
            partial += __shfl_down(partial, offset, 32);
        if (lane == 0) scores[key_row] = partial * scale;
    }
    float inverse = 0.0f;
    if (lane == 0) {
        float maximum = -INFINITY;
        for (uint32_t key_row = 0; key_row < key_count; key_row++)
            maximum = fmaxf(maximum, scores[key_row]);
        float denominator = 0.0f;
        for (uint32_t key_row = 0; key_row < key_count; key_row++) {
            scores[key_row] = expf(scores[key_row] - maximum);
            denominator += scores[key_row];
        }
        inverse = 1.0f / denominator;
    }
    inverse = __shfl(inverse, 0, 32);
    float sum0 = 0.0f;
    float sum32 = 0.0f;
    for (uint32_t key_row = 0; key_row < key_count; key_row++) {
        size_t value_base = batch_base +
            (static_cast<size_t>(key_row) * heads + head) * head_dim;
        float probability = scores[key_row] * inverse;
        sum0 = fmaf(probability, value[value_base + lane], sum0);
        sum32 = fmaf(probability, value[value_base + lane + 32], sum32);
    }
    output[query_base + lane] = sum0;
    output[query_base + lane + 32] = sum32;
}

static int h3_gpu_linear(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight,
                         const h3_gpu_tensor *bias, uint32_t rows,
                         uint32_t input_dim, uint32_t output_dim,
                         h3_gpu_dtype dtype) {
    size_t input_count, weight_count, output_count;
    if (!h3_gpu_require_compute(gpu, "linear") ||
        !h3_gpu_count_2d(gpu, rows, input_dim, &input_count,
                         "linear input") ||
        !h3_gpu_count_2d(gpu, output_dim, input_dim, &weight_count,
                         "linear weight") ||
        !h3_gpu_count_2d(gpu, rows, output_dim, &output_count,
                         "linear output") ||
        !h3_gpu_require_tensor(gpu, input, input_count, dtype,
                               "linear input") ||
        !h3_gpu_require_tensor(gpu, weight, weight_count, dtype,
                               "linear weight") ||
        !h3_gpu_require_tensor(gpu, output, output_count, dtype,
                               "linear output") ||
        (bias && !h3_gpu_require_tensor(gpu, bias, output_dim, dtype,
                                        "linear bias")))
        return 0;

    h3_gpu_profile_scope profile(gpu, H3_HIP_PROFILE_LINEAR);
    rocblas_datatype type = dtype == H3_GPU_F32 ? rocblas_datatype_f32_r :
                                                   rocblas_datatype_bf16_r;
    float alpha = 1.0f;
    float beta = 0.0f;
    rocblas_status status = rocblas_gemm_ex(
        gpu->blas, rocblas_operation_transpose, rocblas_operation_none,
        static_cast<rocblas_int>(output_dim), static_cast<rocblas_int>(rows),
        static_cast<rocblas_int>(input_dim), &alpha, weight->data, type,
        static_cast<rocblas_int>(input_dim), input->data, type,
        static_cast<rocblas_int>(input_dim), &beta, output->data, type,
        static_cast<rocblas_int>(output_dim), output->data, type,
        static_cast<rocblas_int>(output_dim), rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0);
    if (!h3_gpu_check_blas(gpu, status, "rocblas_gemm_ex")) return 0;
    gpu->stats.mps_linear_dispatches++;

    if (bias) {
        if (dtype == H3_GPU_F32) {
            hipLaunchKernelGGL(h3_hip_bias_f32_kernel,
                               h3_gpu_grid_1d(static_cast<uint32_t>(output_count)),
                               dim3(H3_HIP_THREADS), 0, gpu->stream,
                               static_cast<float *>(output->data),
                               static_cast<const float *>(bias->data),
                               static_cast<uint32_t>(output_count), output_dim);
        } else {
            hipLaunchKernelGGL(h3_hip_bias_bf16_kernel,
                               h3_gpu_grid_1d(static_cast<uint32_t>(output_count)),
                               dim3(H3_HIP_THREADS), 0, gpu->stream,
                               static_cast<hip_bfloat16 *>(output->data),
                               static_cast<const hip_bfloat16 *>(bias->data),
                               static_cast<uint32_t>(output_count), output_dim);
        }
        if (!h3_gpu_kernel_enqueued(gpu, "linear bias kernel")) return 0;
    }
    return 1;
}

static int h3_gpu_patch_linear(h3_gpu *gpu, h3_gpu_tensor *output,
                               size_t output_offset,
                               const h3_gpu_tensor *input,
                               size_t input_offset,
                               const h3_gpu_tensor *weight,
                               const h3_gpu_tensor *bias,
                               const h3_gpu_tensor *row_map,
                               uint32_t output_rows, uint32_t rows,
                               uint32_t input_dim, uint32_t output_dim) {
    size_t input_count, weight_count, output_count;
    if (!h3_gpu_require_compute(gpu, "patch projection") ||
        !h3_gpu_count_2d(gpu, rows, input_dim, &input_count,
                         "patch input") ||
        !h3_gpu_count_2d(gpu, output_dim, input_dim, &weight_count,
                         "patch weight") ||
        !h3_gpu_count_2d(gpu, output_rows, output_dim, &output_count,
                         "patch output") ||
        input_offset > SIZE_MAX - input_count ||
        output_offset > SIZE_MAX - output_count ||
        !h3_gpu_require_tensor(gpu, input, input_offset + input_count,
                               H3_GPU_F32, "patch input") ||
        !h3_gpu_require_tensor(gpu, weight, weight_count, H3_GPU_F32,
                               "patch weight") ||
        !h3_gpu_require_tensor(gpu, output, output_offset + output_count,
                               H3_GPU_BF16, "patch output") ||
        (bias && !h3_gpu_require_tensor(gpu, bias, output_dim, H3_GPU_F32,
                                        "patch bias")) ||
        (row_map && !h3_gpu_require_tensor(gpu, row_map, rows, H3_GPU_U32,
                                           "patch row map")))
        return 0;
    dim3 block(H3_HIP_THREADS);
    dim3 grid((output_dim + H3_HIP_THREADS - 1) / H3_HIP_THREADS, rows);
    hipLaunchKernelGGL(h3_hip_patch_linear_bf16_kernel, grid, block, 0,
                       gpu->stream, static_cast<const float *>(input->data),
                       input_offset, static_cast<const float *>(weight->data),
                       bias ? static_cast<const float *>(bias->data) : nullptr,
                       static_cast<hip_bfloat16 *>(output->data), output_offset,
                       row_map ? static_cast<const uint32_t *>(row_map->data) :
                                 nullptr,
                       rows, input_dim, output_dim, bias ? 1 : 0,
                       row_map ? 1 : 0);
    return h3_gpu_kernel_enqueued(gpu, "patch projection kernel");
}

static int h3_gpu_sdpa(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *query,
                       const h3_gpu_tensor *key,
                       const h3_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence, uint32_t heads,
                       uint32_t head_dim, float scale, h3_gpu_dtype dtype,
                       int causal) {
    size_t rows, elements;
    if (!h3_gpu_require_compute(gpu, "SDPA") || !batch || !sequence ||
        !heads ||
        !h3_gpu_count_2d(gpu, batch, sequence, &rows, "SDPA rows") ||
        rows > UINT32_MAX / heads || batch > UINT32_MAX / heads ||
        !h3_gpu_count_2d(gpu, static_cast<uint32_t>(rows * heads), head_dim,
                         &elements, "SDPA tensor") ||
        !h3_gpu_require_tensor(gpu, query, elements, dtype, "SDPA query") ||
        !h3_gpu_require_tensor(gpu, key, elements, dtype, "SDPA key") ||
        !h3_gpu_require_tensor(gpu, value, elements, dtype, "SDPA value") ||
        !h3_gpu_require_tensor(gpu, output, elements, dtype, "SDPA output"))
        return 0;
    size_t shared_bytes = (static_cast<size_t>(sequence) + H3_HIP_THREADS) *
                          sizeof(float);
    int max_shared = 0;
    if (!h3_gpu_check(gpu,
                      hipDeviceGetAttribute(&max_shared,
                          hipDeviceAttributeMaxSharedMemoryPerBlock,
                          gpu->device),
                      "hipDeviceGetAttribute shared memory") ||
        shared_bytes > static_cast<size_t>(max_shared)) {
        h3_gpu_set_error(gpu,
                         "SDPA sequence %u needs %zu shared bytes; device has %d",
                         sequence, shared_bytes, max_shared);
        return 0;
    }
    h3_gpu_profile_scope profile(gpu, H3_HIP_PROFILE_SDPA);
    dim3 grid(sequence, batch * heads);
    const char *wave_value = std::getenv("H3_F32_SDPA_WAVE32");
    const char *scalar_value = std::getenv("H3_F32_SDPA_SCALAR");
    int force_scalar = scalar_value && *scalar_value &&
                       std::strcmp(scalar_value, "0");
    int disable_wave32 = wave_value && *wave_value &&
                         !std::strcmp(wave_value, "0");
    int use_f32_d64_wave32 = dtype == H3_GPU_F32 && head_dim == 64 && !causal &&
        gpu->warp_size == 32 && !force_scalar && !disable_wave32;
    if (use_f32_d64_wave32) {
        hipLaunchKernelGGL(h3_hip_sdpa_f32_d64_wave32_kernel, grid,
                           dim3(32), static_cast<size_t>(sequence) *
                           sizeof(float), gpu->stream,
                           static_cast<const float *>(query->data),
                           static_cast<const float *>(key->data),
                           static_cast<const float *>(value->data),
                           static_cast<float *>(output->data), batch, sequence,
                           heads, scale, causal);
    } else if (dtype == H3_GPU_F32) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(h3_hip_sdpa_kernel<float>), grid,
                           dim3(H3_HIP_THREADS), shared_bytes, gpu->stream,
                           static_cast<const float *>(query->data),
                           static_cast<const float *>(key->data),
                           static_cast<const float *>(value->data),
                           static_cast<float *>(output->data), batch, sequence,
                           heads, head_dim, scale, causal);
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(h3_hip_sdpa_kernel<hip_bfloat16>), grid,
            dim3(H3_HIP_THREADS), shared_bytes, gpu->stream,
            static_cast<const hip_bfloat16 *>(query->data),
            static_cast<const hip_bfloat16 *>(key->data),
            static_cast<const hip_bfloat16 *>(value->data),
            static_cast<hip_bfloat16 *>(output->data), batch, sequence, heads,
            head_dim, scale, causal);
    }
    if (!h3_gpu_kernel_enqueued(gpu, "SDPA kernel")) return 0;
    gpu->stats.mps_sdpa_dispatches++;
    return 1;
}

static int h3_gpu_qkv_rope(h3_gpu *gpu, h3_gpu_tensor *query,
                           h3_gpu_tensor *key, h3_gpu_tensor *value,
                           const h3_gpu_tensor *qkv,
                           const h3_gpu_tensor *q_norm,
                           const h3_gpu_tensor *k_norm,
                           const h3_gpu_tensor *rope_cos,
                           const h3_gpu_tensor *rope_sin,
                           uint32_t sequence, uint32_t heads,
                           uint32_t head_dim, uint32_t rope_half,
                           int grouped, float epsilon) {
    size_t inner, elements, rope_elements;
    if (!h3_gpu_require_compute(gpu, "QKV/RoPE") || !heads || !head_dim ||
        rope_half > head_dim / 2 || epsilon < 0.0f ||
        !h3_gpu_count_2d(gpu, heads, head_dim, &inner, "QKV inner") ||
        inner > UINT32_MAX ||
        !h3_gpu_count_2d(gpu, sequence, static_cast<uint32_t>(inner),
                         &elements, "QKV output") ||
        elements > SIZE_MAX / 3 ||
        !h3_gpu_count_2d(gpu, sequence, rope_half, &rope_elements,
                         "RoPE table") ||
        !h3_gpu_require_tensor(gpu, qkv, elements * 3, H3_GPU_BF16,
                               "QKV input") ||
        !h3_gpu_require_tensor(gpu, q_norm, head_dim, H3_GPU_BF16,
                               "Q norm") ||
        !h3_gpu_require_tensor(gpu, k_norm, head_dim, H3_GPU_BF16,
                               "K norm") ||
        !h3_gpu_require_tensor(gpu, rope_cos, rope_elements, H3_GPU_BF16,
                               "RoPE cosine") ||
        !h3_gpu_require_tensor(gpu, rope_sin, rope_elements, H3_GPU_BF16,
                               "RoPE sine") ||
        !h3_gpu_require_tensor(gpu, query, elements, H3_GPU_BF16,
                               "query") ||
        !h3_gpu_require_tensor(gpu, key, elements, H3_GPU_BF16, "key") ||
        !h3_gpu_require_tensor(gpu, value, elements, H3_GPU_BF16, "value"))
        return 0;
    dim3 grid(heads, sequence);
    hipLaunchKernelGGL(h3_hip_qkv_rope_bf16_kernel, grid,
                       dim3(H3_HIP_THREADS),
                       H3_HIP_THREADS * sizeof(float) * 2, gpu->stream,
                       static_cast<const hip_bfloat16 *>(qkv->data),
                       static_cast<const hip_bfloat16 *>(q_norm->data),
                       static_cast<const hip_bfloat16 *>(k_norm->data),
                       static_cast<const hip_bfloat16 *>(rope_cos->data),
                       static_cast<const hip_bfloat16 *>(rope_sin->data),
                       static_cast<hip_bfloat16 *>(query->data),
                       static_cast<hip_bfloat16 *>(key->data),
                       static_cast<hip_bfloat16 *>(value->data), sequence,
                       heads, head_dim, rope_half, grouped, epsilon);
    return h3_gpu_kernel_enqueued(gpu, "QKV/RoPE kernel");
}

static int h3_gpu_unsupported(h3_gpu *gpu, const char *operation) {
    h3_gpu_set_error(gpu, "%s is not implemented by the HIP backend yet",
                     operation);
    return 0;
}

extern "C" h3_gpu *h3_gpu_create(const char *shader_source_path,
                                  char *error, size_t error_size) {
    (void)shader_source_path;
    if (error && error_size) error[0] = '\0';

    h3_gpu *gpu = static_cast<h3_gpu *>(std::calloc(1, sizeof(*gpu)));
    if (!gpu) {
        if (error && error_size)
            std::snprintf(error, error_size,
                          "out of memory creating HIP context");
        return nullptr;
    }
    if (pthread_mutex_init(&gpu->staging_lock, nullptr) != 0) {
        if (error && error_size)
            std::snprintf(error, error_size,
                          "cannot initialize HIP staging-cache lock");
        std::free(gpu);
        return nullptr;
    }
    gpu->staging_lock_initialized = 1;
    const char *staging_value = std::getenv("H3_HIP_STAGING_CACHE");
    gpu->staging_cache_enabled =
        !(staging_value && !std::strcmp(staging_value, "0"));
    hipError_t status = hipGetDevice(&gpu->device);
    if (status == hipSuccess)
        status = hipStreamCreateWithFlags(&gpu->stream, hipStreamNonBlocking);
    if (status != hipSuccess) {
        if (error && error_size)
            std::snprintf(error, error_size, "cannot create HIP context: %s",
                          hipGetErrorString(status));
        (void)pthread_mutex_destroy(&gpu->staging_lock);
        std::free(gpu);
        return nullptr;
    }
    hipDeviceProp_t properties;
    if (hipGetDeviceProperties(&properties, gpu->device) == hipSuccess) {
        gpu->warp_size = properties.warpSize;
        std::snprintf(gpu->gcn_arch_name, sizeof(gpu->gcn_arch_name), "%s",
                      properties.gcnArchName[0] ? properties.gcnArchName :
                                                  "unknown");
    }
    rocblas_status blas_status = rocblas_create_handle(&gpu->blas);
    if (blas_status == rocblas_status_success)
        blas_status = rocblas_set_stream(gpu->blas, gpu->stream);
    if (blas_status != rocblas_status_success) {
        if (error && error_size)
            std::snprintf(error, error_size,
                          "cannot create rocBLAS context: %s",
                          rocblas_status_to_string(blas_status));
        if (gpu->blas) (void)rocblas_destroy_handle(gpu->blas);
        (void)hipStreamDestroy(gpu->stream);
        (void)pthread_mutex_destroy(&gpu->staging_lock);
        std::free(gpu);
        return nullptr;
    }
    std::snprintf(gpu->profile_label, sizeof(gpu->profile_label),
                  "HIP context");
    gpu->profile_start_wall = h3_gpu_now();
    gpu->profile_mark_wall = gpu->profile_start_wall;
    gpu->profile_start_stats = gpu->stats;
    gpu->profile_mark_stats = gpu->stats;
    return gpu;
}

extern "C" void h3_gpu_free(h3_gpu *gpu) {
    if (!gpu) return;
    (void)hipSetDevice(gpu->device);
    (void)hipStreamSynchronize(gpu->stream);
    h3_gpu_profile_flush_ops(gpu);
    h3_gpu_profile_emit(gpu, "total", gpu->profile_start_stats,
                        gpu->profile_start_wall);
    h3_gpu_profile_emit_ops(gpu);
    h3_gpu_profile_emit_load(gpu);
    h3_gpu_profile_destroy_events(gpu);
    h3_gpu_destroy_staging_events(gpu);
    h3_gpu_purge_staging(gpu);
    (void)hipFree(gpu->sage_workspace);
    (void)rocblas_destroy_handle(gpu->blas);
    (void)hipStreamDestroy(gpu->stream);
    if (gpu->staging_lock_initialized)
        (void)pthread_mutex_destroy(&gpu->staging_lock);
    std::free(gpu);
}

extern "C" int h3_gpu_is_m5(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

extern "C" int h3_gpu_has_nax_mlp(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

extern "C" int h3_gpu_has_int8_mlp(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu,
                                                 size_t elements) {
    return h3_gpu_tensor_new(gpu, nullptr, elements, H3_GPU_F32);
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu,
                                                  size_t elements) {
    return h3_gpu_tensor_new(gpu, nullptr, elements, H3_GPU_BF16);
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu,
                                                size_t elements) {
    return h3_gpu_tensor_new(gpu, nullptr, elements, H3_GPU_I8);
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu,
                                                  const float *values,
                                                  size_t elements) {
    return h3_gpu_tensor_new(gpu, values, elements, H3_GPU_F32);
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu,
                                                   const uint16_t *values,
                                                   size_t elements) {
    return h3_gpu_tensor_new(gpu, values, elements, H3_GPU_BF16);
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu,
                                                  const uint32_t *values,
                                                  size_t elements) {
    return h3_gpu_tensor_new(gpu, values, elements, H3_GPU_U32);
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu,
                                                   const char *path,
                                                   uint64_t file_offset,
                                                   size_t elements) {
    return h3_gpu_tensor_load(gpu, path, file_offset, elements, H3_GPU_BF16);
}

extern "C" h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu,
                                                  const char *path,
                                                  uint64_t file_offset,
                                                  size_t elements) {
    return h3_gpu_tensor_load(gpu, path, file_offset, elements, H3_GPU_F32);
}

extern "C" int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor,
                                             const char *path,
                                             uint64_t file_offset,
                                             size_t elements,
                                             char *error,
                                             size_t error_size) {
    return h3_gpu_read_file(tensor, path, file_offset, elements,
                            H3_GPU_BF16, error, error_size);
}

extern "C" int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor,
                                               const char *path,
                                               uint64_t file_offset,
                                               size_t elements,
                                               char *error,
                                               size_t error_size) {
    return h3_gpu_read_file(tensor, path, file_offset, elements,
                            H3_GPU_BF16, error, error_size);
}

extern "C" void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    if (!tensor) return;
    h3_gpu *gpu = tensor->owner;
    if (gpu) {
        (void)hipSetDevice(gpu->device);
        (void)hipFree(tensor->data);
        gpu->stats.live_bytes -= tensor->bytes;
    }
    std::free(tensor);
}

extern "C" size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->elements : 0;
}

extern "C" h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor) {
    return tensor ? tensor->dtype : H3_GPU_F32;
}

extern "C" int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor,
                                       float *values, size_t elements) {
    return h3_gpu_copy_to_host(tensor, 0, values, elements, H3_GPU_F32);
}

extern "C" int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                             size_t source_offset,
                                             float *values, size_t elements) {
    return h3_gpu_copy_to_host(tensor, source_offset, values, elements,
                               H3_GPU_F32);
}

extern "C" int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor,
                                        uint16_t *values, size_t elements) {
    return h3_gpu_copy_to_host(tensor, 0, values, elements, H3_GPU_BF16);
}

extern "C" int h3_gpu_tensor_read_i8(const h3_gpu_tensor *tensor,
                                      int8_t *values, size_t elements) {
    return h3_gpu_copy_to_host(tensor, 0, values, elements, H3_GPU_I8);
}

extern "C" int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor,
                                        const float *values,
                                        size_t elements) {
    return h3_gpu_copy_from_host(tensor, 0, values, elements, H3_GPU_F32);
}

extern "C" int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                              size_t destination_offset,
                                              const float *values,
                                              size_t elements) {
    return h3_gpu_copy_from_host(tensor, destination_offset, values, elements,
                                 H3_GPU_F32);
}

extern "C" int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor,
                                         const uint16_t *values,
                                         size_t elements) {
    return h3_gpu_copy_from_host(tensor, 0, values, elements, H3_GPU_BF16);
}

extern "C" int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                               size_t destination_offset,
                                               const uint16_t *values,
                                               size_t elements) {
    return h3_gpu_copy_from_host(tensor, destination_offset, values, elements,
                                 H3_GPU_BF16);
}

extern "C" int h3_gpu_begin(h3_gpu *gpu) {
    if (!gpu || gpu->recording) {
        if (gpu) h3_gpu_set_error(gpu, "HIP command stream is already active");
        return 0;
    }
    gpu->recording = 1;
    gpu->error[0] = '\0';
    gpu->command_start_wall = h3_gpu_now();
    return 1;
}

extern "C" int h3_gpu_continue(h3_gpu *gpu) {
    if (!gpu || !gpu->recording) return 0;
    double now = h3_gpu_now();
    if (gpu->command_start_wall > 0.0)
        gpu->stats.command_encode_seconds += now - gpu->command_start_wall;
    double wait_start = h3_gpu_now();
    if (!h3_gpu_check(gpu, hipStreamSynchronize(gpu->stream),
                      "hipStreamSynchronize"))
        return 0;
    double waited = h3_gpu_now() - wait_start;
    gpu->stats.command_wait_seconds += waited;
    gpu->stats.gpu_seconds = gpu->stats.command_wait_seconds;
    gpu->stats.submissions++;
    h3_gpu_profile_flush_ops(gpu);
    gpu->command_start_wall = h3_gpu_now();
    return 1;
}

extern "C" int h3_gpu_submit(h3_gpu *gpu) {
    if (!gpu || !gpu->recording) return 0;
    double now = h3_gpu_now();
    if (gpu->command_start_wall > 0.0)
        gpu->stats.command_encode_seconds += now - gpu->command_start_wall;
    double wait_start = h3_gpu_now();
    int ok = h3_gpu_check(gpu, hipStreamSynchronize(gpu->stream),
                          "hipStreamSynchronize");
    double waited = h3_gpu_now() - wait_start;
    gpu->recording = 0;
    gpu->command_start_wall = 0.0;
    if (ok) {
        gpu->stats.command_wait_seconds += waited;
        gpu->stats.gpu_seconds = gpu->stats.command_wait_seconds;
        gpu->stats.submissions++;
        h3_gpu_profile_flush_ops(gpu);
    }
    return ok;
}

extern "C" const char *h3_gpu_error(const h3_gpu *gpu) {
    return gpu ? gpu->error : "HIP context is null";
}

extern "C" int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats) {
    if (!gpu || !stats) return 0;
    *stats = gpu->stats;
    return 1;
}

extern "C" int h3_gpu_get_memory_info(const h3_gpu *gpu,
                                        uint64_t *free_bytes,
                                        uint64_t *total_bytes) {
    if (!gpu || !free_bytes || !total_bytes) return 0;
    if (hipSetDevice(gpu->device) != hipSuccess) return 0;
    size_t free_size = 0;
    size_t total_size = 0;
    if (hipMemGetInfo(&free_size, &total_size) != hipSuccess) return 0;
    *free_bytes = (uint64_t)free_size;
    *total_bytes = (uint64_t)total_size;
    return 1;
}

extern "C" int h3_gpu_get_profile_stats(
        const h3_gpu *gpu, h3_gpu_profile_stats *stats) {
    if (!gpu || !stats) return 0;
    *stats = gpu->profile_totals;
    stats->enabled = h3_gpu_profile_enabled();
    stats->staging_hits = gpu->staging_hits;
    stats->staging_misses = gpu->staging_misses;
    return 1;
}

extern "C" void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label) {
    if (!gpu || !label) return;
    std::snprintf(gpu->profile_label, sizeof(gpu->profile_label), "%s",
                  label);
}

extern "C" void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase) {
    if (!gpu || !phase || !*phase || !h3_gpu_profile_enabled()) return;
    h3_gpu_profile_flush_ops(gpu);
    h3_gpu_profile_emit(gpu, phase, gpu->profile_mark_stats,
                        gpu->profile_mark_wall);
    h3_gpu_profile_emit_ops(gpu);
    gpu->profile_linear_ms = 0.0;
    gpu->profile_lora_ms = 0.0;
    gpu->profile_sdpa_ms = 0.0;
    gpu->profile_solve_ms = 0.0;
    gpu->profile_scan_ms = 0.0;
    gpu->profile_mark_stats = gpu->stats;
    gpu->profile_mark_wall = h3_gpu_now();
}

extern "C" int h3_gpu_copy_bf16(h3_gpu *gpu,
                                 h3_gpu_tensor *destination,
                                 size_t destination_offset,
                                 const h3_gpu_tensor *source,
                                 size_t source_offset, size_t elements) {
    return h3_gpu_copy_device(gpu, destination, destination_offset,
                              source, source_offset, elements, H3_GPU_BF16);
}

extern "C" int h3_gpu_copy_f32(h3_gpu *gpu,
                                h3_gpu_tensor *destination,
                                size_t destination_offset,
                                const h3_gpu_tensor *source,
                                size_t source_offset, size_t elements) {
    return h3_gpu_copy_device(gpu, destination, destination_offset,
                              source, source_offset, elements, H3_GPU_F32);
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"

extern "C" int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_linear(gpu, output, input, weight, bias, rows, input_dim,
                         output_dim, H3_GPU_F32);
}

extern "C" int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_patch_linear(gpu, output, 0, input, 0, weight, bias, nullptr,
                               rows, rows, input_dim, output_dim);
}

extern "C" int h3_gpu_patch_linear_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                             size_t output_offset,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_patch_linear(gpu, output, output_offset, input, input_offset,
                               weight, bias, nullptr, rows, rows, input_dim,
                               output_dim);
}

extern "C" int h3_gpu_patch_linear_bf16_map(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias,
                             const h3_gpu_tensor *row_map,
                             uint32_t output_rows, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_patch_linear(gpu, output, 0, input, 0, weight, bias, row_map,
                               output_rows, rows, input_dim, output_dim);
}

extern "C" int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_gpu_require_compute(gpu, __func__) ||
        !h3_gpu_require_tensor(gpu, input, elements, H3_GPU_F32,
                               "SiLU input") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_F32,
                               "SiLU output"))
        return 0;
    hipLaunchKernelGGL(h3_hip_silu_f32_kernel, h3_gpu_grid_1d(elements),
                       dim3(H3_HIP_THREADS), 0, gpu->stream,
                       static_cast<const float *>(input->data),
                       static_cast<float *>(output->data), elements);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_gpu_require_compute(gpu, __func__) ||
        !h3_gpu_require_tensor(gpu, input, elements, H3_GPU_F32,
                               "cast input") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_BF16,
                               "cast output"))
        return 0;
    hipLaunchKernelGGL(h3_hip_cast_f32_to_bf16_kernel,
                       h3_gpu_grid_1d(elements), dim3(H3_HIP_THREADS), 0,
                       gpu->stream, static_cast<const float *>(input->data),
                       static_cast<hip_bfloat16 *>(output->data), elements);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_gpu_require_compute(gpu, __func__) ||
        !h3_gpu_require_tensor(gpu, input, elements, H3_GPU_BF16,
                               "cast input") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_F32,
                               "cast output"))
        return 0;
    hipLaunchKernelGGL(h3_hip_cast_bf16_to_f32_kernel,
                       h3_gpu_grid_1d(elements), dim3(H3_HIP_THREADS), 0,
                       gpu->stream,
                       static_cast<const hip_bfloat16 *>(input->data),
                       static_cast<float *>(output->data), elements);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon) {
    size_t elements;
    if (!h3_gpu_require_compute(gpu, __func__) || epsilon < 0.0f ||
        !h3_gpu_count_2d(gpu, rows, width, &elements, "RMSNorm") ||
        !h3_gpu_require_tensor(gpu, input, elements, H3_GPU_F32,
                               "RMSNorm input") ||
        !h3_gpu_require_tensor(gpu, weight, width, H3_GPU_F32,
                               "RMSNorm weight") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_F32,
                               "RMSNorm output"))
        return 0;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(h3_hip_rms_norm_kernel<float>),
                       dim3(rows), dim3(H3_HIP_THREADS),
                       H3_HIP_THREADS * sizeof(float), gpu->stream,
                       static_cast<const float *>(input->data),
                       static_cast<const float *>(weight->data),
                       static_cast<float *>(output->data), rows, width,
                       epsilon);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale) {
    return h3_gpu_sdpa(gpu, output, query, key, value, 1, sequence, heads,
                       head_dim, scale, H3_GPU_F32, 0);
}

extern "C" int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    size_t elements;
    if (!h3_gpu_require_compute(gpu, __func__) ||
        !h3_gpu_count_2d(gpu, rows, width, &elements, "SwiGLU") ||
        elements > UINT32_MAX / 2 ||
        !h3_gpu_require_tensor(gpu, fused, elements * 2, H3_GPU_F32,
                               "SwiGLU input") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_F32,
                               "SwiGLU output"))
        return 0;
    uint32_t count = static_cast<uint32_t>(elements);
    hipLaunchKernelGGL(h3_hip_swiglu_f32_kernel, h3_gpu_grid_1d(count),
                       dim3(H3_HIP_THREADS), 0, gpu->stream,
                       static_cast<const float *>(fused->data),
                       static_cast<float *>(output->data), rows, width);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width) {
    size_t elements;
    if (!h3_gpu_require_compute(gpu, __func__) ||
        !h3_gpu_count_2d(gpu, rows, width, &elements, "scale-add") ||
        elements > UINT32_MAX ||
        !h3_gpu_require_tensor(gpu, residual, elements, H3_GPU_F32,
                               "scale-add residual") ||
        !h3_gpu_require_tensor(gpu, branch, elements, H3_GPU_F32,
                               "scale-add branch") ||
        !h3_gpu_require_tensor(gpu, scale, width, H3_GPU_F32,
                               "scale-add scale") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_F32,
                               "scale-add output"))
        return 0;
    uint32_t count = static_cast<uint32_t>(elements);
    hipLaunchKernelGGL(h3_hip_scale_add_f32_kernel, h3_gpu_grid_1d(count),
                       dim3(H3_HIP_THREADS), 0, gpu->stream,
                       static_cast<const float *>(residual->data),
                       static_cast<const float *>(branch->data),
                       static_cast<const float *>(scale->data),
                       static_cast<float *>(output->data), count, width);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon) {
    size_t elements;
    if (!h3_gpu_require_compute(gpu, __func__) || epsilon < 0.0f ||
        !h3_gpu_count_2d(gpu, rows, width, &elements, "LayerNorm") ||
        !h3_gpu_require_tensor(gpu, input, elements, H3_GPU_F32,
                               "LayerNorm input") ||
        !h3_gpu_require_tensor(gpu, weight, width, H3_GPU_F32,
                               "LayerNorm weight") ||
        !h3_gpu_require_tensor(gpu, bias, width, H3_GPU_F32,
                               "LayerNorm bias") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_F32,
                               "LayerNorm output"))
        return 0;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(h3_hip_layer_norm_kernel<float>),
                       dim3(rows), dim3(H3_HIP_THREADS),
                       H3_HIP_THREADS * sizeof(float), gpu->stream,
                       static_cast<const float *>(input->data),
                       static_cast<const float *>(weight->data),
                       static_cast<const float *>(bias->data),
                       static_cast<float *>(output->data), rows, width,
                       epsilon);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon) {
    size_t inner, elements, rope_elements;
    if (!h3_gpu_require_compute(gpu, __func__) || !sequence || !heads ||
        !head_dim || rope_half > head_dim / 2 || epsilon < 0.0f ||
        !h3_gpu_count_2d(gpu, heads, head_dim, &inner, "video QKV inner") ||
        inner > UINT32_MAX ||
        !h3_gpu_count_2d(gpu, sequence, static_cast<uint32_t>(inner),
                         &elements, "video QKV output") ||
        elements > SIZE_MAX / 3 ||
        !h3_gpu_count_2d(gpu, sequence, rope_half, &rope_elements,
                         "video RoPE table") ||
        !h3_gpu_require_tensor(gpu, qkv, elements * 3, H3_GPU_F32,
                               "video QKV") ||
        !h3_gpu_require_tensor(gpu, rope_cos, rope_elements, H3_GPU_F32,
                               "video RoPE cosine") ||
        !h3_gpu_require_tensor(gpu, rope_sin, rope_elements, H3_GPU_F32,
                               "video RoPE sine") ||
        !h3_gpu_require_tensor(gpu, query, elements, H3_GPU_F32,
                               "video query") ||
        !h3_gpu_require_tensor(gpu, key, elements, H3_GPU_F32,
                               "video key") ||
        !h3_gpu_require_tensor(gpu, value, elements, H3_GPU_F32,
                               "video value"))
        return 0;
    hipLaunchKernelGGL(h3_hip_video_qkv_rope_f32_kernel,
                       dim3(heads, sequence), dim3(H3_HIP_THREADS),
                       H3_HIP_THREADS * sizeof(float) * 2, gpu->stream,
                       static_cast<const float *>(qkv->data),
                       static_cast<const float *>(rope_cos->data),
                       static_cast<const float *>(rope_sin->data),
                       static_cast<float *>(query->data),
                       static_cast<float *>(key->data),
                       static_cast<float *>(value->data), sequence, heads,
                       head_dim, rope_half, epsilon);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation) {
    return h3_gpu_conv1d_stride_f32(
        gpu, output, input, weight, bias, batch, length, input_channels,
        output_channels, kernel, 1, padding, dilation);
}

extern "C" int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding,
                      uint32_t dilation) {
    uint64_t effective = (uint64_t)dilation * (kernel - 1) + 1;
    uint64_t padded = (uint64_t)length + 2 * padding;
    if (!h3_gpu_require_compute(gpu, __func__) || !batch || !length ||
        !input_channels || !output_channels || !kernel || !stride ||
        !dilation || padded < effective) return 0;
    uint64_t output_length_wide = (padded - effective) / stride + 1;
    uint64_t input_count = (uint64_t)batch * length * input_channels;
    uint64_t weight_count =
        (uint64_t)output_channels * input_channels * kernel;
    uint64_t output_count =
        (uint64_t)batch * output_length_wide * output_channels;
    if (output_length_wide > UINT32_MAX || input_count > SIZE_MAX ||
        weight_count > SIZE_MAX || output_count > UINT32_MAX ||
        !h3_gpu_require_tensor(gpu, input, (size_t)input_count, H3_GPU_F32,
                               "Conv1d input") ||
        !h3_gpu_require_tensor(gpu, weight, (size_t)weight_count, H3_GPU_F32,
                               "Conv1d weight") ||
        !h3_gpu_require_tensor(gpu, output, (size_t)output_count, H3_GPU_F32,
                               "Conv1d output") ||
        (bias && !h3_gpu_require_tensor(gpu, bias, output_channels,
                                        H3_GPU_F32, "Conv1d bias")))
        return 0;
    if (kernel == 1 && stride == 1 && padding == 0 && dilation == 1)
        return h3_gpu_linear(
            gpu, output, input, weight, bias,
            static_cast<uint32_t>((uint64_t)batch * length), input_channels,
            output_channels, H3_GPU_F32);
    hipLaunchKernelGGL(h3_hip_conv1d_f32_kernel,
                       h3_gpu_grid_1d(static_cast<uint32_t>(output_count)),
                       dim3(H3_HIP_THREADS), 0, gpu->stream,
                       static_cast<float *>(output->data),
                       static_cast<const float *>(input->data),
                       static_cast<const float *>(weight->data),
                       bias ? static_cast<const float *>(bias->data) : nullptr,
                       static_cast<uint32_t>(output_count), length,
                       static_cast<uint32_t>(output_length_wide),
                       input_channels, output_channels, kernel, stride,
                       padding, dilation, bias ? 1 : 0);
    gpu->stats.mps_conv_dispatches++;
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_conv_transpose1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding) {
    if (!h3_gpu_require_compute(gpu, __func__) || !batch || !length ||
        !input_channels || !output_channels || !kernel || !stride ||
        (uint64_t)(length - 1) * stride + kernel < 2 * padding)
        return 0;
    uint64_t output_length_wide =
        (uint64_t)(length - 1) * stride + kernel - 2 * padding;
    uint64_t input_count = (uint64_t)batch * length * input_channels;
    uint64_t weight_count =
        (uint64_t)input_channels * output_channels * kernel;
    uint64_t output_count =
        (uint64_t)batch * output_length_wide * output_channels;
    if (!output_length_wide || output_length_wide > UINT32_MAX ||
        input_count > SIZE_MAX || weight_count > SIZE_MAX ||
        output_count > UINT32_MAX ||
        !h3_gpu_require_tensor(gpu, input, (size_t)input_count, H3_GPU_F32,
                               "ConvTranspose1d input") ||
        !h3_gpu_require_tensor(gpu, weight, (size_t)weight_count, H3_GPU_F32,
                               "ConvTranspose1d weight") ||
        !h3_gpu_require_tensor(gpu, output, (size_t)output_count, H3_GPU_F32,
                               "ConvTranspose1d output") ||
        (bias && !h3_gpu_require_tensor(gpu, bias, output_channels,
                                        H3_GPU_F32,
                                        "ConvTranspose1d bias")))
        return 0;
    hipLaunchKernelGGL(
        h3_hip_conv_transpose1d_f32_kernel,
        h3_gpu_grid_1d(static_cast<uint32_t>(output_count)),
        dim3(H3_HIP_THREADS), 0, gpu->stream,
        static_cast<float *>(output->data),
        static_cast<const float *>(input->data),
        static_cast<const float *>(weight->data),
        bias ? static_cast<const float *>(bias->data) : nullptr,
        static_cast<uint32_t>(output_count), length,
        static_cast<uint32_t>(output_length_wide), input_channels,
        output_channels, kernel, stride, padding, bias ? 1 : 0);
    gpu->stats.mps_conv_dispatches++;
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner) {
    size_t elements;
    if (!h3_gpu_require_compute(gpu, __func__) || !outer || !inner ||
        !h3_gpu_count_2d(gpu, outer, inner, &elements, "weight norm") ||
        !h3_gpu_require_tensor(gpu, vector, elements, H3_GPU_F32,
                               "weight norm vector") ||
        !h3_gpu_require_tensor(gpu, magnitude, outer, H3_GPU_F32,
                               "weight norm magnitude") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_F32,
                               "weight norm output"))
        return 0;
    hipLaunchKernelGGL(h3_hip_weight_norm_f32_kernel, dim3(outer),
                       dim3(H3_HIP_THREADS), H3_HIP_THREADS * sizeof(float),
                       gpu->stream, static_cast<float *>(output->data),
                       static_cast<const float *>(vector->data),
                       static_cast<const float *>(magnitude->data), outer,
                       inner);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements) {
    if (!h3_gpu_require_compute(gpu, __func__) ||
        !h3_gpu_require_tensor(gpu, left, elements, H3_GPU_F32,
                               "scaled-add left") ||
        !h3_gpu_require_tensor(gpu, right, elements, H3_GPU_F32,
                               "scaled-add right") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_F32,
                               "scaled-add output"))
        return 0;
    hipLaunchKernelGGL(h3_hip_add_scaled_f32_kernel,
                       h3_gpu_grid_1d(elements), dim3(H3_HIP_THREADS), 0,
                       gpu->stream, static_cast<const float *>(left->data),
                       static_cast<const float *>(right->data),
                       static_cast<float *>(output->data), left_scale,
                       right_scale, elements);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_alias_free_snake_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *alpha_log,
                          const h3_gpu_tensor *beta_log,
                          const h3_gpu_tensor *upsample_filter,
                          const h3_gpu_tensor *downsample_filter,
                          uint32_t batch, uint32_t length,
                          uint32_t channels) {
    uint64_t elements_wide = (uint64_t)batch * length * channels;
    if (!h3_gpu_require_compute(gpu, __func__) || !batch || !length ||
        !channels || elements_wide > UINT32_MAX ||
        !h3_gpu_require_tensor(gpu, input, (size_t)elements_wide, H3_GPU_F32,
                               "alias-free input") ||
        !h3_gpu_require_tensor(gpu, output, (size_t)elements_wide,
                               H3_GPU_F32, "alias-free output") ||
        !h3_gpu_require_tensor(gpu, alpha_log, channels, H3_GPU_F32,
                               "alias-free alpha") ||
        !h3_gpu_require_tensor(gpu, beta_log, channels, H3_GPU_F32,
                               "alias-free beta") ||
        !h3_gpu_require_tensor(gpu, upsample_filter, 12, H3_GPU_F32,
                               "alias-free upsample filter") ||
        !h3_gpu_require_tensor(gpu, downsample_filter, 12, H3_GPU_F32,
                               "alias-free downsample filter"))
        return 0;
    uint32_t elements = static_cast<uint32_t>(elements_wide);
    hipLaunchKernelGGL(
        h3_hip_alias_free_snake_f32_kernel, h3_gpu_grid_1d(elements),
        dim3(H3_HIP_THREADS), 0, gpu->stream,
        static_cast<float *>(output->data),
        static_cast<const float *>(input->data),
        static_cast<const float *>(alpha_log->data),
        static_cast<const float *>(beta_log->data),
        static_cast<const float *>(upsample_filter->data),
        static_cast<const float *>(downsample_filter->data), elements,
        length, channels);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels) {
    uint64_t elements_wide = (uint64_t)batch * length * channels;
    if (!h3_gpu_require_compute(gpu, __func__) || !batch || !length ||
        !channels || elements_wide > UINT32_MAX ||
        !h3_gpu_require_tensor(gpu, input, (size_t)elements_wide, H3_GPU_F32,
                               "Snake1d input") ||
        !h3_gpu_require_tensor(gpu, output, (size_t)elements_wide, H3_GPU_F32,
                               "Snake1d output") ||
        !h3_gpu_require_tensor(gpu, alpha, channels, H3_GPU_F32,
                               "Snake1d alpha"))
        return 0;
    uint32_t elements = static_cast<uint32_t>(elements_wide);
    hipLaunchKernelGGL(h3_hip_snake1d_f32_kernel, h3_gpu_grid_1d(elements),
                       dim3(H3_HIP_THREADS), 0, gpu->stream,
                       static_cast<float *>(output->data),
                       static_cast<const float *>(input->data),
                       static_cast<const float *>(alpha->data), elements,
                       channels);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_audio_qkv_split_f32(h3_gpu *gpu,
                       h3_gpu_tensor *query, h3_gpu_tensor *key,
                       h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                       const h3_gpu_tensor *q_bias,
                       const h3_gpu_tensor *k_bias,
                       const h3_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *query,
                       const h3_gpu_tensor *key,
                       const h3_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence, uint32_t heads,
                       uint32_t head_dim, float scale) {
    return h3_gpu_sdpa(gpu, output, query, key, value, batch, sequence, heads,
                       head_dim, scale, H3_GPU_F32, 1);
}

extern "C" int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu,
                       h3_gpu_tensor *output,
                       const h3_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum) {
    if (!h3_gpu_require_compute(gpu, __func__) || minimum > maximum ||
        !h3_gpu_require_tensor(gpu, input, elements, H3_GPU_F32,
                               "clip input") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_F32,
                               "clip output"))
        return 0;
    hipLaunchKernelGGL(h3_hip_clip_f32_kernel, h3_gpu_grid_1d(elements),
                       dim3(H3_HIP_THREADS), 0, gpu->stream,
                       static_cast<const float *>(input->data),
                       static_cast<float *>(output->data), elements, minimum,
                       maximum);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_vae_encoder_pad_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_vae_encoder_group_norm_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_linear(gpu, output, input, weight, bias, rows, input_dim,
                         output_dim, H3_GPU_BF16);
}

extern "C" int h3_gpu_lora_merge_bf16(
                       h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *base,
                       const h3_gpu_tensor *lora_a,
                       const h3_gpu_tensor *lora_b,
                       uint32_t input_dim, uint32_t output_dim,
                       uint32_t rank, float scale) {
    h3_gpu_profile_scope profile(gpu, H3_HIP_PROFILE_LORA);
    size_t weight_count;
    size_t a_count;
    size_t b_count;
    if (!h3_gpu_require_compute(gpu, __func__) || !std::isfinite(scale) ||
        input_dim > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        output_dim > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        rank > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        !h3_gpu_count_2d(gpu, output_dim, input_dim, &weight_count,
                         "LoRA base weight") ||
        !h3_gpu_count_2d(gpu, rank, input_dim, &a_count, "LoRA A") ||
        !h3_gpu_count_2d(gpu, output_dim, rank, &b_count, "LoRA B") ||
        !h3_gpu_require_tensor(gpu, base, weight_count, H3_GPU_BF16,
                               "LoRA base weight") ||
        !h3_gpu_require_tensor(gpu, output, weight_count, H3_GPU_BF16,
                               "LoRA output weight") ||
        !h3_gpu_require_tensor(gpu, lora_a, a_count, H3_GPU_BF16,
                               "LoRA A") ||
        !h3_gpu_require_tensor(gpu, lora_b, b_count, H3_GPU_BF16,
                               "LoRA B"))
        return 0;
    if (output == lora_a || output == lora_b) {
        h3_gpu_set_error(gpu, "LoRA output must not alias adapter matrices");
        return 0;
    }
    if (output != base &&
        !h3_gpu_copy_device(gpu, output, 0, base, 0, weight_count,
                            H3_GPU_BF16))
        return 0;

    /* Row-major B @ A is column-major A^T @ B^T in the same storage. */
    float beta = 1.0f;
    rocblas_status status = rocblas_gemm_ex(
        gpu->blas, rocblas_operation_none, rocblas_operation_none,
        static_cast<rocblas_int>(input_dim),
        static_cast<rocblas_int>(output_dim),
        static_cast<rocblas_int>(rank), &scale,
        lora_a->data, rocblas_datatype_bf16_r,
        static_cast<rocblas_int>(input_dim),
        lora_b->data, rocblas_datatype_bf16_r,
        static_cast<rocblas_int>(rank), &beta,
        output->data, rocblas_datatype_bf16_r,
        static_cast<rocblas_int>(input_dim),
        output->data, rocblas_datatype_bf16_r,
        static_cast<rocblas_int>(input_dim), rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, 0);
    if (!h3_gpu_check_blas(gpu, status, "rocblas_gemm_ex LoRA merge"))
        return 0;
    gpu->stats.mps_linear_dispatches++;
    return 1;
}

extern "C" int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim) {
    size_t fused_count, activated_count;
    if (!h3_gpu_require_compute(gpu, __func__) || hidden_dim > UINT32_MAX / 2 ||
        !h3_gpu_count_2d(gpu, rows, hidden_dim * 2, &fused_count,
                         "MLP fused activation") ||
        !h3_gpu_count_2d(gpu, rows, hidden_dim, &activated_count,
                         "MLP activation"))
        return 0;
    h3_gpu_tensor *fused = h3_gpu_tensor_new_bf16(gpu, fused_count);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_bf16(gpu, activated_count);
    if (!fused || !activated) {
        h3_gpu_tensor_free(activated);
        h3_gpu_tensor_free(fused);
        return 0;
    }
    int ok = h3_gpu_linear_bf16(gpu, fused, input, fc1_weight, nullptr, rows,
                                 input_dim, hidden_dim * 2) &&
             h3_gpu_swiglu_bf16(gpu, activated, fused, rows, hidden_dim) &&
             h3_gpu_linear_bf16(gpu, output, activated, fc2_weight, nullptr,
                                 rows, hidden_dim, output_dim);
    /* hipFree is a synchronization point in this correctness-first fallback.
     * A reusable activation arena replaces it in the streaming optimization
     * phase. */
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(fused);
    return ok;
}

extern "C" int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_linear_int8_head_major_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t output_dim) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         h3_gpu_tensor *activated,
                         h3_gpu_tensor *quantized_activation,
                         h3_gpu_tensor *activation_scales,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *fc1_weight,
                         const h3_gpu_tensor *fc1_scales,
                         const h3_gpu_tensor *fc2_weight,
                         const h3_gpu_tensor *fc2_scales,
                         const h3_gpu_tensor *fc1_bf16,
                         const h3_gpu_tensor *fc2_bf16, uint32_t rows,
                         uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim,
                         int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k,
                         int use_int8_row_fc2,
                         int input_is_quantized) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
    if (!h3_gpu_require_compute(gpu, __func__) ||
        !h3_gpu_require_tensor(gpu, input, elements, H3_GPU_BF16,
                               "SiLU input") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_BF16,
                               "SiLU output"))
        return 0;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(h3_hip_silu_kernel<hip_bfloat16>),
                       h3_gpu_grid_1d(elements), dim3(H3_HIP_THREADS), 0,
                       gpu->stream,
                       static_cast<const hip_bfloat16 *>(input->data),
                       static_cast<hip_bfloat16 *>(output->data), elements);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
    size_t elements;
    if (!h3_gpu_require_compute(gpu, __func__) || epsilon < 0.0f ||
        !h3_gpu_count_2d(gpu, rows, width, &elements, "RMSNorm") ||
        !h3_gpu_require_tensor(gpu, input, elements, H3_GPU_BF16,
                               "RMSNorm input") ||
        !h3_gpu_require_tensor(gpu, weight, width, H3_GPU_BF16,
                               "RMSNorm weight") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_BF16,
                               "RMSNorm output"))
        return 0;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(h3_hip_rms_norm_kernel<hip_bfloat16>),
                       dim3(rows), dim3(H3_HIP_THREADS),
                       H3_HIP_THREADS * sizeof(float), gpu->stream,
                       static_cast<const hip_bfloat16 *>(input->data),
                       static_cast<const hip_bfloat16 *>(weight->data),
                       static_cast<hip_bfloat16 *>(output->data), rows, width,
                       epsilon);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon) {
    size_t elements;
    if (!h3_gpu_require_compute(gpu, __func__) || epsilon < 0.0f ||
        !h3_gpu_count_2d(gpu, rows, width, &elements, "LayerNorm") ||
        !h3_gpu_require_tensor(gpu, input, elements, H3_GPU_BF16,
                               "LayerNorm input") ||
        !h3_gpu_require_tensor(gpu, weight, width, H3_GPU_BF16,
                               "LayerNorm weight") ||
        !h3_gpu_require_tensor(gpu, bias, width, H3_GPU_BF16,
                               "LayerNorm bias") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_BF16,
                               "LayerNorm output"))
        return 0;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(h3_hip_layer_norm_kernel<hip_bfloat16>),
                       dim3(rows), dim3(H3_HIP_THREADS),
                       H3_HIP_THREADS * sizeof(float), gpu->stream,
                       static_cast<const hip_bfloat16 *>(input->data),
                       static_cast<const hip_bfloat16 *>(weight->data),
                       static_cast<const hip_bfloat16 *>(bias->data),
                       static_cast<hip_bfloat16 *>(output->data), rows, width,
                       epsilon);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) {
    if (!h3_gpu_require_compute(gpu, __func__) ||
        !h3_gpu_require_tensor(gpu, input, elements, H3_GPU_BF16,
                               "GELU input") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_BF16,
                               "GELU output"))
        return 0;
    hipLaunchKernelGGL(h3_hip_gelu_bf16_kernel, h3_gpu_grid_1d(elements),
                       dim3(H3_HIP_THREADS), 0, gpu->stream,
                       static_cast<const hip_bfloat16 *>(input->data),
                       static_cast<hip_bfloat16 *>(output->data), elements,
                       approximate ? 1 : 0);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_vision_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                     const h3_gpu_tensor *qkv,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    return h3_gpu_adaln_bf16_offset(
        gpu, output, input, 0, norm_weight, modulation, row_map, rows, width,
        slots, shift_slot, scale_slot, epsilon);
}

extern "C" int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    size_t elements;
    if (!h3_gpu_require_compute(gpu, __func__) || !slots ||
        shift_slot >= slots || scale_slot >= slots || epsilon < 0.0f ||
        !h3_gpu_count_2d(gpu, rows, width, &elements, "AdaLN") ||
        input_offset > SIZE_MAX - elements ||
        !h3_gpu_require_tensor(gpu, input, input_offset + elements,
                               H3_GPU_BF16, "AdaLN input") ||
        !h3_gpu_require_tensor(gpu, norm_weight, width, H3_GPU_BF16,
                               "AdaLN norm weight") ||
        !h3_gpu_require_tensor(gpu, modulation, 1, H3_GPU_BF16,
                               "AdaLN modulation") ||
        !h3_gpu_require_tensor(gpu, row_map, rows, H3_GPU_U32,
                               "AdaLN row map") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_BF16,
                               "AdaLN output"))
        return 0;
    hipLaunchKernelGGL(h3_hip_adaln_bf16_kernel, dim3(rows),
                       dim3(H3_HIP_THREADS), H3_HIP_THREADS * sizeof(float),
                       gpu->stream,
                       static_cast<const hip_bfloat16 *>(input->data),
                       input_offset,
                       static_cast<const hip_bfloat16 *>(norm_weight->data),
                       static_cast<const hip_bfloat16 *>(modulation->data),
                       static_cast<const uint32_t *>(row_map->data),
                       static_cast<hip_bfloat16 *>(output->data), rows, width,
                       slots, shift_slot, scale_slot, epsilon);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_adaln_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                      h3_gpu_tensor *inverse,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t width, uint32_t output_dim, uint32_t slots,
                      uint32_t shift_slot, uint32_t scale_slot,
                      float epsilon) {
    size_t elements;
    if (!h3_gpu_count_2d(gpu, rows, width, &elements,
                         "fused AdaLN linear"))
        return 0;
    if (inverse && !h3_gpu_require_tensor(gpu, inverse, rows, H3_GPU_F32,
                                          "AdaLN inverse scratch"))
        return 0;
    h3_gpu_tensor *normalized = h3_gpu_tensor_new_bf16(gpu, elements);
    if (!normalized) return 0;
    int ok = h3_gpu_adaln_bf16_offset(
                 gpu, normalized, input, input_offset, norm_weight,
                 modulation, row_map, rows, width, slots, shift_slot,
                 scale_slot, epsilon) &&
             h3_gpu_linear_bf16(gpu, output, normalized, weight, bias, rows,
                                width, output_dim);
    h3_gpu_tensor_free(normalized);
    return ok;
}

extern "C" int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) {
    size_t elements;
    if (!h3_gpu_require_compute(gpu, __func__) || !slots ||
        gate_slot >= slots ||
        !h3_gpu_count_2d(gpu, rows, width, &elements, "gate") ||
        elements > UINT32_MAX ||
        !h3_gpu_require_tensor(gpu, residual, elements, H3_GPU_BF16,
                               "gate residual") ||
        !h3_gpu_require_tensor(gpu, branch, elements, H3_GPU_BF16,
                               "gate branch") ||
        !h3_gpu_require_tensor(gpu, modulation, 1, H3_GPU_BF16,
                               "gate modulation") ||
        !h3_gpu_require_tensor(gpu, row_map, rows, H3_GPU_U32,
                               "gate row map") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_BF16,
                               "gate output"))
        return 0;
    uint32_t count = static_cast<uint32_t>(elements);
    hipLaunchKernelGGL(h3_hip_gate_bf16_kernel, h3_gpu_grid_1d(count),
                       dim3(H3_HIP_THREADS), 0, gpu->stream,
                       static_cast<const hip_bfloat16 *>(residual->data),
                       static_cast<const hip_bfloat16 *>(branch->data),
                       static_cast<const hip_bfloat16 *>(modulation->data),
                       static_cast<const uint32_t *>(row_map->data),
                       static_cast<hip_bfloat16 *>(output->data), count, width,
                       slots, gate_slot);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_gate_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot,
                     uint32_t shift_slot, uint32_t scale_slot,
                     float epsilon) {
    size_t elements;
    if (!h3_gpu_require_compute(gpu, __func__) || !slots ||
        gate_slot >= slots || shift_slot >= slots || scale_slot >= slots ||
        epsilon < 0.0f ||
        !h3_gpu_count_2d(gpu, rows, width, &elements, "gate AdaLN") ||
        !h3_gpu_require_tensor(gpu, residual, elements, H3_GPU_BF16,
                               "gate AdaLN residual") ||
        !h3_gpu_require_tensor(gpu, branch, elements, H3_GPU_BF16,
                               "gate AdaLN branch") ||
        !h3_gpu_require_tensor(gpu, norm_weight, width, H3_GPU_BF16,
                               "gate AdaLN norm") ||
        !h3_gpu_require_tensor(gpu, gate_modulation, 1, H3_GPU_BF16,
                               "gate modulation") ||
        !h3_gpu_require_tensor(gpu, norm_modulation, 1, H3_GPU_BF16,
                               "norm modulation") ||
        !h3_gpu_require_tensor(gpu, row_map, rows, H3_GPU_U32,
                               "gate AdaLN row map") ||
        !h3_gpu_require_tensor(gpu, gated_residual, elements, H3_GPU_BF16,
                               "gated residual") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_BF16,
                               "gate AdaLN output"))
        return 0;
    hipLaunchKernelGGL(h3_hip_gate_adaln_bf16_kernel, dim3(rows),
                       dim3(H3_HIP_THREADS), H3_HIP_THREADS * sizeof(float),
                       gpu->stream,
                       static_cast<const hip_bfloat16 *>(residual->data),
                       static_cast<const hip_bfloat16 *>(branch->data),
                       static_cast<const hip_bfloat16 *>(norm_weight->data),
                       static_cast<const hip_bfloat16 *>(gate_modulation->data),
                       static_cast<const hip_bfloat16 *>(norm_modulation->data),
                       static_cast<const uint32_t *>(row_map->data),
                       static_cast<hip_bfloat16 *>(gated_residual->data),
                       static_cast<hip_bfloat16 *>(output->data), rows, width,
                       slots, gate_slot, shift_slot, scale_slot, epsilon);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_gate_adaln_quantize_int8(h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *quantized_output,
                     h3_gpu_tensor *quantized_scales,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t padded_rows, uint32_t width, uint32_t slots,
                     uint32_t gate_slot, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                         h3_gpu_tensor *key, h3_gpu_tensor *value,
                         const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon) {
    return h3_gpu_qkv_rope(gpu, query, key, value, qkv, q_norm, k_norm,
                           rope_cos, rope_sin, sequence, heads, head_dim,
                           rope_half, 0, epsilon);
}

extern "C" int h3_gpu_vdn_qk_rope_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *query,
                     h3_gpu_tensor *key,
                     const h3_gpu_tensor *query_raw,
                     const h3_gpu_tensor *key_raw,
                     const h3_gpu_tensor *q_norm,
                     const h3_gpu_tensor *k_norm,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin,
                     uint32_t sequence, uint32_t heads,
                     uint32_t head_dim, uint32_t rope_half,
                     float epsilon) {
    size_t inner, elements, rope_elements;
    if (!h3_gpu_require_compute(gpu, __func__) || !heads || !head_dim ||
        rope_half > head_dim / 2 || epsilon < 0.0f ||
        !h3_gpu_count_2d(gpu, heads, head_dim, &inner, "VDN QK inner") ||
        inner > UINT32_MAX ||
        !h3_gpu_count_2d(gpu, sequence, static_cast<uint32_t>(inner),
                         &elements, "VDN QK output") ||
        (rope_half && !h3_gpu_count_2d(gpu, sequence, rope_half,
                                      &rope_elements, "VDN RoPE table")) ||
        !h3_gpu_require_tensor(gpu, query_raw, elements, H3_GPU_BF16,
                               "VDN raw query") ||
        !h3_gpu_require_tensor(gpu, key_raw, elements, H3_GPU_BF16,
                               "VDN raw key") ||
        !h3_gpu_require_tensor(gpu, q_norm, head_dim, H3_GPU_BF16,
                               "VDN Q norm") ||
        !h3_gpu_require_tensor(gpu, k_norm, head_dim, H3_GPU_BF16,
                               "VDN K norm") ||
        (rope_half && !h3_gpu_require_tensor(
            gpu, rope_cos, rope_elements, H3_GPU_BF16,
            "VDN RoPE cosine")) ||
        (rope_half && !h3_gpu_require_tensor(
            gpu, rope_sin, rope_elements, H3_GPU_BF16,
            "VDN RoPE sine")) ||
        !h3_gpu_require_tensor(gpu, query, elements, H3_GPU_BF16,
                               "VDN query") ||
        !h3_gpu_require_tensor(gpu, key, elements, H3_GPU_BF16,
                               "VDN key"))
        return 0;
    hipLaunchKernelGGL(h3_hip_vdn_qk_rope_bf16_kernel,
                       dim3(heads, sequence), dim3(H3_HIP_THREADS),
                       H3_HIP_THREADS * sizeof(float) * 2, gpu->stream,
                       static_cast<const hip_bfloat16 *>(query_raw->data),
                       static_cast<const hip_bfloat16 *>(key_raw->data),
                       static_cast<const hip_bfloat16 *>(q_norm->data),
                       static_cast<const hip_bfloat16 *>(k_norm->data),
                       static_cast<const hip_bfloat16 *>(rope_cos->data),
                       static_cast<const hip_bfloat16 *>(rope_sin->data),
                       static_cast<hip_bfloat16 *>(query->data),
                       static_cast<hip_bfloat16 *>(key->data), sequence,
                       heads, head_dim, rope_half, epsilon);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon) {
    return h3_gpu_qkv_rope(gpu, query, key, value, qkv, q_norm, k_norm,
                           rope_cos, rope_sin, sequence, heads, head_dim,
                           rope_half, 1, epsilon);
}

extern "C" int h3_gpu_grouped_qkv_linear_rope_bf16(h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon) {
    if (!heads || head_dim > UINT32_MAX / heads ||
        heads * head_dim > UINT32_MAX / 3) {
        h3_gpu_set_error(gpu, "invalid grouped QKV dimensions");
        return 0;
    }
    uint32_t inner = heads * head_dim;
    return h3_gpu_linear_bf16(gpu, qkv, input, weight, nullptr, rows,
                              input_dim, inner * 3) &&
           h3_gpu_grouped_qkv_rope_bf16(
               gpu, query, key, value, qkv, q_norm, k_norm, rope_cos,
               rope_sin, rows, heads, head_dim, rope_half, epsilon);
}

extern "C" int h3_gpu_grouped_qkv_linear_rope_int8(h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *quantized_input,
                                 h3_gpu_tensor *input_scales,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *weight_scales,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon,
                                 int input_is_quantized,
                                 int use_slower_unfused_qkv_rope,
                                 int use_slower_scalar_qkv_rms,
                                 int use_slower_uncached_int8_scales) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    return h3_gpu_sdpa(gpu, output, query, key, value, 1, sequence, heads,
                       head_dim, scale, H3_GPU_BF16, 0);
}

extern "C" int h3_gpu_vdn_sage_quant_qk_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *query_i8,
                     h3_gpu_tensor *key_i8, h3_gpu_tensor *query_scales,
                     h3_gpu_tensor *key_scales,
                     const h3_gpu_tensor *query_bf16,
                     const h3_gpu_tensor *key_bf16,
                     uint32_t sequence, uint32_t heads,
                     uint32_t head_dim) {
    size_t inner, elements;
    uint32_t query_groups = static_cast<uint32_t>(
        (static_cast<uint64_t>(sequence) + 31) / 32);
    uint32_t key_groups = static_cast<uint32_t>(
        (static_cast<uint64_t>(sequence) + 63) / 64);
    size_t query_scale_elements, key_scale_elements;
    if (!h3_gpu_require_compute(gpu, __func__) || !sequence || !heads ||
        head_dim != 128 ||
        !h3_gpu_count_2d(gpu, heads, head_dim, &inner,
                         "VDN Sage quant inner") ||
        !h3_gpu_count_2d(gpu, sequence, static_cast<uint32_t>(inner),
                         &elements, "VDN Sage quant tensor") ||
        !h3_gpu_count_2d(gpu, heads, query_groups, &query_scale_elements,
                         "VDN Sage Q scales") ||
        !h3_gpu_count_2d(gpu, heads, key_groups, &key_scale_elements,
                         "VDN Sage K scales") ||
        !h3_gpu_require_tensor(gpu, query_bf16, elements, H3_GPU_BF16,
                               "VDN Sage query") ||
        !h3_gpu_require_tensor(gpu, key_bf16, elements, H3_GPU_BF16,
                               "VDN Sage key") ||
        !h3_gpu_require_tensor(gpu, query_i8, elements, H3_GPU_I8,
                               "VDN Sage quantized query") ||
        !h3_gpu_require_tensor(gpu, key_i8, elements, H3_GPU_I8,
                               "VDN Sage quantized key") ||
        !h3_gpu_require_tensor(gpu, query_scales, query_scale_elements,
                               H3_GPU_F32, "VDN Sage query scales") ||
        !h3_gpu_require_tensor(gpu, key_scales, key_scale_elements,
                               H3_GPU_F32, "VDN Sage key scales")) return 0;
    constexpr uint32_t threads = 256;
    size_t shared_bytes = threads * sizeof(float);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(h3_hip_vdn_sage_quant_bf16_i8_kernel<32>),
        dim3(query_groups, heads), dim3(threads), shared_bytes, gpu->stream,
        static_cast<const hip_bfloat16 *>(query_bf16->data),
        static_cast<int8_t *>(query_i8->data),
        static_cast<float *>(query_scales->data), sequence, heads, head_dim,
        query_groups);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(h3_hip_vdn_sage_quant_bf16_i8_kernel<64>),
        dim3(key_groups, heads), dim3(threads), shared_bytes, gpu->stream,
        static_cast<const hip_bfloat16 *>(key_bf16->data),
        static_cast<int8_t *>(key_i8->data),
        static_cast<float *>(key_scales->data), sequence, heads, head_dim,
        key_groups);
    if (!h3_gpu_kernel_enqueued(gpu, __func__)) return 0;
    gpu->stats.direct_dispatches += 2;
    return 1;
}

extern "C" int h3_gpu_vdn_sage_wmma_qk_tile_i8(
                     h3_gpu *gpu, h3_gpu_tensor *scores_f32,
                     const h3_gpu_tensor *query_i8,
                     const h3_gpu_tensor *key_i8,
                     const h3_gpu_tensor *query_scales,
                     const h3_gpu_tensor *key_scales,
                     uint32_t sequence, uint32_t heads,
                     uint32_t head_dim, uint32_t query_start,
                     uint32_t key_start, uint32_t head, float scale) {
    constexpr uint32_t tile = 16;
    size_t inner, elements;
    uint32_t query_groups = static_cast<uint32_t>(
        (static_cast<uint64_t>(sequence) + 31) / 32);
    uint32_t key_groups = static_cast<uint32_t>(
        (static_cast<uint64_t>(sequence) + 63) / 64);
    size_t query_scale_elements, key_scale_elements;
    if (!h3_gpu_require_compute(gpu, __func__) || !sequence || !heads ||
        head_dim != 128 || query_start > sequence || key_start > sequence ||
        sequence - query_start < tile || sequence - key_start < tile ||
        head >= heads || !std::isfinite(scale) || scale <= 0.0f ||
        std::strncmp(gpu->gcn_arch_name, "gfx12", 5) != 0 ||
        gpu->warp_size != 32 ||
        !h3_gpu_count_2d(gpu, heads, head_dim, &inner,
                         "VDN Sage WMMA inner") ||
        !h3_gpu_count_2d(gpu, sequence, static_cast<uint32_t>(inner),
                         &elements, "VDN Sage WMMA tensor") ||
        !h3_gpu_count_2d(gpu, heads, query_groups, &query_scale_elements,
                         "VDN Sage WMMA Q scales") ||
        !h3_gpu_count_2d(gpu, heads, key_groups, &key_scale_elements,
                         "VDN Sage WMMA K scales") ||
        !h3_gpu_require_tensor(gpu, query_i8, elements, H3_GPU_I8,
                               "VDN Sage WMMA query") ||
        !h3_gpu_require_tensor(gpu, key_i8, elements, H3_GPU_I8,
                               "VDN Sage WMMA key") ||
        !h3_gpu_require_tensor(gpu, query_scales, query_scale_elements,
                               H3_GPU_F32, "VDN Sage WMMA Q scales") ||
        !h3_gpu_require_tensor(gpu, key_scales, key_scale_elements,
                               H3_GPU_F32, "VDN Sage WMMA K scales") ||
        !h3_gpu_require_tensor(gpu, scores_f32, tile * tile, H3_GPU_F32,
                               "VDN Sage WMMA scores")) return 0;
    hipLaunchKernelGGL(h3_hip_vdn_sage_wmma_qk_tile_i8_kernel,
                       dim3(1), dim3(32), 0, gpu->stream,
                       static_cast<const int8_t *>(query_i8->data),
                       static_cast<const int8_t *>(key_i8->data),
                       static_cast<const float *>(query_scales->data),
                       static_cast<const float *>(key_scales->data),
                       static_cast<float *>(scores_f32->data), sequence, heads,
                       query_start, key_start, head, query_groups, key_groups,
                       scale);
    if (!h3_gpu_kernel_enqueued(gpu, __func__)) return 0;
    gpu->stats.direct_dispatches++;
    return 1;
}

extern "C" int h3_gpu_vdn_sage_attention_i8_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *output_bf16,
                     const h3_gpu_tensor *query_i8,
                     const h3_gpu_tensor *key_i8,
                     const h3_gpu_tensor *query_scales,
                     const h3_gpu_tensor *key_scales,
                     const h3_gpu_tensor *value_bf16,
                     uint32_t sequence, uint32_t heads,
                     uint32_t head_dim, uint32_t video_start,
                     uint32_t frames, uint32_t tokens_per_frame,
                     uint32_t radius, uint32_t chunk,
                     int anchor_both, float scale) {
    size_t inner, elements;
    uint64_t video_rows = static_cast<uint64_t>(frames) * tokens_per_frame;
    uint64_t video_end_wide = static_cast<uint64_t>(video_start) + video_rows;
    uint32_t query_groups = static_cast<uint32_t>(
        (static_cast<uint64_t>(sequence) + 31) / 32);
    uint32_t key_groups = static_cast<uint32_t>(
        (static_cast<uint64_t>(sequence) + 63) / 64);
    size_t query_scale_elements, key_scale_elements;
    if (!h3_gpu_require_compute(gpu, __func__) || !sequence || !heads ||
        head_dim != 128 || !frames || !tokens_per_frame ||
        (anchor_both != 0 && anchor_both != 1) ||
        video_start > sequence || video_end_wide > sequence ||
        !std::isfinite(scale) || scale <= 0.0f ||
        std::strncmp(gpu->gcn_arch_name, "gfx12", 5) != 0 ||
        gpu->warp_size != 32 ||
        !h3_gpu_count_2d(gpu, heads, head_dim, &inner,
                         "VDN Sage attention inner") ||
        inner > UINT32_MAX ||
        !h3_gpu_count_2d(gpu, sequence, static_cast<uint32_t>(inner),
                         &elements, "VDN Sage attention tensor") ||
        !h3_gpu_count_2d(gpu, heads, query_groups, &query_scale_elements,
                         "VDN Sage attention Q scales") ||
        !h3_gpu_count_2d(gpu, heads, key_groups, &key_scale_elements,
                         "VDN Sage attention K scales") ||
        !h3_gpu_require_tensor(gpu, query_i8, elements, H3_GPU_I8,
                               "VDN Sage attention query") ||
        !h3_gpu_require_tensor(gpu, key_i8, elements, H3_GPU_I8,
                               "VDN Sage attention key") ||
        !h3_gpu_require_tensor(gpu, query_scales, query_scale_elements,
                               H3_GPU_F32, "VDN Sage attention Q scales") ||
        !h3_gpu_require_tensor(gpu, key_scales, key_scale_elements,
                               H3_GPU_F32, "VDN Sage attention K scales") ||
        !h3_gpu_require_tensor(gpu, value_bf16, elements, H3_GPU_BF16,
                               "VDN Sage attention value") ||
        !h3_gpu_require_tensor(gpu, output_bf16, elements, H3_GPU_BF16,
                               "VDN Sage attention output")) return 0;
    const h3_vdn_sage_geometry geometry = {
        sequence, heads, head_dim, video_start, frames, tokens_per_frame,
        radius, chunk, anchor_both};
    h3_vdn_q_task *device_tasks = nullptr;
    uint32_t task_count = 0;
    if (!h3_gpu_prepare_sage_tasks(
            gpu, &geometry, 0, &device_tasks, &task_count)) return 0;
    const hipError_t launch_status = h3_vdn_sage_gfx12_launch_i8_bf16(
        static_cast<const int8_t *>(query_i8->data),
        static_cast<const int8_t *>(key_i8->data),
        static_cast<const float *>(query_scales->data),
        static_cast<const float *>(key_scales->data), value_bf16->data,
        output_bf16->data, device_tasks, task_count, sequence, heads,
        query_groups, key_groups, scale, gpu->stream);
    if (!h3_gpu_check(gpu, launch_status,
                      "VDN Sage E27 attention kernel")) return 0;
    gpu->stats.direct_dispatches++;
    return 1;
}

extern "C" int h3_gpu_vdn_window_sdpa_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query,
                     const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value,
                     uint32_t sequence, uint32_t heads,
                     uint32_t head_dim, uint32_t video_start,
                     uint32_t frames, uint32_t tokens_per_frame,
                     uint32_t radius, uint32_t chunk,
                     int anchor_both, float scale) {
    size_t inner, elements;
    uint64_t video_rows = (uint64_t)frames * tokens_per_frame;
    uint64_t video_end_wide = (uint64_t)video_start + video_rows;
    if (!h3_gpu_require_compute(gpu, __func__) || !sequence || !heads ||
        !head_dim || head_dim > H3_HIP_THREADS || !frames ||
        !tokens_per_frame || (anchor_both != 0 && anchor_both != 1) ||
        video_end_wide > sequence || !std::isfinite(scale) || scale <= 0.0f ||
        !h3_gpu_count_2d(gpu, heads, head_dim, &inner,
                         "VDN window inner") || inner > UINT32_MAX ||
        !h3_gpu_count_2d(gpu, sequence, static_cast<uint32_t>(inner),
                         &elements, "VDN window tensor") ||
        !h3_gpu_require_tensor(gpu, query, elements, H3_GPU_BF16,
                               "VDN window query") ||
        !h3_gpu_require_tensor(gpu, key, elements, H3_GPU_BF16,
                               "VDN window key") ||
        !h3_gpu_require_tensor(gpu, value, elements, H3_GPU_BF16,
                               "VDN window value") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_BF16,
                               "VDN window output"))
        return 0;
    h3_gpu_profile_scope profile(gpu, H3_HIP_PROFILE_SDPA);
    uint32_t video_end = static_cast<uint32_t>(video_end_wide);
    h3_vdn_sdpa_mode mode = H3_VDN_SDPA_AUTO;
    const char *mode_value = std::getenv("H3_VDN_SDPA");
    if (mode_value && *mode_value &&
        !h3_vdn_sdpa_mode_parse(mode_value, &mode)) {
        h3_gpu_set_error(gpu, "invalid H3_VDN_SDPA mode '%s'", mode_value);
        return 0;
    }
    if (mode == H3_VDN_SDPA_SAGE_I8_F16 ||
        mode == H3_VDN_SDPA_SAGE_I8_FP8_E4M3) {
        h3_gpu_set_error(
            gpu, "H3_VDN_SDPA=%s is not enabled yet for %s; use auto, "
            "wave32, scalar, or sage-i8-bf16",
            h3_vdn_sdpa_mode_name(mode),
            gpu->gcn_arch_name[0] ? gpu->gcn_arch_name : "unknown HIP arch");
        return 0;
    }
    if (mode == H3_VDN_SDPA_SAGE_I8_BF16) {
        uint32_t query_groups = static_cast<uint32_t>(
            (static_cast<uint64_t>(sequence) + 31) / 32);
        uint32_t key_groups = static_cast<uint32_t>(
            (static_cast<uint64_t>(sequence) + 63) / 64);
        size_t scale_elements;
        if (head_dim != 128 || gpu->warp_size != 32 ||
            std::strncmp(gpu->gcn_arch_name, "gfx12", 5) != 0) {
            h3_gpu_set_error(
                gpu, "H3_VDN_SDPA=%s requires gfx12 wave32 "
                "and head_dim=128 (got %s wave%d D=%u)",
                h3_vdn_sdpa_mode_name(mode),
                gpu->gcn_arch_name[0] ? gpu->gcn_arch_name : "unknown",
                gpu->warp_size, head_dim);
            return 0;
        }
        if (static_cast<size_t>(query_groups) >
                SIZE_MAX - static_cast<size_t>(key_groups) ||
            !h3_gpu_count_2d(
                gpu, heads, query_groups + key_groups, &scale_elements,
                "VDN Sage workspace scales") ||
            elements > SIZE_MAX / 2 ||
            scale_elements > (SIZE_MAX - elements * 2) / sizeof(float)) {
            h3_gpu_set_error(gpu, "VDN Sage workspace size overflow");
            return 0;
        }
        const size_t tasks_offset = elements * 2 +
                                    scale_elements * sizeof(float);
        const h3_vdn_sage_geometry geometry = {
            sequence, heads, head_dim, video_start, frames,
            tokens_per_frame, radius, chunk, anchor_both};
        h3_vdn_q_task *device_tasks = nullptr;
        uint32_t task_count = 0;
        if (!h3_gpu_prepare_sage_tasks(
                gpu, &geometry, tasks_offset, &device_tasks, &task_count))
            return 0;
        uint8_t *workspace = static_cast<uint8_t *>(gpu->sage_workspace);
        h3_gpu_tensor query_i8 = {
            gpu, workspace, elements, elements, H3_GPU_I8};
        h3_gpu_tensor key_i8 = {
            gpu, workspace + elements, elements, elements, H3_GPU_I8};
        float *scale_data = reinterpret_cast<float *>(
            workspace + elements * 2);
        size_t query_scale_elements =
            static_cast<size_t>(heads) * query_groups;
        h3_gpu_tensor query_scales = {
            gpu, scale_data, query_scale_elements,
            query_scale_elements * sizeof(float), H3_GPU_F32};
        size_t key_scale_elements = static_cast<size_t>(heads) * key_groups;
        h3_gpu_tensor key_scales = {
            gpu, scale_data + query_scale_elements, key_scale_elements,
            key_scale_elements * sizeof(float), H3_GPU_F32};
        if (!h3_gpu_vdn_sage_quant_qk_bf16(
                gpu, &query_i8, &key_i8, &query_scales, &key_scales,
                query, key, sequence, heads, head_dim)) return 0;
        const hipError_t launch_status =
            h3_vdn_sage_gfx12_launch_i8_bf16(
                static_cast<const int8_t *>(query_i8.data),
                static_cast<const int8_t *>(key_i8.data),
                static_cast<const float *>(query_scales.data),
                static_cast<const float *>(key_scales.data), value->data,
                output->data, device_tasks, task_count, sequence, heads,
                query_groups, key_groups, scale, gpu->stream);
        if (!h3_gpu_check(gpu, launch_status,
                          "VDN Sage E27 attention kernel")) return 0;
        gpu->stats.direct_dispatches++;
        gpu->stats.mps_sdpa_dispatches++;
        return 1;
    }
    const char *scalar_value = std::getenv("H3_VDN_SCALAR_SDPA");
    const char *reload_query_value = std::getenv("H3_VDN_RELOAD_QUERY");
    int reload_query = reload_query_value && *reload_query_value &&
        std::strcmp(reload_query_value, "0");
    const char *scan_mask_value = std::getenv("H3_VDN_SCAN_MASK");
    int scan_mask = scan_mask_value && *scan_mask_value &&
        std::strcmp(scan_mask_value, "0");
    const char *lane0_softmax_value = std::getenv("H3_VDN_LANE0_SOFTMAX");
    int lane0_softmax = lane0_softmax_value && *lane0_softmax_value &&
        std::strcmp(lane0_softmax_value, "0");
    int legacy_force_scalar = scalar_value && *scalar_value &&
                              std::strcmp(scalar_value, "0");
    if (mode == H3_VDN_SDPA_WAVE32 &&
        (gpu->warp_size != 32 || head_dim > 256)) {
        h3_gpu_set_error(gpu,
            "H3_VDN_SDPA=wave32 is unsupported for warp=%d head_dim=%u",
            gpu->warp_size, head_dim);
        return 0;
    }
    int use_wave32 = mode != H3_VDN_SDPA_SCALAR && !legacy_force_scalar &&
                     gpu->warp_size == 32 && head_dim <= 256;
    if (use_wave32) {
        const hip_bfloat16 *query_data =
            static_cast<const hip_bfloat16 *>(query->data);
        const hip_bfloat16 *key_data =
            static_cast<const hip_bfloat16 *>(key->data);
        const hip_bfloat16 *value_data =
            static_cast<const hip_bfloat16 *>(value->data);
        hip_bfloat16 *output_data =
            static_cast<hip_bfloat16 *>(output->data);
        if (head_dim == 128) {
            if (reload_query && (scan_mask || !anchor_both)) {
                h3_hip_launch_vdn_window_sdpa_bf16_wave32<
                    true, false, false>(
                    gpu->stream, query_data, key_data, value_data, output_data,
                    sequence, heads, head_dim, video_start, video_end, frames,
                    tokens_per_frame, radius, chunk, anchor_both, scale);
            } else if (reload_query) {
                h3_hip_launch_vdn_window_sdpa_bf16_wave32<true, false, true>(
                    gpu->stream, query_data, key_data, value_data, output_data,
                    sequence, heads, head_dim, video_start, video_end, frames,
                    tokens_per_frame, radius, chunk, anchor_both, scale);
            } else if (scan_mask || !anchor_both) {
                h3_hip_launch_vdn_window_sdpa_bf16_wave32<true, true, false>(
                    gpu->stream, query_data, key_data, value_data, output_data,
                    sequence, heads, head_dim, video_start, video_end, frames,
                    tokens_per_frame, radius, chunk, anchor_both, scale);
            } else {
                if (lane0_softmax) {
                    h3_hip_launch_vdn_window_sdpa_bf16_wave32<
                        true, true, true>(
                        gpu->stream, query_data, key_data, value_data,
                        output_data, sequence, heads, head_dim, video_start,
                        video_end, frames, tokens_per_frame, radius, chunk,
                        anchor_both, scale);
                } else {
                    h3_hip_launch_vdn_window_sdpa_bf16_wave32<
                        true, true, true, true>(
                        gpu->stream, query_data, key_data, value_data,
                        output_data, sequence, heads, head_dim, video_start,
                        video_end, frames, tokens_per_frame, radius, chunk,
                        anchor_both, scale);
                }
            }
        } else {
            h3_hip_launch_vdn_window_sdpa_bf16_wave32<false>(
                gpu->stream, query_data, key_data, value_data, output_data,
                sequence, heads, head_dim, video_start, video_end, frames,
                tokens_per_frame, radius, chunk, anchor_both, scale);
        }
    } else {
        size_t shared_bytes = (H3_HIP_THREADS + 3) * sizeof(float);
        hipLaunchKernelGGL(h3_hip_vdn_window_sdpa_bf16_scalar_kernel,
                           dim3(sequence, heads), dim3(H3_HIP_THREADS),
                           shared_bytes, gpu->stream,
                           static_cast<const hip_bfloat16 *>(query->data),
                           static_cast<const hip_bfloat16 *>(key->data),
                           static_cast<const hip_bfloat16 *>(value->data),
                           static_cast<hip_bfloat16 *>(output->data), sequence,
                           heads, head_dim, video_start, video_end, frames,
                           tokens_per_frame, radius, chunk, anchor_both, scale);
    }
    if (!h3_gpu_kernel_enqueued(gpu, __func__)) return 0;
    gpu->stats.mps_sdpa_dispatches++;
    return 1;
}

extern "C" int h3_gpu_vdn_softmax_gate_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *attended,
                     const h3_gpu_tensor *gate_logits,
                     uint32_t rows, uint32_t heads,
                     uint32_t head_dim) {
    size_t token_heads, elements;
    if (!h3_gpu_require_compute(gpu, __func__) || !heads || !head_dim ||
        !h3_gpu_count_2d(gpu, rows, heads, &token_heads,
                         "VDN gate logits") || token_heads > UINT32_MAX ||
        token_heads > SIZE_MAX / head_dim ||
        (elements = token_heads * head_dim) > UINT32_MAX ||
        !h3_gpu_require_tensor(gpu, attended, elements, H3_GPU_BF16,
                               "VDN attended values") ||
        !h3_gpu_require_tensor(gpu, gate_logits, token_heads, H3_GPU_BF16,
                               "VDN gate logits") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_BF16,
                               "VDN gated output"))
        return 0;
    hipLaunchKernelGGL(h3_hip_vdn_softmax_gate_bf16_kernel,
                       h3_gpu_grid_1d(static_cast<uint32_t>(elements)),
                       dim3(H3_HIP_THREADS), 0, gpu->stream,
                       static_cast<const hip_bfloat16 *>(attended->data),
                       static_cast<const hip_bfloat16 *>(gate_logits->data),
                       static_cast<hip_bfloat16 *>(output->data),
                       static_cast<uint32_t>(elements), head_dim);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_vdn_linear_features_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *query,
                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                     const h3_gpu_tensor *query_raw,
                     const h3_gpu_tensor *key_raw,
                     const h3_gpu_tensor *value_raw,
                     const h3_gpu_tensor *k_spatial,
                     const h3_gpu_tensor *k_temporal,
                     const h3_gpu_tensor *v_spatial,
                     const h3_gpu_tensor *v_temporal,
                     uint32_t frames, uint32_t frame_height,
                     uint32_t frame_width, uint32_t heads,
                     uint32_t head_dim, float epsilon) {
    uint64_t rows_wide = (uint64_t)frames * frame_height * frame_width;
    uint64_t channels_wide = (uint64_t)heads * head_dim;
    uint64_t elements_wide = rows_wide * channels_wide;
    if (!h3_gpu_require_compute(gpu, __func__) || !frames || !frame_height ||
        !frame_width || !heads || !head_dim || head_dim > H3_HIP_THREADS ||
        rows_wide > UINT32_MAX || channels_wide > UINT32_MAX ||
        elements_wide > SIZE_MAX || epsilon < 0.0f ||
        !h3_gpu_require_tensor(gpu, query_raw,
                               static_cast<size_t>(elements_wide),
                               H3_GPU_BF16, "VDN raw linear query") ||
        !h3_gpu_require_tensor(gpu, key_raw,
                               static_cast<size_t>(elements_wide),
                               H3_GPU_BF16, "VDN raw linear key") ||
        !h3_gpu_require_tensor(gpu, value_raw,
                               static_cast<size_t>(elements_wide),
                               H3_GPU_BF16, "VDN raw linear value") ||
        !h3_gpu_require_tensor(gpu, k_spatial,
                               static_cast<size_t>(channels_wide) * 25,
                               H3_GPU_BF16, "VDN K spatial convolution") ||
        !h3_gpu_require_tensor(gpu, k_temporal,
                               static_cast<size_t>(channels_wide) * 5,
                               H3_GPU_BF16, "VDN K temporal convolution") ||
        !h3_gpu_require_tensor(gpu, v_spatial,
                               static_cast<size_t>(channels_wide) * 25,
                               H3_GPU_BF16, "VDN V spatial convolution") ||
        !h3_gpu_require_tensor(gpu, v_temporal,
                               static_cast<size_t>(channels_wide) * 5,
                               H3_GPU_BF16, "VDN V temporal convolution") ||
        !h3_gpu_require_tensor(gpu, query,
                               static_cast<size_t>(elements_wide),
                               H3_GPU_BF16, "VDN linear query") ||
        !h3_gpu_require_tensor(gpu, key,
                               static_cast<size_t>(elements_wide),
                               H3_GPU_BF16, "VDN linear key") ||
        !h3_gpu_require_tensor(gpu, value,
                               static_cast<size_t>(elements_wide),
                               H3_GPU_BF16, "VDN linear value"))
        return 0;
    hipLaunchKernelGGL(h3_hip_vdn_linear_features_bf16_kernel,
                       dim3(static_cast<uint32_t>(rows_wide), heads),
                       dim3(H3_HIP_THREADS),
                       H3_HIP_THREADS * sizeof(float) * 2, gpu->stream,
                       static_cast<const hip_bfloat16 *>(query_raw->data),
                       static_cast<const hip_bfloat16 *>(key_raw->data),
                       static_cast<const hip_bfloat16 *>(value_raw->data),
                       static_cast<const hip_bfloat16 *>(k_spatial->data),
                       static_cast<const hip_bfloat16 *>(k_temporal->data),
                       static_cast<const hip_bfloat16 *>(v_spatial->data),
                       static_cast<const hip_bfloat16 *>(v_temporal->data),
                       static_cast<hip_bfloat16 *>(query->data),
                       static_cast<hip_bfloat16 *>(key->data),
                       static_cast<hip_bfloat16 *>(value->data), frames,
                       frame_height, frame_width, heads, head_dim, epsilon);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_vdn_text_features_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *key,
                     h3_gpu_tensor *value,
                     const h3_gpu_tensor *key_raw,
                     const h3_gpu_tensor *value_raw,
                     uint32_t rows, uint32_t heads,
                     uint32_t head_dim, float epsilon) {
    uint64_t elements_wide = (uint64_t)rows * heads * head_dim;
    if (!h3_gpu_require_compute(gpu, __func__) || !rows || !heads ||
        !head_dim || head_dim > H3_HIP_THREADS ||
        elements_wide > SIZE_MAX || epsilon < 0.0f ||
        !h3_gpu_require_tensor(gpu, key_raw,
                               static_cast<size_t>(elements_wide),
                               H3_GPU_BF16, "VDN raw text key") ||
        !h3_gpu_require_tensor(gpu, value_raw,
                               static_cast<size_t>(elements_wide),
                               H3_GPU_BF16, "VDN raw text value") ||
        !h3_gpu_require_tensor(gpu, key,
                               static_cast<size_t>(elements_wide),
                               H3_GPU_BF16, "VDN text key") ||
        !h3_gpu_require_tensor(gpu, value,
                               static_cast<size_t>(elements_wide),
                               H3_GPU_BF16, "VDN text value"))
        return 0;
    hipLaunchKernelGGL(h3_hip_vdn_text_features_bf16_kernel,
                       dim3(rows, heads), dim3(H3_HIP_THREADS),
                       H3_HIP_THREADS * sizeof(float), gpu->stream,
                       static_cast<const hip_bfloat16 *>(key_raw->data),
                       static_cast<const hip_bfloat16 *>(value_raw->data),
                       static_cast<hip_bfloat16 *>(key->data),
                       static_cast<hip_bfloat16 *>(value->data), rows, heads,
                       head_dim, epsilon);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_vdn_frame_stats_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *a,
                     h3_gpu_tensor *b,
                     const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value,
                     const h3_gpu_tensor *beta_logits,
                     uint32_t frames, uint32_t tokens_per_frame,
                     uint32_t heads, uint32_t head_dim) {
    uint64_t rows = (uint64_t)frames * tokens_per_frame;
    uint64_t feature_elements = rows * heads * head_dim;
    uint64_t beta_elements = rows * heads;
    uint64_t matrix_elements = (uint64_t)head_dim * head_dim;
    uint64_t output_elements = (uint64_t)frames * heads * matrix_elements;
    if (!h3_gpu_require_compute(gpu, __func__) || !frames ||
        !tokens_per_frame || !heads || !head_dim ||
        feature_elements > SIZE_MAX || beta_elements > SIZE_MAX ||
        output_elements > SIZE_MAX || output_elements > UINT32_MAX ||
        !h3_gpu_require_tensor(gpu, key,
                               static_cast<size_t>(feature_elements),
                               H3_GPU_BF16, "VDN statistics key") ||
        !h3_gpu_require_tensor(gpu, value,
                               static_cast<size_t>(feature_elements),
                               H3_GPU_BF16, "VDN statistics value") ||
        !h3_gpu_require_tensor(gpu, beta_logits,
                               static_cast<size_t>(beta_elements),
                               H3_GPU_BF16, "VDN beta logits") ||
        !h3_gpu_require_tensor(gpu, a,
                               static_cast<size_t>(output_elements),
                               H3_GPU_F32, "VDN A statistics") ||
        !h3_gpu_require_tensor(gpu, b,
                               static_cast<size_t>(output_elements),
                               H3_GPU_F32, "VDN B statistics"))
        return 0;
    uint32_t total = static_cast<uint32_t>(output_elements);
    hipLaunchKernelGGL(h3_hip_vdn_frame_stats_bf16_kernel,
                       h3_gpu_grid_1d(total), dim3(H3_HIP_THREADS), 0,
                       gpu->stream,
                       static_cast<const hip_bfloat16 *>(key->data),
                       static_cast<const hip_bfloat16 *>(value->data),
                       static_cast<const hip_bfloat16 *>(beta_logits->data),
                       static_cast<float *>(a->data),
                       static_cast<float *>(b->data), frames,
                       tokens_per_frame, heads, head_dim, matrix_elements);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_vdn_solve_f32(
                     h3_gpu *gpu, h3_gpu_tensor *transition,
                     h3_gpu_tensor *injection, h3_gpu_tensor *a,
                     const h3_gpu_tensor *b,
                     const h3_gpu_tensor *alpha,
                     uint32_t frames, uint32_t heads,
                     uint32_t head_dim) {
    uint64_t batches_wide = (uint64_t)frames * heads;
    uint64_t matrix_elements = (uint64_t)head_dim * head_dim;
    uint64_t elements_wide = batches_wide * matrix_elements;
    uint64_t alpha_elements = batches_wide * head_dim;
    if (!h3_gpu_require_compute(gpu, __func__) || !frames || !heads ||
        !head_dim || batches_wide > INT32_MAX || head_dim > INT32_MAX ||
        elements_wide > SIZE_MAX || elements_wide > UINT32_MAX ||
        alpha_elements > SIZE_MAX ||
        !h3_gpu_require_tensor(gpu, a, static_cast<size_t>(elements_wide),
                               H3_GPU_F32, "VDN A solve input") ||
        !h3_gpu_require_tensor(gpu, b, static_cast<size_t>(elements_wide),
                               H3_GPU_F32, "VDN B solve input") ||
        !h3_gpu_require_tensor(gpu, alpha,
                               static_cast<size_t>(alpha_elements),
                               H3_GPU_F32, "VDN alpha") ||
        !h3_gpu_require_tensor(gpu, transition,
                               static_cast<size_t>(elements_wide),
                               H3_GPU_F32, "VDN transition") ||
        !h3_gpu_require_tensor(gpu, injection,
                               static_cast<size_t>(elements_wide),
                               H3_GPU_F32, "VDN injection"))
        return 0;
    h3_gpu_profile_scope profile(gpu, H3_HIP_PROFILE_SOLVE);
    uint32_t batches = static_cast<uint32_t>(batches_wide);
    uint32_t total = static_cast<uint32_t>(elements_wide);
    hipLaunchKernelGGL(h3_hip_vdn_add_identity_f32_kernel,
                       h3_gpu_grid_1d(batches), dim3(H3_HIP_THREADS), 0,
                       gpu->stream, static_cast<float *>(a->data), batches,
                       head_dim);
    if (!h3_gpu_kernel_enqueued(gpu, "VDN add identity")) return 0;

    rocblas_int *device_info = nullptr;
    rocblas_int *host_info = static_cast<rocblas_int *>(
        std::calloc(batches, sizeof(*host_info)));
    if (!host_info || !h3_gpu_check(
            gpu, hipMalloc(reinterpret_cast<void **>(&device_info),
                           batches * sizeof(*device_info)),
            "allocate VDN Cholesky info")) {
        std::free(host_info);
        return 0;
    }
    rocblas_stride stride = static_cast<rocblas_stride>(matrix_elements);
    int ok = h3_gpu_check_blas(
        gpu, rocsolver_spotrf_strided_batched(
            gpu->blas, rocblas_fill_lower, static_cast<rocblas_int>(head_dim),
            static_cast<float *>(a->data), static_cast<rocblas_int>(head_dim),
            stride, device_info, static_cast<rocblas_int>(batches)),
        "VDN batched Cholesky") &&
        h3_gpu_check(gpu, hipMemcpyAsync(
            host_info, device_info, batches * sizeof(*host_info),
            hipMemcpyDeviceToHost, gpu->stream),
            "read VDN Cholesky info") &&
        h3_gpu_check(gpu, hipStreamSynchronize(gpu->stream),
                     "wait for VDN Cholesky");
    if (ok) {
        for (uint32_t index = 0; index < batches; index++)
            if (host_info[index] != 0) {
                h3_gpu_set_error(gpu,
                    "VDN Cholesky batch %u failed at leading minor %d",
                    index, host_info[index]);
                ok = 0;
                break;
            }
    }
    if (ok) {
        std::memset(host_info, 0, batches * sizeof(*host_info));
        ok = h3_gpu_check_blas(
            gpu, rocsolver_spotri_strided_batched(
                gpu->blas, rocblas_fill_lower,
                static_cast<rocblas_int>(head_dim),
                static_cast<float *>(a->data),
                static_cast<rocblas_int>(head_dim), stride, device_info,
                static_cast<rocblas_int>(batches)),
            "VDN batched Cholesky inverse") &&
            h3_gpu_check(gpu, hipMemcpyAsync(
                host_info, device_info, batches * sizeof(*host_info),
                hipMemcpyDeviceToHost, gpu->stream),
                "read VDN inverse info") &&
            h3_gpu_check(gpu, hipStreamSynchronize(gpu->stream),
                         "wait for VDN inverse");
    }
    if (ok) {
        for (uint32_t index = 0; index < batches; index++)
            if (host_info[index] != 0) {
                h3_gpu_set_error(gpu,
                    "VDN inverse batch %u failed at diagonal %d",
                    index, host_info[index]);
                ok = 0;
                break;
            }
    }
    (void)hipFree(device_info);
    std::free(host_info);
    if (!ok) return 0;

    hipLaunchKernelGGL(h3_hip_vdn_symmetrize_inverse_f32_kernel,
                       h3_gpu_grid_1d(total), dim3(H3_HIP_THREADS), 0,
                       gpu->stream, static_cast<float *>(a->data),
                       elements_wide, head_dim);
    if (!h3_gpu_kernel_enqueued(gpu, "VDN inverse symmetrization")) return 0;
    hipLaunchKernelGGL(h3_hip_vdn_transition_f32_kernel,
                       h3_gpu_grid_1d(total), dim3(H3_HIP_THREADS), 0,
                       gpu->stream, static_cast<const float *>(a->data),
                       static_cast<const float *>(alpha->data),
                       static_cast<float *>(transition->data), elements_wide,
                       head_dim);
    if (!h3_gpu_kernel_enqueued(gpu, "VDN transition construction")) return 0;
    const float one = 1.0f;
    const float zero = 0.0f;
    return h3_gpu_check_blas(
        gpu, rocblas_sgemm_strided_batched(
            gpu->blas, rocblas_operation_none, rocblas_operation_none,
            static_cast<rocblas_int>(head_dim),
            static_cast<rocblas_int>(head_dim),
            static_cast<rocblas_int>(head_dim), &one,
            static_cast<const float *>(a->data),
            static_cast<rocblas_int>(head_dim), stride,
            static_cast<const float *>(b->data),
            static_cast<rocblas_int>(head_dim), stride, &zero,
            static_cast<float *>(injection->data),
            static_cast<rocblas_int>(head_dim), stride,
            static_cast<rocblas_int>(batches)),
        "VDN injection batched GEMM");
}

extern "C" int h3_gpu_vdn_frame_mean_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, size_t input_offset,
                     uint32_t frames, uint32_t tokens_per_frame,
                     uint32_t width) {
    uint64_t input_elements = (uint64_t)frames * tokens_per_frame * width;
    uint64_t output_elements = (uint64_t)frames * width;
    if (!h3_gpu_require_compute(gpu, __func__) || !frames ||
        !tokens_per_frame || !width || output_elements > UINT32_MAX ||
        input_elements > SIZE_MAX || input_offset > SIZE_MAX - input_elements ||
        !h3_gpu_require_tensor(gpu, input,
                               input_offset +
                               static_cast<size_t>(input_elements),
                               H3_GPU_BF16, "VDN frame-mean input") ||
        !h3_gpu_require_tensor(gpu, output,
                               static_cast<size_t>(output_elements),
                               H3_GPU_F32, "VDN frame mean"))
        return 0;
    uint32_t total = static_cast<uint32_t>(output_elements);
    hipLaunchKernelGGL(h3_hip_vdn_frame_mean_bf16_kernel,
                       h3_gpu_grid_1d(total), dim3(H3_HIP_THREADS), 0,
                       gpu->stream,
                       static_cast<const hip_bfloat16 *>(input->data),
                       input_offset, static_cast<float *>(output->data),
                       tokens_per_frame, width, output_elements);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_vdn_alpha_f32(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *delta,
                     const h3_gpu_tensor *dt_bias,
                     const h3_gpu_tensor *a_log,
                     uint32_t frames, uint32_t heads,
                     uint32_t head_dim) {
    uint64_t channels = (uint64_t)heads * head_dim;
    uint64_t elements = (uint64_t)frames * channels;
    if (!h3_gpu_require_compute(gpu, __func__) || !frames || !heads ||
        !head_dim || channels > SIZE_MAX || elements > UINT32_MAX ||
        !h3_gpu_require_tensor(gpu, delta, static_cast<size_t>(elements),
                               H3_GPU_F32, "VDN alpha delta") ||
        !h3_gpu_require_tensor(gpu, dt_bias,
                               static_cast<size_t>(channels), H3_GPU_BF16,
                               "VDN alpha dt bias") ||
        !h3_gpu_require_tensor(gpu, a_log, heads, H3_GPU_BF16,
                               "VDN alpha A log") ||
        !h3_gpu_require_tensor(gpu, output, static_cast<size_t>(elements),
                               H3_GPU_F32, "VDN alpha"))
        return 0;
    uint32_t total = static_cast<uint32_t>(elements);
    hipLaunchKernelGGL(h3_hip_vdn_alpha_f32_kernel,
                       h3_gpu_grid_1d(total), dim3(H3_HIP_THREADS), 0,
                       gpu->stream, static_cast<const float *>(delta->data),
                       static_cast<const hip_bfloat16 *>(dt_bias->data),
                       static_cast<const hip_bfloat16 *>(a_log->data),
                       static_cast<float *>(output->data), elements, heads,
                       head_dim);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_vdn_scan_f32(
                     h3_gpu *gpu, h3_gpu_tensor *prefix,
                     h3_gpu_tensor *suffix,
                     const h3_gpu_tensor *transition,
                     const h3_gpu_tensor *injection,
                     const h3_gpu_tensor *text_state,
                     float text_state_scale,
                     uint32_t frames, uint32_t heads,
                     uint32_t head_dim) {
    uint64_t matrix_elements = (uint64_t)head_dim * head_dim;
    uint64_t head_bank_elements = (uint64_t)heads * matrix_elements;
    uint64_t elements = (uint64_t)frames * head_bank_elements;
    if (!h3_gpu_require_compute(gpu, __func__) || !frames || !heads ||
        !head_dim || heads > INT32_MAX || head_dim > INT32_MAX ||
        elements > SIZE_MAX || head_bank_elements > SIZE_MAX ||
        !std::isfinite(text_state_scale) ||
        !h3_gpu_require_tensor(gpu, transition,
                               static_cast<size_t>(elements), H3_GPU_F32,
                               "VDN scan transition") ||
        !h3_gpu_require_tensor(gpu, injection,
                               static_cast<size_t>(elements), H3_GPU_F32,
                               "VDN scan injection") ||
        !h3_gpu_require_tensor(gpu, text_state,
                               static_cast<size_t>(head_bank_elements),
                               H3_GPU_F32, "VDN scan text state") ||
        !h3_gpu_require_tensor(gpu, prefix, static_cast<size_t>(elements),
                               H3_GPU_F32, "VDN prefix states") ||
        !h3_gpu_require_tensor(gpu, suffix, static_cast<size_t>(elements),
                               H3_GPU_F32, "VDN suffix states"))
        return 0;
    h3_gpu_profile_scope profile(gpu, H3_HIP_PROFILE_SCAN);
    size_t bank_bytes = static_cast<size_t>(head_bank_elements) * sizeof(float);
    rocblas_stride stride = static_cast<rocblas_stride>(matrix_elements);
    const float one = 1.0f;
    for (uint32_t frame = 0; frame < frames; frame++) {
        size_t offset = static_cast<size_t>(frame) * head_bank_elements;
        if (!h3_gpu_check(gpu, hipMemcpyAsync(
                static_cast<float *>(prefix->data) + offset,
                static_cast<const float *>(injection->data) + offset,
                bank_bytes, hipMemcpyDeviceToDevice, gpu->stream),
                "copy VDN prefix injection")) return 0;
        const float *state = frame ?
            static_cast<const float *>(prefix->data) +
                offset - static_cast<size_t>(head_bank_elements) :
            static_cast<const float *>(text_state->data);
        const float initial_scale = frame ? one : text_state_scale;
        if (!h3_gpu_check_blas(gpu, rocblas_sgemm_strided_batched(
                gpu->blas, rocblas_operation_none, rocblas_operation_none,
                static_cast<rocblas_int>(head_dim),
                static_cast<rocblas_int>(head_dim),
                static_cast<rocblas_int>(head_dim), &initial_scale,
                static_cast<const float *>(transition->data) + offset,
                static_cast<rocblas_int>(head_dim), stride, state,
                static_cast<rocblas_int>(head_dim), stride, &one,
                static_cast<float *>(prefix->data) + offset,
                static_cast<rocblas_int>(head_dim), stride,
                static_cast<rocblas_int>(heads)),
                "VDN prefix scan GEMM")) return 0;
    }
    for (uint32_t reverse = frames; reverse > 0; reverse--) {
        uint32_t frame = reverse - 1;
        size_t offset = static_cast<size_t>(frame) * head_bank_elements;
        if (!h3_gpu_check(gpu, hipMemcpyAsync(
                static_cast<float *>(suffix->data) + offset,
                static_cast<const float *>(injection->data) + offset,
                bank_bytes, hipMemcpyDeviceToDevice, gpu->stream),
                "copy VDN suffix injection")) return 0;
        const float *state = frame + 1 < frames ?
            static_cast<const float *>(suffix->data) +
                offset + static_cast<size_t>(head_bank_elements) :
            static_cast<const float *>(text_state->data);
        const float initial_scale = frame + 1 < frames ? one : text_state_scale;
        if (!h3_gpu_check_blas(gpu, rocblas_sgemm_strided_batched(
                gpu->blas, rocblas_operation_none, rocblas_operation_none,
                static_cast<rocblas_int>(head_dim),
                static_cast<rocblas_int>(head_dim),
                static_cast<rocblas_int>(head_dim), &initial_scale,
                static_cast<const float *>(transition->data) + offset,
                static_cast<rocblas_int>(head_dim), stride, state,
                static_cast<rocblas_int>(head_dim), stride, &one,
                static_cast<float *>(suffix->data) + offset,
                static_cast<rocblas_int>(head_dim), stride,
                static_cast<rocblas_int>(heads)),
                "VDN suffix scan GEMM")) return 0;
    }
    return 1;
}

extern "C" int h3_gpu_vdn_readout_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query,
                     const h3_gpu_tensor *prefix,
                     const h3_gpu_tensor *suffix,
                     const h3_gpu_tensor *alpha,
                     const h3_gpu_tensor *text_state,
                     float text_state_scale,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_logits,
                     uint32_t frames, uint32_t tokens_per_frame,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t radius, uint32_t chunk,
                     float epsilon) {
    uint64_t tokens = (uint64_t)frames * tokens_per_frame;
    uint64_t feature_elements = tokens * heads * head_dim;
    uint64_t state_elements =
        (uint64_t)frames * heads * head_dim * head_dim;
    uint64_t text_elements = (uint64_t)heads * head_dim * head_dim;
    uint64_t alpha_elements = (uint64_t)frames * heads * head_dim;
    if (!h3_gpu_require_compute(gpu, __func__) || !frames ||
        !tokens_per_frame || !heads || !head_dim ||
        head_dim > H3_HIP_THREADS || tokens > UINT32_MAX ||
        feature_elements > SIZE_MAX || state_elements > SIZE_MAX ||
        text_elements > SIZE_MAX || alpha_elements > SIZE_MAX ||
        !std::isfinite(text_state_scale) || epsilon < 0.0f ||
        !h3_gpu_require_tensor(gpu, query,
                               static_cast<size_t>(feature_elements),
                               H3_GPU_BF16, "VDN readout query") ||
        !h3_gpu_require_tensor(gpu, prefix,
                               static_cast<size_t>(state_elements),
                               H3_GPU_F32, "VDN prefix states") ||
        !h3_gpu_require_tensor(gpu, suffix,
                               static_cast<size_t>(state_elements),
                               H3_GPU_F32, "VDN suffix states") ||
        !h3_gpu_require_tensor(gpu, alpha,
                               static_cast<size_t>(alpha_elements),
                               H3_GPU_F32, "VDN readout alpha") ||
        !h3_gpu_require_tensor(gpu, text_state,
                               static_cast<size_t>(text_elements),
                               H3_GPU_F32, "VDN readout text state") ||
        !h3_gpu_require_tensor(gpu, norm_weight, head_dim, H3_GPU_BF16,
                               "VDN readout norm") ||
        !h3_gpu_require_tensor(gpu, gate_logits,
                               static_cast<size_t>(feature_elements),
                               H3_GPU_BF16, "VDN output-gate logits") ||
        !h3_gpu_require_tensor(gpu, output,
                               static_cast<size_t>(feature_elements),
                               H3_GPU_BF16, "VDN linear readout"))
        return 0;
    size_t shared_bytes = (static_cast<size_t>(head_dim) * 2 +
                           H3_HIP_THREADS) * sizeof(float);
    hipLaunchKernelGGL(h3_hip_vdn_readout_bf16_kernel,
                       dim3(static_cast<uint32_t>(tokens), heads),
                       dim3(H3_HIP_THREADS), shared_bytes, gpu->stream,
                       static_cast<const hip_bfloat16 *>(query->data),
                       static_cast<const float *>(prefix->data),
                       static_cast<const float *>(suffix->data),
                       static_cast<const float *>(alpha->data),
                       static_cast<const float *>(text_state->data),
                       text_state_scale,
                       static_cast<const hip_bfloat16 *>(norm_weight->data),
                       static_cast<const hip_bfloat16 *>(gate_logits->data),
                       static_cast<hip_bfloat16 *>(output->data), frames,
                       tokens_per_frame, heads, head_dim, radius, chunk,
                       epsilon);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_sdpa_bf16_head_major_output(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) {
    size_t elements;
    if (!h3_gpu_require_compute(gpu, __func__) ||
        !h3_gpu_count_2d(gpu, rows, width, &elements, "SwiGLU") ||
        elements > UINT32_MAX / 2 ||
        !h3_gpu_require_tensor(gpu, fused, elements * 2, H3_GPU_BF16,
                               "SwiGLU input") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_BF16,
                               "SwiGLU output"))
        return 0;
    uint32_t count = static_cast<uint32_t>(elements);
    hipLaunchKernelGGL(h3_hip_swiglu_bf16_kernel, h3_gpu_grid_1d(count),
                       dim3(H3_HIP_THREADS), 0, gpu->stream,
                       static_cast<const hip_bfloat16 *>(fused->data),
                       static_cast<hip_bfloat16 *>(output->data), rows, width);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu,
                             h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value,
                           uint32_t sequence, uint32_t query_heads,
                           uint32_t kv_heads, uint32_t head_dim,
                           float scale) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    if (!h3_gpu_require_compute(gpu, __func__) ||
        !h3_gpu_require_tensor(gpu, left, elements, H3_GPU_BF16,
                               "add left") ||
        !h3_gpu_require_tensor(gpu, right, elements, H3_GPU_BF16,
                               "add right") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_BF16,
                               "add output"))
        return 0;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(h3_hip_binary_kernel<hip_bfloat16>),
                       h3_gpu_grid_1d(elements), dim3(H3_HIP_THREADS), 0,
                       gpu->stream,
                       static_cast<const hip_bfloat16 *>(left->data),
                       static_cast<const hip_bfloat16 *>(right->data),
                       static_cast<hip_bfloat16 *>(output->data), elements, 0);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    if (!h3_gpu_require_compute(gpu, __func__) ||
        !h3_gpu_require_tensor(gpu, left, elements, H3_GPU_BF16,
                               "subtract left") ||
        !h3_gpu_require_tensor(gpu, right, elements, H3_GPU_BF16,
                               "subtract right") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_BF16,
                               "subtract output"))
        return 0;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(h3_hip_binary_kernel<hip_bfloat16>),
                       h3_gpu_grid_1d(elements), dim3(H3_HIP_THREADS), 0,
                       gpu->stream,
                       static_cast<const hip_bfloat16 *>(left->data),
                       static_cast<const hip_bfloat16 *>(right->data),
                       static_cast<hip_bfloat16 *>(output->data), elements, 1);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           size_t input_offset,
                           h3_gpu_tensor *original,
                           size_t original_offset,
                           h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_token_pool_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t input_rows, uint32_t rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_token_expand_delta_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents, uint32_t rows,
                           uint32_t reduced_rows, uint32_t baseline_rows,
                           uint32_t width,
                           uint32_t exact_prefix_rows,
                           float update_scale) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_token_expand_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t rows, uint32_t reduced_rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t exact_prefix_rows, float update_scale,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon) {
    return h3_gpu_unsupported(gpu, __func__);
}

extern "C" int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio) {
    if (!h3_gpu_require_compute(gpu, __func__) ||
        !sample || sample->owner != gpu || sample->dtype != H3_GPU_F32 ||
        sample_offset > sample->elements ||
        elements > sample->elements - sample_offset ||
        !h3_gpu_require_tensor(gpu, last, elements, H3_GPU_BF16,
                               "Euler last velocity") ||
        !h3_gpu_require_tensor(gpu, previous, elements, H3_GPU_BF16,
                               "Euler previous velocity")) {
        if (gpu && (!sample || sample->owner != gpu ||
                    sample->dtype != H3_GPU_F32 ||
                    sample_offset > (sample ? sample->elements : 0) ||
                    (sample && sample_offset <= sample->elements &&
                     elements > sample->elements - sample_offset)))
            h3_gpu_set_error(gpu, "invalid Euler F32 sample range");
        return 0;
    }
    hipLaunchKernelGGL(h3_hip_euler_bf16_kernel,
                       h3_gpu_grid_1d(elements), dim3(H3_HIP_THREADS), 0,
                       gpu->stream, static_cast<float *>(sample->data),
                       sample_offset,
                       static_cast<const hip_bfloat16 *>(last->data),
                       static_cast<const hip_bfloat16 *>(previous->data),
                       elements, delta, ratio);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_euler_f32(h3_gpu *gpu, h3_gpu_tensor *sample,
                      const h3_gpu_tensor *velocity, uint32_t elements,
                      float velocity_scale) {
    if (!h3_gpu_require_compute(gpu, __func__) ||
        !std::isfinite(velocity_scale) ||
        !h3_gpu_require_tensor(gpu, sample, elements, H3_GPU_F32,
                               "Euler F32 sample") ||
        !h3_gpu_require_tensor(gpu, velocity, elements, H3_GPU_F32,
                               "Euler F32 velocity"))
        return 0;
    hipLaunchKernelGGL(h3_hip_euler_f32_kernel,
                       h3_gpu_grid_1d(elements), dim3(H3_HIP_THREADS), 0,
                       gpu->stream, static_cast<float *>(sample->data),
                       static_cast<const float *>(velocity->data), elements,
                       velocity_scale);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}

extern "C" int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate,
                         const h3_gpu_tensor *up, uint32_t elements) {
    if (!h3_gpu_require_compute(gpu, __func__) ||
        !h3_gpu_require_tensor(gpu, gate, elements, H3_GPU_BF16,
                               "SiLU gate") ||
        !h3_gpu_require_tensor(gpu, up, elements, H3_GPU_BF16,
                               "SiLU up") ||
        !h3_gpu_require_tensor(gpu, output, elements, H3_GPU_BF16,
                               "SiLU product"))
        return 0;
    hipLaunchKernelGGL(h3_hip_silu_mul_bf16_kernel,
                       h3_gpu_grid_1d(elements), dim3(H3_HIP_THREADS), 0,
                       gpu->stream,
                       static_cast<const hip_bfloat16 *>(gate->data),
                       static_cast<const hip_bfloat16 *>(up->data),
                       static_cast<hip_bfloat16 *>(output->data), elements);
    return h3_gpu_kernel_enqueued(gpu, __func__);
}


#pragma clang diagnostic pop
