#include "h3_backend.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

static void h3_hip_error(char *error, size_t error_size,
                         const char *operation, hipError_t status) {
    if (error && error_size) {
        std::snprintf(error, error_size, "%s: %s", operation,
                      hipGetErrorString(status));
    }
}

extern "C" int h3_backend_device_count(char *error, size_t error_size) {
    int count = 0;
    hipError_t status = hipGetDeviceCount(&count);
    if (status != hipSuccess) {
        h3_hip_error(error, error_size, "hipGetDeviceCount", status);
        return -1;
    }
    return count;
}

extern "C" int h3_backend_probe(int device_index, h3_device_info *info,
                                char *error, size_t error_size) {
    if (!info) {
        if (error && error_size)
            std::snprintf(error, error_size, "device info output is required");
        return 0;
    }

    int count = h3_backend_device_count(error, error_size);
    if (count < 0) return 0;
    if (device_index < 0 || device_index >= count) {
        if (error && error_size) {
            std::snprintf(error, error_size,
                          "HIP device index %d is out of range (count %d)",
                          device_index, count);
        }
        return 0;
    }

    hipDeviceProp_t properties;
    hipError_t status = hipGetDeviceProperties(&properties, device_index);
    if (status != hipSuccess) {
        h3_hip_error(error, error_size, "hipGetDeviceProperties", status);
        return 0;
    }
    status = hipSetDevice(device_index);
    if (status != hipSuccess) {
        h3_hip_error(error, error_size, "hipSetDevice", status);
        return 0;
    }

    std::memset(info, 0, sizeof(*info));
    std::snprintf(info->backend, sizeof(info->backend), "hip");
    std::snprintf(info->name, sizeof(info->name), "%s", properties.name);
    std::snprintf(info->architecture, sizeof(info->architecture), "%s",
                  properties.gcnArchName[0] ? properties.gcnArchName : "unknown");
    status = hipDeviceGetPCIBusId(info->pci_bus_id,
                                  static_cast<int>(sizeof(info->pci_bus_id)),
                                  device_index);
    if (status != hipSuccess) info->pci_bus_id[0] = '\0';
    info->device_index = device_index;
    info->physical_memory = static_cast<uint64_t>(properties.totalGlobalMem);
    info->recommended_working_set = info->physical_memory > UINT64_C(2147483648)
        ? info->physical_memory - UINT64_C(2147483648)
        : info->physical_memory;
    info->max_buffer_length = info->physical_memory;
    info->unified_memory = 0;
    return 1;
}
