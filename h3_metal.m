#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "h3_backend.h"

#include <stdio.h>
#include <string.h>

static void h3_copy_string(char *destination, size_t size, NSString *value) {
    if (!destination || !size) return;
    const char *source = value ? value.UTF8String : "unknown";
    snprintf(destination, size, "%s", source ? source : "unknown");
}

int h3_backend_device_count(char *error, size_t error_size) {
    (void)error;
    (void)error_size;
    @autoreleasepool {
        return (int)MTLCopyAllDevices().count;
    }
}

int h3_backend_probe(int device_index, h3_device_info *info,
                     char *error, size_t error_size) {
    if (!info) return 0;
    memset(info, 0, sizeof(*info));
    @autoreleasepool {
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        if (device_index < 0 || (NSUInteger)device_index >= devices.count) {
            if (error && error_size)
                snprintf(error, error_size,
                         "Metal device index %d is out of range (count %zu)",
                         device_index, (size_t)devices.count);
            return 0;
        }
        id<MTLDevice> device = devices[(NSUInteger)device_index];
        snprintf(info->backend, sizeof(info->backend), "metal");
        info->device_index = device_index;
        h3_copy_string(info->name, sizeof(info->name), device.name);
        if (@available(macOS 14.0, *)) {
            h3_copy_string(info->architecture, sizeof(info->architecture),
                           device.architecture.name);
        } else {
            h3_copy_string(info->architecture, sizeof(info->architecture), @"unknown");
        }
        info->physical_memory = NSProcessInfo.processInfo.physicalMemory;
        info->recommended_working_set = device.recommendedMaxWorkingSetSize;
        info->max_buffer_length = device.maxBufferLength;
        info->unified_memory = device.hasUnifiedMemory ? 1 : 0;
        for (int family = 10; family >= 1; family--) {
            MTLGPUFamily candidate = (MTLGPUFamily)(1000 + family);
            if ([device supportsFamily:candidate]) {
                info->apple_gpu_family = family;
                break;
            }
        }
        if (@available(macOS 26.0, *)) {
            info->metal4 = [device supportsFamily:MTLGPUFamilyMetal4] ? 1 : 0;
        }
    }
    return 1;
}
