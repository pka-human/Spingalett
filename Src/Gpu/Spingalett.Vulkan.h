/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The GPU through Vulkan compute, as the rest of the library sees it: a device opened on first use
 * (the Vulkan loader is opened at run time, so the library needs no Vulkan to load or to run on the
 * CPU), buffers in device memory known to shaders by their addresses, and command buffers of
 * kernel dispatches that can be recorded once and submitted again and again.
 *
 * Kernels take their parameters as push constants: buffer addresses (64-bit, through
 * VK_KHR_buffer_device_address, core in Vulkan 1.2) and sizes, so no descriptor sets are needed.
 * Tile sizes, modes and activations are specialization constants: each set of them is a pipeline
 * of its own, made when first dispatched and kept for the life of the process.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* A buffer of device memory; host-visible ones are mapped for their whole life. */
typedef struct {
    void *buffer, *memory;          /* VkBuffer, VkDeviceMemory */
    uint64_t address;               /* VkDeviceAddress */
    void *mapped;                   /* host-visible buffers */
    size_t size;
} SpgGpuBuffer;

/* The kernels (Src/Gpu/Shaders/<name>.comp). */
typedef enum {
#define SPG_KERNEL(name) SPG_KERNEL_##name,
#include "Spingalett.Kernels.def"
#undef SPG_KERNEL
    SPG_KERNEL_COUNT
} SpgKernel;

/* Most specialization constants a kernel takes (constant_id 0 to 15) and bytes of push constants. */
#define SPG_SPEC_MAX   16u
#define SPG_PUSH_BYTES 128u

typedef struct SpgGpuCommands SpgGpuCommands;

/* Opens the device on first use (thread-safe); false when there is no usable Vulkan device. */
bool spg_gpu_open(void);
/* The device's name, or NULL without one. */
const char *spg_gpu_device_name(void);
/* Bytes of the device's largest device-local memory heap (0 without a device). */
uint64_t spg_gpu_memory(void);
/* Bytes of shared memory a workgroup may use; the subgroup size. */
uint32_t spg_gpu_shared_memory(void);
uint32_t spg_gpu_subgroup_size(void);
/* Whether the device multiplies bfloat16 cooperative matrices (16 x 16 x 16, sums in float). */
bool spg_gpu_mma_bf16(void);

bool spg_gpu_buffer_create(SpgGpuBuffer *buffer, size_t bytes, bool host_visible);
void spg_gpu_buffer_free(SpgGpuBuffer *buffer);

/* A command buffer of dispatches; record, end, then submit as often as needed. */
SpgGpuCommands *spg_gpu_commands_create(void);
void spg_gpu_commands_free(SpgGpuCommands *commands);
/* Leaves the commands out of SPINGALETT_GPU_PROFILE's times (trial runs). */
void spg_gpu_commands_untimed(SpgGpuCommands *commands);
bool spg_gpu_record_begin(SpgGpuCommands *commands);
/* spec: the kernel's specialization constants 0 .. spec_count - 1; push: its parameters; groups in
   x, y and z (nothing is recorded when one is 0) */
void spg_gpu_dispatch(SpgGpuCommands *commands, SpgKernel kernel, const uint32_t *spec, uint32_t spec_count,
                      const void *push, uint32_t push_size, uint32_t gx, uint32_t gy, uint32_t gz);
/* Every dispatch and copy recorded before it completes before any recorded after it starts, and
   their results are visible to the host once the commands have run. */
void spg_gpu_barrier(SpgGpuCommands *commands);
/* The results of what was recorded before it are visible to the host once the commands have run;
   commands submitted later need not wait for them (only their own barriers order them). */
void spg_gpu_barrier_host(SpgGpuCommands *commands);
void spg_gpu_copy(SpgGpuCommands *commands, const SpgGpuBuffer *src, size_t src_offset, const SpgGpuBuffer *dst,
                  size_t dst_offset, size_t bytes);
/* bytes and offset multiples of 4 */
void spg_gpu_fill(SpgGpuCommands *commands, const SpgGpuBuffer *dst, size_t offset, size_t bytes, uint32_t value);
/* false when recording failed (a pipeline could not be made, among others) */
bool spg_gpu_record_end(SpgGpuCommands *commands);
/* Submits the recorded commands; spg_gpu_wait blocks until they have run. A command buffer is
   submitted again only after its previous run finished (submit waits for it). */
bool spg_gpu_submit(SpgGpuCommands *commands);
bool spg_gpu_wait(SpgGpuCommands *commands);
