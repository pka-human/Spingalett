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

/* A buffer of device memory; host-visible ones are mapped for their whole life. The buffers of an
   arena are ranges of a VkBuffer they share, from `offset` on (their memory belongs to the arena). */
typedef struct {
    void *buffer, *memory;          /* VkBuffer, VkDeviceMemory (NULL in an arena) */
    uint64_t address;               /* VkDeviceAddress */
    void *mapped;                   /* host-visible buffers */
    size_t size, offset;
} SpgGpuBuffer;

/* Where a buffer's memory is: on the device; visible to the host and cached where the device has
   such (results read back); or on the device and visible to the host, for data the host writes and
   never reads, which the device then reads at its own memory's speed (only where
   spg_gpu_host_writes()). */
typedef enum { SPG_MEMORY_DEVICE, SPG_MEMORY_HOST, SPG_MEMORY_HOST_WRITES } SpgMemory;

/* Buffers made together, in a few allocations: spg_gpu_arena_add() asks for one, whose struct
   spg_gpu_arena_commit() fills, and spg_gpu_arena_free() frees them all. An allocation of its own per
   buffer costs a vkAllocateMemory and a vkFreeMemory, about 0.15 ms each on some drivers, and a
   network on the GPU has dozens of buffers. */
typedef struct SpgGpuArena SpgGpuArena;

/* The kernels (Src/Gpu/Shaders/<name>.comp); a kernel's bfloat16 variant, SPG_KERNEL_<name>_h, follows
   it. */
typedef enum {
#define SPG_KERNEL(name) SPG_KERNEL_##name,
#define SPG_KERNEL_H(name) SPG_KERNEL_##name, SPG_KERNEL_##name##_h,
#include "Spingalett.Kernels.def"
#undef SPG_KERNEL
#undef SPG_KERNEL_H
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
/* Workgroups a dispatch may have along axis 0 (x), 1 (y) or 2 (z): 65535 at least. */
uint32_t spg_gpu_max_workgroups(uint32_t axis);
/* Whether the device multiplies bfloat16 cooperative matrices (16 x 16 x 16, sums in float). */
bool spg_gpu_mma_bf16(void);
/* Whether kernels may keep bfloat16 values in memory (the matrix units, and 16-bit storage). */
bool spg_gpu_bf16_storage(void);
/* Whether the host can write into all of the device's memory (resizable BAR, unified memory), for
   SPG_MEMORY_HOST_WRITES: the largest device-local heap's memory is host-visible too. */
bool spg_gpu_host_writes(void);

bool spg_gpu_buffer_create(SpgGpuBuffer *buffer, size_t bytes, bool host_visible);
void spg_gpu_buffer_free(SpgGpuBuffer *buffer);
SpgGpuArena *spg_gpu_arena_create(void);
/* bytes 0 asks for nothing (the buffer stays empty) */
void spg_gpu_arena_add(SpgGpuArena *arena, SpgGpuBuffer *buffer, size_t bytes, SpgMemory memory);
/* false when memory ran out (no buffer is made then) */
bool spg_gpu_arena_commit(SpgGpuArena *arena);
void spg_gpu_arena_free(SpgGpuArena *arena);

/* A command buffer of dispatches; record, end, then submit as often as needed. */
SpgGpuCommands *spg_gpu_commands_create(void);
void spg_gpu_commands_free(SpgGpuCommands *commands);
/* Leaves the commands out of SPINGALETT_GPU_PROFILE's times (trial runs). */
void spg_gpu_commands_untimed(SpgGpuCommands *commands);
/* Timestamps for measurements of the commands' own: room for `count` (false without timestamps on
   the device's queue); spg_gpu_timestamp() records index once everything recorded before it has
   run, and spg_gpu_timestamps() reads them in nanoseconds after spg_gpu_wait(). */
bool spg_gpu_commands_stamps(SpgGpuCommands *commands, uint32_t count);
void spg_gpu_timestamp(SpgGpuCommands *commands, uint32_t index);
bool spg_gpu_timestamps(SpgGpuCommands *commands, double *ns, uint32_t count);
bool spg_gpu_record_begin(SpgGpuCommands *commands);
/* spec: the kernel's specialization constants 0 .. spec_count - 1; push: its parameters; groups in
   x, y and z (nothing is recorded when one is 0) */
void spg_gpu_dispatch(SpgGpuCommands *commands, SpgKernel kernel, const uint32_t *spec, uint32_t spec_count,
                      const void *push, uint32_t push_size, uint32_t gx, uint32_t gy, uint32_t gz);
/* Makes the pipelines of n sets of specialization constants (specs: n x count) that do not exist yet,
   on several threads, ahead of the dispatches that will use them. */
void spg_gpu_prepare(SpgKernel kernel, const uint32_t *specs, uint32_t count, uint32_t n);
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
