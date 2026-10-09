/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The device interface of Spingalett.Device.h over the backends built: the calling thread's backend
 * (spg_gpu_use()) for what makes objects or asks about the device, the object's own for the rest.
 * Arenas and commands are handles that name their backend; buffers carry it.
 */

#include "Spingalett/Spingalett.Config.h"
#include "Spingalett.Device.h"
#include <stdlib.h>
#include <string.h>

const char *const spg_kernel_names[SPG_KERNEL_COUNT] = {
#define SPG_KERNEL(name) #name,
#define SPG_KERNEL_H(name) #name, #name "_h",
#include "Spingalett.Kernels.def"
#undef SPG_KERNEL
#undef SPG_KERNEL_H
};

static const SpgGpuOps *backend_ops(SpgBackend backend) {
    switch (backend) {
#if defined(SPINGALETT_HAS_VULKAN)
        case SPG_BACKEND_VULKAN: return &spg_vulkan_ops;
#endif
#if defined(SPINGALETT_HAS_CUDA)
        case SPG_BACKEND_CUDA: return &spg_cuda_ops;
#endif
        default: return NULL;
    }
}

static _Thread_local SpgBackend current = SPG_BACKEND_VULKAN;

void spg_gpu_use(SpgBackend backend) {
    current = backend;
}

SpgBackend spg_gpu_using(void) {
    return current;
}

bool spg_gpu_built(SpgBackend backend) {
    return backend_ops(backend) != NULL;
}

/* An arena or commands: the backend's object and the backend. */
struct SpgGpuArena {
    const SpgGpuOps *ops;
    void *impl;
};
struct SpgGpuCommands {
    const SpgGpuOps *ops;
    void *impl;
    SpgBackend backend;
};

#define OPS backend_ops(current)

bool spg_gpu_open(void) { return OPS && OPS->open(); }
const char *spg_gpu_device_name(void) { return OPS ? OPS->device_name() : NULL; }
uint64_t spg_gpu_memory(void) { return OPS ? OPS->memory() : 0; }
uint32_t spg_gpu_shared_memory(void) { return OPS ? OPS->shared_memory() : 0; }
uint32_t spg_gpu_subgroup_size(void) { return OPS ? OPS->subgroup_size() : 32u; }
uint32_t spg_gpu_max_workgroups(uint32_t axis) { return OPS ? OPS->max_workgroups(axis) : 65535u; }
bool spg_gpu_mma_bf16(void) { return OPS && OPS->mma_bf16(); }
bool spg_gpu_bf16_storage(void) { return OPS && OPS->bf16_storage(); }
bool spg_gpu_host_writes(void) { return OPS && OPS->host_writes(); }

bool spg_gpu_buffer_create(SpgGpuBuffer *buffer, size_t bytes, bool host_visible) {
    memset(buffer, 0, sizeof *buffer);
    if (!OPS || !OPS->buffer_create(buffer, bytes, host_visible)) return false;
    buffer->backend = current;
    return true;
}

void spg_gpu_buffer_free(SpgGpuBuffer *buffer) {
    const SpgGpuOps *ops = buffer ? backend_ops(buffer->backend) : NULL;
    if (ops) ops->buffer_free(buffer);
}

SpgGpuArena *spg_gpu_arena_create(void) {
    if (!OPS) return NULL;
    SpgGpuArena *a = (SpgGpuArena *)calloc(1, sizeof *a);
    if (a && !(a->impl = OPS->arena_create())) {
        free(a);
        return NULL;
    }
    if (a) a->ops = OPS;
    return a;
}

void spg_gpu_arena_add(SpgGpuArena *arena, SpgGpuBuffer *buffer, size_t bytes, SpgMemory memory) {
    arena->ops->arena_add(arena->impl, buffer, bytes, memory);
}

bool spg_gpu_arena_commit(SpgGpuArena *arena) {
    return arena->ops->arena_commit(arena->impl);
}

void spg_gpu_arena_free(SpgGpuArena *arena) {
    if (!arena) return;
    arena->ops->arena_free(arena->impl);
    free(arena);
}

SpgGpuCommands *spg_gpu_commands_create(void) {
    if (!OPS) return NULL;
    SpgGpuCommands *c = (SpgGpuCommands *)calloc(1, sizeof *c);
    if (c && !(c->impl = OPS->commands_create())) {
        free(c);
        return NULL;
    }
    if (c) {
        c->ops = OPS;
        c->backend = current;
    }
    return c;
}

void spg_gpu_commands_free(SpgGpuCommands *c) {
    if (!c) return;
    c->ops->commands_free(c->impl);
    free(c);
}

void spg_gpu_commands_untimed(SpgGpuCommands *c) { c->ops->commands_untimed(c->impl); }
bool spg_gpu_commands_stamps(SpgGpuCommands *c, uint32_t count) { return c->ops->commands_stamps(c->impl, count); }
void spg_gpu_timestamp(SpgGpuCommands *c, uint32_t index) { c->ops->timestamp(c->impl, index); }
bool spg_gpu_timestamps(SpgGpuCommands *c, double *ns, uint32_t count) { return c->ops->timestamps(c->impl, ns, count); }
bool spg_gpu_record_begin(SpgGpuCommands *c) { return c->ops->record_begin(c->impl); }

void spg_gpu_dispatch(SpgGpuCommands *c, SpgKernel kernel, const uint32_t *spec, uint32_t spec_count, const void *push,
                      uint32_t push_size, uint32_t gx, uint32_t gy, uint32_t gz) {
    c->ops->dispatch(c->impl, kernel, spec, spec_count, push, push_size, gx, gy, gz);
}

void spg_gpu_prepare(SpgKernel kernel, const uint32_t *specs, uint32_t count, uint32_t n) {
    if (OPS) OPS->prepare(kernel, specs, count, n);
}

void spg_gpu_barrier(SpgGpuCommands *c) { c->ops->barrier(c->impl); }
void spg_gpu_barrier_host(SpgGpuCommands *c) { c->ops->barrier_host(c->impl); }

void spg_gpu_copy(SpgGpuCommands *c, const SpgGpuBuffer *src, size_t src_offset, const SpgGpuBuffer *dst,
                  size_t dst_offset, size_t bytes) {
    c->ops->copy(c->impl, src, src_offset, dst, dst_offset, bytes);
}

void spg_gpu_fill(SpgGpuCommands *c, const SpgGpuBuffer *dst, size_t offset, size_t bytes, uint32_t value) {
    c->ops->fill(c->impl, dst, offset, bytes, value);
}

bool spg_gpu_record_end(SpgGpuCommands *c) { return c->ops->record_end(c->impl); }
bool spg_gpu_submit(SpgGpuCommands *c) { return c->ops->submit(c->impl); }
bool spg_gpu_wait(SpgGpuCommands *c) { return c->ops->wait(c->impl); }
