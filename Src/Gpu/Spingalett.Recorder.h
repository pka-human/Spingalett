/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Commands are recorded through a Recorder, which keeps the memory read and written since the last
 * barrier: every dispatch states what it reads and writes, and a barrier is recorded before it only
 * when it reads what an earlier one wrote or writes what an earlier one read or wrote. Independent
 * dispatches, such as a layer's weight gradient and its data gradient, then run side by side.
 * Consecutive dispatches of one nonzero `group` access the same ranges and write disjoint parts of
 * them (the phases of a data gradient): they do not wait for each other. A group belongs to one layer
 * and purpose (Spingalett.Gpu.c's group_of()), so that the next layer's dispatches, which read what these
 * wrote, wait. Shared by the executors of networks (Spingalett.Gpu.c) and of integer deployment models
 * (Spingalett.GpuModel.c).
 */

#pragma once

#include "Spingalett.Device.h"
#include <string.h>

struct SpgGpuNet;

typedef struct { uint64_t lo, hi; } Range;

#define TRACKED 96u
typedef struct {
    SpgGpuCommands *c;
    const struct SpgGpuNet *g;      /* the network whose activations may be kept as bfloat16, or NULL */
    Range reads[TRACKED], writes[TRACKED];
    uint32_t nr, nw, group;
} Recorder;

typedef struct {
    Range read[8], write[4];
    uint32_t nr, nw, group;
} Access;

static inline Range whole(const SpgGpuBuffer *b) { return (Range){b->address, b->address + b->size}; }
static inline Range span(uint64_t address, uint64_t floats) { return (Range){address, address + 4u * floats}; }

static inline void reads(Access *a, Range r) { if (r.hi > r.lo) a->read[a->nr++] = r; }
static inline void writes(Access *a, Range r) { if (r.hi > r.lo) a->write[a->nw++] = r; }

static inline bool hits(const Range *list, uint32_t n, Range r) {
    for (uint32_t k = 0; k < n; k++)
        if (r.lo < list[k].hi && list[k].lo < r.hi) return true;
    return false;
}

static inline void barrier(Recorder *r) {
    spg_gpu_barrier(r->c);
    r->nr = r->nw = 0;
    r->group = 0;
}

/* A barrier if the access conflicts with what was recorded since the last one; then it is noted. */
static inline void need(Recorder *r, const Access *a) {
    bool wait = r->nr + a->nr > TRACKED || r->nw + a->nw > TRACKED;
    if (!(a->group != 0 && a->group == r->group)) {
        for (uint32_t k = 0; k < a->nr && !wait; k++) wait = hits(r->writes, r->nw, a->read[k]);
        for (uint32_t k = 0; k < a->nw && !wait; k++)
            wait = hits(r->writes, r->nw, a->write[k]) || hits(r->reads, r->nr, a->write[k]);
    }
    if (wait) barrier(r);
    memcpy(r->reads + r->nr, a->read, a->nr * sizeof(Range));
    memcpy(r->writes + r->nw, a->write, a->nw * sizeof(Range));
    r->nr += a->nr;
    r->nw += a->nw;
    r->group = a->group;
}
