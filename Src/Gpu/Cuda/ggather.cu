/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* ggather.comp: Gather of graphs, y[o][j][e] = x[o][idx[j]][e], elements of SIZE bytes (spec 0: 1, 4 or 8), indices of
   type spec 1 (SPG_T_I32 or SPG_T_I64). 256 threads, a grid-stride loop. */

#include "common.cuh"

template <typename T>
DEVICE void gather_all(const SpgGatherPush &p, bool wide) {
    const uint32_t stride = blocks_x() * threads_x();
    for (uint32_t i = global_x(); i < p.total; i += stride) {
        const uint32_t e = i % p.inner, t = i / p.inner, j = t % p.count, o = t / p.count;
        int64_t k = wide ? ((const int64_t *)p.idx)[j] : (int64_t)((const int32_t *)p.idx)[j];
        if (k < 0) k += p.axis;
        T v = T();
        if (k >= 0 && k < (int64_t)p.axis) v = ((const T *)p.x)[((uint64_t)o * p.axis + (uint64_t)k) * p.inner + e];
        ((T *)p.y)[i] = v;
    }
}

KERNEL(ggather, SpgGatherPush) {
    const bool wide = spec.v[1] == SPG_T_I64;
    switch (spec.v[0]) {
        case 1u: gather_all<uint8_t>(p, wide); break;
        case 8u: gather_all<uint64_t>(p, wide); break;
        default: gather_all<uint32_t>(p, wide); break;
    }
}
