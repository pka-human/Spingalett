/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* gcopy.comp: values of graphs copied through strides, elements of SIZE bytes (spec 0: 1, 2, 4, 8 or 16). 256 threads,
   a grid-stride loop. */

#include "common.cuh"

struct int4_ {
    int32_t x, y, z, w;
} __attribute__((aligned(16)));

template <typename T>
DEVICE void copy_all(const SpgCopyPush &p) {
    const uint32_t stride = blocks_x() * threads_x();
    for (uint32_t i = global_x(); i < p.total; i += stride) {
        uint32_t r = i;
        int64_t ax = p.base, ay = 0;
#pragma unroll
        for (uint32_t d = SPG_COPY_AXES; d-- > 0;) {
            const uint32_t c = d ? r % p.dims[d] : r;
            if (d) r /= p.dims[d];
            ax += (int64_t)c * p.sx[d];
            ay += (int64_t)c * p.sy[d];
        }
        ((T *)p.y)[ay] = ((const T *)p.x)[ax];
    }
}

KERNEL(gcopy, SpgCopyPush) {
    switch (spec.v[0]) {
        case 1u: copy_all<uint8_t>(p); break;
        case 2u: copy_all<uint16_t>(p); break;
        case 8u: copy_all<uint64_t>(p); break;
        case 16u: copy_all<int4_>(p); break;
        default: copy_all<uint32_t>(p); break;
    }
}
