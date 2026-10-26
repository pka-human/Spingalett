/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* gcvt.comp: floats of graphs copied through strides as gcopy.cu copies them, rounded to bfloat16 as they are written.
   256 threads, a grid-stride loop. */

#include "common.cuh"

KERNEL(gcvt, SpgCopyPush) {
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
        H(p.y)[ay] = (uint16_t)to_bf16(F(p.x)[ax]);
    }
}
