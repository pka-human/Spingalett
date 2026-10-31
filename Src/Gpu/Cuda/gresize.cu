/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* gresize.comp: Resize of graphs through tables of taps, the axes' taps in an odometer, the first axis outermost.
   256 threads, a grid-stride loop. */

#include "common.cuh"

KERNEL(gresize, SpgResizePush) {
    const int32_t *taps = (const int32_t *)p.taps;
    const uint32_t stride = blocks_x() * threads_x();
    for (uint32_t e = global_x(); e < p.total; e += stride) {
        uint32_t o[4], r = e;
        for (uint32_t d = 4; d-- > 1;) {
            o[d] = r % p.dims[d];
            r /= p.dims[d];
        }
        o[0] = r;
        uint32_t c[4] = {0, 0, 0, 0};
        float sum = 0.0f;
        bool outside = false;
        for (;;) {
            float w = 1.0f;
            int64_t at = 0;
            for (uint32_t d = 0; d < 4; d++) {
                const uint32_t k = p.at[d] + o[d] * p.T[d] + c[d];
                const int32_t i = taps[2u * k];
                if (i < 0) outside = true;
                at += (int64_t)i * p.xs[d];
                w *= bits_float((uint32_t)taps[2u * k + 1u]);
            }
            if (outside) break;
            sum = fma_(w, F(p.x)[at], sum);
            uint32_t d = 4;
            while (d-- > 0) {
                if (++c[d] < p.T[d]) break;
                c[d] = 0;
            }
            if (d == 0xFFFFFFFFu) break;
        }
        F(p.y)[e] = outside ? p.extrapolation : sum;
    }
}
