/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* wtrans.comp: a convolution's weights regrouped for its data gradient. 256 threads, a grid-stride loop. */

#include "common.cuh"

KERNEL(wtrans, SpgWtransPush) {
    const uint32_t stride = blocks_x() * 256u;
    for (uint32_t i = global_x(); i < p.total; i += stride) {
        const uint32_t c = i % p.CG, q = i / p.CG, tap = q % p.taps, row = q / p.taps;
        const uint32_t g = row / p.OG, f = row % p.OG;
        F(p.wt)[((g * p.taps + U(p.order)[tap]) * p.OG + f) * p.CG + c] = F(p.w)[i];
    }
}
