/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The epilogues of the products whose activation (or derivative) is no slope (act_slope()): sigmoid, tanh
   and the like. The products (gemm.cu, gemm_mma.cu) hold the sloped epilogues only; Spingalett.Cuda.c runs
   the others as the product with EPI_STORE, then this pass over its outputs in C, with the product's push
   constants and spec (EPI 7, ACT 8, PHASED 10, HALF 13: C 4, e0 8): block z the group, a grid-stride loop. */

#include "gemm_common.cuh"

KERNEL(epi, SpgGemmPush) {
    const uint32_t EPI = spec.v[7], ACT = spec.v[8], PHASED = spec.v[10], HALF = spec.v[13];
    const uint32_t coff = block_z() * p.c_group, stride = blocks_x() * 256u;
    const uint32_t *geo = U(p.geo);
    const uint32_t RH = PHASED ? geo[GEO_RH] : 1u, RW = PHASED ? geo[GEO_RW] : 1u;
    const uint64_t total = (uint64_t)p.M * p.N;
    for (uint64_t i = global_x(); i < total; i += stride) {
        const uint32_t m = (uint32_t)(i / p.N), n = (uint32_t)(i % p.N);
        uint32_t row = m;
        if (PHASED) {       /* (gemm.cu's c_row()) */
            const uint32_t CH = geo[GEO_CH], CW = geo[GEO_CW], CY = geo[GEO_CY], CX = geo[GEO_CX];
            const uint32_t CSH = geo[GEO_CS] & 0xFFFFu, CSW = geo[GEO_CS] >> 16;
            const uint32_t s = m / (RH * RW), r = m % (RH * RW);
            row = (s * CH + CY + CSH * (r / RW)) * CW + CX + CSW * (r % RW);
        }
        const uint32_t at = coff + row * p.ldc + n;
        float v = ld(p.c, at, 2u, HALF);
        if (EPI == EPI_BIAS_ACT) {
            if (p.flags & FLAG_BIAS) v += F(p.e0)[coff + n];
            v = activate(v, ACT);
        } else if (EPI == EPI_SCALE_ACT) {
            v = activate(v * F(p.e0)[coff + n] + F(p.e1)[coff + n], ACT);
        } else if (EPI == EPI_DERIV) {
            v *= derivative(ld(p.e0, at, 3u, HALF), ACT);
        }
        st(p.c, at, 2u, HALF, v);
    }
}
