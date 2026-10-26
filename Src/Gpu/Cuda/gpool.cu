/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* gpool.comp: MaxPool, AveragePool and LpPool of graphs over channels-first maps, an output value a thread, its window
   in order (as the CPU's k_pool). 256 threads, a grid-stride loop. */

#include "common.cuh"

KERNEL(gpool, SpgPoolGPush) {
    const bool max = (p.flags & SPG_GPOOL_MAX) != 0u, lp = (p.flags & SPG_GPOOL_LP) != 0u;
    const uint32_t opix = p.out[0] * p.out[1] * p.out[2], ipix = p.in[0] * p.in[1] * p.in[2];
    const uint32_t stride = blocks_x() * threads_x();
    for (uint32_t e = global_x(); e < p.total; e += stride) {
        const uint32_t q = e / opix, o = e % opix;
        const uint32_t o2 = o % p.out[2], o1 = (o / p.out[2]) % p.out[1], o0 = o / (p.out[1] * p.out[2]);
        const float *x = F(p.x) + (uint64_t)q * ipix;
        float acc = max ? -__builtin_inff() : 0.0f;
        bool any = false;
        uint32_t count = 0, padded = 0;
        for (uint32_t k0 = 0; k0 < p.k[0]; k0++) {
            const int32_t i0 = (int32_t)(o0 * p.s[0] + k0 * p.dil[0]) - (int32_t)p.pb[0];
            for (uint32_t k1 = 0; k1 < p.k[1]; k1++) {
                const int32_t i1 = (int32_t)(o1 * p.s[1] + k1 * p.dil[1]) - (int32_t)p.pb[1];
                for (uint32_t k2 = 0; k2 < p.k[2]; k2++) {
                    const int32_t i2 = (int32_t)(o2 * p.s[2] + k2 * p.dil[2]) - (int32_t)p.pb[2];
                    padded += i0 < (int32_t)(p.in[0] + p.pe[0]) && i1 < (int32_t)(p.in[1] + p.pe[1]) &&
                              i2 < (int32_t)(p.in[2] + p.pe[2]);
                    if (i0 < 0 || i0 >= (int32_t)p.in[0] || i1 < 0 || i1 >= (int32_t)p.in[1] || i2 < 0 ||
                        i2 >= (int32_t)p.in[2])
                        continue;
                    const float v = x[((uint32_t)i0 * p.in[1] + (uint32_t)i1) * p.in[2] + (uint32_t)i2];
                    if (max) {
                        if (!any || v > acc || (isnan_(v) && !isnan_(acc))) acc = v;
                        any = true;
                    } else {
                        acc += lp ? __nvvm_ex2_approx_f(p.p * __nvvm_lg2_approx_f(abs_(v))) : v;
                    }
                    count++;
                }
            }
        }
        float y = acc;
        if (lp) y = acc > 0.0f ? __nvvm_ex2_approx_f(__nvvm_lg2_approx_f(acc) / p.p) : 0.0f;
        else if (!max) y = acc / (float)((p.flags & SPG_GPOOL_INCLUDE) ? padded : count ? count : 1u);
        F(p.y)[e] = y;
    }
}
