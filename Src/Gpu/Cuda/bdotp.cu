/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * bdotp.comp: bdot's product for weights of blocks of 32 values kept packed on the device (bmmp's layout: the codes of
 * every row, N rows of K signed bytes, then the blocks' scales as floats, N x K / 32), the CPU's bits: a warp a row of
 * weights, a lane a block of each round of 32 (its codes and the activations' in sixteen bytes' loads), the blocks'
 * integer sums by dp4a, their terms (d dx) I1, each product and sum rounded alone, added in their order through
 * shuffles; the bias then the sum, the activation of spec 2, the product by e0's value. A row of activations a pass
 * (r0); a unit a TYPE (Q8_0: bdotp_q8_0).
 */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_bdotp_q8_0
#define TYPE 0u
#endif

struct uint4_ {
    uint32_t x, y, z, w;
} __attribute__((aligned(16)));

DEVICE float shfl_f(float v, uint32_t lane) { return __nvvm_shfl_sync_idx_f32(0xFFFFFFFFu, v, lane, 0x1F); }
DEVICE int32_t dp4a(uint32_t a, uint32_t b, int32_t c) {
    int32_t d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

extern "C" __global__ void __launch_bounds__(256) SPG_ENTRY(const SpgBdotPush p, const Spec spec) {
    const uint32_t lane = thread_x() & 31u, j = block_x() * 8u + (thread_x() >> 5);
    if (j >= p.n) return;                       /* (a whole warp) */
    const uint32_t blocks = p.k / 32u;
    const uint8_t *row = (const uint8_t *)p.w + (uint64_t)j * p.k, *x = (const uint8_t *)p.q + (uint64_t)p.r0 * p.k;
    const float *scales = (const float *)(p.w + (uint64_t)p.n * p.k) + (uint64_t)j * blocks;
    const float *dx = F(p.dx) + (uint64_t)p.r0 * blocks;
    float acc = 0.0f;
    for (uint32_t b0 = 0; b0 < blocks; b0 += 32u) {
        const uint32_t b = b0 + lane;
        float term = 0.0f;
        if (b < blocks) {
            const uint4_ w0 = ((const uint4_ *)(row + 32u * b))[0], w1 = ((const uint4_ *)(row + 32u * b))[1];
            const uint4_ x0 = ((const uint4_ *)(x + 32u * b))[0], x1 = ((const uint4_ *)(x + 32u * b))[1];
            int32_t i1 = 0;
            i1 = dp4a(w0.x, x0.x, i1);
            i1 = dp4a(w0.y, x0.y, i1);
            i1 = dp4a(w0.z, x0.z, i1);
            i1 = dp4a(w0.w, x0.w, i1);
            i1 = dp4a(w1.x, x1.x, i1);
            i1 = dp4a(w1.y, x1.y, i1);
            i1 = dp4a(w1.z, x1.z, i1);
            i1 = dp4a(w1.w, x1.w, i1);
            term = (scales[b] * dx[b]) * (float)i1;
        }
        /* the round's blocks' terms in their order */
        const uint32_t count = blocks - b0 < 32u ? blocks - b0 : 32u;
        for (uint32_t g = 0; g < count; g++) acc = acc + shfl_f(term, g);
    }
    if (lane == 0u) {
        const uint32_t act = spec.v[2];
        const uint64_t at = (uint64_t)p.r0 * p.ldy + j;
        float v = F(p.bias)[j] + acc;
        if (act) v = activate(v, act);
        if (p.e0) v = v * F(p.e0)[at];
        F(p.y)[at] = v;
    }
}
