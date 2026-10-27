/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * bdotn.comp: bdot's product for weights of blocks of 32 4-bit codes (Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL) packed as sessions
 * keep them (Spingalett.GpuPack.c: rows `stride` bytes apart), the CPU's bits: a warp a row of weights, a lane a block of each round of 32 (its 16
 * bytes of codes and the activations' 32 in sixteen bytes' loads): its products by dp4a (the low nibbles with values 0
 * to 15, the high ones with 16 to 31; the fifth bits at bit 4; IQ4_NL's values from its table) less the offset times
 * the rows' sum of 32 (I1), its term (d dx) I1 (Q4_1, Q5_1: + (m dx) times the sum), each product and sum rounded
 * alone, added in their order through shuffles; the bias then the sum, the activation of spec 2, the product by e0's
 * value. A row of activations a pass (r0); a unit a TYPE (bdotp_q4_0, ...).
 */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_bdotp_q4_0
#define TYPE 1u
#endif

#define ONE (TYPE == SPG_B_Q4_1 || TYPE == SPG_B_Q5_1)
#define FIVE (TYPE == SPG_B_Q5_0 || TYPE == SPG_B_Q5_1)
#define NL (TYPE == SPG_B_IQ4_NL)
#define OFF (TYPE == SPG_B_Q4_0 ? 8 : TYPE == SPG_B_Q5_0 ? 16 : 0)

struct uint4_ {
    uint32_t x, y, z, w;
} __attribute__((aligned(16)));

DEVICE float shfl_f(float v, uint32_t lane) { return __nvvm_shfl_sync_idx_f32(0xFFFFFFFFu, v, lane, 0x1F); }
DEVICE int32_t dp4a(uint32_t a, uint32_t b, int32_t c) {
    int32_t d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
DEVICE uint32_t spread1(uint32_t x) { return ((x * 0x00204081u) & 0x01010101u) << 4; }
DEVICE uint32_t nl_values(uint32_t q) {
    const uint32_t y = q | q >> 4, sel = ((y & 0xFFu) | ((y >> 8) & 0xFF00u)) & 0x7777u;
    uint32_t lo, hi;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo) : "r"(0xBFAD9881u), "r"(0xF6EADDCFu), "r"(sel));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi) : "r"(0x26190D01u), "r"(0x71594535u), "r"(sel));
    const uint32_t mask = ((q >> 3) & 0x01010101u) * 0xFFu;
    return (lo & ~mask) | (hi & mask);
}

extern "C" __global__ void __launch_bounds__(256) SPG_ENTRY(const SpgBdotPush p, const Spec spec) {
    const uint32_t lane = thread_x() & 31u, j = block_x() * 8u + (thread_x() >> 5);
    if (j >= p.n) return;                       /* (a whole warp) */
    const uint32_t blocks = p.k / 32u;
    const uint8_t *row = (const uint8_t *)p.w + (uint64_t)j * p.stride, *x = (const uint8_t *)p.q + (uint64_t)p.r0 * p.k;
    const uint32_t *fifth = (const uint32_t *)(row + p.k / 2u);
    const uint16_t *scales = (const uint16_t *)(row + p.k / 2u + (FIVE ? 4u * blocks : 0u));
    const int32_t *xs = (const int32_t *)p.xs + (uint64_t)p.r0 * (p.k / 16u);
    const float *dx = F(p.dx) + (uint64_t)p.r0 * blocks;
    float acc = 0.0f;
    for (uint32_t b0 = 0; b0 < blocks; b0 += 32u) {
        const uint32_t b = b0 + lane;
        float term = 0.0f;
        if (b < blocks) {
            const uint4_ w = *(const uint4_ *)(row + 16u * b);
            const uint4_ x0 = ((const uint4_ *)(x + 32u * b))[0], x1 = ((const uint4_ *)(x + 32u * b))[1];
            const uint32_t words[4] = {w.x, w.y, w.z, w.w}, xa[4] = {x0.x, x0.y, x0.z, x0.w}, xb[4] = {x1.x, x1.y, x1.z, x1.w};
            const uint32_t f = FIVE ? fifth[b] : 0u;
            int32_t s = 0;
#pragma unroll
            for (uint32_t k = 0; k < 4u; k++) {
                uint32_t lo = words[k] & 0x0F0F0F0Fu, hi = (words[k] >> 4) & 0x0F0F0F0Fu;
                if (FIVE) {
                    lo |= spread1((f >> (4u * k)) & 0xFu);
                    hi |= spread1((f >> (16u + 4u * k)) & 0xFu);
                }
                if (NL) {
                    lo = nl_values(lo);
                    hi = nl_values(hi);
                }
                s = dp4a(lo, xa[k], s);
                s = dp4a(hi, xb[k], s);
            }
            const int32_t sum = xs[2u * b] + xs[2u * b + 1u], i1 = s - OFF * sum;
            const float d = from_half(scales[(ONE ? 2u : 1u) * b]);
            term = (d * dx[b]) * (float)i1;
            if (ONE) term = term + (from_half(scales[2u * b + 1u]) * dx[b]) * (float)sum;
        }
        /* the round's blocks' terms in their order */
        const uint32_t count = blocks - b0 < 32u ? blocks - b0 : 32u;
        for (uint32_t k = 0; k < count; k++) acc = acc + shfl_f(term, k);
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
