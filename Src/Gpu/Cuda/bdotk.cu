/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * bdotk.comp: bdot's product for weights of Q4_K packed as sessions keep them (Spingalett.GpuPack.c), the CPU's bits:
 * a warp a row of weights, a lane a sub-block of 32 values of each round of 32 (its 16 bytes of codes and the
 * activations' 32 in sixteen bytes' loads): its products by dp4a (the low nibbles with values 0 to 15, the high ones
 * with 16 to 31) times its scale (I1), its minimum times the rows' sum of 32 (I2), both summed over the eight lanes of
 * a super-block (exact), then the super-blocks' terms (d dx) I1 + (m dx) I2, each product and sum rounded alone, added
 * in their order through shuffles; the bias then the sum, the activation of spec 2, the product by e0's value. Q6_K: each
 * half of 16 its codes' products less 32 times the activations' sum of 16, times its signed scale (I1 alone). A row of
 * activations a pass (r0); a unit a TYPE (bdotp_q4_k, bdotp_q6_k).
 */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_bdotp_q4_k
#define TYPE 7u
#endif

#define SIX (TYPE == SPG_B_Q6_K)

struct uint4_ {
    uint32_t x, y, z, w;
} __attribute__((aligned(16)));

DEVICE float shfl_f(float v, uint32_t lane) { return __nvvm_shfl_sync_idx_f32(0xFFFFFFFFu, v, lane, 0x1F); }
DEVICE int32_t shfl_add(int32_t v, uint32_t mask) { return v + (int32_t)__nvvm_shfl_sync_bfly_i32(0xFFFFFFFFu, v, mask, 0x1F); }
DEVICE int32_t dp4a(uint32_t a, uint32_t b, int32_t c) {
    int32_t d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

extern "C" __global__ void __launch_bounds__(256) SPG_ENTRY(const SpgBdotPush p, const Spec spec) {
    const uint32_t lane = thread_x() & 31u, j = block_x() * 8u + (thread_x() >> 5);
    if (j >= p.n) return;                       /* (a whole warp) */
    const uint32_t subs = p.k / 32u, blocks = p.k / 256u;
    const uint8_t *row = (const uint8_t *)p.w + (uint64_t)j * (p.k / 2u), *x = (const uint8_t *)p.q + (uint64_t)p.r0 * p.k;
    const uint8_t *highs = (const uint8_t *)p.w + (uint64_t)p.n * (p.k / 2u);
    const uint16_t *pairs0 = (const uint16_t *)(highs + (SIX ? (uint64_t)p.n * (p.k / 4u) : 0u));
    const uint16_t *pairs = pairs0 + (uint64_t)j * subs;
    const float *dm = (const float *)((const uint8_t *)pairs0 + (uint64_t)p.n * subs * 2u) + 2u * (uint64_t)j * blocks;
    highs += (uint64_t)j * (p.k / 4u);
    const int32_t *xs = (const int32_t *)p.xs + (uint64_t)p.r0 * (p.k / 16u);
    const float *dx = F(p.dx) + (uint64_t)p.r0 * blocks;
    float acc = 0.0f;
    for (uint32_t c0 = 0; c0 < subs; c0 += 32u) {
        const uint32_t c = c0 + lane;
        int32_t i1 = 0, i2 = 0;
        if (c < subs) {
            const uint4_ w = *(const uint4_ *)(row + 16u * c);
            const uint4_ x0 = ((const uint4_ *)(x + 32u * c))[0], x1 = ((const uint4_ *)(x + 32u * c))[1];
            const uint32_t m4 = 0x0F0F0F0Fu;
            if (SIX) {
                /* (each half of 16 alone: its four words of codes, a lane's low nibbles and the high two bits) */
                const uint2_ hb = *(const uint2_ *)(highs + 8u * c);
                const uint32_t pieces[4] = {hb.x & 0xFFFFu, hb.x >> 16, hb.y & 0xFFFFu, hb.y >> 16};
                const uint32_t words[4] = {w.x, w.y, w.z, w.w}, xa[4] = {x0.x, x0.y, x0.z, x0.w}, xb[4] = {x1.x, x1.y, x1.z, x1.w};
                int32_t lo = 0, hi = 0;
#pragma unroll
                for (uint32_t k = 0; k < 4u; k++) {
                    uint32_t h = pieces[k] & 0xFFu;
                    h = h | h << 6;
                    h = h | h << 12;
                    lo = dp4a((words[k] & m4) | (h & 0x03030303u) << 4, xa[k], lo);
                    h = pieces[k] >> 8;
                    h = h | h << 6;
                    h = h | h << 12;
                    hi = dp4a(((words[k] >> 4) & m4) | (h & 0x03030303u) << 4, xb[k], hi);
                }
                const uint32_t pr = pairs[c];
                i1 = (int32_t)(signed char)(pr & 0xFFu) * (lo - 32 * xs[2u * c]);
                i1 += (int32_t)(signed char)(pr >> 8) * (hi - 32 * xs[2u * c + 1u]);
            }
            int32_t s = 0;
            s = dp4a(w.x & m4, x0.x, s);
            s = dp4a(w.y & m4, x0.y, s);
            s = dp4a(w.z & m4, x0.z, s);
            s = dp4a(w.w & m4, x0.w, s);
            s = dp4a((w.x >> 4) & m4, x1.x, s);
            s = dp4a((w.y >> 4) & m4, x1.y, s);
            s = dp4a((w.z >> 4) & m4, x1.z, s);
            s = dp4a((w.w >> 4) & m4, x1.w, s);
            if (!SIX) {
                const uint32_t pr = pairs[c];
                i1 = (int32_t)(pr & 0xFFu) * s;
                i2 = (int32_t)(pr >> 8) * (xs[2u * c] + xs[2u * c + 1u]);
            }
        }
        /* (a super-block's sub-blocks: their integers summed across their eight lanes, exact) */
        for (uint32_t h = 1u; h < 8u; h <<= 1) {
            i1 = shfl_add(i1, h);
            i2 = shfl_add(i2, h);
        }
        float term = 0.0f;
        const uint32_t b = c / 8u;
        if (lane % 8u == 0u && b < blocks) {
            const float d = dm[2u * b], m = dm[2u * b + 1u], dxv = dx[b];
            term = (d * dxv) * (float)i1;
            if (!SIX) term = term + (m * dxv) * (float)i2;
        }
        /* the round's super-blocks' terms in their order */
        const uint32_t first = c0 / 8u, count = blocks - first < 4u ? blocks - first : 4u;
        for (uint32_t k = 0; k < count; k++) acc = acc + shfl_f(term, 8u * k);
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
