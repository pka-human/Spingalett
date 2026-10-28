/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * bdotk.comp: bdot's product for weights of Q4_K, Q5_K, Q6_K and IQ4_XS packed as sessions keep them
 * (Spingalett.GpuPack.c: rows `stride` bytes apart), the CPU's bits:
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
#define FIVE (TYPE == SPG_B_Q5_K)
#define NLX (TYPE == SPG_B_IQ4_XS)

struct uint4_ {
    uint32_t x, y, z, w;
} __attribute__((aligned(16)));

DEVICE float shfl_f(float v, uint32_t lane) { return __nvvm_shfl_sync_idx_f32(0xFFFFFFFFu, v, lane, 0x1F); }
DEVICE int32_t shfl_add(int32_t v, uint32_t mask) { return v + (int32_t)__nvvm_shfl_sync_bfly_i32(0xFFFFFFFFu, v, mask, 0x1F); }
DEVICE uint32_t spread1(uint32_t x) { return ((x * 0x00204081u) & 0x01010101u) << 4; }
DEVICE uint32_t nl_values(uint32_t q) {
    const uint32_t y = q | q >> 4, sel = ((y & 0xFFu) | ((y >> 8) & 0xFF00u)) & 0x7777u;
    uint32_t lo, hi;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo) : "r"(0xBFAD9881u), "r"(0xF6EADDCFu), "r"(sel));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi) : "r"(0x26190D01u), "r"(0x71594535u), "r"(sel));
    const uint32_t mask = ((q >> 3) & 0x01010101u) * 0xFFu;
    return (lo & ~mask) | (hi & mask);
}
DEVICE int32_t dp4a(uint32_t a, uint32_t b, int32_t c) {
    int32_t d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

extern "C" __global__ void __launch_bounds__(256) SPG_ENTRY(const SpgBdotPush p, const Spec spec) {
    const bool pair = spec.v[4] != 0u;          /* (a gated unit's pair of rows a warp: block_out()) */
    const uint32_t lane = thread_x() & 31u, j = block_x() * 8u + (thread_x() >> 5);
    if (j >= (pair ? p.n / 2u : p.n)) return;   /* (a whole warp) */
    /* the sum of the weights' row r */
    auto dot = [&](uint32_t r) {
        const uint32_t subs = p.k / 32u, blocks = p.k / 256u;
        const uint8_t *row = (const uint8_t *)p.w + (uint64_t)r * p.stride, *x = (const uint8_t *)p.q + (uint64_t)p.r0 * p.k;
        const uint8_t *highs = row + p.k / 2u, *aux = row + SPG_PACK_CODES(TYPE, p.k);
        const uint32_t *fifth = (const uint32_t *)aux;
        const uint16_t *pairs = (const uint16_t *)(aux + (FIVE ? 4u * subs : 0u));
        const uint16_t *dm = (const uint16_t *)(pairs + subs);       /* (Q6_K: the scales alone) */
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
                    /* (each half of 16 alone: its four words of codes, a lane's low nibbles and the high two bits, the
                       words of its pair of sub-blocks, bits 0 to 3 of their bytes the first's, 4 to 7 the second's) */
                    const uint4_ hb = *(const uint4_ *)(highs + 16u * (c / 2u));
                    const uint32_t hw[4] = {hb.x, hb.y, hb.z, hb.w}, shift = 4u * (c % 2u);
                    const uint32_t words[4] = {w.x, w.y, w.z, w.w}, xa[4] = {x0.x, x0.y, x0.z, x0.w}, xb[4] = {x1.x, x1.y, x1.z, x1.w};
                    int32_t lo = 0, hi = 0;
    #pragma unroll
                    for (uint32_t k = 0; k < 4u; k++) {
                        const uint32_t h = hw[k] >> shift;
                        lo = dp4a((words[k] & m4) | (h & 0x03030303u) << 4, xa[k], lo);
                        hi = dp4a(((words[k] >> 4) & m4) | (h & 0x0C0C0C0Cu) << 2, xb[k], hi);
                    }
                    const uint32_t pr = pairs[c];
                    i1 = (int32_t)(signed char)(pr & 0xFFu) * (lo - 32 * xs[2u * c]);
                    i1 += (int32_t)(signed char)(pr >> 8) * (hi - 32 * xs[2u * c + 1u]);
                }
                int32_t s = 0;
                const uint32_t f = FIVE ? fifth[c] : 0u;
                const uint32_t ws[4] = {w.x, w.y, w.z, w.w}, xa[4] = {x0.x, x0.y, x0.z, x0.w}, xb[4] = {x1.x, x1.y, x1.z, x1.w};
    #pragma unroll
                for (uint32_t k = 0; k < 4u; k++) {
                    uint32_t lo = ws[k] & m4, hi = (ws[k] >> 4) & m4;
                    if (FIVE) {
                        lo |= spread1((f >> (4u * k)) & 0xFu);
                        hi |= spread1((f >> (16u + 4u * k)) & 0xFu);
                    }
                    if (NLX) {
                        lo = nl_values(lo);
                        hi = nl_values(hi);
                    }
                    s = dp4a(lo, xa[k], s);
                    s = dp4a(hi, xb[k], s);
                }
                if (!SIX) {
                    const uint32_t pr = pairs[c];
                    i1 = (NLX ? (int32_t)(signed char)(pr & 0xFFu) : (int32_t)(pr & 0xFFu)) * s;
                    i2 = NLX ? 0 : (int32_t)(pr >> 8) * (xs[2u * c] + xs[2u * c + 1u]);
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
                const float d = from_half(dm[SIX ? b : 2u * b]), m = SIX ? 0.0f : -from_half(dm[2u * b + 1u]), dxv = dx[b];
                term = (d * dxv) * (float)i1;
                if (!SIX && !NLX) term = term + (m * dxv) * (float)i2;
            }
            /* the round's super-blocks' terms in their order */
            const uint32_t first = c0 / 8u, count = blocks - first < 4u ? blocks - first : 4u;
            for (uint32_t k = 0; k < count; k++) acc = acc + shfl_f(term, 8u * k);
        }
        return acc;
    };
    const float a0 = dot(pair ? 2u * j : j), a1 = pair ? dot(2u * j + 1u) : 0.0f;
    if (lane == 0u) block_out(p, spec.v[2], pair, 0u, j, a0, a1);
}
