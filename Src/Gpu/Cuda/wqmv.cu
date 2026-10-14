/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * wqmv.comp: products of a few rows of activations with a linear layer's weights kept as an integer model's codes
 * (INT8, INT4, INT2) and a scale a row of weights, the codes widened in registers. A unit a BITS and ROWS
 * (wqmv_b<BITS>_r<ROWS>): a block 32 rows of weights, a warp NR of them, a lane 16 bytes of each of its rows' codes
 * in turn (a value of x read once for the NR rows); a lane's sums in the order of k, then the lanes' by a butterfly
 * of shuffles (a fixed order). Spec: BITS, ROWS, ACT, HALF (words: x 0, y 1).
 */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_wqmv_b8_r1
#define BITS 8u
#define ROWS 1u
#endif

#define NR SPG_WQ_NR(ROWS)                   /* rows of weights a warp */

DEVICE float shfl_xor(float v, uint32_t mask) { return __nvvm_shfl_sync_bfly_f32(0xFFFFFFFFu, v, mask, 0x1F); }

struct uint4_ {
    uint32_t x, y, z, w;
} __attribute__((aligned(16)));

/* code j of a word of codes as a float: the code with its top bit flipped (code + half its range, unsigned) in the
   significand of 2^23, less 2^23 + half the range (exact; no cvt from integers, a quarter of the rate of an add) */
DEVICE float code_of(uint32_t w, uint32_t j) {
    constexpr uint32_t MASK = (1u << BITS) - 1u, MID = 1u << (BITS - 1u);
    return bits_float(0x4B000000u | (((w >> (BITS * j)) & MASK) ^ MID)) - (8388608.0f + (float)MID);
}

extern "C" __global__ void __launch_bounds__(256) SPG_ENTRY(const SpgWqPush p, const Spec spec) {
    const uint32_t ACT = spec.v[2], HALF = spec.v[3];
    constexpr uint32_t PER = 32u / BITS, CODES = 4u * PER;
    const uint32_t lane = thread_x() & 31u, n0 = (block_x() * 8u + (thread_x() >> 5)) * NR;
    if (n0 >= p.N) return;                  /* (a whole warp) */
    float acc[NR][ROWS];
    for (uint32_t i = 0; i < NR; i++)
        for (uint32_t r = 0; r < ROWS; r++) acc[i][r] = 0.0f;
    const uint32_t chunks = (p.K + CODES - 1u) / CODES, rows = p.rows < ROWS ? p.rows : ROWS;
    const uint32_t live = p.N - n0 < NR ? p.N - n0 : NR;   /* (rows of weights past N: none) */
    const bool fours = (p.K & 3u) == 0u;
    for (uint32_t c = lane; c < chunks; c += 32u) {
        uint32_t q[NR][4];
#pragma unroll
        for (uint32_t i = 0; i < NR; i++) {
            const uint4_ q4 = i < live ? ((const uint4_ *)(p.codes + (uint64_t)(n0 + i) * p.stride))[c] : uint4_{0, 0, 0, 0};
            q[i][0] = q4.x; q[i][1] = q4.y; q[i][2] = q4.z; q[i][3] = q4.w;
        }
        const uint32_t k0 = c * CODES;
        if (fours && k0 + CODES <= p.K) {
            /* (the words unrolled, their codes in a loop: the loads of x of every step at once would take more registers
               than a thread may have) */
#pragma unroll
            for (uint32_t wi = 0; wi < 4u; wi++)
#pragma unroll 1
                for (uint32_t j = 0; j < PER; j += 4u) {
                    const uint32_t k = k0 + wi * PER + j;
                    float4_ x[ROWS];
#pragma unroll
                    for (uint32_t r = 0; r < ROWS; r++) x[r] = r < rows ? ld4(p.x, (p.r0 + r) * p.K + k, 0u, HALF) : float4_{0, 0, 0, 0};
#pragma unroll
                    for (uint32_t i = 0; i < NR; i++) {
                        const float c0 = code_of(q[i][wi], j), c1 = code_of(q[i][wi], j + 1u), c2 = code_of(q[i][wi], j + 2u),
                                    c3 = code_of(q[i][wi], j + 3u);
#pragma unroll
                        for (uint32_t r = 0; r < ROWS; r++)
                            acc[i][r] = fma_(x[r].w, c3, fma_(x[r].z, c2, fma_(x[r].y, c1, fma_(x[r].x, c0, acc[i][r]))));
                    }
                }
        } else {
            /* (the last chunk, or rows not in fours: a value at a time, every array's index a constant: in registers) */
#pragma unroll
            for (uint32_t wi = 0; wi < 4u; wi++)
#pragma unroll 1
                for (uint32_t j = 0; j < PER; j++) {
                    const uint32_t k = k0 + wi * PER + j;
                    if (k >= p.K) continue;
#pragma unroll
                    for (uint32_t r = 0; r < ROWS; r++) {
                        if (r >= rows) break;
                        const float x = ld(p.x, (p.r0 + r) * p.K + k, 0u, HALF);
#pragma unroll
                        for (uint32_t i = 0; i < NR; i++) acc[i][r] = fma_(x, code_of(q[i][wi], j), acc[i][r]);
                    }
                }
        }
    }
    /* each sum over the lanes, then lane i + NR r writes row i's output of x's row r */
    float s = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < NR; i++)
#pragma unroll
        for (uint32_t r = 0; r < ROWS; r++) {
            float v = acc[i][r];
            for (uint32_t m = 16u; m > 0u; m >>= 1) v += shfl_xor(v, m);
            if (lane == i + NR * r) s = v;
        }
    const uint32_t i = lane % NR, r = lane / NR;
    if (r < rows && i < live)
        st(p.y, (p.r0 + r) * p.N + n0 + i, 1u, HALF, activate(F(p.bias)[n0 + i] + F(p.scale)[n0 + i] * s, ACT));
}
