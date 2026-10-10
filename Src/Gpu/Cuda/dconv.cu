/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * dconv: the matrix products of few k whose A is a convolution's windows (A_CONV) and B its filters by rows
 * (B_COL), over channels that do not come in fours: a network's first convolution (images of one or three
 * channels, k = 9 or 27), which the tiles of gemm.cu and gemm_mma.cu read a value at a time through the
 * geometry, and then wait on. Spingalett.Cuda.c runs this kernel in place of every tile of such a product
 * (k at most 48, at most 128 columns, no slices, no phases).
 *
 * A thread computes a row of C (an output pixel): its window is read once, into registers (its loads issued
 * before the block reads the filters, so that their latencies pass together); the filters are in shared
 * memory as [k][column], padded to a multiple of sixteen columns with zeros, read by every thread at once.
 * Sixteen columns a round: the products, then through shared memory the epilogue and stores of the block's
 * rows side by side. Each output adds its products in the order of k with fused multiply-adds from zero, as
 * gemm.cu's tiles do (the same bits in single precision); with ROUND (a product on the matrix units) the
 * operands are rounded to bfloat16 first, so that the products are those of the matrix units, summed in
 * that order. The epilogues are gemm_mma.cu's (sloped activations and derivatives: the others run as
 * EPI_STORE and epi.cu's pass). Units for k of exactly KMAX (EXACT: no tests of k in the loops) and up to it.
 *
 * Spec: EPI, ACT, HALF (A 0, B 1, C 2, e0 3), ROUND, the first row and the end of the rows of the dispatch,
 * the rounds of sixteen columns. T threads; shared memory: T TP + k 16 rounds words.
 */

#include "gemm_common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_dconv32
#define KMAX 32
#endif

#ifndef EXACT
#define EXACT 0
#endif
#define T 128u                  /* threads a block, a row each (six blocks an SM keep their phases apart) */
#define TP 20u                  /* floats a row of a round's outputs in shared memory (16 and padding) */

/* Outputs n .. n + 3 of row m (at: their index in C) through the epilogue, four at once (C and e0 aligned) or
   one at a time */
DEVICE float4_ epilogue4(const SpgGemmPush &p, uint32_t epi, uint32_t half, float slope, uint32_t coff, uint32_t at,
                         uint32_t n, float4_ v) {
    if (epi == EPI_BIAS_ACT) {
        if (p.flags & FLAG_BIAS) v = add4(v, ((const float4_ *)p.e0)[(coff + n) >> 2]);
        return sloped4(v, slope);
    }
    if (epi == EPI_SCALE_ACT)
        return sloped4(add4(mul4(v, ((const float4_ *)p.e0)[(coff + n) >> 2]), ((const float4_ *)p.e1)[(coff + n) >> 2]),
                       slope);
    v = scale4(v, p.alpha);
    if (p.beta != 0.0f) v = add4(v, scale4(ld4(p.c, at, 2u, half), p.beta));
    if (epi == EPI_DERIV) {
        const float4_ d = ld4(p.e0, at, 3u, half);
        v = mul4(v, float4_{sloped_derivative(d.x, slope), sloped_derivative(d.y, slope), sloped_derivative(d.z, slope),
                            sloped_derivative(d.w, slope)});
    }
    return v;
}
DEVICE void finish_one(const SpgGemmPush &p, uint32_t epi, uint32_t half, float slope, uint32_t coff, uint32_t at,
                       uint32_t n, float v) {
    if (epi == EPI_BIAS_ACT) {
        if (p.flags & FLAG_BIAS) v += F(p.e0)[coff + n];
        v = sloped(v, slope);
    } else if (epi == EPI_SCALE_ACT) {
        v = sloped(v * F(p.e0)[coff + n] + F(p.e1)[coff + n], slope);
    } else {
        v *= p.alpha;
        if (p.beta != 0.0f) v += p.beta * ld(p.c, at, 2u, half);
        if (epi == EPI_DERIV) v *= sloped_derivative(ld(p.e0, at, 3u, half), slope);
    }
    st(p.c, at, 2u, half, v);
}

extern "C" __global__ void __attribute__((launch_bounds(T, KMAX <= 32 ? 6 : 4))) SPG_ENTRY(const SpgGemmPush p,
                                                                                         const Spec spec) {
    extern __shared__ float4_ smem[];
    const uint32_t EPI = spec.v[0], ACT = spec.v[1], HALF = spec.v[2], ROUND = spec.v[3];
    const uint32_t first = spec.v[4], last = spec.v[5], chunks = spec.v[6];
    const uint32_t z = block_z(), K = p.K, N = p.N, NP = 16u * chunks, tid = thread_x();
    const uint32_t aoff = z * p.a_group, boff = z * p.b_group, coff = z * p.c_group;
    const uint32_t *geo = U(p.geo);
    float *out = (float *)smem;                     /* a round's outputs: T rows of TP floats */
    float *ws = out + T * TP;

    /* the row's window (zeros outside the input), its loads first: their latency then passes with the filters' */
    const uint32_t m0 = first + block_x() * T, m = m0 + tid;
    const uint32_t RH = geo[GEO_RH], RW = geo[GEO_RW], GH = geo[GEO_GH], GW = geo[GEO_GW], GC = geo[GEO_GC];
    const uint32_t sample = m / (RH * RW), r = m % (RH * RW);
    const int y0 = (int)((r / RW) * geo[GEO_SH]) - (int)geo[GEO_PH], x0 = (int)((r % RW) * geo[GEO_SW]) - (int)geo[GEO_PW];
    const uint32_t base = sample * GH * GW * GC + aoff;
    const bool live = m < last;
    float a[KMAX];
    bool inside[KMAX];
#pragma unroll
    for (uint32_t k = 0; k < KMAX; k++) {
        const uint32_t t = EXACT || k < K ? geo[GEO_TAPS + k] : 0u;
        const int y = y0 + (int)(t & 255u), x = x0 + (int)((t >> 8) & 255u);
        inside[k] = live && (EXACT || k < K) && (uint32_t)y < GH && (uint32_t)x < GW;
        const uint32_t at = inside[k] ? base + ((uint32_t)y * GW + (uint32_t)x) * GC + (t >> 16) : 0u;
        a[k] = (HALF & 1u) ? from_bf16(H(p.a)[at]) : F(p.a)[at];
    }

    /* the group's filters, read in their order into [k][column] (rounded with ROUND, zeros beyond N) */
    for (uint32_t i = tid; i < K * NP; i += T) {
        const uint32_t n = i / K, k = i % K;
        const float w = n < N ? ld(p.b, boff + n * p.ldb + k, 1u, HALF) : 0.0f;
        ws[k * NP + n] = ROUND ? from_bf16(to_bf16(w)) : w;
    }
    const bool round_a = ROUND && !(HALF & 1u);
#pragma unroll
    for (uint32_t k = 0; k < KMAX; k++) a[k] = !inside[k] ? 0.0f : round_a ? from_bf16(to_bf16(a[k])) : a[k];
    barrier();

    const float slope = EPI == EPI_DERIV ? act_slope(ACT) : act_slope_forward(ACT);
    /* four outputs at once where C (and what the epilogue reads with it) is aligned */
    const bool fours = (p.c % 16u) == 0u && (p.ldc % 4u) == 0u && (coff % 4u) == 0u &&
                       (EPI == EPI_STORE || p.e0 % 16u == 0u) && (EPI != EPI_SCALE_ACT || p.e1 % 16u == 0u);
    /* sixteen columns a round, the filters the same for every thread (read by all at once): the products, then
       the epilogue, through shared memory to stores of the block's rows side by side */
    for (uint32_t c = 0; c < chunks; c++) {
        const uint32_t n0 = 16u * c;
        float4_ s[4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
        const float4_ *w = (const float4_ *)(ws + n0);
#pragma unroll
        for (uint32_t k = 0; k < KMAX; k++) {
            if (EXACT || k < K) {
                const float4_ a4 = float4_{a[k], a[k], a[k], a[k]};
#pragma unroll
                for (uint32_t q = 0; q < 4u; q++) s[q] = fma4(a4, w[k * (NP / 4u) + q], s[q]);
            }
        }
        if (c) barrier();                           /* the round before stored */
#pragma unroll
        for (uint32_t q = 0; q < 4u; q++) ((float4_ *)(out + tid * TP))[q] = s[q];
        barrier();
        for (uint32_t i = tid; i < T * 4u; i += T) {
            const uint32_t row = i / 4u, n = n0 + 4u * (i % 4u), mm = m0 + row;
            if (mm >= last) continue;
            const uint32_t at = coff + mm * p.ldc + n;
            const float4_ v = ((const float4_ *)(out + row * TP))[i % 4u];
            if (fours && n + 3u < N) {
                st4(p.c, at, 2u, HALF, epilogue4(p, EPI, HALF, slope, coff, at, n, v));
                continue;
            }
            for (uint32_t j = 0; j < 4u && n + j < N; j++) finish_one(p, EPI, HALF, slope, coff, at + j, n + j, get4(v, j));
        }
    }
}
