/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * What the products on the matrix units (gemm_mma.cu, gemm_pipe.cu) share: fragments loaded by ldmatrix,
 * mma.sync m16n8k16 of bfloat16 adding in single precision, and the epilogue of an output (or of four
 * consecutive outputs of a row): bias and activation, scale and shift, alpha and beta, the derivative, the
 * slices' partial sums. Units built with GENERAL hold every activation; the others those that are a slope
 * (act_slope()).
 */

#pragma once

#include "gemm_common.cuh"

DEVICE void ldsm4(uint32_t at, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at) : "memory");
}
DEVICE void ldsm4t(uint32_t at, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at) : "memory");
}
DEVICE void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVICE void st_shared16(uint32_t at, uint32_t v) { asm volatile("st.shared.u16 [%0], %1;" ::"r"(at), "h"((uint16_t)v) : "memory"); }
DEVICE void st_shared64(uint32_t at, uint32_t a, uint32_t b) {
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::"r"(at), "r"(a), "r"(b) : "memory");
}

#if !defined(GENERAL)
#define GENERAL 0
#endif

/* What the epilogue needs. */
struct Epi {
    uint64_t c, e0, e1;
    uint32_t M, N, ldc, epi, act, flags, half;
    float alpha, beta;
};

/* Output (m, n) of block z's product, v, through the epilogue of a sloped activation or derivative (row: the row
   of C it goes to). */
DEVICE void store_one(const Epi &e, float slope, uint32_t z, uint32_t coff, uint32_t m, uint32_t row, uint32_t n,
                      float v) {
    if (e.epi == EPI_PARTIAL) {
        F(e.c)[(z * e.M + m) * e.N + n] = v;
        return;
    }
    const uint32_t at = coff + row * e.ldc + n;
#if GENERAL
    if (e.epi == EPI_BIAS_ACT) {
        if (e.flags & FLAG_BIAS) v += F(e.e0)[coff + n];
        if (e.flags & FLAG_PRE) st(e.e1, at, 2u, e.half, v);
        v = activate(v, e.act);
    } else if (e.epi == EPI_SCALE_ACT) {
        v = activate(v * F(e.e0)[coff + n] + F(e.e1)[coff + n], e.act);
    } else {
        v *= e.alpha;
        if (e.beta != 0.0f) v += e.beta * ld(e.c, at, 2u, e.half);
        if (e.epi == EPI_DERIV) v *= derivative(ld(e.e0, at, 3u, e.half), e.act);
    }
    (void)slope;
#else
    if (e.epi == EPI_BIAS_ACT) {
        if (e.flags & FLAG_BIAS) v += F(e.e0)[coff + n];
        v = sloped(v, slope);
    } else if (e.epi == EPI_SCALE_ACT) {
        v = sloped(v * F(e.e0)[coff + n] + F(e.e1)[coff + n], slope);
    } else {
        v *= e.alpha;
        if (e.beta != 0.0f) v += e.beta * ld(e.c, at, 2u, e.half);
        if (e.epi == EPI_DERIV) v *= sloped_derivative(ld(e.e0, at, 3u, e.half), slope);
    }
#endif
    st(e.c, at, 2u, e.half, v);
}

/* and outputs n .. n + 3 of row m at once */
DEVICE void store_four(const Epi &e, float slope, uint32_t z, uint32_t coff, uint32_t m, uint32_t row, uint32_t n,
                       float4_ v) {
    if (e.epi == EPI_PARTIAL) {
        ((float4_ *)e.c)[((z * e.M + m) * e.N + n) >> 2] = v;
        return;
    }
    const uint32_t at = coff + row * e.ldc + n;
#if GENERAL
    if (e.epi == EPI_BIAS_ACT) {
        if (e.flags & FLAG_BIAS) v = add4(v, ((const float4_ *)e.e0)[(coff + n) >> 2]);
        if (e.flags & FLAG_PRE) st4(e.e1, at, 2u, e.half, v);
        v = activate4(v, e.act);
    } else if (e.epi == EPI_SCALE_ACT) {
        v = activate4(add4(mul4(v, ((const float4_ *)e.e0)[(coff + n) >> 2]), ((const float4_ *)e.e1)[(coff + n) >> 2]),
                      e.act);
    } else {
        v = scale4(v, e.alpha);
        if (e.beta != 0.0f) v = add4(v, scale4(ld4(e.c, at, 2u, e.half), e.beta));
        if (e.epi == EPI_DERIV) {
            const float4_ d = ld4(e.e0, at, 3u, e.half);
            v = mul4(v, float4_{derivative(d.x, e.act), derivative(d.y, e.act), derivative(d.z, e.act), derivative(d.w, e.act)});
        }
    }
    (void)slope;
#else
    if (e.epi == EPI_BIAS_ACT) {
        if (e.flags & FLAG_BIAS) v = add4(v, ((const float4_ *)e.e0)[(coff + n) >> 2]);
        v = sloped4(v, slope);
    } else if (e.epi == EPI_SCALE_ACT) {
        v = sloped4(add4(mul4(v, ((const float4_ *)e.e0)[(coff + n) >> 2]), ((const float4_ *)e.e1)[(coff + n) >> 2]), slope);
    } else {
        v = scale4(v, e.alpha);
        if (e.beta != 0.0f) v = add4(v, scale4(ld4(e.c, at, 2u, e.half), e.beta));
        if (e.epi == EPI_DERIV) {
            const float4_ d = ld4(e.e0, at, 3u, e.half);
            v = mul4(v, float4_{sloped_derivative(d.x, slope), sloped_derivative(d.y, slope), sloped_derivative(d.z, slope),
                                sloped_derivative(d.w, slope)});
        }
    }
#endif
    st4(e.c, at, 2u, e.half, v);
}
