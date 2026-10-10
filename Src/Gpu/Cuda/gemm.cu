/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The matrix product in single precision: C = A B over tiles of BM x BN outputs a block, BK steps of
 * the sum at a time, TM x TN outputs a thread, with gemm.comp's operand modes (A_ROW, A_COL, A_CONV;
 * B_ROW, B_COL, B_CONV), convolution geometry, PHASED rows, slices of split sums and epilogues
 * (gemm_common.glsl). Two buffers of shared memory: step s is multiplied from one while step s + 1
 * arrives in the other, one barrier a step. Operands read along k (A_ROW, A_CONV, B_COL) are loaded
 * into registers a step ahead and stored to shared memory after the step's products, as gemm.comp
 * does; the others, contiguous as they are stored (A_COL, B_ROW, B_CONV), go there by asynchronous
 * copies (cp.async), values outside the operands filled with zeros by the copies themselves. Every
 * output is the same chain of fused multiply-adds in the order of k whatever the tile, so the tile
 * changes the speed only.
 *
 * The tile, the operand modes and the vector loads are template parameters (SPG_ENTRY and GEMM name the
 * instance, cmake/Cuda.cmake compiles one a unit): the convolution modes that read four channels at once
 * are modes of their own, A_CONV4 and B_CONV4, and VEC says that the other operands read four values at
 * once. The epilogue, the activation, PHASED and the vector stores of C come from the spec: BM, BN, BK,
 * TM, TN, AMODE, BMODE, EPI, ACT, THREADS, PHASED, VEC. Its shared memory is dynamic:
 * SPG_CUDA_GEMM_SHARED(BM, BN, BK) bytes (Spingalett.GpuPush.h).
 *
 * The epilogues are inline for every output, of the activations (and derivatives) that are a slope
 * below zero only (act_slope(): none, ReLU, leaky ReLU). Units built with GENERAL (dense layers' modes:
 * Kernels.def) hold the others instead (GELU, SiLU, ...), and with FLAG_PRE EPI_BIAS_ACT also stores its
 * values before the activation at e1; Spingalett.Cuda.c runs the others of the other units as EPI_STORE
 * and the pass of epi.cu.
 */

#include "gemm_common.cuh"

#if !defined(GENERAL)
#define GENERAL 0
#endif

/* What the epilogue needs. */
struct Epilogue {
    uint64_t c, e0, e1;
    uint32_t M, N, ldc, epi, act, flags;
    float alpha, beta;
};

/* Outputs n .. n + 3 of row m of block z's product (row: the row of C they go to) through the epilogue, its
   activation (or derivative) a slope (C and what the epilogue reads with it aligned for vectors) */
DEVICE void store_c4(const Epilogue &e, float slope, uint32_t z, uint32_t coff, uint32_t m, uint32_t row, uint32_t n,
                     float4_ v) {
    float4_ *C4 = (float4_ *)e.c;
    if (e.epi == EPI_PARTIAL) {
        C4[((z * e.M + m) * e.N + n) >> 2] = v;
        return;
    }
    const uint32_t at = coff + row * e.ldc + n;
#if GENERAL
    if (e.epi == EPI_BIAS_ACT) {
        if (e.flags & FLAG_BIAS) v = add4(v, ((const float4_ *)e.e0)[(coff + n) >> 2]);
        if (e.flags & FLAG_PRE) ((float4_ *)e.e1)[at >> 2] = v;
        v = activate4(v, e.act);
    } else if (e.epi == EPI_SCALE_ACT) {
        v = activate4(add4(mul4(v, ((const float4_ *)e.e0)[(coff + n) >> 2]), ((const float4_ *)e.e1)[(coff + n) >> 2]),
                      e.act);
    } else {
        v = scale4(v, e.alpha);
        if (e.beta != 0.0f) v = add4(v, scale4(C4[at >> 2], e.beta));
        if (e.epi == EPI_DERIV) {
            const float4_ d = ((const float4_ *)e.e0)[at >> 2];
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
        if (e.beta != 0.0f) v = add4(v, scale4(C4[at >> 2], e.beta));
        if (e.epi == EPI_DERIV) {
            const float4_ d = ((const float4_ *)e.e0)[at >> 2];
            v = mul4(v, float4_{sloped_derivative(d.x, slope), sloped_derivative(d.y, slope),
                                sloped_derivative(d.z, slope), sloped_derivative(d.w, slope)});
        }
    }
#endif
    C4[at >> 2] = v;
}

/* and output (m, n) alone (the outputs at the edge of C, or of rows not in fours) */
DEVICE void store_c(const Epilogue &e, float slope, uint32_t z, uint32_t coff, uint32_t m, uint32_t row, uint32_t n,
                    float v) {
    float *C = F(e.c);
    if (e.epi == EPI_PARTIAL) {
        C[(z * e.M + m) * e.N + n] = v;
        return;
    }
    const uint32_t at = coff + row * e.ldc + n;
#if GENERAL
    if (e.epi == EPI_BIAS_ACT) {
        if (e.flags & FLAG_BIAS) v += F(e.e0)[coff + n];
        if (e.flags & FLAG_PRE) F(e.e1)[at] = v;
        v = activate(v, e.act);
    } else if (e.epi == EPI_SCALE_ACT) {
        v = activate(v * F(e.e0)[coff + n] + F(e.e1)[coff + n], e.act);
    } else {
        v *= e.alpha;
        if (e.beta != 0.0f) v += e.beta * C[at];
        if (e.epi == EPI_DERIV) v *= derivative(F(e.e0)[at], e.act);
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
        if (e.beta != 0.0f) v += e.beta * C[at];
        if (e.epi == EPI_DERIV) v *= sloped_derivative(F(e.e0)[at], slope);
    }
#endif
    C[at] = v;
}

template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t TM, uint32_t TN, uint32_t AMODE, uint32_t BMODE, uint32_t VEC>
struct Gemm {
    static constexpr uint32_t THREADS = (BM / TM) * (BN / TN);
    static constexpr uint32_t SA = BM + 4u, SB = BN + 4u;           /* floats a shared row: [k][m], [k][n] */
    static constexpr uint32_t STAGE = BK * (SA + SB);               /* floats a buffer */
    static constexpr uint32_t LA = (BM * BK + THREADS - 1u) / THREADS;    /* scalar loads a step */
    static constexpr uint32_t LB = (BN * BK + THREADS - 1u) / THREADS;
    static constexpr uint32_t LA4 = (BM * BK / 4u + THREADS - 1u) / THREADS;  /* vector loads a step */
    static constexpr uint32_t LB4 = (BN * BK / 4u + THREADS - 1u) / THREADS;
    static constexpr bool A_REG = AMODE != A_COL, B_REG = BMODE == B_COL;        /* through registers */
    static constexpr bool A_CV = AMODE == A_CONV || AMODE == A_CONV4, B_CV = BMODE == B_CONV || BMODE == B_CONV4;
    static constexpr uint32_t RA = A_REG ? (LA > 4u * LA4 ? LA : 4u * LA4) : 1u;
    static constexpr uint32_t RB = B_REG ? (LB > 4u * LB4 ? LB : 4u * LB4) : 1u;
    static constexpr uint32_t GA = AMODE == A_CONV ? LA : AMODE == A_CONV4 ? LA4 : 1u;     /* conv state */
    static constexpr uint32_t GB = BMODE == B_CONV ? LB : BMODE == B_CONV4 ? LB4 : 1u;

    /* four values a load: the modes A_CONV4 and B_CONV4, and with VEC the operands that are no convolutions */
    static constexpr bool VA = AMODE == A_CONV4 || (VEC && (AMODE == A_ROW || AMODE == A_COL));
    static constexpr bool VB = BMODE == B_CONV4 || (VEC && (BMODE == B_ROW || BMODE == B_COL));

    const SpgGemmPush &p;
    uint32_t EPI, ACT, PHASED;
    bool VC;
    uint32_t tid, m0, n0, kbeg, kend, aoff, boff;
    uint32_t RH, RW, GH, GW, GC, SH, SW, PH, PW;
    const uint32_t *geo;
    float ra[RA], rb[RB];
    /* A_CONV: each loaded row's sample and window; B_CONV: each loaded column's tap, and the output pixel
       (sample offset, y, x) of its k, followed from step to step */
    int abase[GA], ah[GA], aw[GA];
    int btap_h[GB], btap_w[GB], btap_c[GB];
    uint32_t bsample[GB], by[GB], bx[GB];

    MEMBER Gemm(const SpgGemmPush &push, const Spec &s) : p(push) {
        EPI = s.v[7]; ACT = s.v[8]; PHASED = s.v[10];
        VC = (s.v[11] & 16u) != 0u;
        geo = U(p.geo);
    }

    /* load i of a step (vector loads: four values of the contiguous axis): its row or column, and k */
    MEMBER uint32_t a_row(uint32_t i, bool vec) const {
        const uint32_t e = tid + i * THREADS;
        return AMODE == A_COL ? (vec ? (e % (BM / 4u)) * 4u : e % BM) : vec ? e / (BK / 4u) : e / BK;
    }
    MEMBER uint32_t a_k(uint32_t i, bool vec) const {
        const uint32_t e = tid + i * THREADS;
        return AMODE == A_COL ? (vec ? e / (BM / 4u) : e / BM) : vec ? (e % (BK / 4u)) * 4u : e % BK;
    }
    MEMBER uint32_t b_col(uint32_t i, bool vec) const {
        const uint32_t e = tid + i * THREADS;
        return BMODE == B_COL ? (vec ? e / (BK / 4u) : e / BK) : vec ? (e % (BN / 4u)) * 4u : e % BN;
    }
    MEMBER uint32_t b_k(uint32_t i, bool vec) const {
        const uint32_t e = tid + i * THREADS;
        return BMODE == B_COL ? (vec ? (e % (BK / 4u)) * 4u : e % BK) : vec ? e / (BN / 4u) : e / BN;
    }
    MEMBER bool a_live(uint32_t i, bool vec) const { return tid + i * THREADS < (vec ? BM * BK / 4u : BM * BK); }
    MEMBER bool b_live(uint32_t i, bool vec) const { return tid + i * THREADS < (vec ? BN * BK / 4u : BN * BK); }

    MEMBER void prepare() {
        if (A_CV || B_CV) {
            RH = geo[GEO_RH]; RW = geo[GEO_RW];
            GH = geo[GEO_GH]; GW = geo[GEO_GW]; GC = geo[GEO_GC];
            SH = geo[GEO_SH]; SW = geo[GEO_SW]; PH = geo[GEO_PH]; PW = geo[GEO_PW];
        }
        if (A_CV) {
            constexpr bool vec = AMODE == A_CONV4;
#pragma unroll
            for (uint32_t i = 0; i < GA; i++) {
                const uint32_t m = m0 + a_row(i, vec);
                const uint32_t n = m / (RH * RW), r = m % (RH * RW), y = r / RW, x = r % RW;
                abase[i] = (int)(n * GH * GW * GC);
                ah[i] = (int)(y * SH) - (int)PH;
                aw[i] = (int)(x * SW) - (int)PW;
                if (m >= p.M || !a_live(i, vec)) ah[i] = -(1 << 28);        /* never inside */
            }
        }
        if (B_CV) {
            constexpr bool vec = BMODE == B_CONV4;
#pragma unroll
            for (uint32_t i = 0; i < GB; i++) {
                const uint32_t n = n0 + b_col(i, vec);
                const uint32_t t = n < p.N ? geo[GEO_TAPS + n] : 0u;
                btap_h[i] = (int)(t & 255u);
                btap_w[i] = n < p.N ? (int)((t >> 8) & 255u) : (1 << 28);
                btap_c[i] = (int)(t >> 16);
                const uint32_t k = kbeg + b_k(i, vec), r = k % (RH * RW);
                bsample[i] = (k / (RH * RW)) * GH * GW * GC;
                by[i] = r / RW;
                bx[i] = r % RW;
            }
        }
    }

    /* A of step k0: into registers (A_REG), or copied into buffer As */
    MEMBER void load_a(float *As, uint32_t k0) {
        const float *A = F(p.a);
        if (VA) {
#pragma unroll
            for (uint32_t i = 0; i < LA4; i++) {
                if (!a_live(i, true)) continue;
                const uint32_t m = a_row(i, true), k = a_k(i, true), gm = m0 + m, gk = k0 + k;
                bool inside = gm < p.M && gk < kend;
                uint32_t at = 0u;
                if (AMODE == A_ROW) {
                    at = aoff + gm * p.lda + gk;
                } else if (AMODE == A_COL) {
                    at = aoff + gk * p.lda + gm;
                } else if (AMODE == A_CONV4) {
                    const uint32_t t = inside ? geo[GEO_TAPS + gk] : 0u;
                    const int y = ah[i] + (int)(t & 255u), x = aw[i] + (int)((t >> 8) & 255u);
                    inside = inside && (uint32_t)y < GH && (uint32_t)x < GW;
                    at = (uint32_t)abase[i] + ((uint32_t)y * GW + (uint32_t)x) * GC + (t >> 16) + aoff;
                }
                if (A_REG) {
                    const float4_ v = inside ? *(const float4_ *)(A + at) : float4_{0.0f, 0.0f, 0.0f, 0.0f};
                    ra[4u * i] = v.x; ra[4u * i + 1u] = v.y; ra[4u * i + 2u] = v.z; ra[4u * i + 3u] = v.w;
                } else {
                    copy16(shared_address(As + k * SA + m), inside ? A + at : A, inside ? 16u : 0u);
                }
            }
            return;
        }
#pragma unroll
        for (uint32_t i = 0; i < LA; i++) {
            if (!a_live(i, false)) continue;
            const uint32_t m = a_row(i, false), k = a_k(i, false), gm = m0 + m, gk = k0 + k;
            bool inside = gm < p.M && gk < kend;
            uint32_t at = 0u;
            if (AMODE == A_ROW) {
                at = aoff + gm * p.lda + gk;
            } else if (AMODE == A_COL) {
                at = aoff + gk * p.lda + gm;
            } else if (AMODE == A_CONV) {
                const uint32_t t = inside ? geo[GEO_TAPS + gk] : 0u;
                const int y = ah[i] + (int)(t & 255u), x = aw[i] + (int)((t >> 8) & 255u);
                inside = inside && (uint32_t)y < GH && (uint32_t)x < GW;
                at = (uint32_t)abase[i] + ((uint32_t)y * GW + (uint32_t)x) * GC + (t >> 16) + aoff;
            }
            if (A_REG) ra[i] = inside ? A[at] : 0.0f;
            else copy4(shared_address(As + k * SA + m), inside ? A + at : A, inside ? 4u : 0u);
        }
    }

    /* the registers of A_REG into buffer As ([k][m]) */
    MEMBER void store_a(float *As) const {
        if (!A_REG) return;
        if (VA) {
#pragma unroll
            for (uint32_t i = 0; i < LA4; i++) {
                if (!a_live(i, true)) continue;
                const uint32_t m = a_row(i, true), k = a_k(i, true);
#pragma unroll
                for (uint32_t j = 0; j < 4u; j++) As[(k + j) * SA + m] = ra[4u * i + j];
            }
            return;
        }
#pragma unroll
        for (uint32_t i = 0; i < LA; i++)
            if (a_live(i, false)) As[a_k(i, false) * SA + a_row(i, false)] = ra[i];
    }

    MEMBER void load_b(float *Bs, uint32_t k0) {
        const float *B = F(p.b);
        if (VB) {
#pragma unroll
            for (uint32_t i = 0; i < LB4; i++) {
                if (!b_live(i, true)) continue;
                const uint32_t n = b_col(i, true), k = b_k(i, true), gn = n0 + n, gk = k0 + k;
                bool inside = gn < p.N && gk < kend;
                uint32_t at = 0u;
                if (BMODE == B_ROW) {
                    at = boff + gk * p.ldb + gn;
                } else if (BMODE == B_COL) {
                    at = boff + gn * p.ldb + gk;
                } else if (BMODE == B_CONV4) {
                    const int y = (int)(by[i] * SH) - (int)PH + btap_h[i], x = (int)(bx[i] * SW) - (int)PW + btap_w[i];
                    inside = inside && (uint32_t)y < GH && (uint32_t)x < GW;
                    at = bsample[i] + ((uint32_t)y * GW + (uint32_t)x) * GC + (uint32_t)btap_c[i] + boff;
                    advance(i);
                }
                if (B_REG) {
                    const float4_ v = inside ? *(const float4_ *)(B + at) : float4_{0.0f, 0.0f, 0.0f, 0.0f};
                    rb[4u * i] = v.x; rb[4u * i + 1u] = v.y; rb[4u * i + 2u] = v.z; rb[4u * i + 3u] = v.w;
                } else {
                    copy16(shared_address(Bs + k * SB + n), inside ? B + at : B, inside ? 16u : 0u);
                }
            }
            return;
        }
#pragma unroll
        for (uint32_t i = 0; i < LB; i++) {
            if (!b_live(i, false)) continue;
            const uint32_t n = b_col(i, false), k = b_k(i, false), gn = n0 + n, gk = k0 + k;
            bool inside = gn < p.N && gk < kend;
            uint32_t at = 0u;
            if (BMODE == B_ROW) {
                at = boff + gk * p.ldb + gn;
            } else if (BMODE == B_COL) {
                at = boff + gn * p.ldb + gk;
            } else if (BMODE == B_CONV) {
                const int y = (int)(by[i] * SH) - (int)PH + btap_h[i], x = (int)(bx[i] * SW) - (int)PW + btap_w[i];
                inside = inside && (uint32_t)y < GH && (uint32_t)x < GW;
                at = bsample[i] + ((uint32_t)y * GW + (uint32_t)x) * GC + (uint32_t)btap_c[i] + boff;
                advance(i);
            }
            if (B_REG) rb[i] = inside ? B[at] : 0.0f;
            else copy4(shared_address(Bs + k * SB + n), inside ? B + at : B, inside ? 4u : 0u);
        }
    }

    /* B_CONV: load i's pixel, BK further (followed rather than divided out every step) */
    MEMBER void advance(uint32_t i) {
        bx[i] += BK;
        while (bx[i] >= RW) {
            bx[i] -= RW;
            if (++by[i] == RH) { by[i] = 0u; bsample[i] += GH * GW * GC; }
        }
    }

    MEMBER void store_b(float *Bs) const {
        if (!B_REG) return;
        if (VB) {
#pragma unroll
            for (uint32_t i = 0; i < LB4; i++) {
                if (!b_live(i, true)) continue;
                const uint32_t n = b_col(i, true), k = b_k(i, true);
#pragma unroll
                for (uint32_t j = 0; j < 4u; j++) Bs[(k + j) * SB + n] = rb[4u * i + j];
            }
            return;
        }
#pragma unroll
        for (uint32_t i = 0; i < LB; i++)
            if (b_live(i, false)) Bs[b_k(i, false) * SB + b_col(i, false)] = rb[i];
    }

    /* the row of C that row m of the product goes to */
    MEMBER uint32_t c_row(uint32_t m) const {
        if (PHASED == 0u) return m;
        const uint32_t CH = geo[GEO_CH], CW = geo[GEO_CW], CY = geo[GEO_CY], CX = geo[GEO_CX];
        const uint32_t CSH = geo[GEO_CS] & 0xFFFFu, CSW = geo[GEO_CS] >> 16;
        const uint32_t n = m / (RH * RW), r = m % (RH * RW);
        return (n * CH + CY + CSH * (r / RW)) * CW + CX + CSW * (r % RW);
    }

    MEMBER void run(float *smem) {
        tid = thread_x();
        const uint32_t z = block_z(), g = z / p.slices, s = z % p.slices;
        m0 = (p.m_tile0 + block_x()) * BM;
        n0 = block_y() * BN;
        kbeg = s * p.slice_k;
        kend = umin(p.K, kbeg + p.slice_k);
        aoff = g * p.a_group;
        boff = g * p.b_group;
        prepare();

        /* thread (tr, tc) computes rows 4 tr + RS i' + (0 .. 3) and columns 4 tc + CS j' + (0 .. 3) of the
           tile; a warp covers 4 x 8 threads' blocks where the tile allows, so that its reads of shared
           memory are four vectors of A and 128 consecutive bytes of B */
        constexpr bool WARPED = THREADS % 32u == 0u && (BM / TM) % 4u == 0u && (BN / TN) % 8u == 0u;
        constexpr uint32_t RS = 4u * (BM / TM), CS = 4u * (BN / TN), WCOLS = (BN / TN) / 8u;
        const uint32_t tr = WARPED ? (tid / 32u / WCOLS) * 4u + (tid % 32u) / 8u : tid / (BN / TN);
        const uint32_t tc = WARPED ? (tid / 32u % WCOLS) * 8u + tid % 8u : tid % (BN / TN);
        float4_ acc[TM * TN / 4];
#pragma unroll
        for (uint32_t i = 0; i < TM * TN / 4u; i++) acc[i] = float4_{0.0f, 0.0f, 0.0f, 0.0f};

        /* two buffers: step s computes from one while step s + 1 arrives in the other (async copies,
           and loads into registers stored after the products), one barrier a step */
        const uint32_t steps = kbeg < kend ? (kend - kbeg + BK - 1u) / BK : 0u;
        if (steps) {
            load_a(smem, kbeg);
            load_b(smem + BK * SA, kbeg);
            copies_commit();
            store_a(smem);
            store_b(smem + BK * SA);
        }
        for (uint32_t step = 0; step < steps; step++) {
            float *now = smem + (step & 1u) * STAGE, *next = smem + ((step + 1u) & 1u) * STAGE;
            copies_wait<0>();
            barrier();
            const bool more = step + 1u < steps;
            if (more) {
                load_a(next, kbeg + (step + 1u) * BK);
                load_b(next + BK * SA, kbeg + (step + 1u) * BK);
            }
            copies_commit();
            const float4_ *A4 = (const float4_ *)now, *B4 = (const float4_ *)(now + BK * SA);
#pragma unroll
            for (uint32_t kk = 0; kk < BK; kk++) {
                float4_ av[TM / 4], bv[TN / 4];
#pragma unroll
                for (uint32_t i = 0; i < TM / 4u; i++) av[i] = A4[kk * (SA / 4u) + i * (BM / TM) + tr];
#pragma unroll
                for (uint32_t j = 0; j < TN / 4u; j++) bv[j] = B4[kk * (SB / 4u) + j * (BN / TN) + tc];
#pragma unroll
                for (uint32_t i = 0; i < TM; i++) {
                    const float a = get4(av[i >> 2], i & 3u);
#pragma unroll
                    for (uint32_t j = 0; j < TN / 4u; j++)
                        acc[i * (TN / 4u) + j] = fma4(float4_{a, a, a, a}, bv[j], acc[i * (TN / 4u) + j]);
                }
            }
            if (more) {
                store_a(next);
                store_b(next + BK * SA);
            }
        }

        const uint32_t coff = g * p.c_group;
        const Epilogue e = {p.c, p.e0, p.e1, p.M, p.N, p.ldc, EPI, ACT, p.flags, p.alpha, p.beta};
        /* (the epilogues whose activation or derivative is no slope run as EPI_STORE and epi.cu's pass) */
        const float slope = EPI == EPI_DERIV ? act_slope(ACT) : act_slope_forward(ACT);
#pragma unroll
        for (uint32_t i = 0; i < TM; i++) {
            const uint32_t m = m0 + 4u * tr + RS * (i >> 2) + (i & 3u);
            if (m >= p.M) continue;
            const uint32_t row = c_row(m);
#pragma unroll
            for (uint32_t q = 0; q < TN / 4u; q++) {
                const uint32_t n = n0 + 4u * tc + CS * q;
                if (VC && n + 3u < p.N) {
                    store_c4(e, slope, z, coff, m, row, n, acc[i * (TN / 4u) + q]);
                    continue;
                }
#pragma unroll
                for (uint32_t j = 0; j < 4u; j++)
                    if (n + j < p.N) store_c(e, slope, z, coff, m, row, n + j, get4(acc[i * (TN / 4u) + q], j));
            }
        }
    }
};

/* the unit's instance: SPG_ENTRY and GEMM (BM, BN, BK, TM, TN, AMODE, BMODE, VEC) from cmake/Cuda.cmake */
#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_gemm_128x64x16_8x4_a0b1v
#define GEMM 128, 64, 16, 8, 4, 0, 1, 1
#endif

/* (at most 128 registers a thread: two blocks of 256 threads an SM, whose warps hide each other's waits) */
extern "C" __global__ void __attribute__((launch_bounds(Gemm<GEMM>::THREADS, 512u / Gemm<GEMM>::THREADS)))
SPG_ENTRY(const SpgGemmPush p, const Spec spec) {
    extern __shared__ float4_ smem[];
    Gemm<GEMM> gemm(p, spec);
    gemm.run((float *)smem);
}
