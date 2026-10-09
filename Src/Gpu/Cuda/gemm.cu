/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * gemm.comp in single precision: C = A B over tiles of BM x BN outputs a block, BK steps of the sum at a
 * time through shared memory (two buffers, one barrier a step), TM x TN outputs a thread, the next step's
 * operands loaded into registers while the current one is multiplied. The operand modes (A_ROW, A_COL,
 * A_CONV; B_ROW, B_COL, B_CONV), the convolutions' geometry, PHASED rows, the slices of split sums and
 * the epilogues are gemm.comp's (gemm_common.glsl); every output is the same chain of fused
 * multiply-adds in the order of k whatever the tile.
 *
 * The tile is a template parameter (SPG_ENTRY names the instance, cmake/Cuda.cmake compiles one a unit);
 * the rest comes from the spec: BM, BN, BK, TM, TN, AMODE, BMODE, EPI, ACT, THREADS, PHASED, VEC.
 */

#include "common.cuh"

#define A_ROW   0u
#define A_COL   1u
#define A_CONV  2u
#define B_ROW   0u
#define B_COL   1u
#define B_CONV  2u
#define EPI_STORE       0u
#define EPI_BIAS_ACT    1u
#define EPI_SCALE_ACT   2u
#define EPI_PARTIAL     3u
#define EPI_DERIV       4u
#define FLAG_BIAS       1u

#define GEO_RH 0u
#define GEO_RW 1u
#define GEO_GH 2u
#define GEO_GW 3u
#define GEO_GC 4u
#define GEO_SH 5u
#define GEO_SW 6u
#define GEO_PH 7u
#define GEO_PW 8u
#define GEO_CH 11u
#define GEO_CW 12u
#define GEO_CY 13u
#define GEO_CX 14u
#define GEO_CS 15u
#define GEO_TAPS 16u

template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t TM, uint32_t TN>
struct Gemm {
    static constexpr uint32_t BK_ = BK;
    static constexpr uint32_t THREADS = (BM / TM) * (BN / TN);
    static constexpr uint32_t LA = (BM * BK + THREADS - 1u) / THREADS;    /* loads a step, scalar at most */
    static constexpr uint32_t LB = (BN * BK + THREADS - 1u) / THREADS;
    static constexpr uint32_t SA = BM / 4u + 1u, SB = BN / 4u + 1u;       /* float4 a shared row, padded */

    const SpgGemmPush &p;
    uint32_t AMODE, BMODE, EPI, ACT, PHASED;
    bool VA, VB, VC, A_KFAST, B_KFAST;
    uint32_t WA, WB;
    uint32_t tid, m0, n0, kend, aoff, boff;
    uint32_t RH, RW, GH, GW, GC, SH, SW, PH, PW;
    const uint32_t *geo;
    int abase[LA], ah[LA], aw[LA];
    int btap_h[LB], btap_w[LB], btap_c[LB];
    uint32_t bsample[LB], by[LB], bx[LB];
    float4_ fa[LA], fb[LB];

    __device__ Gemm(const SpgGemmPush &push, const Spec &s) : p(push) {
        AMODE = s.v[5]; BMODE = s.v[6]; EPI = s.v[7]; ACT = s.v[8]; PHASED = s.v[10];
        const uint32_t vec = s.v[11];
        VA = vec & 1u; VB = (vec & 2u) != 0u; VC = (vec & 16u) != 0u;
        A_KFAST = AMODE != A_COL; B_KFAST = BMODE == B_COL;
        WA = VA ? 4u : 1u; WB = VB ? 4u : 1u;
        geo = U(p.geo);
    }

    __device__ uint32_t a_row(uint32_t i) const {
        const uint32_t e = tid + i * THREADS;
        return A_KFAST ? e / (BK / WA) : (e % (BM / WA)) * WA;
    }
    __device__ uint32_t a_k(uint32_t i) const {
        const uint32_t e = tid + i * THREADS;
        return A_KFAST ? (e % (BK / WA)) * WA : e / (BM / WA);
    }
    __device__ uint32_t b_col(uint32_t i) const {
        const uint32_t e = tid + i * THREADS;
        return B_KFAST ? e / (BK / WB) : (e % (BN / WB)) * WB;
    }
    __device__ uint32_t b_k(uint32_t i) const {
        const uint32_t e = tid + i * THREADS;
        return B_KFAST ? (e % (BK / WB)) * WB : e / (BN / WB);
    }
    __device__ bool a_live(uint32_t i) const { return tid + i * THREADS < BM * BK / WA; }
    __device__ bool b_live(uint32_t i) const { return tid + i * THREADS < BN * BK / WB; }

    __device__ void prepare(uint32_t kbeg) {
        if (AMODE == A_CONV || BMODE == B_CONV) {
            RH = geo[GEO_RH]; RW = geo[GEO_RW];
            GH = geo[GEO_GH]; GW = geo[GEO_GW]; GC = geo[GEO_GC];
            SH = geo[GEO_SH]; SW = geo[GEO_SW]; PH = geo[GEO_PH]; PW = geo[GEO_PW];
        }
        if (AMODE == A_CONV) {
            for (uint32_t i = 0; i < LA; i++) {
                const uint32_t m = m0 + a_row(i);
                const uint32_t n = m / (RH * RW), r = m % (RH * RW), y = r / RW, x = r % RW;
                abase[i] = (int)(n * GH * GW * GC);
                ah[i] = (int)(y * SH) - (int)PH;
                aw[i] = (int)(x * SW) - (int)PW;
                if (m >= p.M || !a_live(i)) ah[i] = -(1 << 28);
            }
        }
        if (BMODE == B_CONV) {
            for (uint32_t i = 0; i < LB; i++) {
                const uint32_t n = n0 + b_col(i);
                const uint32_t t = n < p.N ? geo[GEO_TAPS + n] : 0u;
                btap_h[i] = (int)(t & 255u);
                btap_w[i] = n < p.N ? (int)((t >> 8) & 255u) : (1 << 28);
                btap_c[i] = (int)(t >> 16);
                const uint32_t k = kbeg + b_k(i), r = k % (RH * RW);
                bsample[i] = (k / (RH * RW)) * GH * GW * GC;
                by[i] = r / RW;
                bx[i] = r % RW;
            }
        }
    }

    DEVICE float4_ load(uint64_t x, uint32_t at, bool vec) {
        if (vec) return ((const float4_ *)x)[at >> 2];
        return float4_{F(x)[at], 0.0f, 0.0f, 0.0f};
    }

    __device__ void fetch(uint32_t k0) {
        for (uint32_t i = 0; i < LA; i++) {
            if (i * THREADS >= BM * BK / WA) break;
            const uint32_t m = m0 + a_row(i), k = k0 + a_k(i);
            uint32_t at = 0u;
            bool inside = a_live(i) && m < p.M && k < kend;
            if (inside) {
                if (AMODE == A_ROW) {
                    at = aoff + m * p.lda + k;
                } else if (AMODE == A_COL) {
                    at = aoff + k * p.lda + m;
                } else {
                    const uint32_t t = geo[GEO_TAPS + k];
                    const int y = ah[i] + (int)(t & 255u), x = aw[i] + (int)((t >> 8) & 255u);
                    inside = (uint32_t)y < GH && (uint32_t)x < GW;
                    at = (uint32_t)abase[i] + ((uint32_t)y * GW + (uint32_t)x) * GC + (t >> 16) + aoff;
                }
            }
            fa[i] = inside ? load(p.a, at, VA) : float4_{0.0f, 0.0f, 0.0f, 0.0f};
        }
        for (uint32_t i = 0; i < LB; i++) {
            if (i * THREADS >= BN * BK / WB) break;
            const uint32_t n = n0 + b_col(i), k = k0 + b_k(i);
            uint32_t at = 0u;
            bool inside = b_live(i) && n < p.N && k < kend;
            if (inside) {
                if (BMODE == B_ROW) {
                    at = boff + k * p.ldb + n;
                } else if (BMODE == B_COL) {
                    at = boff + n * p.ldb + k;
                } else {
                    const int y = (int)(by[i] * SH) - (int)PH + btap_h[i], x = (int)(bx[i] * SW) - (int)PW + btap_w[i];
                    inside = (uint32_t)y < GH && (uint32_t)x < GW;
                    at = bsample[i] + ((uint32_t)y * GW + (uint32_t)x) * GC + (uint32_t)btap_c[i] + boff;
                }
            }
            fb[i] = inside ? load(p.b, at, VB) : float4_{0.0f, 0.0f, 0.0f, 0.0f};
            if (BMODE == B_CONV) {
                bx[i] += BK;
                while (bx[i] >= RW) {
                    bx[i] -= RW;
                    if (++by[i] == RH) { by[i] = 0u; bsample[i] += GH * GW * GC; }
                }
            }
        }
    }

    __device__ void store_tiles(float *As, float *Bs) const {
        for (uint32_t i = 0; i < LA; i++) {
            if (!a_live(i)) continue;
            const uint32_t k = a_k(i), m = a_row(i);
            if (!VA) As[k * 4u * SA + m] = fa[i].x;
            else if (A_KFAST) {
                As[k * 4u * SA + m] = fa[i].x; As[(k + 1u) * 4u * SA + m] = fa[i].y;
                As[(k + 2u) * 4u * SA + m] = fa[i].z; As[(k + 3u) * 4u * SA + m] = fa[i].w;
            } else ((float4_ *)As)[k * SA + (m >> 2)] = fa[i];
        }
        for (uint32_t i = 0; i < LB; i++) {
            if (!b_live(i)) continue;
            const uint32_t k = b_k(i), n = b_col(i);
            if (!VB) Bs[k * 4u * SB + n] = fb[i].x;
            else if (B_KFAST) {
                Bs[k * 4u * SB + n] = fb[i].x; Bs[(k + 1u) * 4u * SB + n] = fb[i].y;
                Bs[(k + 2u) * 4u * SB + n] = fb[i].z; Bs[(k + 3u) * 4u * SB + n] = fb[i].w;
            } else ((float4_ *)Bs)[k * SB + (n >> 2)] = fb[i];
        }
    }

    /* the row of C that row m of the product goes to */
    __device__ uint32_t c_row(uint32_t m) const {
        if (PHASED == 0u) return m;
        const uint32_t CH = geo[GEO_CH], CW = geo[GEO_CW], CY = geo[GEO_CY], CX = geo[GEO_CX];
        const uint32_t CSH = geo[GEO_CS] & 0xFFFFu, CSW = geo[GEO_CS] >> 16;
        const uint32_t n = m / (RH * RW), r = m % (RH * RW);
        return (n * CH + CY + CSH * (r / RW)) * CW + CX + CSW * (r % RW);
    }

    __device__ void store_c(uint32_t z, uint32_t coff, uint32_t m, uint32_t row, uint32_t n, float v) const {
        float *C = F(p.c);
        if (EPI == EPI_PARTIAL) {
            C[(z * p.M + m) * p.N + n] = v;
            return;
        }
        const uint32_t at = coff + row * p.ldc + n;
        if (EPI == EPI_STORE) {
            v *= p.alpha;
            if (p.beta != 0.0f) v += p.beta * C[at];
        } else if (EPI == EPI_BIAS_ACT) {
            if (p.flags & FLAG_BIAS) v += F(p.e0)[coff + n];
            v = activate(v, ACT);
        } else if (EPI == EPI_SCALE_ACT) {
            v = activate(v * F(p.e0)[coff + n] + F(p.e1)[coff + n], ACT);
        } else if (EPI == EPI_DERIV) {
            v *= p.alpha;
            if (p.beta != 0.0f) v += p.beta * C[at];
            v *= derivative(F(p.e0)[at], ACT);
        }
        C[at] = v;
    }

    __device__ void store_c4(uint32_t z, uint32_t coff, uint32_t m, uint32_t row, uint32_t n, float4_ v) const {
        float4_ *C4 = (float4_ *)p.c;
        if (EPI == EPI_PARTIAL) {
            C4[((z * p.M + m) * p.N + n) >> 2] = v;
            return;
        }
        const uint32_t at = coff + row * p.ldc + n;
        if (EPI == EPI_STORE) {
            v = scale4(v, p.alpha);
            if (p.beta != 0.0f) v = add4(v, scale4(C4[at >> 2], p.beta));
        } else if (EPI == EPI_BIAS_ACT) {
            if (p.flags & FLAG_BIAS) v = add4(v, ((const float4_ *)p.e0)[(coff + n) >> 2]);
            v = activate4(v, ACT);
        } else if (EPI == EPI_SCALE_ACT) {
            v = activate4(add4(mul4(v, ((const float4_ *)p.e0)[(coff + n) >> 2]), ((const float4_ *)p.e1)[(coff + n) >> 2]), ACT);
        } else if (EPI == EPI_DERIV) {
            v = scale4(v, p.alpha);
            if (p.beta != 0.0f) v = add4(v, scale4(C4[at >> 2], p.beta));
            const float4_ e = ((const float4_ *)p.e0)[at >> 2];
            v = mul4(v, float4_{derivative(e.x, ACT), derivative(e.y, ACT), derivative(e.z, ACT), derivative(e.w, ACT)});
        }
        C4[at >> 2] = v;
    }

    __device__ void run(float *smem) {
        tid = thread_x();
        const uint32_t z = block_z(), g = z / p.slices, s = z % p.slices;
        m0 = (p.m_tile0 + block_x()) * BM;
        n0 = block_y() * BN;
        const uint32_t kbeg = s * p.slice_k;
        kend = umin(p.K, kbeg + p.slice_k);
        aoff = g * p.a_group;
        boff = g * p.b_group;
        prepare(kbeg);

        /* threads as gemm.comp lays them out (WARPED where the tile allows) */
        const bool WARPED = THREADS % 32u == 0u && (BM / TM) % 4u == 0u && (BN / TN) % 8u == 0u;
        const uint32_t RS = 4u * (BM / TM), CS = 4u * (BN / TN), WCOLS = (BN / TN) / 8u;
        const uint32_t tr = WARPED ? (tid / 32u / WCOLS) * 4u + (tid % 32u) / 8u : tid / (BN / TN);
        const uint32_t tc = WARPED ? (tid / 32u % WCOLS) * 8u + tid % 8u : tid % (BN / TN);
        float4_ acc[TM * TN / 4];
        for (uint32_t i = 0; i < TM * TN / 4u; i++) acc[i] = float4_{0.0f, 0.0f, 0.0f, 0.0f};

        float *As[2] = {smem, smem + BK * 4u * SA};
        float *Bs[2] = {smem + 2u * BK * 4u * SA, smem + 2u * BK * 4u * SA + BK * 4u * SB};
        uint32_t buf = 0;
        if (kbeg < kend) fetch(kbeg);
        for (uint32_t k0 = kbeg; k0 < kend; k0 += BK, buf ^= 1u) {
            store_tiles(As[buf], Bs[buf]);
            barrier();
            if (k0 + BK < kend) fetch(k0 + BK);
            const float4_ *A4 = (const float4_ *)As[buf], *B4 = (const float4_ *)Bs[buf];
#pragma unroll
            for (uint32_t kk = 0; kk < BK; kk++) {
                float4_ av[TM / 4], bv[TN / 4];
#pragma unroll
                for (uint32_t i = 0; i < TM / 4u; i++) av[i] = A4[kk * SA + i * (BM / TM) + tr];
#pragma unroll
                for (uint32_t j = 0; j < TN / 4u; j++) bv[j] = B4[kk * SB + j * (BN / TN) + tc];
#pragma unroll
                for (uint32_t i = 0; i < TM; i++) {
                    const float a = get4(av[i >> 2], i & 3u);
#pragma unroll
                    for (uint32_t j = 0; j < TN / 4u; j++) acc[i * (TN / 4u) + j] = fma4(float4_{a, a, a, a}, bv[j], acc[i * (TN / 4u) + j]);
                }
            }
            /* (two buffers: the next step stores into the other one, so one barrier a step) */
        }

        const uint32_t coff = g * p.c_group;
        for (uint32_t i = 0; i < TM; i++) {
            const uint32_t m = m0 + 4u * tr + RS * (i >> 2) + (i & 3u);
            if (m >= p.M) continue;
            const uint32_t row = c_row(m);
            for (uint32_t q = 0; q < TN / 4u; q++) {
                const uint32_t n = n0 + 4u * tc + CS * q;
                if (VC && n + 3u < p.N) {
                    store_c4(z, coff, m, row, n, acc[i * (TN / 4u) + q]);
                    continue;
                }
                for (uint32_t j = 0; j < 4u; j++)
                    if (n + j < p.N) store_c(z, coff, m, row, n + j, get4(acc[i * (TN / 4u) + q], j));
            }
        }
    }
};

/* the unit's instance: SPG_ENTRY and TILE (BM, BN, BK, TM, TN) from cmake/Cuda.cmake */
#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_gemm_128x64x16_8x4
#define TILE 128, 64, 16, 8, 4
#endif

extern "C" __global__ void __launch_bounds__(Gemm<TILE>::THREADS) SPG_ENTRY(const SpgGemmPush p, const Spec spec) {
    typedef Gemm<TILE> G;
    __shared__ float4_ smem[2u * G::BK_ * (G::SA + G::SB)];
    G gemm(p, spec);
    gemm.run((float *)smem);
}
