/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The matrix product on the matrix units: mma.sync m16n8k16, operands in bfloat16, products added in
 * single precision. gemm.cu's operand modes, convolution geometry, PHASED rows, slices and epilogues,
 * with operands and results kept as bfloat16 where HALF says (bits: A 1, B 2, C 4, e0 8, as for
 * gemm_mma.comp, whose constants it takes: BM, BN, BK, TM, TN, AMODE, BMODE, EPI, ACT, THREADS,
 * PHASED, VEC, SG, HALF). A warp computes 16 TM x 16 TN outputs, the block's warps tile BM x BN.
 *
 * Each operand is read one of three ways (KA, KB, template parameters): kept as bfloat16 eight values at
 * a time, by asynchronous copies (WIDE) that run SPG_CUDA_MMA_STAGES - 1 steps ahead; or four floats at
 * a time (FLOAT4), or a value at a time (ONE), into registers a step ahead, rounded to bfloat16 and stored
 * after the step's products. Shared tiles keep each operand's layout in memory, rows along its
 * contiguous axis padded by eight values (so that the eight rows ldmatrix reads fall in distinct banks),
 * and ldmatrix gives the fragments, transposing those of the other layout.
 *
 * An output's products are added 16 values of k at a time, in the order of k, by the same instruction
 * whatever the tile: tiles change the speed only. The results go to the epilogue through shared memory,
 * four outputs of a row a thread: inline for the activations that are a slope (act_slope()), through
 * store_one() for the others.
 */

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

/* What the epilogue needs, by value (store_one() is a call). */
struct Epi {
    uint64_t c, e0, e1;
    uint32_t M, N, ldc, epi, act, flags, half;
    float alpha, beta;
};

/* Output (m, n) of block z's product, v, through the epilogue (row: the row of C it goes to). */
static __device__ __attribute__((noinline)) void store_one(Epi e, uint32_t z, uint32_t coff, uint32_t m, uint32_t row,
                                                           uint32_t n, float v) {
    if (e.epi == EPI_PARTIAL) {
        F(e.c)[(z * e.M + m) * e.N + n] = v;
        return;
    }
    const uint32_t at = coff + row * e.ldc + n;
    if (e.epi == EPI_STORE) {
        v *= e.alpha;
        if (e.beta != 0.0f) v += e.beta * ld(e.c, at, 2u, e.half);
    } else if (e.epi == EPI_BIAS_ACT) {
        if (e.flags & FLAG_BIAS) v += F(e.e0)[coff + n];
        v = activate(v, e.act);
    } else if (e.epi == EPI_SCALE_ACT) {
        v = activate(v * F(e.e0)[coff + n] + F(e.e1)[coff + n], e.act);
    } else if (e.epi == EPI_DERIV) {
        v *= e.alpha;
        if (e.beta != 0.0f) v += e.beta * ld(e.c, at, 2u, e.half);
        v *= derivative(ld(e.e0, at, 3u, e.half), e.act);
    }
    st(e.c, at, 2u, e.half, v);
}

/* Outputs n .. n + 3 of row m through the epilogue of a sloped activation (EPI_SCALE_ACT not among them). */
DEVICE void store_four(const Epi &e, float slope, uint32_t z, uint32_t coff, uint32_t m, uint32_t row, uint32_t n,
                       float4_ v) {
    if (e.epi == EPI_PARTIAL) {
        ((float4_ *)e.c)[((z * e.M + m) * e.N + n) >> 2] = v;
        return;
    }
    const uint32_t at = coff + row * e.ldc + n;
    if (e.epi == EPI_BIAS_ACT) {
        if (e.flags & FLAG_BIAS) v = add4(v, ((const float4_ *)e.e0)[(coff + n) >> 2]);
        v = sloped4(v, slope);
    } else {
        v = scale4(v, e.alpha);
        if (e.beta != 0.0f) v = add4(v, scale4(ld4(e.c, at, 2u, e.half), e.beta));
        if (e.epi == EPI_DERIV) {
            const float4_ d = ld4(e.e0, at, 3u, e.half);
            v = mul4(v, float4_{sloped_derivative(d.x, slope), sloped_derivative(d.y, slope), sloped_derivative(d.z, slope),
                                sloped_derivative(d.w, slope)});
        }
    }
    st4(e.c, at, 2u, e.half, v);
}

#define ONE    0u               /* the ways of reading an operand */
#define FLOAT4 1u
#define WIDE   2u

template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t TM, uint32_t TN, uint32_t AMODE, uint32_t BMODE, uint32_t KA, uint32_t KB>
struct Mma {
    static constexpr uint32_t WARPS_M = BM / (16u * TM), WARPS_N = BN / (16u * TN), THREADS = WARPS_M * WARPS_N * 32u;
    static constexpr bool A_KFAST = AMODE != A_COL, B_KFAST = BMODE == B_COL;     /* rows along k */
    static constexpr uint32_t SA = A_KFAST ? BK + 8u : BM + 8u, ROWS_A = A_KFAST ? BM : BK;
    static constexpr uint32_t SB = B_KFAST ? BK + 8u : BN + 8u, ROWS_B = B_KFAST ? BN : BK;
    static constexpr uint32_t STAGE_A = ROWS_A * SA, STAGE = STAGE_A + ROWS_B * SB;      /* bfloat16 a stage */
    static constexpr bool COPIES = KA == WIDE || KB == WIDE, REGS = KA != WIDE || KB != WIDE;
    static constexpr uint32_t STAGES = COPIES ? SPG_CUDA_MMA_STAGES : 2u;
    /* values a load (along the operand's contiguous axis), loads a step */
    static constexpr uint32_t WA = KA == WIDE ? 8u : KA == FLOAT4 ? 4u : 1u, WB = KB == WIDE ? 8u : KB == FLOAT4 ? 4u : 1u;
    static constexpr uint32_t EA = BM * BK / WA, EB = BN * BK / WB;
    static constexpr uint32_t LA = (EA + THREADS - 1u) / THREADS, LB = (EB + THREADS - 1u) / THREADS;
    static constexpr uint32_t GA = AMODE == A_CONV ? LA : 1u, GB = BMODE == B_CONV ? LB : 1u;
    /* registers of the next step's values (bfloat16, two to a register with FLOAT4) */
    static constexpr uint32_t RA = KA == WIDE ? 1u : KA == FLOAT4 ? 2u * LA : LA, RB = KB == WIDE ? 1u : KB == FLOAT4 ? 2u * LB : LB;

    const SpgGemmPush &p;
    uint32_t EPI, ACT, PHASED, HALF;
    uint32_t tid, m0, n0, kbeg, kend, aoff, boff;
    uint32_t RH, RW, GH, GW, GC, SH, SW, PH, PW;
    const uint32_t *geo;
    /* A_CONV: each loaded row's sample and window; B_CONV: each loaded column's tap, and the output pixel
       of its k, followed from step to step */
    int abase[GA], ah[GA], aw[GA];
    int btap_h[GB], btap_w[GB], btap_c[GB];
    uint32_t bsample[GB], by[GB], bx[GB];
    uint32_t ra[RA], rb[RB];

    MEMBER Mma(const SpgGemmPush &push, const Spec &s) : p(push) {
        EPI = s.v[7]; ACT = s.v[8]; PHASED = s.v[10]; HALF = s.v[13];
        geo = U(p.geo);
    }

    /* load i of a step: its row (column) and k */
    MEMBER uint32_t a_row(uint32_t i) const {
        const uint32_t e = tid + i * THREADS;
        return A_KFAST ? e / (BK / WA) : (e % (BM / WA)) * WA;
    }
    MEMBER uint32_t a_k(uint32_t i) const {
        const uint32_t e = tid + i * THREADS;
        return A_KFAST ? (e % (BK / WA)) * WA : e / (BM / WA);
    }
    MEMBER uint32_t b_col(uint32_t i) const {
        const uint32_t e = tid + i * THREADS;
        return B_KFAST ? e / (BK / WB) : (e % (BN / WB)) * WB;
    }
    MEMBER uint32_t b_k(uint32_t i) const {
        const uint32_t e = tid + i * THREADS;
        return B_KFAST ? (e % (BK / WB)) * WB : e / (BN / WB);
    }
    MEMBER bool a_live(uint32_t i) const { return tid + i * THREADS < EA; }
    MEMBER bool b_live(uint32_t i) const { return tid + i * THREADS < EB; }
    /* bytes into a stage of value (k, m) of A, (k, n) of B */
    MEMBER uint32_t sh_a(uint32_t k, uint32_t m) const { return 2u * (A_KFAST ? m * SA + k : k * SA + m); }
    MEMBER uint32_t sh_b(uint32_t k, uint32_t n) const { return 2u * (STAGE_A + (B_KFAST ? n * SB + k : k * SB + n)); }

    MEMBER void prepare() {
        if (AMODE == A_CONV || BMODE == B_CONV) {
            RH = geo[GEO_RH]; RW = geo[GEO_RW];
            GH = geo[GEO_GH]; GW = geo[GEO_GW]; GC = geo[GEO_GC];
            SH = geo[GEO_SH]; SW = geo[GEO_SW]; PH = geo[GEO_PH]; PW = geo[GEO_PW];
        }
        if (AMODE == A_CONV) {
#pragma unroll
            for (uint32_t i = 0; i < GA; i++) {
                const uint32_t m = m0 + a_row(i);
                const uint32_t n = m / (RH * RW), r = m % (RH * RW), y = r / RW, x = r % RW;
                abase[i] = (int)(n * GH * GW * GC);
                ah[i] = (int)(y * SH) - (int)PH;
                aw[i] = (int)(x * SW) - (int)PW;
                if (m >= p.M || !a_live(i)) ah[i] = -(1 << 28);        /* never inside */
            }
        }
        if (BMODE == B_CONV) {
#pragma unroll
            for (uint32_t i = 0; i < GB; i++) {
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

    /* the element of A (B) load i reads at step k0, and whether it is inside the operand */
    MEMBER bool a_at(uint32_t i, uint32_t k0, uint32_t &at) const {
        const uint32_t gm = m0 + a_row(i), gk = k0 + a_k(i);
        bool inside = a_live(i) && gm < p.M && gk < kend;
        if (AMODE == A_ROW) {
            at = aoff + gm * p.lda + gk;
        } else if (AMODE == A_COL) {
            at = aoff + gk * p.lda + gm;
        } else {
            const uint32_t t = inside ? geo[GEO_TAPS + gk] : 0u;
            const int y = ah[i] + (int)(t & 255u), x = aw[i] + (int)((t >> 8) & 255u);
            inside = inside && (uint32_t)y < GH && (uint32_t)x < GW;
            at = (uint32_t)abase[i] + ((uint32_t)y * GW + (uint32_t)x) * GC + (t >> 16) + aoff;
        }
        return inside;
    }
    MEMBER bool b_at(uint32_t i, uint32_t k0, uint32_t &at) {
        const uint32_t gn = n0 + b_col(i), gk = k0 + b_k(i);
        bool inside = b_live(i) && gn < p.N && gk < kend;
        if (BMODE == B_ROW) {
            at = boff + gk * p.ldb + gn;
        } else if (BMODE == B_COL) {
            at = boff + gn * p.ldb + gk;
        } else {
            const int y = (int)(by[i] * SH) - (int)PH + btap_h[i], x = (int)(bx[i] * SW) - (int)PW + btap_w[i];
            inside = inside && (uint32_t)y < GH && (uint32_t)x < GW;
            at = bsample[i] + ((uint32_t)y * GW + (uint32_t)x) * GC + (uint32_t)btap_c[i] + boff;
            bx[i] += BK;                    /* the pixel BK further, followed rather than divided out */
            while (bx[i] >= RW) {
                bx[i] -= RW;
                if (++by[i] == RH) { by[i] = 0u; bsample[i] += GH * GW * GC; }
            }
        }
        return inside;
    }

    /* the WIDE operands' chunks of step k0 copied into the stage at byte `stage` of shared memory */
    MEMBER void copy_step(uint32_t stage, uint32_t k0) {
        const uint16_t *A = H(p.a), *B = H(p.b);
#pragma unroll
        for (uint32_t i = 0; i < (KA == WIDE ? LA : 0u); i++) {
            if (!a_live(i)) continue;
            uint32_t at = 0u;
            const bool inside = a_at(i, k0, at);
            copy16(stage + sh_a(a_k(i), a_row(i)), inside ? A + at : A, inside ? 16u : 0u);
        }
#pragma unroll
        for (uint32_t i = 0; i < (KB == WIDE ? LB : 0u); i++) {
            if (!b_live(i)) continue;
            uint32_t at = 0u;
            const bool inside = b_at(i, k0, at);
            copy16(stage + sh_b(b_k(i), b_col(i)), inside ? B + at : B, inside ? 16u : 0u);
        }
    }

    /* the other operands' values of step k0 into registers, as bfloat16 (FLOAT4: four floats a load); then
       into a stage */
    MEMBER void load_step(uint32_t k0) {
#pragma unroll
        for (uint32_t i = 0; i < (KA == WIDE ? 0u : LA); i++) {
            uint32_t at = 0u;
            const bool inside = a_at(i, k0, at);
            if (KA == FLOAT4) {
                const float4_ v = inside ? *(const float4_ *)(F(p.a) + at) : float4_{0.0f, 0.0f, 0.0f, 0.0f};
                ra[2u * i] = to_bf16(v.x) | to_bf16(v.y) << 16;
                ra[2u * i + 1u] = to_bf16(v.z) | to_bf16(v.w) << 16;
            } else {
                ra[i] = !inside ? 0u : (HALF & 1u) ? (uint32_t)H(p.a)[at] : to_bf16(F(p.a)[at]);
            }
        }
#pragma unroll
        for (uint32_t i = 0; i < (KB == WIDE ? 0u : LB); i++) {
            uint32_t at = 0u;
            const bool inside = b_at(i, k0, at);
            if (KB == FLOAT4) {
                const float4_ v = inside ? *(const float4_ *)(F(p.b) + at) : float4_{0.0f, 0.0f, 0.0f, 0.0f};
                rb[2u * i] = to_bf16(v.x) | to_bf16(v.y) << 16;
                rb[2u * i + 1u] = to_bf16(v.z) | to_bf16(v.w) << 16;
            } else {
                rb[i] = !inside ? 0u : (HALF & 2u) ? (uint32_t)H(p.b)[at] : to_bf16(F(p.b)[at]);
            }
        }
    }
    MEMBER void store_step(uint32_t stage) const {
#pragma unroll
        for (uint32_t i = 0; i < (KA == WIDE ? 0u : LA); i++) {
            if (!a_live(i)) continue;
            if (KA == FLOAT4) st_shared64(stage + sh_a(a_k(i), a_row(i)), ra[2u * i], ra[2u * i + 1u]);
            else st_shared16(stage + sh_a(a_k(i), a_row(i)), ra[i]);
        }
#pragma unroll
        for (uint32_t i = 0; i < (KB == WIDE ? 0u : LB); i++) {
            if (!b_live(i)) continue;
            if (KB == FLOAT4) st_shared64(stage + sh_b(b_k(i), b_col(i)), rb[2u * i], rb[2u * i + 1u]);
            else st_shared16(stage + sh_b(b_k(i), b_col(i)), rb[i]);
        }
    }

    /* the products of a stage */
    MEMBER void multiply(uint32_t stage, uint32_t wm, uint32_t wn, uint32_t lane, float (&acc)[TM][2 * TN][4]) const {
#pragma unroll
        for (uint32_t kk = 0; kk < BK; kk += 16u) {
            uint32_t af[TM][4], bf[TN][4];
#pragma unroll
            for (uint32_t i = 0; i < TM; i++) {
                const uint32_t mb = (wm * TM + i) * 16u;
                if (A_KFAST) ldsm4(stage + sh_a(kk + (lane / 16u) * 8u, mb + lane % 16u), af[i]);
                else ldsm4t(stage + sh_a(kk + lane % 8u + (lane / 16u) * 8u, mb + (lane / 8u % 2u) * 8u), af[i]);
            }
#pragma unroll
            for (uint32_t j = 0; j < TN; j++) {
                const uint32_t nb = (wn * TN + j) * 16u;
                if (B_KFAST) ldsm4(stage + sh_b(kk + (lane / 8u % 2u) * 8u, nb + lane % 8u + (lane / 16u) * 8u), bf[j]);
                else ldsm4t(stage + sh_b(kk + lane % 8u + (lane / 8u % 2u) * 8u, nb + (lane / 16u) * 8u), bf[j]);
            }
#pragma unroll
            for (uint32_t i = 0; i < TM; i++)
#pragma unroll
                for (uint32_t j = 0; j < TN; j++) {
                    mma(acc[i][2u * j], af[i], bf[j][0], bf[j][1]);
                    mma(acc[i][2u * j + 1u], af[i], bf[j][2], bf[j][3]);
                }
        }
    }

    /* the row of C that row m of the product goes to */
    MEMBER uint32_t c_row(uint32_t m) const {
        if (PHASED == 0u) return m;
        const uint32_t CH = geo[GEO_CH], CW = geo[GEO_CW], CY = geo[GEO_CY], CX = geo[GEO_CX];
        const uint32_t CSH = geo[GEO_CS] & 0xFFFFu, CSW = geo[GEO_CS] >> 16;
        const uint32_t n = m / (RH * RW), r = m % (RH * RW);
        return (n * CH + CY + CSH * (r / RW)) * CW + CX + CSW * (r % RW);
    }

    /* rows of results a round of the epilogue, in the stages' shared memory: floats of BN + 4 a row */
    static constexpr uint32_t CP = BN + 4u, SMEM = STAGES * STAGE * 2u;
    static constexpr uint32_t RCH = BM * CP * 4u <= SMEM ? BM : BM / 2u * CP * 4u <= SMEM ? BM / 2u : BM / 4u * CP * 4u <= SMEM ? BM / 4u : BM / 8u;
    float *cs_base;

    MEMBER void run(uint32_t smem) {
        tid = thread_x();
        const uint32_t z = block_z(), g = z / p.slices, s = z % p.slices;
        m0 = (p.m_tile0 + block_x()) * BM;
        n0 = block_y() * BN;
        kbeg = s * p.slice_k;
        kend = umin(p.K, kbeg + p.slice_k);
        aoff = g * p.a_group;
        boff = g * p.b_group;
        if (AMODE == A_CONV || BMODE == B_CONV || PHASED) {
            RH = geo[GEO_RH]; RW = geo[GEO_RW];
        }
        prepare();

        const uint32_t warp = tid / 32u, lane = tid % 32u, wm = warp % WARPS_M, wn = warp / WARPS_M;
        float acc[TM][2 * TN][4];
#pragma unroll
        for (uint32_t i = 0; i < TM; i++)
#pragma unroll
            for (uint32_t j = 0; j < 2u * TN; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

        /* the WIDE operands STAGES - 1 steps ahead, the others one step ahead in registers, stored to the
           next step's stage after the products (which the barrier of every step makes safe: that stage was
           last read STAGES - 1 steps before) */
        const uint32_t steps = kbeg < kend ? (kend - kbeg + BK - 1u) / BK : 0u;
#pragma unroll
        for (uint32_t s0 = 0; COPIES && s0 + 1u < STAGES; s0++) {
            if (s0 < steps) copy_step(smem + s0 * STAGE * 2u, kbeg + s0 * BK);
            copies_commit();
        }
        if (REGS && steps) {
            load_step(kbeg);
            store_step(smem);
        }
        for (uint32_t step = 0; step < steps; step++) {
            if (COPIES) copies_wait<STAGES - 2u>();
            barrier();
            if (COPIES) {
                const uint32_t ahead = step + STAGES - 1u;
                if (ahead < steps) copy_step(smem + (ahead % STAGES) * STAGE * 2u, kbeg + ahead * BK);
                copies_commit();
            }
            const bool more = step + 1u < steps;
            if (REGS && more) load_step(kbeg + (step + 1u) * BK);
            multiply(smem + (step % STAGES) * STAGE * 2u, wm, wn, lane, acc);
            if (REGS && more) store_step(smem + ((step + 1u) % STAGES) * STAGE * 2u);
        }

        /* the results through shared memory, RCH rows of the tile at a time: fragment (i, j) holds rows lane / 4
           and + 8 of its 16, columns 2 (lane % 4) and + 1 of its 8, which as they are would make stores of 8
           rows of 16 bytes a warp; read back, a thread finishes four consecutive outputs of a row */
        const uint32_t coff = g * p.c_group;
        const Epi e = {p.c, p.e0, p.e1, p.M, p.N, p.ldc, EPI, ACT, p.flags, HALF, p.alpha, p.beta};
        const float slope = act_slope(ACT);
        const bool sloped_epi = EPI == EPI_PARTIAL || EPI == EPI_STORE || (EPI != EPI_SCALE_ACT && slope >= 0.0f);
        /* four outputs at once: at indices of C (and of e0 with it) that are multiples of four, at 16 bytes */
        const bool fours = sloped_epi && (p.c % 16u) == 0u &&
                           (EPI == EPI_PARTIAL ? p.N % 4u == 0u
                                               : (p.ldc % 4u) == 0u && (coff % 4u) == 0u &&
                                                 (EPI == EPI_STORE || p.e0 % 16u == 0u));
        float *Cs = (float *)__builtin_assume_aligned(cs_base, 16);
#pragma unroll
        for (uint32_t r0 = 0; r0 < BM; r0 += RCH) {
            barrier();
#pragma unroll
            for (uint32_t i = 0; i < TM; i++)
#pragma unroll
                for (uint32_t h = 0; h < 2u; h++) {
                    const uint32_t lr = (wm * TM + i) * 16u + lane / 4u + 8u * h;
                    if (lr < r0 || lr >= r0 + RCH) continue;
#pragma unroll
                    for (uint32_t j = 0; j < 2u * TN; j++) {
                        float *at = Cs + (lr - r0) * CP + wn * TN * 16u + 8u * j + 2u * (lane % 4u);
                        at[0] = acc[i][j][2u * h];
                        at[1] = acc[i][j][2u * h + 1u];
                    }
                }
            barrier();
#pragma unroll 1
            for (uint32_t q = tid; q < RCH * (BN / 4u); q += THREADS) {
                const uint32_t lr = q / (BN / 4u), ln = 4u * (q % (BN / 4u));
                const uint32_t m = m0 + r0 + lr, n = n0 + ln;
                if (m >= p.M || n >= p.N) continue;
                const float4_ v = *(const float4_ *)(Cs + lr * CP + ln);
                const uint32_t row = c_row(m);
                if (fours && n + 3u < p.N) {
                    store_four(e, slope, z, coff, m, row, n, v);
                    continue;
                }
                for (uint32_t k = 0; k < 4u && n + k < p.N; k++) store_one(e, z, coff, m, row, n + k, get4(v, k));
            }
        }
    }
};

/* the unit's instance: SPG_ENTRY and MMA (BM, BN, BK, TM, TN, AMODE, BMODE, KA, KB) from cmake/Cuda.cmake */
#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_gemm_mma_128x128x32_4x2_a0b1_wf
#define MMA 128, 128, 32, 4, 2, 0, 1, WIDE, FLOAT4
#endif

/* (at most 128 registers a thread: two blocks of 256 threads an SM) */
extern "C" __global__ void __attribute__((launch_bounds(Mma<MMA>::THREADS, 512u / Mma<MMA>::THREADS)))
SPG_ENTRY(const SpgGemmPush p, const Spec spec) {
    extern __shared__ float4_ smem[];
    Mma<MMA> mma_(p, spec);
    mma_.cs_base = (float *)smem;
    mma_.run(shared_address(smem));
}
