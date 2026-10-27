/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * bmmk.comp: bmm's product for weights of Q4_K, Q5_K, Q6_K and IQ4_XS packed as sessions keep them (Spingalett.GpuPack.c:
 * rows `stride` bytes apart), the CPU's bits:
 * tiles of 64 rows by 64 weights, four warps of 32 x 32 outputs (three workgroups an SM), k in stages of two sub-blocks of 32 values, STAGES
 * of them in shared memory by asynchronous copies two ahead (the rows' codes in chunks of 16 bytes swizzled for
 * ldmatrix; the weights' codes, a sub-block's 16 bytes, swizzled so that a warp's words fall in distinct banks); each
 * super-block's scales, minima, the rows' sums of 16 and scales in a buffer of its own (two, by the super-block's
 * parity). Each sub-block's products by mma.sync m16n8k32 of 8-bit integers (a lane's word of codes gives both halves of
 * its fragment: the low nibbles, then the high ones), times the sub-block's scale, added in integers (I1); at a
 * super-block's end its minima times the rows' sums of 32 (I2), then each output's term (d dx) I1 + (m dx) I2, every
 * product and sum rounded alone, added to its sum. The bias then the sum, the activation of spec 2, the product by
 * e0's value: bdot's epilogue. Push: SpgBdotPush (r0 the first row, rows its rows); groups: the tiles of rows, then
 * of weights. A unit a TYPE (Q4_K: bmmp_q4_k).
 */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_bmmp_q4_k
#define TYPE 7u
#endif

#define WN 2u                               /* warps along the weights (two along the rows) */
#define TN 4u                               /* tiles of 8 columns a warp */
#define THREADS (64u * WN)
#define SIX (TYPE == SPG_B_Q6_K)            /* Q6_K: halves of 16 of their own scales, two high bits, no minima */
#define FIVE (TYPE == SPG_B_Q5_K)           /* Q5_K: Q4_K's and a fifth bit */
#define NLX (TYPE == SPG_B_IQ4_XS)          /* IQ4_XS: IQ4_NL's values, signed scales, no minima */
#define BM 64u
#define BN (8u * TN * WN)
#define SB 2u                               /* sub-blocks of 32 values a stage */
#define STAGES 3u
#define PER (8u / SB)                       /* stages a super-block of 256 */

typedef signed char int8_t;

struct uint4_ {
    uint32_t x, y, z, w;
} __attribute__((aligned(16)));

DEVICE uint32_t shared_address(const void *p) {
    uint64_t r;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(r) : "l"(p));
    return (uint32_t)r;
}
DEVICE void copy4(uint32_t dst, const void *src, uint32_t bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
DEVICE void copy8(uint32_t dst, const void *src, uint32_t bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
DEVICE void copy16(uint32_t dst, const void *src, uint32_t bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
DEVICE void copies_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <uint32_t N>
DEVICE void copies_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
DEVICE void ldsm4(uint32_t at, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(at) : "memory");
}
/* d = c + a b of 8-bit integers (16 x 16 signed by 16 x 8 unsigned), exact: c's rows g (c01) and g + 8 (c23) */
DEVICE void mma_s8u8_k16(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t b, int32_t c01, int32_t c23) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %7, %8, %8};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a0), "r"(a1), "r"(b), "r"(c01), "r"(c23));
}
/* four bits of x at bit 4 of four bytes */
DEVICE uint32_t spread1(uint32_t x) { return ((x * 0x00204081u) & 0x01010101u) << 4; }
/* four 2-bit fields of x at bits 4 and 5 of four bytes */
DEVICE uint32_t spread2(uint32_t x) {
    x = x | x << 6;
    x = x | x << 12;
    return (x & 0x03030303u) << 4;
}
/* d = a b of 8-bit integers (16 x 32 signed by 32 x 8 unsigned; IQ4_XS: signed), exact */
DEVICE void mma_s8u8_k32(int32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    if (NLX)
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%10, %10, %10, %10};"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
    else
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%10, %10, %10, %10};"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
/* four codes (bytes of 0 to 15) as IQ4_NL's values (llama.cpp's kvalues_iq4nl), by two byte permutations */
DEVICE uint32_t nl_values(uint32_t q) {
    const uint32_t y = q | q >> 4, sel = ((y & 0xFFu) | ((y >> 8) & 0xFF00u)) & 0x7777u;
    uint32_t lo, hi;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo) : "r"(0xBFAD9881u), "r"(0xF6EADDCFu), "r"(sel));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi) : "r"(0x26190D01u), "r"(0x71594535u), "r"(sel));
    const uint32_t mask = ((q >> 3) & 0x01010101u) * 0xFFu;
    return (lo & ~mask) | (hi & mask);
}

/* chunk g of row r of the rows' codes (rows of 64 bytes), swizzled: ldmatrix's eight rows in distinct banks */
DEVICE uint32_t chunk_a(uint32_t r, uint32_t g) {
    const uint32_t idx = r * 4u + g;
    return 16u * (idx ^ ((idx >> 3) & 7u));
}
/* sub-block cb of row r of the weights' codes (rows of 32 bytes), swizzled: rows 4-7 of every 8 the other way round */
DEVICE uint32_t chunk_b(uint32_t r, uint32_t cb) { return 32u * r + 16u * (cb ^ ((r >> 2) & 1u)); }

struct __attribute__((aligned(16))) Stage {       /* (a multiple of 16 bytes: the next stage's chunks aligned) */
    uint8_t a[BM * 64u], b[BN * 32u];       /* the rows' codes, the weights' (Q6_K: their low four bits) */
    uint8_t h[SIX ? BN * 16u : 16u];        /* Q6_K: their high two bits, 8 bytes a sub-block */
    uint32_t f[FIVE ? BN : 4u][SB];         /* Q5_K: their fifth bits */
};
struct Super {
    uint16_t pairs[BN][8];                  /* the sub-blocks' scale | minimum << 8 (Q6_K: the halves' signed scales) */
    int32_t xs[BM][16];                     /* the rows' sums of 16 codes */
    uint32_t dm[BN];                        /* the scale and minimum (Q6_K: two scales), float16 */
    float dx[BM];                           /* the rows' scales */
};
struct Shared {
    Stage st[STAGES];
    Super su[2];
};

extern "C" __global__ void __launch_bounds__(THREADS) SPG_ENTRY(const SpgBdotPush p, const Spec spec) {
    __shared__ __attribute__((aligned(16))) Shared sh;
    const uint32_t tid = thread_x(), lane = tid & 31u, warp = tid >> 5, wm = warp / WN, wn = warp % WN;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t m0 = block_x() * BM, n0 = block_y() * BN, subs = p.k / 32u, blocks = p.k / 256u, stages = subs / SB;
    /* (a row's codes, Q6_K's high bits, then its fifth bits, scales and minima, the super-blocks' scales) */
    const uint8_t *weights = (const uint8_t *)p.w;
    const uint32_t aux = SPG_PACK_CODES(TYPE, p.k), pairs = aux + (FIVE ? 4u * subs : 0u), dm = pairs + 2u * subs;
    /* the copies of stage s (zeros past the rows and the weights) */
    auto issue = [&](uint32_t s) {
        Stage &st = sh.st[s % STAGES];
        const uint32_t sb = s / PER, j = s % PER;
        Super &su = sh.su[sb & 1u];
#pragma unroll
        for (uint32_t i = 0; i < BM * 4u / THREADS; i++) {
            const uint32_t u = tid + THREADS * i, r = u >> 2, c = u & 3u, m = m0 + r;
            const bool live = m < p.rows;
            copy16(shared_address(st.a + chunk_a(r, c)),
                   (const int8_t *)p.q + (live ? (uint64_t)(p.r0 + m) * p.k + 32u * SB * s + 16u * c : 0u), live ? 16u : 0u);
        }
#pragma unroll
        for (uint32_t i = 0; i < BN * SB / THREADS; i++) {
            const uint32_t u = tid + THREADS * i, r = u / SB, cb = u % SB, n = n0 + r;
            const bool live = n < p.n;
            copy16(shared_address(st.b + chunk_b(r, cb)), weights + (live ? (uint64_t)n * p.stride + 16u * (SB * s + cb) : 0u),
                   live ? 16u : 0u);
        }
        if (FIVE && tid < BN) {
            const uint32_t n = n0 + tid;
            const bool live = n < p.n;
            copy8(shared_address(&st.f[FIVE ? tid : 0u][0]), weights + (live ? (uint64_t)n * p.stride + aux + 4u * SB * s : 0u),
                  live ? 8u : 0u);
        }
        if (SIX && tid < BN) {
            const uint32_t n = n0 + tid;
            const bool live = n < p.n;
            copy16(shared_address(st.h + 16u * tid), weights + (live ? (uint64_t)n * p.stride + p.k / 2u + 8u * SB * s : 0u),
                   live ? 16u : 0u);
        }
        if (tid < BN) {
            const uint32_t n = n0 + tid;
            const bool live = n < p.n;
            const uint8_t *row = weights + (live ? (uint64_t)n * p.stride : 0u);
            copy4(shared_address(&su.pairs[tid][SB * j]), row + pairs + 2u * SB * s, live ? 4u : 0u);
            /* (Q6_K: a float16 scale a super-block, the word of this one's and its neighbour's) */
            if (j == 0u) copy4(shared_address(&su.dm[tid]), row + dm + (SIX ? 4u * (sb / 2u) : 4u * sb), live ? 4u : 0u);
        }
        if (tid < BM) {
            const uint32_t m = m0 + tid;
            const bool live = m < p.rows;
            copy16(shared_address(&su.xs[tid][4u * j]),
                   (const int32_t *)p.xs + (live ? (uint64_t)(p.r0 + m) * (p.k / 16u) + 2u * SB * s : 0u), live ? 16u : 0u);
            if (j == 0u)
                copy4(shared_address(&su.dx[tid]), F(p.dx) + (live ? (uint64_t)(p.r0 + m) * blocks + sb : 0u), live ? 4u : 0u);
        }
    };
    float acc[2][TN][4];
    int32_t i1[2][TN][4];
#pragma unroll
    for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
        for (uint32_t b = 0; b < TN; b++)
#pragma unroll
            for (uint32_t e = 0; e < 4u; e++) acc[a][b][e] = 0.0f, i1[a][b][e] = 0;
#pragma unroll
    for (uint32_t s = 0; s + 1u < STAGES; s++) {
        if (s < stages) issue(s);
        copies_commit();
    }
    for (uint32_t s = 0; s < stages; s++) {
        copies_wait<STAGES - 2u>();
        barrier();
        if (s + STAGES - 1u < stages) issue(s + STAGES - 1u);
        copies_commit();
        const Stage &st = sh.st[s % STAGES];
        const uint32_t sb = s / PER, j = s % PER;
        const Super &su = sh.su[sb & 1u];
        /* the scales of this thread's 16 columns, both sub-blocks of the stage (a word each) */
        uint32_t pw[TN][2];
#pragma unroll
        for (uint32_t b = 0; b < TN; b++)
#pragma unroll
            for (uint32_t e = 0; e < 2u; e++) pw[b][e] = *(const uint32_t *)&su.pairs[8u * TN * wn + 8u * b + 2u * t + e][SB * j];
#pragma unroll
        for (uint32_t cb = 0; cb < SB; cb++) {
            /* A fragments (two tiles of 16 rows: rows 0-7 k 0-15, rows 8-15 k 0-15, rows 0-7 k 16-31, rows 8-15 k 16-31),
               B's of each tile of 8 columns from a word of codes: k 4t to 4t + 3 the low nibbles, 16 + 4t on the high */
            uint32_t af[2][4], b0[TN], b1[TN];
#pragma unroll
            for (uint32_t a = 0; a < 2u; a++)
                ldsm4(shared_address(st.a + chunk_a(32u * wm + 16u * a + (lane & 15u), 2u * cb + (lane >> 4))), af[a]);
#pragma unroll
            for (uint32_t b = 0; b < TN; b++) {
                const uint32_t r = 8u * TN * wn + 8u * b + g, w = *(const uint32_t *)(st.b + chunk_b(r, cb) + 4u * t);
                b0[b] = w & 0x0F0F0F0Fu;
                b1[b] = (w >> 4) & 0x0F0F0F0Fu;
                if (FIVE) {
                    const uint32_t f = st.f[FIVE ? r : 0u][cb];
                    b0[b] |= spread1((f >> (4u * t)) & 0xFu);
                    b1[b] |= spread1((f >> (16u + 4u * t)) & 0xFu);
                }
                if (NLX) {
                    b0[b] = nl_values(b0[b]);
                    b1[b] = nl_values(b1[b]);
                }
                if (SIX) {
                    /* (the high two bits of the four values of each half) */
                    const uint32_t piece = *(const uint16_t *)(st.h + 16u * r + 8u * cb + 2u * t);
                    b0[b] |= spread2(piece & 0xFFu);
                    b1[b] |= spread2(piece >> 8);
                }
            }
            if (SIX) {
                /* the halves of 16, each from minus 32 times the rows' sums of 16 (the codes less 32), times its scale */
                const uint32_t c16 = 2u * (SB * j + cb);
                int32_t off[2][2][2];
#pragma unroll
                for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
                    for (uint32_t h = 0; h < 2u; h++)
#pragma unroll
                        for (uint32_t half = 0; half < 2u; half++)
                            off[a][h][half] = -32 * su.xs[32u * wm + 16u * a + g + 8u * h][c16 + half];
#pragma unroll
                for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
                    for (uint32_t b = 0; b < TN; b++) {
                        int32_t lo[4], hi[4];
                        mma_s8u8_k16(lo, af[a][0], af[a][1], b0[b], off[a][0][0], off[a][1][0]);
                        mma_s8u8_k16(hi, af[a][2], af[a][3], b1[b], off[a][0][1], off[a][1][1]);
#pragma unroll
                        for (uint32_t e = 0; e < 4u; e++) {
                            const uint32_t pr = pw[b][e & 1u] >> (16u * cb);
                            i1[a][b][e] += (int32_t)(int8_t)(pr & 0xFFu) * lo[e];
                            i1[a][b][e] += (int32_t)(int8_t)((pr >> 8) & 0xFFu) * hi[e];
                        }
                    }
                continue;
            }
#pragma unroll
            for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
                for (uint32_t b = 0; b < TN; b++) {
                    int32_t s4[4];
                    mma_s8u8_k32(s4, af[a], b0[b], b1[b]);
#pragma unroll
                    for (uint32_t e = 0; e < 4u; e++) {
                        const uint32_t sc = (pw[b][e & 1u] >> (16u * cb)) & 0xFFu;
                        i1[a][b][e] += (NLX ? (int32_t)(int8_t)sc : (int32_t)sc) * s4[e];
                    }
                }
        }
        if (j == PER - 1u) {
            /* the super-block's end: I2 from the minima and the rows' sums of 32, then the terms, in order */
#pragma unroll
            for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
                for (uint32_t h = 0; h < 2u; h++) {
                    /* a row's sums of 32, then each of its outputs' I2 and term */
                    const uint32_t rm = 32u * wm + 16u * a + g + 8u * h;
                    const int32_t *xs = su.xs[rm];
                    int32_t x32[8];
#pragma unroll
                    for (uint32_t c = 0; c < 8u; c++) x32[c] = xs[2u * c] + xs[2u * c + 1u];
                    const float dx = su.dx[rm];
#pragma unroll
                    for (uint32_t b = 0; b < TN; b++)
#pragma unroll
                        for (uint32_t e1 = 0; e1 < 2u; e1++) {
                            const uint32_t cn = 8u * TN * wn + 8u * b + 2u * t + e1, e = e1 + 2u * h;
                            const uint4_ pr = *(const uint4_ *)su.pairs[cn];
                            const uint32_t words[4] = {pr.x, pr.y, pr.z, pr.w};
                            int32_t i2 = 0;
#pragma unroll
                            for (uint32_t c = 0; c < 8u; c++) i2 += (int32_t)((words[c / 2u] >> (16u * (c & 1u) + 8u)) & 0xFFu) * x32[c];
                            const uint32_t w = SIX ? su.dm[cn] >> (16u * (sb & 1u)) : su.dm[cn];
                            float term = (from_half(w & 0xFFFFu) * dx) * (float)i1[a][b][e];
                            if (!SIX && !NLX) term = term + (-from_half(w >> 16) * dx) * (float)i2;
                            acc[a][b][e] = acc[a][b][e] + term;
                            i1[a][b][e] = 0;
                        }
                }
        }
    }
    const uint32_t act = spec.v[2];
#pragma unroll
    for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
        for (uint32_t b = 0; b < TN; b++)
#pragma unroll
            for (uint32_t e = 0; e < 4u; e++) {
                const uint32_t n = n0 + 8u * TN * wn + 8u * b + 2u * t + (e & 1u), m = m0 + 32u * wm + 16u * a + g + 8u * (e >> 1);
                if (n >= p.n || m >= p.rows) continue;
                const uint64_t at = (uint64_t)(p.r0 + m) * p.ldy + n;
                float v = F(p.bias)[n] + acc[a][b][e];
                if (act) v = activate(v, act);
                if (p.e0) v = v * F(p.e0)[at];
                F(p.y)[at] = v;
            }
}
