/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * bmmk.comp: bmm's product for weights of Q4_K, Q5_K, Q6_K and IQ4_XS packed as sessions keep them (Spingalett.GpuPack.c:
 * rows `stride` bytes apart), the CPU's bits:
 * tiles of 64 rows by BN weights, four warps of 32 x BN / 2 outputs, k in stages of two sub-blocks of 32 values, STAGES
 * of them in shared memory by asynchronous copies two ahead (the rows' codes in chunks of 16 bytes swizzled for
 * ldmatrix; the weights' codes, a sub-block's 16 bytes, swizzled so that a warp's words fall in distinct banks); each
 * super-block's scales, minima, the rows' sums of 16 and scales in a buffer of its own (two, by the super-block's
 * parity). Each sub-block's products by mma.sync m16n8k32 of 8-bit integers (a lane's word of codes gives both halves of
 * its fragment: the low nibbles, then the high ones), times the sub-block's scale, added in integers (I1); at a
 * super-block's end its minima times the rows' sums of 32 (I2, on the matrix units too: the sums of 16 as 64 h + l, h
 * and l bytes, by each minimum twice, the products by h 64 times those by l), then each output's term
 * (d dx) I1 + (m dx) I2, every product and sum rounded alone, added to its sum. The bias then the sum, the activation of spec 2, the product by
 * e0's value: bdot's epilogue. Push: SpgBdotPush (r0 the first row, rows its rows); groups: the tiles of rows, then
 * of weights. A unit a TYPE (Q4_K: bmmp_q4_k).
 */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_bmmp_q4_k
#define TYPE 7u
#endif

#define WN 2u                               /* warps along the weights (two along the rows) */
#if !defined(TN)
#define TN 4u                               /* tiles of 8 columns a warp */
#endif
#define THREADS (64u * WN)
#define SIX (TYPE == SPG_B_Q6_K)            /* Q6_K: halves of 16 of their own scales, two high bits, no minima */
#define FIVE (TYPE == SPG_B_Q5_K)           /* Q5_K: Q4_K's and a fifth bit */
#define NLX (TYPE == SPG_B_IQ4_XS)          /* IQ4_XS: IQ4_NL's values, signed scales, no minima */
#define MINIMA (!SIX && !NLX)               /* Q4_K, Q5_K */
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
/* d = a b of 8-bit integers (16 x 16 signed by 16 x 8 unsigned), exact */
DEVICE void mma_s8u8_k16(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %7, %7, %7};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a0), "r"(a1), "r"(b), "r"(0));
}
DEVICE uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
/* four bits of x at bit 4 of four bytes */
DEVICE uint32_t spread1(uint32_t x) { return ((x * 0x00204081u) & 0x01010101u) << 4; }
/* d = c + a b of 8-bit integers (16 x 16 signed by 16 x 8 unsigned), exact */
DEVICE void mma_k16(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t b, const int32_t (&c)[4]) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}
/* d = c + a b of 8-bit integers (16 x 16 by 16 x 8 signed: a signed (S), else unsigned), exact */
template <bool S>
DEVICE void mma_k16_s(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t b, const int32_t (&c)[4]) {
    if (S)
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
            : "r"(a0), "r"(a1), "r"(b), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
    else
        asm("mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
            : "r"(a0), "r"(a1), "r"(b), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
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
/* sub-block cb of row r of the weights' codes (rows of 8 words), its first word, swizzled: rows 4-7 of every 8 the
   other way round */
DEVICE uint32_t chunk_b(uint32_t r, uint32_t cb) { return 8u * r + 4u * (cb ^ ((r >> 2) & 1u)); }

/* (the arrays read a word or more at a time are of words, not bytes: the compiler loaded bytes of byte arrays) */
struct __attribute__((aligned(16))) Stage {       /* (a multiple of 16 bytes: the next stage's chunks aligned) */
    uint8_t a[BM * 64u];                    /* the rows' codes */
    uint32_t b[BN * 8u];                    /* the weights' (Q6_K: their low four bits) */
    uint32_t h[SIX ? BN * 4u : 4u];         /* Q6_K: their high two bits, a word a lane of the stage's two sub-blocks */
    uint32_t f[FIVE ? BN : 4u][SB];         /* Q5_K: their fifth bits */
};
struct Super {
    uint32_t pairs[BN][4];                  /* the sub-blocks' scale | minimum << 8 (IQ4_XS: the signed scale; Q6_K: the
                                               halves' signed scales), two sub-blocks a word */
    int32_t xs[BM][16];                     /* the rows' sums of 16 codes (Q4_K, Q5_K: then in their place those sums as
                                               64 h + l, the bytes h then the bytes l: three workgroups an SM) */
    uint32_t dm[BN];                        /* the scale and minimum (Q6_K: two scales), float16 */
    float dx[BM];                           /* the rows' scales */
};
struct Shared {
    Stage st[STAGES];
    Super su[2];
    uint32_t heads[SIX ? 1u : BN][4];       /* a super-block's first 16 bytes as GGUF keeps them (IQ4_XS: 8), copied a
                                               stage ahead of the super-block and spread into its pairs and dm */
};

extern "C" __global__ void __launch_bounds__(THREADS) SPG_ENTRY(const SpgBdotPush p, const Spec spec) {
    __shared__ __attribute__((aligned(16))) Shared sh;
    const uint32_t tid = thread_x(), lane = tid & 31u, warp = tid >> 5, wm = warp / WN, wn = warp % WN;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t m0 = block_x() * BM, n0 = block_y() * BN, subs = p.k / 32u, blocks = p.k / 256u, stages = subs / SB;
    /* (a row's codes, Q6_K's high bits, then its fifth bits, the super-blocks' first bytes; Q6_K: the halves' scales,
       then the super-blocks') */
    const uint8_t *weights = (const uint8_t *)p.w;
    const uint32_t aux = SPG_PACK_CODES(TYPE, p.k), pairs = aux + (FIVE ? 4u * subs : 0u), dm = pairs + 2u * subs;
    const uint32_t head = NLX ? 8u : 16u;   /* (bytes of a super-block's first, at `pairs`) */
    /* this thread's copies, their sources at stage 0 and their sizes, computed once (past the rows and the weights the
       first row's, zero bytes of it) */
    const uint32_t base = shared_address(&sh);
    const int8_t *src_a[BM * 4u / THREADS];
    uint32_t to_a[BM * 4u / THREADS], size_a[BM * 4u / THREADS];
#pragma unroll
    for (uint32_t i = 0; i < BM * 4u / THREADS; i++) {
        const uint32_t u = tid + THREADS * i, r = u >> 2, c = u & 3u, m = m0 + r;
        src_a[i] = (const int8_t *)p.q + (m < p.rows ? (uint64_t)(p.r0 + m) * p.k : 0u) + 16u * c;
        to_a[i] = __builtin_offsetof(Stage, a) + chunk_a(r, c);
        size_a[i] = m < p.rows ? 16u : 0u;
    }
#define CB (BN * SB / THREADS)              /* copies of the weights' codes a thread */
    const uint8_t *src_b[CB];
    uint32_t to_b[CB], size_b[CB];
#pragma unroll
    for (uint32_t i = 0; i < CB; i++) {
        const uint32_t u = tid + THREADS * i, rb = u / SB, nb = n0 + rb;
        src_b[i] = weights + (nb < p.n ? (uint64_t)nb * p.stride : 0u) + 16u * (u % SB);
        to_b[i] = __builtin_offsetof(Stage, b) + 4u * chunk_b(rb, u % SB);
        size_b[i] = nb < p.n ? 16u : 0u;
    }
    /* (the first BN threads: a weight's scales and bits; the first BM: a row's sums and scales) */
    static_assert(BN <= THREADS && BM <= THREADS, "a weight and a row a thread");
    const uint32_t ns = n0 + tid, ms = m0 + tid, size_w = ns < p.n ? 4u : 0u, size_x = ms < p.rows ? 4u : 0u;
    const uint8_t *row = weights + (ns < p.n ? (uint64_t)ns * p.stride : 0u);
    const float *dx_row = F(p.dx) + (ms < p.rows ? (uint64_t)(p.r0 + ms) * blocks : 0u);
    /* (the rows' sums of 16 of a super-block, 64 bytes a row: four threads a row, a warp's copies eight rows) */
#define CX (BM * 4u / THREADS)
    const int32_t *src_x[CX];
    uint32_t to_x[CX], size_xs[CX];
#pragma unroll
    for (uint32_t i = 0; i < CX; i++) {
        const uint32_t u = tid + THREADS * i, r = u >> 2, m = m0 + r;
        src_x[i] = (const int32_t *)p.xs + (m < p.rows ? (uint64_t)(p.r0 + m) * (p.k / 16u) : 0u) + 4u * (u & 3u);
        to_x[i] = __builtin_offsetof(Super, xs) + 64u * r + 16u * (u & 3u);
        size_xs[i] = m < p.rows ? 16u : 0u;
    }
    /* the copies of stage s, to the buffer `slot` */
    auto issue = [&](uint32_t s, uint32_t slot) {
        const uint32_t st = base + __builtin_offsetof(Shared, st) + slot * (uint32_t)sizeof(Stage);
#pragma unroll
        for (uint32_t i = 0; i < BM * 4u / THREADS; i++) copy16(st + to_a[i], src_a[i] + 32u * SB * s, size_a[i]);
#pragma unroll
        for (uint32_t i = 0; i < CB; i++) copy16(st + to_b[i], src_b[i] + 16u * SB * s, size_b[i]);
        const uint32_t sb = s / PER, j = s % PER;
        const uint32_t su = base + __builtin_offsetof(Shared, su) + (sb & 1u) * (uint32_t)sizeof(Super);
        if (tid < BN) {
            if (FIVE) copy8(st + __builtin_offsetof(Stage, f) + 4u * SB * tid, row + aux + 4u * SB * s, 2u * size_w);
            if (SIX) copy16(st + __builtin_offsetof(Stage, h) + 16u * tid, row + p.k / 2u + 8u * SB * s, 4u * size_w);
            /* (a super-block's scales at its first stage, 16 bytes: copies of few bytes from many rows cost as much as
               the codes; Q6_K: a float16 scale a super-block, the word of this one's and its neighbour's. The others:
               its first bytes a stage before its first, the first super-block's with the first stage) */
            if (SIX && j == 0u) {
                copy16(su + __builtin_offsetof(Super, pairs) + 16u * tid, row + pairs + 16u * sb, 4u * size_w);
                copy4(su + __builtin_offsetof(Super, dm) + 4u * tid, row + dm + 4u * (sb / 2u), size_w);
            }
            const uint32_t next = s == 0u ? 0u : (s + 1u) / PER;
            if (!SIX && (s == 0u || (j == PER - 1u && next < blocks))) {
                const uint32_t to = base + __builtin_offsetof(Shared, heads) + 16u * tid;
                if (NLX) copy8(to, row + pairs + 8u * next, 2u * size_w);
                else copy16(to, row + pairs + 16u * next, 4u * size_w);
            }
        }
        if (!NLX && j == 0u)
#pragma unroll
            for (uint32_t i = 0; i < CX; i++) copy16(su + to_x[i], src_x[i] + 16u * sb, size_xs[i]);
        if (tid < BM) {
            if (j == 0u) {
                copy4(su + __builtin_offsetof(Super, dx) + 4u * tid, dx_row + sb, size_x);
            }
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
        if (s < stages) issue(s, s);
        copies_commit();
    }
    /* (the buffers of the stage read and of the one written, counted round) */
    uint32_t rd = 0u, wr = STAGES - 1u;
    for (uint32_t s = 0; s < stages; s++) {
        copies_wait<STAGES - 2u>();
        barrier();
        if (s + STAGES - 1u < stages) issue(s + STAGES - 1u, wr);
        copies_commit();
        const Stage &st = sh.st[rd];
        rd = rd == STAGES - 1u ? 0u : rd + 1u;
        wr = wr == STAGES - 1u ? 0u : wr + 1u;
        const uint32_t sb = s / PER, j = s % PER;
        const Super &su = sh.su[sb & 1u];
        if (!SIX && (s == 0u || (j == PER - 1u && sb + 1u < blocks))) {
            /* the next super-block's first bytes (with the first stage the first's) spread into its pairs and dm, a
               thread half a column (sub-blocks 4 part to 4 part + 3; shown to the other warps by the next stage's
               barrier, the first super-block's by one of its own) */
            static_assert(THREADS == 2u * BN, "half a column a thread");
            Super &sw = sh.su[(s == 0u ? 0u : sb + 1u) & 1u];
            const uint32_t col = tid >> 1, part = tid & 1u;
            const uint32_t w0 = sh.heads[col][0], w1 = sh.heads[col][1];
            uint32_t lo, hi;
            if (NLX) {
                lo = ((uint32_t)iq4xs_scale(w0, w1, 4u * part) & 0xFFu) | ((uint32_t)iq4xs_scale(w0, w1, 4u * part + 1u) & 0xFFu) << 16;
                hi = ((uint32_t)iq4xs_scale(w0, w1, 4u * part + 2u) & 0xFFu) | ((uint32_t)iq4xs_scale(w0, w1, 4u * part + 3u) & 0xFFu) << 16;
                if (part == 0u) sw.dm[col] = w0 & 0xFFFFu;
            } else {
                uint32_t sc, mn;
                k4_scales(w1, sh.heads[col][2], sh.heads[col][3], part, sc, mn);
                lo = prmt(sc, mn, 0x5140u);
                hi = prmt(sc, mn, 0x7362u);
                if (part == 0u) sw.dm[col] = w0;
            }
            sw.pairs[col][2u * part] = lo;
            sw.pairs[col][2u * part + 1u] = hi;
            if (s == 0u) barrier();
        }
        if (!NLX && j == 0u && tid < BM) {
            /* this row's sums of 16 codes of the super-block as 64 h + l (h of -32 to 31, l of 0 to 63), the bytes of the
               minima's products (Q6_K: of the halves' scales'; shown to the other warps by the next stage's barrier: a
               super-block has four) */
            Super &sw = sh.su[sb & 1u];
            uint32_t hl[8];
#pragma unroll
            for (uint32_t c = 0; c < 4u; c++) {
                const int32_t *x = &sw.xs[tid][4u * c];
                hl[c] = ((uint32_t)(x[0] >> 6) & 0xFFu) | ((uint32_t)(x[1] >> 6) & 0xFFu) << 8 |
                        ((uint32_t)(x[2] >> 6) & 0xFFu) << 16 | (uint32_t)(x[3] >> 6) << 24;
                hl[4u + c] = ((uint32_t)x[0] & 63u) | ((uint32_t)x[1] & 63u) << 8 | ((uint32_t)x[2] & 63u) << 16 |
                             ((uint32_t)x[3] & 63u) << 24;
            }
            *(uint4_ *)&sw.xs[tid][0] = uint4_{hl[0], hl[1], hl[2], hl[3]};
            *(uint4_ *)&sw.xs[tid][4] = uint4_{hl[4], hl[5], hl[6], hl[7]};
        }
        /* the scales of this thread's 16 columns, both sub-blocks of the stage (a word each) */
        uint32_t pw[TN][2];
#pragma unroll
        for (uint32_t b = 0; b < TN; b++)
#pragma unroll
            for (uint32_t e = 0; e < 2u; e++) pw[b][e] = su.pairs[8u * TN * wn + 8u * b + 2u * t + e][j];
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
                const uint32_t r = 8u * TN * wn + 8u * b + g, w = st.b[chunk_b(r, cb) + t];
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
                    /* (the high two bits of the four values of each half: bits 0 to 3 of the word's bytes the first
                       sub-block's, 4 to 7 the second's) */
                    const uint32_t hw = st.h[4u * r + t];
                    b0[b] |= cb ? hw & 0x30303030u : (hw << 4) & 0x30303030u;
                    b1[b] |= cb ? (hw >> 2) & 0x30303030u : (hw << 2) & 0x30303030u;
                }
            }
            if (SIX) {
                /* the halves of 16 of the codes as they are (0 to 63), each times its scale (the codes less 32: the
                   super-block's end takes 32 times the scales by the rows' sums of 16 away) */
#pragma unroll
                for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
                    for (uint32_t b = 0; b < TN; b++) {
                        int32_t lo[4], hi[4];
                        mma_s8u8_k16(lo, af[a][0], af[a][1], b0[b]);
                        mma_s8u8_k16(hi, af[a][2], af[a][3], b1[b]);
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
            /* the super-block's end: I2, the minima times the rows' sums of 32, on the matrix units (k the 16 sums of 16,
               each minimum twice: by the h bytes, 64 times, then by the l bytes; Q6_K: the halves' scales by the sums of
               16, I1 less 32 times them); then the terms, in order */
            /* (a tile at a time: its I2 lives four registers) */
#pragma unroll
            for (uint32_t a = 0; a < 2u; a++) {
                const uint32_t r = 32u * wm + 16u * a + g;
                const float dx[2] = {su.dx[r], su.dx[r + 8u]};
                uint32_t h0 = 0u, h1 = 0u, l0 = 0u, l1 = 0u;
                if (!NLX) {
                    h0 = (uint32_t)su.xs[r][t], h1 = (uint32_t)su.xs[r + 8u][t];
                    l0 = (uint32_t)su.xs[r][4u + t], l1 = (uint32_t)su.xs[r + 8u][4u + t];
                }
#pragma unroll
                for (uint32_t b = 0; b < TN; b++) {
                    int32_t i2[4] = {0, 0, 0, 0};
                    if (MINIMA) {
                        const uint32_t mb = prmt(su.pairs[8u * TN * wn + 8u * b + g][t], 0u, 0x3311u);
                        int32_t d[4];
                        mma_k16(d, h0, h1, mb, i2);
                        const int32_t c[4] = {d[0] * 64, d[1] * 64, d[2] * 64, d[3] * 64};
                        mma_k16(i2, l0, l1, mb, c);
                    }
                    if (SIX) {
                        /* (the scales of halves 4t to 4t + 3 of the column g, signed) */
                        const uint32_t sb4 = su.pairs[8u * TN * wn + 8u * b + g][t];
                        int32_t d[4];
                        mma_k16_s<true>(d, h0, h1, sb4, i2);
                        const int32_t c[4] = {d[0] * 64, d[1] * 64, d[2] * 64, d[3] * 64};
                        mma_k16_s<false>(i2, l0, l1, sb4, c);
#pragma unroll
                        for (uint32_t e = 0; e < 4u; e++) i1[a][b][e] -= 32 * i2[e];
                    }
#pragma unroll
                    for (uint32_t e = 0; e < 4u; e++) {
                        const uint32_t cn = 8u * TN * wn + 8u * b + 2u * t + (e & 1u);
                        const uint32_t w = SIX ? su.dm[cn] >> (16u * (sb & 1u)) : su.dm[cn];
                        float term = (from_half(w & 0xFFFFu) * dx[e >> 1]) * (float)i1[a][b][e];
                        if (MINIMA) term = term + (-from_half(w >> 16) * dx[e >> 1]) * (float)i2[e];
                        acc[a][b][e] = acc[a][b][e] + term;
                        i1[a][b][e] = 0;
                    }
                }
            }
        }
    }
    const uint32_t act = spec.v[2];
    const bool pair = spec.v[3] != 0u;          /* (a gated unit's pair: a thread's columns 2j and 2j + 1, block_out()) */
#pragma unroll
    for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
        for (uint32_t b = 0; b < TN; b++)
#pragma unroll
            for (uint32_t h = 0; h < 2u; h++) {
                const uint32_t n = n0 + 8u * TN * wn + 8u * b + 2u * t, m = m0 + 32u * wm + 16u * a + g + 8u * h;
                if (m >= p.rows) continue;
                if (pair) {
                    if (n < p.n) block_out(p, act, true, m, n / 2u, acc[a][b][2u * h], acc[a][b][2u * h + 1u]);
                    continue;
                }
#pragma unroll
                for (uint32_t e = 0; e < 2u; e++)
                    if (n + e < p.n) block_out(p, act, false, m, n + e, acc[a][b][2u * h + e], 0.0f);
            }
}
