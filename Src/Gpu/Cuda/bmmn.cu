/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * bmmn.comp: bmm's product for weights of blocks of 32 4-bit codes (Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL) packed as sessions
 * keep them (Spingalett.GpuPack.c: rows `stride` bytes apart), the CPU's bits: tiles of 64 rows by 64 weights, four warps of 32 x 32 outputs, k in
 * stages of two blocks, STAGES of them in shared memory by asynchronous copies two ahead (the scales, fifth bits and
 * rows' sums of GB blocks at once, in buffers of their own, a warp's copies a few rows: copies of a few bytes from many
 * rows cost as much as the codes). A lane's word of a block's
 * codes gives both halves of its fragment (the low nibbles values 4t to 4t + 3, the high ones 16 + 4t on), with the
 * fifth bits (Q5_0, Q5_1) at bit 4 of each byte, or IQ4_NL's values from its table. Each block's products by mma.sync
 * m16n8k32 of 8-bit integers (exact) from an accumulator of 1.5 2^23's bits less the codes' offset (8: Q4_0, 16: Q5_0)
 * times the rows' sums of 32 (the CPU's I1 = the codes less the offset times the rows' codes); its float then
 * 1.5 2^23 less, exactly; then each output's term (d dx) I1 (Q4_1, Q5_1: + (m dx) times the rows' sum), every product
 * and sum rounded alone, added to its sum block after block. The bias then the sum, the activation of spec 2, the
 * product by e0's value: bdot's epilogue. Push: SpgBdotPush (r0 the first row, rows its rows); groups: the tiles of
 * rows, then of weights. A unit a TYPE (bmmp_q4_0, ...).
 */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_bmmp_q4_0
#define TYPE 1u
#endif

#define ONE (TYPE == SPG_B_Q4_1 || TYPE == SPG_B_Q5_1)       /* a minimum a block */
#define FIVE (TYPE == SPG_B_Q5_0 || TYPE == SPG_B_Q5_1)      /* fifth bits */
#define NL (TYPE == SPG_B_IQ4_NL)                            /* IQ4_NL's table of values */
#define OFF (TYPE == SPG_B_Q4_0 ? 8 : TYPE == SPG_B_Q5_0 ? 16 : 0)
#define WN 2u
#define TN 4u
#define THREADS (64u * WN)
#define BM 64u
#define BN (8u * TN * WN)
#define SB 2u                               /* blocks of 32 a stage */
#define STAGES 3u
#define GB 4u                               /* blocks of a group of scales */
#define GS (GB / SB)                        /* stages a group */

typedef signed char int8_t;

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
/* d = c + a b of 8-bit integers (16 x 32 signed by 32 x 8, unsigned or signed), exact: c's rows g (c01), g + 8 (c23) */
DEVICE void mma_k32(int32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, int32_t c01, int32_t c23) {
    if (NL)
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%10, %10, %11, %11};"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c01), "r"(c23));
    else
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
            "{%10, %10, %11, %11};"
            : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c01), "r"(c23));
}
/* four bits of x at bit 4 of four bytes */
DEVICE uint32_t spread1(uint32_t x) { return ((x * 0x00204081u) & 0x01010101u) << 4; }
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
/* block cb of row r of the weights' codes (rows of 8 words), its first word, swizzled: rows 4-7 of every 8 the other
   way round */
DEVICE uint32_t chunk_b(uint32_t r, uint32_t cb) { return 8u * r + 4u * (cb ^ ((r >> 2) & 1u)); }

/* (the arrays read a word at a time are of words, not bytes: the compiler loaded bytes of byte arrays) */
struct __attribute__((aligned(16))) Stage {       /* (a multiple of 16 bytes: the next stage's chunks aligned) */
    uint8_t a[BM * 64u];                    /* the rows' codes */
    uint32_t b[BN * 8u];                    /* the weights' */
};
#define DW (ONE ? GB : GB / 2u)             /* words of scales a weight a group */
struct __attribute__((aligned(16))) Group {
    uint32_t f[FIVE ? BN : 1u][GB];         /* the weights' fifth bits */
    uint32_t d[BN][DW];                     /* their blocks' scales (and minima), float16 */
    float dx[BM][GB];                       /* the rows' */
    int32_t xs[BM][2u * GB];                /* the rows' sums of 16 codes */
};
struct Shared {
    Stage st[STAGES];
    Group gr[2];                            /* (by the group's parity) */
};

extern "C" __global__ void __launch_bounds__(THREADS) SPG_ENTRY(const SpgBdotPush p, const Spec spec) {
    __shared__ __attribute__((aligned(16))) Shared sh;
    const uint32_t tid = thread_x(), lane = tid & 31u, warp = tid >> 5, wm = warp / WN, wn = warp % WN;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t m0 = block_x() * BM, n0 = block_y() * BN, blocks = p.k / 32u, stages = (blocks + SB - 1u) / SB;
    const uint8_t *weights = (const uint8_t *)p.w;
    const uint32_t aux = p.k / 2u, scales = aux + (FIVE ? 4u * blocks : 0u);  /* (their offsets in a row) */
    /* the copies of stage s (zeros past the rows, the weights and the blocks) */
    auto issue = [&](uint32_t s) {
        Stage &st = sh.st[s % STAGES];
#pragma unroll
        for (uint32_t i = 0; i < BM * 4u / THREADS; i++) {
            const uint32_t u = tid + THREADS * i, r = u >> 2, c = u & 3u, m = m0 + r;
            const bool live = m < p.rows && SB * s + c / 2u < blocks;
            copy16(shared_address(st.a + chunk_a(r, c)),
                   (const int8_t *)p.q + (live ? (uint64_t)(p.r0 + m) * p.k + 32u * SB * s + 16u * c : 0u), live ? 16u : 0u);
        }
#pragma unroll
        for (uint32_t i = 0; i < BN * SB / THREADS; i++) {
            const uint32_t u = tid + THREADS * i, r = u / SB, cb = u % SB, n = n0 + r;
            const bool live = n < p.n && SB * s + cb < blocks;
            copy16(shared_address(&st.b[chunk_b(r, cb)]), weights + (live ? (uint64_t)n * p.stride + 16u * (SB * s + cb) : 0u),
                   live ? 16u : 0u);
        }
        if (s % GS != 0u) return;
        /* a group's scales, fifth bits and sums: words of a row in consecutive lanes (past the weights, the rows and
           the blocks: zeros) */
        Group &gr = sh.gr[(s / GS) & 1u];
        const uint32_t b0 = SB * s;
        auto rows = [&](uint32_t words, auto &&copy) {
#pragma unroll
            for (uint32_t i = 0; i < (64u * words + THREADS - 1u) / THREADS; i++) {
                const uint32_t u = tid + THREADS * i;
                if (u < 64u * words) copy(u / words, u % words);
            }
        };
        if (FIVE)
            rows(GB, [&](uint32_t r, uint32_t c) {
                const uint32_t n = n0 + r;
                const bool live = n < p.n && b0 + c < blocks;
                copy4(shared_address(&gr.f[FIVE ? r : 0u][c]), weights + (live ? (uint64_t)n * p.stride + aux + 4u * (b0 + c) : 0u),
                      live ? 4u : 0u);
            });
        /* (a word of two float16 scales past the last block: the next bytes of the row, unread) */
        rows(DW, [&](uint32_t r, uint32_t c) {
            const uint32_t n = n0 + r;
            const bool live = n < p.n && b0 + (ONE ? c : 2u * c) < blocks;
            copy4(shared_address(&gr.d[r][c]), weights + (live ? (uint64_t)n * p.stride + scales + 4u * (b0 / (ONE ? 1u : 2u) + c) : 0u),
                  live ? 4u : 0u);
        });
        rows(GB, [&](uint32_t r, uint32_t c) {
            const uint32_t m = m0 + r;
            const bool live = m < p.rows && b0 + c < blocks;
            copy4(shared_address(&gr.dx[r][c]), F(p.dx) + (live ? (uint64_t)(p.r0 + m) * blocks + b0 + c : 0u), live ? 4u : 0u);
        });
        rows(GB, [&](uint32_t r, uint32_t c) {
            const uint32_t m = m0 + r;
            const bool live = m < p.rows && b0 + c < blocks;
            copy8(shared_address(&gr.xs[r][2u * c]), (const int32_t *)p.xs + (live ? (uint64_t)(p.r0 + m) * (p.k / 16u) + 2u * (b0 + c) : 0u),
                  live ? 8u : 0u);
        });
    };
    float acc[2][TN][4];
#pragma unroll
    for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
        for (uint32_t b = 0; b < TN; b++)
#pragma unroll
            for (uint32_t e = 0; e < 4u; e++) acc[a][b][e] = 0.0f;
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
        const Group &gr = sh.gr[(s / GS) & 1u];
        const uint32_t gc = SB * (s % GS);        /* (the stage's first block in its group) */
#pragma unroll
        for (uint32_t cb = 0; cb < SB; cb++) {
            if (SB * s + cb >= blocks) continue;
            /* A fragments (two tiles of 16 rows: rows 0-7 k 0-15, rows 8-15 k 0-15, rows 0-7 k 16-31, rows 8-15 k 16-31),
               B's of each tile of 8 columns from a word of codes */
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
                    const uint32_t f = gr.f[FIVE ? r : 0u][gc + cb];
                    b0[b] |= spread1((f >> (4u * t)) & 0xFu);
                    b1[b] |= spread1((f >> (16u + 4u * t)) & 0xFu);
                }
                if (NL) {
                    b0[b] = nl_values(b0[b]);
                    b1[b] = nl_values(b1[b]);
                }
            }
            /* the rows' scales, their accumulators' start (1.5 2^23's bits less the offset times the rows' sums of 32),
               the sums as floats (the minima's I2) */
            float dx[2][2], x2[2][2];
            int32_t c0[2][2];
#pragma unroll
            for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
                for (uint32_t h = 0; h < 2u; h++) {
                    const uint32_t rm = 32u * wm + 16u * a + g + 8u * h;
                    const int32_t sum = gr.xs[rm][2u * (gc + cb)] + gr.xs[rm][2u * (gc + cb) + 1u];
                    dx[a][h] = gr.dx[rm][gc + cb];
                    c0[a][h] = 0x4B400000 - OFF * sum;
                    x2[a][h] = (float)sum;
                }
#pragma unroll
            for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
                for (uint32_t b = 0; b < TN; b++) {
                    int32_t s4[4];
                    mma_k32(s4, af[a], b0[b], b1[b], c0[a][0], c0[a][1]);
#pragma unroll
                    for (uint32_t e = 0; e < 4u; e++) {
                        const uint32_t cn = 8u * TN * wn + 8u * b + 2u * t + (e & 1u);
                        const float i1 = __builtin_bit_cast(float, s4[e]) - 12582912.0f;
                        const uint32_t dw = ONE ? gr.d[cn][gc + cb] : gr.d[cn][(gc + cb) / 2u] >> (16u * (cb & 1u));
                        float term = (from_half(dw & 0xFFFFu) * dx[a][e >> 1]) * i1;
                        if (ONE) term = term + (from_half(dw >> 16) * dx[a][e >> 1]) * x2[a][e >> 1];
                        acc[a][b][e] = acc[a][b][e] + term;
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
