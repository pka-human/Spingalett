/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * bmmp.comp: bmm's product for weights of blocks of 32 values kept packed on the device (the session's copy of a
 * layer: the codes of every row in a matrix of N rows of K signed bytes, then the blocks' scales as floats, N x K / 32;
 * the values of the image, rearranged), the CPU's bits: a workgroup takes BM rows of the quantized activations and BN
 * rows of weights, four warps of 32 x 64 outputs, over k in stages of SB blocks, STAGES of them in shared memory,
 * copied STAGES - 1 ahead by asynchronous copies (rows of codes in chunks of 16 bytes, swizzled so that ldmatrix's
 * eight rows fall in distinct banks). Each block's products by mma.sync m16n8k32 of 8-bit integers (exact) from an
 * accumulator of 1.5 2^23's bits (the sum's float then 1.5 2^23 less: exact below 2^22), then each output's block term
 * (d dx) I1, every product and sum rounded alone, added to its sum block after block. The bias then the sum, the
 * activation of spec 2, the product by e0's value: bdot's epilogue. Push: SpgBdotPush (r0 the first row, rows its rows,
 * stride unused); groups: the tiles of rows, then of weights. A unit a TYPE (Q8_0: bmmp_q8_0).
 */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_bmmp_q8_0
#define TYPE 0u
#endif

#define BM 64u
#define BN 128u
#define SB 2u                               /* blocks of 32 a stage */
#define STAGES 3u
#define ROWB (32u * SB)                     /* bytes of a row of codes a stage */
#define GR (ROWB / 16u)                     /* its chunks of 16 bytes */

typedef signed char int8_t;

DEVICE uint32_t shared_address(const void *p) {
    uint64_t r;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(r) : "l"(p));
    return (uint32_t)r;
}
DEVICE void copy4(uint32_t dst, const void *src, uint32_t bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
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
/* d = c + a b of 8-bit integers (16 x 32 by 32 x 8), exact */
DEVICE void mma_s8_k32(int32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, int32_t c) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c));
}

/* chunk g of row r of a tile's rows of codes: its place, swizzled (the chunk's index among the eight of 128 bytes
   XORed with the pair of rows: ldmatrix's eight rows in eight distinct groups of banks) */
DEVICE uint32_t chunk_at(uint32_t r, uint32_t g) {
    const uint32_t idx = r * GR + g;
    return 16u * (idx ^ ((idx >> 3) & 7u));
}

struct __attribute__((aligned(16))) Stage {       /* (a multiple of 16 bytes: the next stage's chunks aligned) */
    uint8_t a[BM * ROWB], b[BN * ROWB];     /* the rows' codes, the weights' */
    float dx[BM][SB], dw[BN][SB];           /* their blocks' scales */
};

extern "C" __global__ void __launch_bounds__(128) SPG_ENTRY(const SpgBdotPush p, const Spec spec) {
    __shared__ __attribute__((aligned(16))) Stage sh[STAGES];
    const uint32_t tid = thread_x(), lane = tid & 31u, warp = tid >> 5, wm = warp >> 1, wn = warp & 1u;
    const uint32_t g = lane >> 2, t = lane & 3u;
    const uint32_t m0 = block_x() * BM, n0 = block_y() * BN, blocks = p.k / 32u, stages = (blocks + SB - 1u) / SB;
    const int8_t *codes = (const int8_t *)p.w;
    const float *scales = (const float *)(p.w + (uint64_t)p.n * p.k);
    /* the copies of stage s: chunks of 16 bytes of the rows' and the weights' codes, eight bytes of their scales */
    auto issue = [&](uint32_t s) {
        Stage &st = sh[s % STAGES];
        const uint32_t b0 = s * SB, k0 = 32u * b0;
#pragma unroll
        for (uint32_t i = 0; i < BM * GR / 128u; i++) {
            const uint32_t u = tid + 128u * i, r = u / GR, c = u % GR, m = m0 + r;
            const bool live = m < p.rows && k0 + 16u * c < p.k;
            copy16(shared_address(st.a + chunk_at(r, c)),
                   (const int8_t *)p.q + (uint64_t)(p.r0 + (live ? m : 0u)) * p.k + (live ? k0 + 16u * c : 0u), live ? 16u : 0u);
        }
#pragma unroll
        for (uint32_t i = 0; i < BN * GR / 128u; i++) {
            const uint32_t u = tid + 128u * i, r = u / GR, c = u % GR, n = n0 + r;
            const bool live = n < p.n && k0 + 16u * c < p.k;
            copy16(shared_address(st.b + chunk_at(r, c)), codes + (uint64_t)(live ? n : 0u) * p.k + (live ? k0 + 16u * c : 0u),
                   live ? 16u : 0u);
        }
        /* (the scales of blocks past the end zeros) */
#pragma unroll
        for (uint32_t cb = 0; cb < SB; cb++) {
            const bool in = b0 + cb < blocks;
            if (tid < BM) {
                const uint32_t m = m0 + tid;
                const bool live = m < p.rows && in;
                copy4(shared_address(&st.dx[tid][cb]), F(p.dx) + (live ? (uint64_t)(p.r0 + m) * blocks + b0 + cb : 0u),
                      live ? 4u : 0u);
            }
            const uint32_t n = n0 + tid;
            const bool live = n < p.n && in;
            copy4(shared_address(&st.dw[tid][cb]), scales + (live ? (uint64_t)n * blocks + b0 + cb : 0u), live ? 4u : 0u);
        }
    };
    float acc[2][8][4];
#pragma unroll
    for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
        for (uint32_t b = 0; b < 8u; b++)
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
        const Stage &st = sh[s % STAGES];
#pragma unroll
        for (uint32_t cb = 0; cb < SB; cb++) {
            if (s * SB + cb >= blocks) continue;
            /* the block's A fragments (two tiles of 16 rows: ldmatrix x4 of rows 0-7 k 0-15, rows 8-15 k 0-15, rows 0-7
               k 16-31, rows 8-15 k 16-31) and B fragments (four of 16 columns: n 0-7 k 0-15, n 0-7 k 16-31, n 8-15 k
               0-15, n 8-15 k 16-31) */
            uint32_t af[2][4], bf[4][4];
#pragma unroll
            for (uint32_t a = 0; a < 2u; a++)
                ldsm4(shared_address(st.a + chunk_at(32u * wm + 16u * a + (lane & 15u), 2u * cb + (lane >> 4))), af[a]);
#pragma unroll
            for (uint32_t b = 0; b < 4u; b++)
                ldsm4(shared_address(st.b + chunk_at(64u * wn + 16u * b + (lane & 7u) + 8u * (lane >> 4),
                                                     2u * cb + ((lane >> 3) & 1u))), bf[b]);
            float dx[2][2], dw[8][2];
#pragma unroll
            for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
                for (uint32_t e = 0; e < 2u; e++) dx[a][e] = st.dx[32u * wm + 16u * a + g + 8u * e][cb];
#pragma unroll
            for (uint32_t b = 0; b < 8u; b++)
#pragma unroll
                for (uint32_t e = 0; e < 2u; e++) dw[b][e] = st.dw[64u * wn + 8u * b + 2u * t + e][cb];
#pragma unroll
            for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
                for (uint32_t b = 0; b < 8u; b++) {
                    int32_t s4[4];
                    mma_s8_k32(s4, af[a], bf[b >> 1][2u * (b & 1u)], bf[b >> 1][2u * (b & 1u) + 1u], 0x4B400000);
                    /* the block's terms, in order: (float)I1 the bits of 1.5 2^23 + I1, less 1.5 2^23 */
#pragma unroll
                    for (uint32_t e = 0; e < 4u; e++) {
                        const float i1 = __builtin_bit_cast(float, s4[e]) - 12582912.0f;
                        const float term = (dw[b][e & 1u] * dx[a][e >> 1]) * i1;
                        acc[a][b][e] = acc[a][b][e] + term;
                    }
                }
        }
    }
    const uint32_t act = spec.v[2];
#pragma unroll
    for (uint32_t a = 0; a < 2u; a++)
#pragma unroll
        for (uint32_t b = 0; b < 8u; b++)
#pragma unroll
            for (uint32_t e = 0; e < 4u; e++) {
                const uint32_t n = n0 + 64u * wn + 8u * b + 2u * t + (e & 1u), m = m0 + 32u * wm + 16u * a + g + 8u * (e >> 1);
                if (n >= p.n || m >= p.rows) continue;
                const uint64_t at = (uint64_t)(p.r0 + m) * p.ldy + n;
                float v = F(p.bias)[n] + acc[a][b][e];
                if (act) v = activate(v, act);
                if (p.e0) v = v * F(p.e0)[at];
                F(p.y)[at] = v;
            }
}
