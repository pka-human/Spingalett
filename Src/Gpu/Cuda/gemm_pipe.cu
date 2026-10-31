/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The pipelined tiles of the product on the matrix units: blocks of BM x BN outputs over warps of 16 TI x 16 TJ, both
 * operands kept as bfloat16 and read eight values at a time (dense layers' modes: A_ROW or A_COL, B_ROW or
 * B_COL), STAGES steps of BK in shared memory, copied STAGES - 1 steps ahead by asynchronous copies. Shared
 * tiles keep each operand's layout in memory, chunks of 16 bytes XOR-swizzled within their rows (swizzle())
 * so that the eight rows ldmatrix reads, and the chunks a warp's copies write, fall in distinct banks without
 * padding. Fragments come from ldmatrix a k-step of 16 ahead of their products, and the results go to the
 * epilogue from the registers, eight consecutive outputs of a row a thread (units with GENERAL: through shared
 * memory, as gemm_mma.cu's).
 *
 * An output's products are added 16 values of k at a time, in the order of k, by the same instruction as in
 * gemm_mma.cu's tiles, over the same steps of 32 (BK is 32: zeros beyond the sum's end in the same places),
 * and the epilogue computes what gemm_mma.cu's does: the tiles change the speed only.
 */

#include "mma.cuh"

/* A tile's 16-byte chunk idx (row r, chunk c of rows of `len` chunks) at chunk swizzle(idx) of shared memory:
   its place among the eight of its 128 bytes XORed with the row (with rows of four chunks, two rows to the
   128 bytes: with the pair of rows). The eight rows of 16 bytes ldmatrix reads, from rows of a multiple of
   eight, land in eight distinct groups of banks. */
template <uint32_t LEN>
DEVICE uint32_t swizzle(uint32_t idx) {
    constexpr uint32_t SHIFT = LEN <= 8u ? 3u : LEN == 16u ? 4u : LEN == 32u ? 5u : 6u;
    return idx ^ ((idx >> SHIFT) & 7u);
}

struct uint4_ {
    uint32_t x, y, z, w;
} __attribute__((aligned(16)));

DEVICE float shfl_xor(float v, uint32_t mask) { return __nvvm_shfl_sync_bfly_f32(0xFFFFFFFFu, v, mask, 0x1F); }

/* eight values from i (a multiple of eight) on of buffer x, whose push-constant word is w (ld(), st()): sixteen
   bytes of bfloat16, or two of floats */
DEVICE void ld8(uint64_t x, uint32_t i, uint32_t w, uint32_t half, float (&v)[8]) {
    if ((half >> w) & 1u) {
        const uint4_ u = ((const uint4_ *)x)[i >> 3];
        const uint32_t words[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++) v[2u * k] = bits_float(words[k] << 16), v[2u * k + 1u] = bits_float(words[k] & 0xFFFF0000u);
        return;
    }
    const float4_ a = ((const float4_ *)x)[i >> 2], b = ((const float4_ *)x)[(i >> 2) + 1u];
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
DEVICE void st8(uint64_t x, uint32_t i, uint32_t w, uint32_t half, const float (&v)[8]) {
    if ((half >> w) & 1u) {
        ((uint4_ *)x)[i >> 3] = uint4_{to_bf16(v[0]) | to_bf16(v[1]) << 16, to_bf16(v[2]) | to_bf16(v[3]) << 16,
                                       to_bf16(v[4]) | to_bf16(v[5]) << 16, to_bf16(v[6]) | to_bf16(v[7]) << 16};
        return;
    }
    ((float4_ *)x)[i >> 2] = float4_{v[0], v[1], v[2], v[3]};
    ((float4_ *)x)[(i >> 2) + 1u] = float4_{v[4], v[5], v[6], v[7]};
}

/* Outputs n .. n + 7 of row m (n a multiple of eight) through the epilogue at once, as store_one() computes each
   (with GENERAL every activation, and with FLAG_PRE EPI_BIAS_ACT's values before it at e1) */
#define ANY_ACT 0xFFFFFFFFu        /* store_eight()'s ACTC: the activation of the spec, chosen as each value needs it */
template <uint32_t ACTC = ANY_ACT>
DEVICE void store_eight(const Epi &e, float slope, uint32_t z, uint32_t coff, uint32_t m, uint32_t n, float (&v)[8]) {
    const uint32_t act = ACTC == ANY_ACT ? e.act : ACTC;
    (void)act;
    if (e.epi == EPI_PARTIAL) {
        float4_ *at = (float4_ *)e.c + (((z * e.M + m) * e.N + n) >> 2);
        at[0] = float4_{v[0], v[1], v[2], v[3]};
        at[1] = float4_{v[4], v[5], v[6], v[7]};
        return;
    }
    const uint32_t at = coff + m * e.ldc + n;
    if (e.epi == EPI_BIAS_ACT) {
        float b[8];
        if (e.flags & FLAG_BIAS) {
            ld8(e.e0, coff + n, 0u, 0u, b);
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++) v[k] += b[k];
        }
#if GENERAL
        if (e.flags & FLAG_PRE) st8(e.e1, at, 2u, e.half, v);
#pragma unroll
        for (uint32_t k = 0; k < 8u; k++) v[k] = activate(v[k], act);
#else
#pragma unroll
        for (uint32_t k = 0; k < 8u; k++) v[k] = sloped(v[k], slope);
#endif
    } else if (e.epi == EPI_SCALE_ACT) {
        float sc[8], sh[8];
        ld8(e.e0, coff + n, 0u, 0u, sc);
        ld8(e.e1, coff + n, 0u, 0u, sh);
#pragma unroll
        for (uint32_t k = 0; k < 8u; k++) {
#if GENERAL
            v[k] = activate(v[k] * sc[k] + sh[k], act);
#else
            v[k] = sloped(v[k] * sc[k] + sh[k], slope);
#endif
        }
    } else {
#pragma unroll
        for (uint32_t k = 0; k < 8u; k++) v[k] *= e.alpha;
        if (e.beta != 0.0f) {
            float old[8];
            ld8(e.c, at, 2u, e.half, old);
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++) v[k] += e.beta * old[k];
        }
        if (e.epi == EPI_DERIV) {
            float d[8];
            ld8(e.e0, at, 3u, e.half, d);
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++) {
#if GENERAL
                v[k] *= derivative(d[k], act);
#else
                v[k] *= sloped_derivative(d[k], slope);
#endif
            }
        }
    }
    (void)slope;
    st8(e.c, at, 2u, e.half, v);
}

template <uint32_t V>
struct Const {
    static constexpr uint32_t value = V;
};

template <uint32_t BM, uint32_t BN, uint32_t BK, uint32_t TI, uint32_t TJ, uint32_t STAGES, uint32_t AMODE, uint32_t BMODE>
struct Pipe {
    /* warps of 16 TI x 16 TJ outputs */
    static constexpr uint32_t WARPS_M = BM / (16u * TI), WARPS_N = BN / (16u * TJ), THREADS = WARPS_M * WARPS_N * 32u;
    /* blocks an SM the registers allow: two of 128 threads with warps of 64 x 64 (up to 255 registers a thread), four
       with warps of half as many outputs (128 registers) */
    static constexpr uint32_t MIN_BLOCKS = (TI * TJ >= 16u ? 256u : 512u) / THREADS > 0u ? (TI * TJ >= 16u ? 256u : 512u) / THREADS : 1u;
    static constexpr bool A_KFAST = AMODE == A_ROW, B_KFAST = BMODE == B_COL;     /* rows along k */
    /* each operand's tile: rows of LEN chunks of eight bfloat16 */
    static constexpr uint32_t LEN_A = A_KFAST ? BK / 8u : BM / 8u, LEN_B = B_KFAST ? BK / 8u : BN / 8u;
    static constexpr uint32_t CHUNKS_A = BM * BK / 8u, CHUNKS_B = BN * BK / 8u;
    static constexpr uint32_t LA = CHUNKS_A / THREADS, LB = CHUNKS_B / THREADS;  /* copies a thread a step */
    static constexpr uint32_t STAGE_A = CHUNKS_A * 16u, STAGE = STAGE_A + CHUNKS_B * 16u;    /* bytes */
    static_assert(CHUNKS_A % THREADS == 0 && CHUNKS_B % THREADS == 0, "whole copies a thread");

    /* a thread's groups of eight outputs of a row, and (GENERAL) those whose outputs the stages' shared memory holds
       at once */
    static constexpr uint32_t GROUPS = TI * TJ, GS = GROUPS * THREADS * 32u <= STAGES * STAGE ? GROUPS
                                                   : GROUPS / 2u * THREADS * 32u <= STAGES * STAGE ? GROUPS / 2u : GROUPS / 4u;
    /* rows of results a round of the epilogue through shared memory: floats of BN + 4 a row */
    static constexpr uint32_t CP = BN + 4u, SMEM = STAGES * STAGE;
    static constexpr uint32_t RCH = BM * CP * 4u <= SMEM ? BM : BM / 2u * CP * 4u <= SMEM ? BM / 2u : BM / 4u * CP * 4u <= SMEM ? BM / 4u : BM / 8u;

    const SpgGemmPush &p;
    uint32_t EPI, ACT, HALF;
    float *cs_base;

    MEMBER Pipe(const SpgGemmPush &push, const Spec &s) : p(push) {
        EPI = s.v[7]; ACT = s.v[8]; HALF = s.v[13];
    }

    /* the bytes into a stage of A's (B's) chunk of value (k, m) ((k, n)) */
    MEMBER static uint32_t at_a(uint32_t k, uint32_t m) {
        return 16u * swizzle<LEN_A>(A_KFAST ? m * LEN_A + k / 8u : k * LEN_A + m / 8u);
    }
    MEMBER static uint32_t at_b(uint32_t k, uint32_t n) {
        return STAGE_A + 16u * swizzle<LEN_B>(B_KFAST ? n * LEN_B + k / 8u : k * LEN_B + n / 8u);
    }

    /* The copies of a tile: copy i of a thread takes chunk tid + i THREADS of the operand's tile, row r0 + i R (R the
       rows THREADS chunks take) at chunk c of the operand's layout, the same chunk c for every i; its address at the
       sum's first k (A_ROW: m * lda + k; A_COL: k * lda + m; B_COL: n * ldb + k; B_ROW: k * ldb + n) moves by i R rows
       and by BK (or BK rows) a step. Each operand: a base address, its rows' stride, and which copies are inside. */
    struct Copies {
        const uint16_t *a, *b;
        uint32_t live_a, live_b;    /* bit i: copy i's row (k-fast) or columns (otherwise) are inside the matrix */
    };
    static constexpr uint32_t RA = THREADS / LEN_A, RB = THREADS / LEN_B;   /* rows a round of copies takes */
    static_assert(THREADS % LEN_A == 0 && THREADS % LEN_B == 0, "copies of whole rows");

    MEMBER void locate(uint32_t tid, uint32_t m0, uint32_t n0, uint32_t kbeg, const uint16_t *A, const uint16_t *B,
                       Copies &c) const {
        const uint32_t ra = tid / LEN_A, ca = tid % LEN_A, rb = tid / LEN_B, cb = tid % LEN_B;
        c.a = A + (A_KFAST ? (uint64_t)(m0 + ra) * p.lda + kbeg + 8u * ca : (uint64_t)(kbeg + ra) * p.lda + m0 + 8u * ca);
        c.b = B + (B_KFAST ? (uint64_t)(n0 + rb) * p.ldb + kbeg + 8u * cb : (uint64_t)(kbeg + rb) * p.ldb + n0 + 8u * cb);
        c.live_a = c.live_b = 0u;
#pragma unroll
        for (uint32_t i = 0; i < LA; i++)
            if (A_KFAST ? m0 + ra + i * RA < p.M : m0 + 8u * ca < p.M) c.live_a |= 1u << i;
#pragma unroll
        for (uint32_t i = 0; i < LB; i++)
            if (B_KFAST ? n0 + rb + i * RB < p.N : n0 + 8u * cb < p.N) c.live_b |= 1u << i;
    }

    /* the copies of step `step` (of the sum from kbeg to kend) into the stage at byte `stage`: zeros beyond the matrices
       and the sum */
    MEMBER void copy_step(uint32_t tid, const Copies &c, uint32_t stage, uint32_t step, uint32_t kbeg, uint32_t kend) const {
        const uint32_t k0 = kbeg + step * BK, ra = tid / LEN_A, ca = tid % LEN_A, rb = tid / LEN_B, cb = tid % LEN_B;
        const uint64_t lda = A_KFAST ? p.lda : (uint64_t)p.lda * RA, ldb = B_KFAST ? p.ldb : (uint64_t)p.ldb * RB;
        const uint16_t *a = c.a + (A_KFAST ? (uint64_t)step * BK : (uint64_t)step * BK * p.lda);
        const uint16_t *b = c.b + (B_KFAST ? (uint64_t)step * BK : (uint64_t)step * BK * p.ldb);
#pragma unroll
        for (uint32_t i = 0; i < LA; i++) {
            const bool inside = ((c.live_a >> i) & 1u) && k0 + (A_KFAST ? 8u * ca : ra + i * RA) < kend;
            const uint16_t *at = a + (A_KFAST ? (uint64_t)(i * RA) * lda : i * lda);
            copy16(stage + 16u * swizzle<LEN_A>((ra + i * RA) * LEN_A + ca), inside ? at : c.a, inside ? 16u : 0u);
        }
#pragma unroll
        for (uint32_t i = 0; i < LB; i++) {
            const bool inside = ((c.live_b >> i) & 1u) && k0 + (B_KFAST ? 8u * cb : rb + i * RB) < kend;
            const uint16_t *at = b + (B_KFAST ? (uint64_t)(i * RB) * ldb : i * ldb);
            copy16(stage + STAGE_A + 16u * swizzle<LEN_B>((rb + i * RB) * LEN_B + cb), inside ? at : c.b,
                   inside ? 16u : 0u);
        }
    }

    MEMBER void run(uint32_t smem) {
        const uint32_t tid = thread_x(), z = block_z(), g = z / p.slices, s = z % p.slices;
        uint32_t tm, tn;
        swizzled(tm, tn);
        const uint32_t m0 = (p.m_tile0 + tm) * BM, n0 = tn * BN;
        const uint32_t kbeg = s * p.slice_k, kend = umin(p.K, kbeg + p.slice_k);
        const uint32_t steps = kbeg < kend ? (kend - kbeg + BK - 1u) / BK : 0u;
        Copies c;
        locate(tid, m0, n0, kbeg, H(p.a) + g * p.a_group, H(p.b) + g * p.b_group, c);
        const uint32_t warp = tid / 32u, lane = tid % 32u, wm = warp % WARPS_M, wn = warp / WARPS_M;

        float acc[TI][2 * TJ][4];
#pragma unroll
        for (uint32_t i = 0; i < TI; i++)
#pragma unroll
            for (uint32_t j = 0; j < 2u * TJ; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

        /* fragments of k-step kk of the stage at byte `stage`: A's TI tiles of 16 rows, B's TJ of 16 columns (as
           gemm_mma.cu reads them) */
        auto fragments = [&](uint32_t stage, uint32_t kk, uint32_t (&af)[TI][4], uint32_t (&bf)[TJ][4]) {
#pragma unroll
            for (uint32_t i = 0; i < TI; i++) {
                const uint32_t mb = (wm * TI + i) * 16u;
                if (A_KFAST) ldsm4(stage + at_a(kk + (lane / 16u) * 8u, mb + lane % 16u), af[i]);
                else ldsm4t(stage + at_a(kk + lane % 8u + (lane / 16u) * 8u, mb + (lane / 8u % 2u) * 8u), af[i]);
            }
#pragma unroll
            for (uint32_t j = 0; j < TJ; j++) {
                const uint32_t nb = (wn * TJ + j) * 16u;
                if (B_KFAST) ldsm4(stage + at_b(kk + (lane / 8u % 2u) * 8u, nb + lane % 8u + (lane / 16u) * 8u), bf[j]);
                else ldsm4t(stage + at_b(kk + lane % 8u + (lane / 8u % 2u) * 8u, nb + (lane / 16u) * 8u), bf[j]);
            }
        };

#pragma unroll
        for (uint32_t s0 = 0; s0 + 1u < STAGES; s0++) {
            if (s0 < steps) copy_step(tid, c, smem + s0 * STAGE, s0, kbeg, kend);
            copies_commit();
        }
        uint32_t read = 0, write = STAGES - 1u;          /* the stages of this step and of the copies issued */
        for (uint32_t step = 0; step < steps; step++) {
            copies_wait<STAGES - 2u>();
            barrier();          /* this step's stage written; the one copied to next read by every warp */
            if (step + STAGES - 1u < steps) copy_step(tid, c, smem + write * STAGE, step + STAGES - 1u, kbeg, kend);
            copies_commit();
            const uint32_t stage = smem + read * STAGE;
            uint32_t af[2][TI][4], bf[2][TJ][4];
            fragments(stage, 0u, af[0], bf[0]);
#pragma unroll
            for (uint32_t kk = 0; kk < BK / 16u; kk++) {
                if (kk + 1u < BK / 16u) fragments(stage, (kk + 1u) * 16u, af[(kk + 1u) % 2u], bf[(kk + 1u) % 2u]);
#pragma unroll
                for (uint32_t i = 0; i < TI; i++)
#pragma unroll
                    for (uint32_t j = 0; j < TJ; j++) {
                        mma(acc[i][2u * j], af[kk % 2u][i], bf[kk % 2u][j][0], bf[kk % 2u][j][1]);
                        mma(acc[i][2u * j + 1u], af[kk % 2u][i], bf[kk % 2u][j][2], bf[kk % 2u][j][3]);
                    }
            }
            read = read + 1u == STAGES ? 0u : read + 1u;
            write = write + 1u == STAGES ? 0u : write + 1u;
        }

        const uint32_t coff = g * p.c_group;
        const Epi e = {p.c, p.e0, p.e1, p.M, p.N, p.ldc, EPI, ACT, p.flags, HALF, p.alpha, p.beta};
        const float slope = EPI == EPI_DERIV ? act_slope(ACT) : act_slope_forward(ACT);
        /* eight outputs at once where C (and what the epilogue reads with it) has them at indices that are multiples
           of eight, at 16 bytes, in rows of N a multiple of eight; else through shared memory, as gemm_mma.cu's */
        const bool eights = TJ % 2u == 0u && (p.c % 16u) == 0u && p.N % 8u == 0u &&
                            (EPI == EPI_PARTIAL || ((p.ldc % 8u) == 0u && (coff % 8u) == 0u &&
                                                    (EPI == EPI_STORE || p.e0 % 16u == 0u) &&
                                                    (EPI != EPI_SCALE_ACT || p.e1 % 16u == 0u) &&
                                                    (EPI != EPI_BIAS_ACT || !(p.flags & FLAG_PRE) || p.e1 % 16u == 0u)));
        if (eights) {
            /* fragment (i, j) holds rows lane / 4 and + 8 of its 16, columns 2 (lane % 4) and + 1 of its 8: the four
               threads of a row exchange their pairs of four fragments in two steps of shuffles (a 4 x 4 transpose),
               after which thread q of the four holds the eight columns of the group's fragment q, which it finishes
               at once (two outputs a thread, stores of 8 rows of 16 bytes a warp, ran a tile's epilogue 40% longer) */
            const uint32_t q = lane % 4u;
            /* group (i, h, a): rows of fragments i, half h, fragments 4a to 4a + 3 */
            auto transpose = [&](uint32_t i, uint32_t h, uint32_t a, float (&v)[8]) __attribute__((always_inline)) {
                float x[4][2];
#pragma unroll
                for (uint32_t b = 0; b < 4u; b++) x[b][0] = acc[i][4u * a + b][2u * h], x[b][1] = acc[i][4u * a + b][2u * h + 1u];
#pragma unroll
                for (uint32_t b = 0; b < 4u; b += 2u)       /* with the thread q ^ 1 */
#pragma unroll
                    for (uint32_t k = 0; k < 2u; k++) {
                        const float r = shfl_xor((q & 1u) ? x[b][k] : x[b + 1u][k], 1u);
                        if (q & 1u) x[b][k] = r;
                        else x[b + 1u][k] = r;
                    }
#pragma unroll
                for (uint32_t d = 0; d < 2u; d++)           /* with the thread q ^ 2 */
#pragma unroll
                    for (uint32_t k = 0; k < 2u; k++) {
                        const float r = shfl_xor((q & 2u) ? x[d][k] : x[2u + d][k], 2u);
                        if (q & 2u) x[d][k] = r;
                        else x[2u + d][k] = r;
                    }
#pragma unroll
                for (uint32_t b = 0; b < 4u; b++) v[2u * b] = x[b][0], v[2u * b + 1u] = x[b][1];
            };
            auto row = [&](uint32_t i, uint32_t h) { return m0 + (wm * TI + i) * 16u + lane / 4u + 8u * h; };
            auto column = [&](uint32_t a) { return n0 + wn * 16u * TJ + 8u * (4u * a + q); };
#if GENERAL
            /* every activation, too long to repeat for each group: the groups' outputs through the stages' shared memory,
               each thread's eight of a group at a place of its own (so that no thread waits for another), GS groups a
               round, then finished one group at a time */
            barrier();          /* every warp past its last fragments: the stages are free */
            float4_ *scratch = (float4_ *)cs_base;
#pragma unroll
            for (uint32_t g0 = 0; g0 < GROUPS; g0 += GS) {
#pragma unroll
                for (uint32_t gi = 0; gi < GS; gi++) {
                    const uint32_t gr = g0 + gi, i = gr / TJ, h = gr / (TJ / 2u) % 2u, a = gr % (TJ / 2u);
                    float v[8];
                    transpose(i, h, a, v);
                    scratch[2u * (gi * THREADS + tid)] = float4_{v[0], v[1], v[2], v[3]};
                    scratch[2u * (gi * THREADS + tid) + 1u] = float4_{v[4], v[5], v[6], v[7]};
                }
                /* the activations of language models with their own loops, the others' chosen value by value
                   (a jump a value cost GELU's products in inference a fifth of their time) */
                auto finish = [&](auto actc) __attribute__((always_inline)) {
#pragma unroll 1
                    for (uint32_t gi = 0; gi < GS; gi++) {
                        const uint32_t gr = g0 + gi, i = gr / TJ, h = gr / (TJ / 2u) % 2u, a = gr % (TJ / 2u);
                        const uint32_t m = row(i, h), n = column(a);
                        const float4_ lo = scratch[2u * (gi * THREADS + tid)], hi = scratch[2u * (gi * THREADS + tid) + 1u];
                        float v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                        if (m < p.M && n < p.N) store_eight<decltype(actc)::value>(e, slope, z, coff, m, n, v);
                    }
                };
                if (ACT == ACT_GELU) finish(Const<ACT_GELU>{});
                else if (ACT == ACT_GELU_TANH) finish(Const<ACT_GELU_TANH>{});
                else if (ACT == ACT_SILU) finish(Const<ACT_SILU>{});
                else finish(Const<ANY_ACT>{});
            }
#else
            /* each epilogue a loop of its own (an unrolled loop that branched between them all ran 18,000 instructions
               out of the instruction cache) */
            auto eight = [&](uint32_t epi) __attribute__((always_inline)) {
                Epi f = e;
                f.epi = epi;
#pragma unroll
                for (uint32_t i = 0; i < TI; i++)
#pragma unroll
                    for (uint32_t h = 0; h < 2u; h++)
#pragma unroll
                        for (uint32_t a = 0; a < TJ / 2u; a++) {
                            float v[8];
                            transpose(i, h, a, v);
                            const uint32_t m = row(i, h), n = column(a);
                            if (m < p.M && n < p.N) store_eight(f, slope, z, coff, m, n, v);
                        }
            };
            if (EPI == EPI_STORE) eight(EPI_STORE);
            else if (EPI == EPI_BIAS_ACT) eight(EPI_BIAS_ACT);
            else if (EPI == EPI_DERIV) eight(EPI_DERIV);
            else if (EPI == EPI_PARTIAL) eight(EPI_PARTIAL);
            else eight(EPI_SCALE_ACT);
#endif
            return;
        }
        /* (and every activation's epilogue, with GENERAL) through the stages' shared memory, RCH rows of the tile at a
           time, as gemm_mma.cu's results go: a thread finishes four consecutive outputs of a row */
        copies_wait<0>();
        float *Cs = (float *)__builtin_assume_aligned(cs_base, 16);
        const bool fours = (p.c % 16u) == 0u &&
                           (EPI == EPI_PARTIAL ? p.N % 4u == 0u
                                               : (p.ldc % 4u) == 0u && (coff % 4u) == 0u &&
                                                 (EPI == EPI_STORE || p.e0 % 16u == 0u) &&
                                                 (EPI != EPI_SCALE_ACT || p.e1 % 16u == 0u));
#pragma unroll
        for (uint32_t r0 = 0; r0 < BM; r0 += RCH) {
            barrier();
#pragma unroll
            for (uint32_t i = 0; i < TI; i++)
#pragma unroll
                for (uint32_t h = 0; h < 2u; h++) {
                    const uint32_t lr = (wm * TI + i) * 16u + lane / 4u + 8u * h;
                    if (lr < r0 || lr >= r0 + RCH) continue;
#pragma unroll
                    for (uint32_t j = 0; j < 2u * TJ; j++) {
                        float *at = Cs + (lr - r0) * CP + wn * 16u * TJ + 8u * j + 2u * (lane % 4u);
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
                if (fours && n + 3u < p.N) {
                    store_four(e, slope, z, coff, m, m, n, v);
                    continue;
                }
                for (uint32_t k = 0; k < 4u && n + k < p.N; k++) store_one(e, slope, z, coff, m, m, n + k, get4(v, k));
            }
        }
    }
};

/* the unit's instance: SPG_ENTRY and PIPE (BM, BN, BK, TI, TJ, STAGES, AMODE, BMODE) from cmake/Cuda.cmake */
#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_gemm_pipe_128x128x32_4x4_s3_a0b1
#define PIPE 128, 128, 32, 4, 4, 3, 0, 1
#endif

extern "C" __global__ void __attribute__((launch_bounds(Pipe<PIPE>::THREADS, Pipe<PIPE>::MIN_BLOCKS)))
SPG_ENTRY(const SpgGemmPush p, const Spec spec) {
    extern __shared__ float4_ smem[];
    Pipe<PIPE> pipe(p, spec);
    pipe.cs_base = (float *)smem;
    pipe.run(shared_address(smem));
}
