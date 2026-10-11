/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * flash: attn.comp's FORWARD, DQ and DKV on the matrix units (mma.sync m16n8k16, bfloat16 operands,
 * single-precision sums), for networks whose products are in bfloat16. A warp takes 16 rows (queries, or
 * keys in DKV), a block WARPS of them; the other side's tiles of BT rows pass through shared memory (rows
 * of D + 8 bfloat16, so that the eight rows ldmatrix reads fall in distinct banks). Queries and keys are
 * rotated (ROPE) and rounded to bfloat16 as they are loaded; the scores are scaled in single precision.
 *
 *   FORWARD  S = Q K^T a tile of keys at a time, the softmax online in registers (a row's maximum and sum
 *            combined over the four threads that hold it, in a fixed pattern), O += P V with P rounded to
 *            bfloat16 as the matrix units take it; y = O / sum, and the log sum with STATS.
 *   DQ       S and P = exp(S - lse) again, dP = dO V^T, dS = P (dP - rowsum(dO o)), dQ += dS K.
 *   DKV      per tile of keys, over the group's query heads and tiles of queries in order: S^T = K Q^T,
 *            P^T, dP^T = V dO^T, dS^T, dV += P^T dO, dK += dS^T Q.
 *
 * Tiles of activations kept as bfloat16 load sixteen bytes a thread at a time (eight values, or a chunk
 * and its pair of the other half of the vector where rotated), the next tile of the loop into registers
 * while the block computes on the current one; tiles of floats load a value at a time, as they are needed.
 *
 * Every product adds its 16 values of k at a time in the order of k, every row's sums are combined in the
 * same pattern: the same bits on every run. A unit per head size D (32, 64, 128) and pass (OPC: each with
 * the registers of its own). Spec: OP, BQ, DMAX, THREADS, HALF (words: x 0, y 1, dy 2, dx 3); the
 * executor's tiles of attn.cu do not apply here.
 */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_flash_d64_o0
#define D 64u
#define OPC 0u
#endif
#define WARPS (D <= 64u ? 4u : 2u)         /* warps a block: 16 rows each */
#define ROWS (16u * WARPS)                 /* rows of a block */
#define BT (D <= 64u ? 64u : 32u)          /* rows of the other side's tiles */
#define SD (D + 8u)                        /* bfloat16 a row in shared memory */
#define NT (BT / 8u)                       /* n-tiles of 8 of a tile of scores */
#define DT (D / 8u)                        /* n-tiles of 8 of a row of D */
#define KS (D / 16u)                       /* k-steps of 16 over D */
#define NCH (BT * (D / 8u) / (32u * WARPS))  /* chunks of eight of a tile of BT rows a thread loads */

#define FORWARD 0u
#define DQ      2u
#define DKV     3u
#define FLAG_CAUSAL 1u
#define FLAG_ROPE   2u
#define FLAG_STATS  4u

DEVICE uint32_t lane_id() { return thread_x() & 31u; }
DEVICE uint32_t warp_id() { return thread_x() >> 5; }
DEVICE uint32_t shared_address(const void *p) {
    uint64_t r;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(r) : "l"(p));
    return (uint32_t)r;
}
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
DEVICE uint32_t pack2(float lo, float hi) { return to_bf16(lo) | to_bf16(hi) << 16; }
DEVICE float shfl_xor(float v, uint32_t mask) { return __nvvm_shfl_sync_bfly_f32(0xFFFFFFFFu, v, mask, 0x1F); }

/* the maximum (sum) of a row's values over the four threads of a quad that hold it: xor 1, then xor 2 */
DEVICE float quad_max(float v) { v = max_(v, shfl_xor(v, 1u)); return max_(v, shfl_xor(v, 2u)); }
DEVICE float quad_sum(float v) { v += shfl_xor(v, 1u); return v + shfl_xor(v, 2u); }

/* Rows r0 .. r0 + count - 1 of a head (float `head` of x's cells) into a shared tile [row][SD] of bfloat16:
   rotated with rotate (ROPE), zeros past the cells. Eight values a thread-load, with their pair of the
   other half of the vector where rotated. */
DEVICE void load_tile(const SpgAttnPush &p, uint32_t HALF, uint16_t *tile, uint32_t s, uint32_t r0, uint32_t count,
                      uint32_t head, uint32_t stride, uint64_t base, uint32_t word, bool rotate, uint32_t threads) {
    const bool rope = rotate && (p.flags & FLAG_ROPE);
    const uint32_t chunks = rope ? D / 16u : D / 8u;       /* (rotated: a chunk and its pair a thread) */
    for (uint32_t e = thread_x(); e < count * chunks; e += threads) {
        const uint32_t row = e / chunks, c = 8u * (e % chunks), pos = r0 + row;
        float v[8], w[8];
        const bool live = pos < p.cells;
        const uint32_t at = (s * p.cells + pos) * stride + head;
#pragma unroll
        for (uint32_t k = 0; k < 8u; k++) {
            v[k] = live ? ld(base, at + c + k, word, HALF) : 0.0f;
            w[k] = live && rope ? ld(base, at + c + k + D / 2u, word, HALF) : 0.0f;
        }
        if (rope && live) {
            const float *cs = F(p.table) + pos * (D / 2u) + c, *sn = F(p.table) + p.cells * (D / 2u) + pos * (D / 2u) + c;
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++) {
                const float a = v[k], b = w[k];
                v[k] = a * cs[k] - b * sn[k];
                w[k] = b * cs[k] + a * sn[k];
            }
        }
        uint32_t *dst = (uint32_t *)(tile + row * SD + c);
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++) dst[k] = pack2(v[2u * k], v[2u * k + 1u]);
        if (rope) {
            uint32_t *pair = (uint32_t *)(tile + row * SD + c + D / 2u);
#pragma unroll
            for (uint32_t k = 0; k < 4u; k++) pair[k] = pack2(w[2u * k], w[2u * k + 1u]);
        }
    }
}

struct uint4_ {
    uint32_t x, y, z, w;
} __attribute__((aligned(16)));

/* Eight bfloat16 of a row of a tile (two a word, the first value in the low half), as loaded */
struct Raw {
    uint4_ v;
};

/* A tile of BT rows from r0 of a head kept as bfloat16 into this thread's registers: chunk e of the tile is row
   e / (D / 8), values 8 (e % (D / 8)) on; rotated (rope) chunk pairs, row e / (D / 16), values c = 8 (e % (D / 16))
   and c + D / 2, in v[2k] and v[2k + 1]. Zeros past the cells. */
DEVICE void fetch_tile(const SpgAttnPush &p, uint32_t s, uint32_t r0, uint32_t head, uint32_t stride, uint64_t base,
                       bool rope, Raw (&v)[NCH]) {
    const uint4_ zero = {0u, 0u, 0u, 0u};
#pragma unroll
    for (uint32_t k = 0; k < NCH; k++) {
        const uint32_t e = rope ? thread_x() + 32u * WARPS * (k / 2u) : thread_x() + 32u * WARPS * k;
        const uint32_t row = rope ? e / (D / 16u) : e / (D / 8u);
        const uint32_t c = rope ? 8u * (e % (D / 16u)) + (k & 1u) * (D / 2u) : 8u * (e % (D / 8u)), pos = r0 + row;
        v[k].v = pos < p.cells ? *(const uint4_ *)(H(base) + (s * p.cells + pos) * stride + head + c) : zero;
    }
}

/* and from the registers into a shared tile [row][SD], rotated with rope (ROPE) as load_tile() rotates */
DEVICE void put_tile(const SpgAttnPush &p, uint16_t *tile, uint32_t r0, bool rope, const Raw (&v)[NCH]) {
    if (!(rope && (p.flags & FLAG_ROPE))) {
#pragma unroll
        for (uint32_t k = 0; k < NCH; k++) {
            const uint32_t e = rope ? thread_x() + 32u * WARPS * (k / 2u) : thread_x() + 32u * WARPS * k;
            const uint32_t row = rope ? e / (D / 16u) : e / (D / 8u);
            const uint32_t c = rope ? 8u * (e % (D / 16u)) + (k & 1u) * (D / 2u) : 8u * (e % (D / 8u));
            *(uint4_ *)(tile + row * SD + c) = v[k].v;
        }
        return;
    }
#pragma unroll
    for (uint32_t k = 0; k < NCH; k += 2u) {
        const uint32_t e = thread_x() + 32u * WARPS * (k / 2u), row = e / (D / 16u), c = 8u * (e % (D / 16u));
        const uint32_t pos = r0 + row;
        const uint32_t a[4] = {v[k].v.x, v[k].v.y, v[k].v.z, v[k].v.w}, b[4] = {v[k + 1u].v.x, v[k + 1u].v.y,
                                                                                v[k + 1u].v.z, v[k + 1u].v.w};
        uint32_t ra[4], rb[4];
        if (pos < p.cells) {
            const float *cs = F(p.table) + pos * (D / 2u) + c, *sn = F(p.table) + p.cells * (D / 2u) + pos * (D / 2u) + c;
#pragma unroll
            for (uint32_t w = 0; w < 4u; w++) {
                const float x0 = bits_float(a[w] << 16), x1 = bits_float(a[w] & 0xFFFF0000u);
                const float y0 = bits_float(b[w] << 16), y1 = bits_float(b[w] & 0xFFFF0000u);
                const float c0 = cs[2u * w], c1 = cs[2u * w + 1u], s0 = sn[2u * w], s1 = sn[2u * w + 1u];
                ra[w] = pack2(x0 * c0 - y0 * s0, x1 * c1 - y1 * s1);
                rb[w] = pack2(y0 * c0 + x0 * s0, y1 * c1 + x1 * s1);
            }
        } else {
#pragma unroll
            for (uint32_t w = 0; w < 4u; w++) ra[w] = rb[w] = 0u;
        }
        *(uint4_ *)(tile + row * SD + c) = uint4_{ra[0], ra[1], ra[2], ra[3]};
        *(uint4_ *)(tile + row * SD + c + D / 2u) = uint4_{rb[0], rb[1], rb[2], rb[3]};
    }
}

/* A operand fragments (16 rows x 16 k) of a warp's rows r0.. of a shared tile, k-step ks */
DEVICE void frag_a(const uint16_t *tile, uint32_t r0, uint32_t ks, uint32_t (&a)[4]) {
    const uint32_t l = lane_id();
    ldsm4(shared_address(tile + (r0 + (l & 15u)) * SD + 16u * ks + 8u * (l >> 4)), a);
}
/* B fragments of two n-tiles (n0, n0 + 8) of k-step ks, from a tile stored [n][k] */
DEVICE void frag_b(const uint16_t *tile, uint32_t n0, uint32_t ks, uint32_t (&b)[4]) {
    const uint32_t l = lane_id();
    ldsm4(shared_address(tile + (n0 + (l & 7u) + 8u * (l >> 4)) * SD + 16u * ks + 8u * ((l >> 3) & 1u)), b);
}
/* B fragments of two n-tiles (n0, n0 + 8) of k-step ks (rows k0 = 16 ks), from a tile stored [k][n] */
DEVICE void frag_bt(const uint16_t *tile, uint32_t n0, uint32_t ks, uint32_t (&b)[4]) {
    const uint32_t l = lane_id();
    ldsm4t(shared_address(tile + (16u * ks + (l & 7u) + 8u * ((l >> 3) & 1u)) * SD + n0 + 8u * (l >> 4)), b);
}

/* The rows of the scores [16][BT] (c-fragments of NT n-tiles) as A fragments of k-step j (keys 16 j..) */
DEVICE void scores_a(const float (&s)[NT][4], uint32_t j, uint32_t (&a)[4]) {
    a[0] = pack2(s[2u * j][0], s[2u * j][1]);
    a[1] = pack2(s[2u * j][2], s[2u * j][3]);
    a[2] = pack2(s[2u * j + 1u][0], s[2u * j + 1u][1]);
    a[3] = pack2(s[2u * j + 1u][2], s[2u * j + 1u][3]);
}

struct Shared {
    uint16_t a[ROWS * SD];          /* the block's rows: queries (FORWARD, DQ) or keys (DKV) */
    uint16_t b[ROWS * SD];          /* DQ: the outputs' gradient of the queries; DKV: the values */
    uint16_t t[BT * SD], u[BT * SD];    /* the other side's tile: keys and values, or queries and dO */
    float l[BT], d[BT];             /* DKV: the tile's queries' log sums and rowsum(dO o) */
};

extern "C" __global__ void __launch_bounds__(32u * WARPS) SPG_ENTRY(const SpgAttnPush p, const Spec spec) {
    const uint32_t OP = OPC, HALF = spec.v[4], threads = 32u * WARPS;
    __shared__ __attribute__((aligned(16))) Shared sh;
    const uint32_t l = lane_id(), w = warp_id(), g = l >> 2, t = l & 3u;
    const uint32_t C = (p.heads + 2u * p.kv) * D, out_c = p.heads * D, group = p.heads / p.kv;
    const bool causal = (p.flags & FLAG_CAUSAL) != 0u;
    const float NEG = -3.402823466e38f, scale = p.scale;
    const uint32_t s = block_z(), rows_all = p.n * p.heads * p.cells;
    if (OP == FORWARD || OP == DQ) {
        const uint32_t h = block_y(), kh = h / group, i0 = block_x() * ROWS, r0 = 16u * w;
        load_tile(p, HALF, sh.a, s, i0, ROWS, h * D, C, p.x, 0u, true, threads);
        if (OP == DQ) load_tile(p, HALF, sh.b, s, i0, ROWS, h * D, out_c, p.dy, 2u, false, threads);
        barrier();
        uint32_t qa[KS][4], da[KS][4];
#pragma unroll
        for (uint32_t k = 0; k < KS; k++) {
            frag_a(sh.a, r0, k, qa[k]);
            if (OP == DQ) frag_a(sh.b, r0, k, da[k]);
        }
        /* this thread's rows: g and g + 8 of the warp's */
        const uint32_t ra = i0 + r0 + g, rb = ra + 8u;
        const uint32_t sa = (s * p.heads + h) * p.cells;
        float lse_a = 0.0f, lse_b = 0.0f, del_a = 0.0f, del_b = 0.0f;
        if (OP == DQ) {
            if (ra < p.cells) { lse_a = F(p.stats)[sa + ra]; del_a = F(p.stats)[rows_all + sa + ra]; }
            if (rb < p.cells) { lse_b = F(p.stats)[sa + rb]; del_b = F(p.stats)[rows_all + sa + rb]; }
        }
        float o[DT][4];
#pragma unroll
        for (uint32_t j = 0; j < DT; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
        float m_a = NEG, m_b = NEG, l_a = 0.0f, l_b = 0.0f;
        const uint32_t keys = causal ? umin(p.cells, i0 + ROWS) : p.cells;
        const bool fast = (HALF & 1u) != 0u;        /* (x kept as bfloat16: its tiles a step ahead) */
        Raw rk[NCH], rv[NCH];
        if (fast) {
            fetch_tile(p, s, 0u, (p.heads + kh) * D, C, p.x, true, rk);
            fetch_tile(p, s, 0u, (p.heads + p.kv + kh) * D, C, p.x, false, rv);
        }
        for (uint32_t j0 = 0; j0 < keys; j0 += BT) {
            barrier();
            if (fast) {
                put_tile(p, sh.t, j0, true, rk);
                put_tile(p, sh.u, j0, false, rv);
            } else {
                load_tile(p, HALF, sh.t, s, j0, BT, (p.heads + kh) * D, C, p.x, 0u, true, threads);
                load_tile(p, HALF, sh.u, s, j0, BT, (p.heads + p.kv + kh) * D, C, p.x, 0u, false, threads);
            }
            barrier();
            if (fast && j0 + BT < keys) {
                fetch_tile(p, s, j0 + BT, (p.heads + kh) * D, C, p.x, true, rk);
                fetch_tile(p, s, j0 + BT, (p.heads + p.kv + kh) * D, C, p.x, false, rv);
            }
            float sc[NT][4];
#pragma unroll
            for (uint32_t j = 0; j < NT; j++) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.0f;
#pragma unroll
            for (uint32_t k = 0; k < KS; k++)
#pragma unroll
                for (uint32_t j = 0; j < NT; j += 2u) {
                    uint32_t b[4];
                    frag_b(sh.t, 8u * j, k, b);
                    mma(sc[j], qa[k], b[0], b[1]);
                    mma(sc[j + 1u], qa[k], b[2], b[3]);
                }
            /* the scale, the mask (keys past the cells, after the query when causal) */
#pragma unroll
            for (uint32_t j = 0; j < NT; j++)
#pragma unroll
                for (uint32_t e = 0; e < 4u; e++) {
                    const uint32_t key = j0 + 8u * j + 2u * t + (e & 1u), row = e < 2u ? ra : rb;
                    const bool valid = key < p.cells && row < p.cells && (!causal || key <= row);
                    sc[j][e] = valid ? sc[j][e] * scale : NEG;
                }
            if (OP == FORWARD) {
                float ta = NEG, tb = NEG;
#pragma unroll
                for (uint32_t j = 0; j < NT; j++) {
                    ta = max_(ta, max_(sc[j][0], sc[j][1]));
                    tb = max_(tb, max_(sc[j][2], sc[j][3]));
                }
                ta = quad_max(ta);
                tb = quad_max(tb);
                const float na = max_(m_a, ta), nb = max_(m_b, tb);
                const float aa = na == NEG ? 1.0f : exp_(m_a - na), ab = nb == NEG ? 1.0f : exp_(m_b - nb);
                float suma = 0.0f, sumb = 0.0f;
#pragma unroll
                for (uint32_t j = 0; j < NT; j++) {
                    sc[j][0] = sc[j][0] == NEG ? 0.0f : exp_(sc[j][0] - na);
                    sc[j][1] = sc[j][1] == NEG ? 0.0f : exp_(sc[j][1] - na);
                    sc[j][2] = sc[j][2] == NEG ? 0.0f : exp_(sc[j][2] - nb);
                    sc[j][3] = sc[j][3] == NEG ? 0.0f : exp_(sc[j][3] - nb);
                    suma += sc[j][0] + sc[j][1];
                    sumb += sc[j][2] + sc[j][3];
                }
                l_a = l_a * aa + quad_sum(suma);
                l_b = l_b * ab + quad_sum(sumb);
                m_a = na;
                m_b = nb;
#pragma unroll
                for (uint32_t j = 0; j < DT; j++) {
                    o[j][0] *= aa; o[j][1] *= aa;
                    o[j][2] *= ab; o[j][3] *= ab;
                }
                /* O += P V: P's rows as A fragments, V's [key][d] tile transposed into B fragments */
#pragma unroll
                for (uint32_t k = 0; k < BT / 16u; k++) {
                    uint32_t pa[4];
                    scores_a(sc, k, pa);
#pragma unroll
                    for (uint32_t j = 0; j < DT; j += 2u) {
                        uint32_t b[4];
                        frag_bt(sh.u, 8u * j, k, b);
                        mma(o[j], pa, b[0], b[1]);
                        mma(o[j + 1u], pa, b[2], b[3]);
                    }
                }
                continue;
            }
            /* DQ: P = exp(S - lse), dP = dO V^T, dS = P (dP - delta), dQ += dS K */
            float dp[NT][4];
#pragma unroll
            for (uint32_t j = 0; j < NT; j++) dp[j][0] = dp[j][1] = dp[j][2] = dp[j][3] = 0.0f;
#pragma unroll
            for (uint32_t k = 0; k < KS; k++)
#pragma unroll
                for (uint32_t j = 0; j < NT; j += 2u) {
                    uint32_t b[4];
                    frag_b(sh.u, 8u * j, k, b);
                    mma(dp[j], da[k], b[0], b[1]);
                    mma(dp[j + 1u], da[k], b[2], b[3]);
                }
#pragma unroll
            for (uint32_t j = 0; j < NT; j++)
#pragma unroll
                for (uint32_t e = 0; e < 4u; e++) {
                    const float lse = e < 2u ? lse_a : lse_b, del = e < 2u ? del_a : del_b;
                    const float pv = sc[j][e] == NEG ? 0.0f : exp_(sc[j][e] - lse);
                    sc[j][e] = pv * (dp[j][e] - del);
                }
#pragma unroll
            for (uint32_t k = 0; k < BT / 16u; k++) {
                uint32_t pa[4];
                scores_a(sc, k, pa);
#pragma unroll
                for (uint32_t j = 0; j < DT; j += 2u) {
                    uint32_t b[4];
                    frag_bt(sh.t, 8u * j, k, b);
                    mma(o[j], pa, b[0], b[1]);
                    mma(o[j + 1u], pa, b[2], b[3]);
                }
            }
        }
        if (OP == FORWARD) {
            const float ia = 1.0f / l_a, ib = 1.0f / l_b;
#pragma unroll
            for (uint32_t j = 0; j < DT; j++) {
                const uint32_t c = 8u * j + 2u * t;
                if (ra < p.cells) {
                    st(p.y, (s * p.cells + ra) * out_c + h * D + c, 1u, HALF, o[j][0] * ia);
                    st(p.y, (s * p.cells + ra) * out_c + h * D + c + 1u, 1u, HALF, o[j][1] * ia);
                }
                if (rb < p.cells) {
                    st(p.y, (s * p.cells + rb) * out_c + h * D + c, 1u, HALF, o[j][2] * ib);
                    st(p.y, (s * p.cells + rb) * out_c + h * D + c + 1u, 1u, HALF, o[j][3] * ib);
                }
            }
            if ((p.flags & FLAG_STATS) && t == 0u) {
                if (ra < p.cells) F(p.stats)[sa + ra] = m_a + log_(l_a);
                if (rb < p.cells) F(p.stats)[sa + rb] = m_b + log_(l_b);
            }
            return;
        }
        /* dQ: the scale, then rotated back through shared memory (as floats over the rows' tile) */
        barrier();
        float *dq = (float *)sh.t;          /* ROWS x D floats: within t and u */
#pragma unroll
        for (uint32_t j = 0; j < DT; j++) {
            const uint32_t c = 8u * j + 2u * t;
            dq[(r0 + g) * D + c] = o[j][0] * scale;
            dq[(r0 + g) * D + c + 1u] = o[j][1] * scale;
            dq[(r0 + g + 8u) * D + c] = o[j][2] * scale;
            dq[(r0 + g + 8u) * D + c + 1u] = o[j][3] * scale;
        }
        barrier();
        for (uint32_t e = thread_x(); e < ROWS * D; e += threads) {
            const uint32_t row = e / D, c = e % D, i = i0 + row;
            if (i >= p.cells) continue;
            float v = dq[row * D + c];
            if (p.flags & FLAG_ROPE) {
                const uint32_t hd = D / 2u, ii = c < hd ? c : c - hd;
                const float cs = F(p.table)[i * hd + ii], sn = F(p.table)[p.cells * hd + i * hd + ii];
                v = c < hd ? v * cs + dq[row * D + c + hd] * sn : v * cs - dq[row * D + c - hd] * sn;
            }
            st(p.dx, (s * p.cells + i) * C + h * D + c, 3u, HALF, v);
        }
        return;
    }
    /* DKV: the block's keys (and values) of head kh, the warp's 16 of them; queries of the group's heads */
    const uint32_t kh = block_y(), j0 = block_x() * ROWS, r0 = 16u * w;
    load_tile(p, HALF, sh.a, s, j0, ROWS, (p.heads + kh) * D, C, p.x, 0u, true, threads);
    load_tile(p, HALF, sh.b, s, j0, ROWS, (p.heads + p.kv + kh) * D, C, p.x, 0u, false, threads);
    barrier();
    uint32_t ka[KS][4], va[KS][4];
#pragma unroll
    for (uint32_t k = 0; k < KS; k++) {
        frag_a(sh.a, r0, k, ka[k]);
        frag_a(sh.b, r0, k, va[k]);
    }
    const uint32_t ka_row = j0 + r0 + g, kb_row = ka_row + 8u;     /* this thread's keys */
    float dk[DT][4], dv[DT][4];
#pragma unroll
    for (uint32_t j = 0; j < DT; j++) {
        dk[j][0] = dk[j][1] = dk[j][2] = dk[j][3] = 0.0f;
        dv[j][0] = dv[j][1] = dv[j][2] = dv[j][3] = 0.0f;
    }
    const bool fast = (HALF & 5u) == 5u;            /* (x and dy kept as bfloat16: their tiles a step ahead) */
    Raw rq[NCH], rg[NCH];
    for (uint32_t h = kh * group; h < (kh + 1u) * group; h++) {
        const uint32_t sa = (s * p.heads + h) * p.cells, first = causal ? (j0 / BT) * BT : 0u;
        if (fast) {
            fetch_tile(p, s, first, h * D, C, p.x, true, rq);
            fetch_tile(p, s, first, h * D, out_c, p.dy, false, rg);
        }
        for (uint32_t q0 = first; q0 < p.cells; q0 += BT) {
            barrier();
            if (fast) {
                put_tile(p, sh.t, q0, true, rq);
                put_tile(p, sh.u, q0, false, rg);
            } else {
                load_tile(p, HALF, sh.t, s, q0, BT, h * D, C, p.x, 0u, true, threads);
                load_tile(p, HALF, sh.u, s, q0, BT, h * D, out_c, p.dy, 2u, false, threads);
            }
            for (uint32_t e = thread_x(); e < BT; e += threads) {
                const bool live = q0 + e < p.cells;
                sh.l[e] = live ? F(p.stats)[sa + q0 + e] : 0.0f;
                sh.d[e] = live ? F(p.stats)[rows_all + sa + q0 + e] : 0.0f;
            }
            barrier();
            if (fast && q0 + BT < p.cells) {
                fetch_tile(p, s, q0 + BT, h * D, C, p.x, true, rq);
                fetch_tile(p, s, q0 + BT, h * D, out_c, p.dy, false, rg);
            }
            /* S^T = K Q^T and dP^T = V dO^T: the warp's keys against the tile's queries */
            float st_[NT][4], dpt[NT][4];
#pragma unroll
            for (uint32_t j = 0; j < NT; j++) {
                st_[j][0] = st_[j][1] = st_[j][2] = st_[j][3] = 0.0f;
                dpt[j][0] = dpt[j][1] = dpt[j][2] = dpt[j][3] = 0.0f;
            }
#pragma unroll
            for (uint32_t k = 0; k < KS; k++)
#pragma unroll
                for (uint32_t j = 0; j < NT; j += 2u) {
                    uint32_t b[4], c[4];
                    frag_b(sh.t, 8u * j, k, b);
                    frag_b(sh.u, 8u * j, k, c);
                    mma(st_[j], ka[k], b[0], b[1]);
                    mma(st_[j + 1u], ka[k], b[2], b[3]);
                    mma(dpt[j], va[k], c[0], c[1]);
                    mma(dpt[j + 1u], va[k], c[2], c[3]);
                }
            /* P^T = exp(S^T scale - lse[query]), dS^T = P^T (dP^T - delta[query]) */
#pragma unroll
            for (uint32_t j = 0; j < NT; j++)
#pragma unroll
                for (uint32_t e = 0; e < 4u; e++) {
                    const uint32_t qc = 8u * j + 2u * t + (e & 1u), query = q0 + qc, key = e < 2u ? ka_row : kb_row;
                    const bool valid = query < p.cells && key < p.cells && (!causal || key <= query);
                    const float pv = valid ? exp_(st_[j][e] * scale - sh.l[qc]) : 0.0f;
                    st_[j][e] = pv;
                    dpt[j][e] = pv * (dpt[j][e] - sh.d[qc]);
                }
            /* dV += P^T dO, dK += dS^T Q (the queries' rows of the tiles transposed into B fragments) */
#pragma unroll
            for (uint32_t k = 0; k < BT / 16u; k++) {
                uint32_t pa[4], sa_[4];
                scores_a(st_, k, pa);
                scores_a(dpt, k, sa_);
#pragma unroll
                for (uint32_t j = 0; j < DT; j += 2u) {
                    uint32_t b[4], c[4];
                    frag_bt(sh.u, 8u * j, k, b);
                    frag_bt(sh.t, 8u * j, k, c);
                    mma(dv[j], pa, b[0], b[1]);
                    mma(dv[j + 1u], pa, b[2], b[3]);
                    mma(dk[j], sa_, c[0], c[1]);
                    mma(dk[j + 1u], sa_, c[2], c[3]);
                }
            }
        }
    }
    /* dK times the scale, rotated back through shared memory; dV as it is */
    barrier();
    float *dkf = (float *)sh.t;             /* ROWS x D floats */
#pragma unroll
    for (uint32_t j = 0; j < DT; j++) {
        const uint32_t c = 8u * j + 2u * t;
        dkf[(r0 + g) * D + c] = dk[j][0] * scale;
        dkf[(r0 + g) * D + c + 1u] = dk[j][1] * scale;
        dkf[(r0 + g + 8u) * D + c] = dk[j][2] * scale;
        dkf[(r0 + g + 8u) * D + c + 1u] = dk[j][3] * scale;
        const uint32_t at_a = (s * p.cells + ka_row) * C + (p.heads + p.kv + kh) * D + c;
        const uint32_t at_b = (s * p.cells + kb_row) * C + (p.heads + p.kv + kh) * D + c;
        if (ka_row < p.cells) {
            st(p.dx, at_a, 3u, HALF, dv[j][0]);
            st(p.dx, at_a + 1u, 3u, HALF, dv[j][1]);
        }
        if (kb_row < p.cells) {
            st(p.dx, at_b, 3u, HALF, dv[j][2]);
            st(p.dx, at_b + 1u, 3u, HALF, dv[j][3]);
        }
    }
    barrier();
    for (uint32_t e = thread_x(); e < ROWS * D; e += threads) {
        const uint32_t row = e / D, c = e % D, j = j0 + row;
        if (j >= p.cells) continue;
        float v = dkf[row * D + c];
        if (p.flags & FLAG_ROPE) {
            const uint32_t hd = D / 2u, ii = c < hd ? c : c - hd;
            const float cs = F(p.table)[j * hd + ii], sn = F(p.table)[p.cells * hd + j * hd + ii];
            v = c < hd ? v * cs + dkf[row * D + c + hd] * sn : v * cs - dkf[row * D + c - hd] * sn;
        }
        st(p.dx, (s * p.cells + j) * C + (p.heads + kh) * D + c, 3u, HALF, v);
    }
}
