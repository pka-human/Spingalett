/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * gflash: FusedAttention of graphs on the matrix units (mma.sync m16n8k16, bfloat16 operands, single-precision sums)
 * when their products are in bfloat16, as flash.cu's FORWARD computes attention: a warp takes MT tiles of 16 queries
 * of a batch and head, a block WARPS of them; tiles of BT keys and their values pass through shared memory (rows of
 * D + 8 bfloat16), with blocks of 128 queries the next tile's copied while the products of one run (asynchronous copies,
 * where keys and values are kept as bfloat16 in rows of 16 bytes; the second stage in the queries' room once they are
 * in registers).
 * Queries, keys and values are read where strides put them (SpgGattnPush, as gattn.cu reads them), floats rounded
 * to bfloat16 as they are loaded; the scores are scaled in single precision, then the keep mask and the mask apply
 * as in gattn.cu, the softmax online in registers (a row's maximum and sum combined over the four threads that hold
 * it, in a fixed pattern), O += P V with P rounded to bfloat16. Every product adds its 16 values of k at a time in
 * the order of k: the same bits on every run. A unit per head size D (32, 64, 128), and for
 * heads of 32 and 64 one of blocks of half the queries (_m1: MTS 1), for sequences too short to fill the device.
 */

#include "common.cuh"

#if !defined(SPG_ENTRY)
#define SPG_ENTRY spg_gflash_d64
#define D 64u
#endif
#define WARPS (D <= 64u ? 4u : 2u)         /* warps a block */
#if defined(MTS)
#define MT MTS                             /* tiles of 16 queries a warp: units _m1 for short sequences (more blocks) */
#else
#define MT (D <= 64u ? 2u : 1u)
#endif
#define ROWS (16u * MT * WARPS)            /* queries of a block */
#define BT (D <= 64u ? 64u : 32u)          /* keys a tile */
#define SD (D + 8u)                        /* bfloat16 a row in shared memory */
#define NT (BT / 8u)                       /* n-tiles of 8 of a tile of scores */
#define DT (D / 8u)                        /* n-tiles of 8 of a row of D */
#define KS (D / 16u)                       /* k-steps of 16 over D */
#define STAGES (ROWS >= 2u * BT ? 2u : 1u)  /* of keys and values: the second where the queries' room holds it */

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
/* two values rounded to bfloat16 (to the nearest, ties to even, as to_bf16() rounds all but NaN) in a word, the
   first in the low half: one instruction */
DEVICE uint32_t pack2(float lo, float hi) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
DEVICE float shfl_xor(float v, uint32_t mask) { return __nvvm_shfl_sync_bfly_f32(0xFFFFFFFFu, v, mask, 0x1F); }
/* asynchronous copies of 16 bytes to shared memory (`bytes` of them from src, the rest zeros) */
DEVICE void copy16(uint32_t dst, const void *src, uint32_t bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
DEVICE void copies_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <uint32_t N>
DEVICE void copies_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

/* x[b] (b < 4): this thread's pair of values of fragment b of a group of four, of a row the four threads of a quad
   hold (thread t its columns 2t and 2t + 1 of each fragment); after the exchange (a 4 x 4 transpose in two steps of
   shuffles), x[b] holds columns 2b and 2b + 1 of fragment t: the eight columns of one fragment */
DEVICE void quad_transpose(float (&x)[4][2], uint32_t t) {
#pragma unroll
    for (uint32_t b = 0; b < 4u; b += 2u)       /* with the thread t ^ 1 */
#pragma unroll
        for (uint32_t k = 0; k < 2u; k++) {
            const float r = shfl_xor((t & 1u) ? x[b][k] : x[b + 1u][k], 1u);
            if (t & 1u) x[b][k] = r;
            else x[b + 1u][k] = r;
        }
#pragma unroll
    for (uint32_t d = 0; d < 2u; d++)           /* with the thread t ^ 2 */
#pragma unroll
        for (uint32_t k = 0; k < 2u; k++) {
            const float r = shfl_xor((t & 2u) ? x[d][k] : x[2u + d][k], 2u);
            if (t & 2u) x[d][k] = r;
            else x[2u + d][k] = r;
        }
}

/* the maximum (sum) of a row's values over the four threads of a quad that hold it: xor 1, then xor 2 */
DEVICE float quad_max(float v) { v = max_(v, shfl_xor(v, 1u)); return max_(v, shfl_xor(v, 2u)); }
DEVICE float quad_sum(float v) { v += shfl_xor(v, 1u); return v + shfl_xor(v, 2u); }

struct uint4_ {
    uint32_t x, y, z, w;
} __attribute__((aligned(16)));

/* Rows r0 .. r0 + count - 1 (of `rows`) of floats into a shared tile [row][SD] of bfloat16: row r at base + r stride,
   its value c at c cs; zeros past the rows and the head's d values. Eight values a thread at a time, two loads of four
   where the rows are of exactly D contiguous values at 16 bytes (vec). */
DEVICE void load_tile(const SpgGattnPush &p, uint16_t *tile, uint64_t base, uint32_t stride, uint32_t cs, uint32_t r0,
                      uint32_t count, uint32_t rows, bool vec, bool half) {
    for (uint32_t e = thread_x(); e < count * (D / 8u); e += 32u * WARPS) {
        const uint32_t row = e / (D / 8u), c = 8u * (e % (D / 8u)), pos = r0 + row;
        if (half) {
            /* (kept as bfloat16: as it is, sixteen bytes at once where the rows allow) */
            uint4_ u = {0u, 0u, 0u, 0u};
            if (vec) {
                if (pos < rows) u = *(const uint4_ *)(H(base) + (uint64_t)pos * stride + c);
            } else {
                uint32_t w[4] = {0u, 0u, 0u, 0u};
#pragma unroll
                for (uint32_t k = 0; k < 8u; k++)
                    if (pos < rows && c + k < p.d)
                        w[k / 2u] |= (uint32_t)H(base)[(uint64_t)pos * stride + (uint64_t)(c + k) * cs] << (16u * (k & 1u));
                u = uint4_{w[0], w[1], w[2], w[3]};
            }
            *(uint4_ *)(tile + row * SD + c) = u;
            continue;
        }
        float v[8];
        if (vec) {
            const float4_ zero = {0.0f, 0.0f, 0.0f, 0.0f};
            const float *at = F(base) + (uint64_t)pos * stride + c;
            const float4_ a = pos < rows ? ((const float4_ *)at)[0] : zero, b = pos < rows ? ((const float4_ *)at)[1] : zero;
            v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
        } else {
#pragma unroll
            for (uint32_t k = 0; k < 8u; k++)
                v[k] = pos < rows && c + k < p.d ? F(base)[(uint64_t)pos * stride + (uint64_t)(c + k) * cs] : 0.0f;
        }
        *(uint4_ *)(tile + row * SD + c) = uint4_{pack2(v[0], v[1]), pack2(v[2], v[3]), pack2(v[4], v[5]), pack2(v[6], v[7])};
    }
}

/* Rows r0 .. r0 + BT - 1 (of `rows`) of bfloat16 kept so, D contiguous values at 16 bytes, into a shared tile by
   asynchronous copies (zeros past the rows) */
DEVICE void fetch_tile(uint16_t *tile, uint64_t base, uint32_t stride, uint32_t r0, uint32_t rows) {
    for (uint32_t e = thread_x(); e < BT * (D / 8u); e += 32u * WARPS) {
        const uint32_t row = e / (D / 8u), c = 8u * (e % (D / 8u)), pos = r0 + row;
        copy16(shared_address(tile + row * SD + c), pos < rows ? H(base) + (uint64_t)pos * stride + c : H(base),
               pos < rows ? 16u : 0u);
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

/* the block's queries, then (once in registers) the second of the stages of a tile of keys and one of values (with
   blocks of fewer queries one stage: a second would take room enough to leave the device fewer blocks) */
struct Shared {
    union {
        uint16_t a[ROWS * SD];
        uint16_t s1[STAGES > 1u ? 2u * BT * SD : 1u];
    } x;
    uint16_t s0[2u * BT * SD];
};

/* (blocks of 64 queries three to a multiprocessor: left to itself the compiler takes registers enough for two) */
#define MIN_BLOCKS (MT == 1u && D <= 64u ? 3u : 1u)

extern "C" __global__ void __attribute__((launch_bounds(32u * WARPS, MIN_BLOCKS))) SPG_ENTRY(const SpgGattnPush p, const Spec spec) {
    (void)spec;
    __shared__ __attribute__((aligned(16))) Shared sh;
    const uint32_t *st = U(p.strides);
    const uint32_t l = lane_id(), w = warp_id(), g = l >> 2, t = l & 3u;
    const uint32_t bh = block_y(), b = bh / p.heads, h = bh % p.heads, i0 = block_x() * ROWS, r0 = 16u * MT * w;
    const bool hq = (p.flags & SPG_GATTN_HALF_Q) != 0u, hk = (p.flags & SPG_GATTN_HALF_K) != 0u;
    const bool hv = (p.flags & SPG_GATTN_HALF_V) != 0u, hy = (p.flags & SPG_GATTN_HALF_Y) != 0u;
    /* (a session's prompt: its sequence's caches, the head of keys and values of its group, keys up to its last row's
       position) */
    const bool cache = (p.flags & SPG_GATTN_CACHE) != 0u;
    const uint32_t seq = cache ? U(p.info)[0] : b, p0 = cache ? U(p.info)[1] : 0u, hkv = cache ? h / p.group : h;
    const uint32_t keys = cache ? p0 + p.queries : p.keys;
    const uint64_t qb = p.q + (hq ? 2ull : 4ull) * ((uint64_t)b * st[0] + (uint64_t)h * st[1]);
    const uint64_t kb = p.k + (hk ? 2ull : 4ull) * ((uint64_t)seq * st[3] + (uint64_t)hkv * st[4]);
    const uint64_t vb = p.v + (hv ? 2ull : 4ull) * ((uint64_t)seq * st[6] + (uint64_t)hkv * st[7]);
    const uint64_t mbase = (uint64_t)b * st[12] + (uint64_t)h * st[13], wbase = (uint64_t)b * st[16] + (uint64_t)h * st[17];
    const bool vec = (p.flags & SPG_GATTN_VEC) != 0u, masked = (p.flags & (SPG_GATTN_MASK | SPG_GATTN_BOOL | SPG_GATTN_KEEP)) != 0u;
    const float NEG = -__builtin_inff(), scale = p.scale;
    load_tile(p, sh.x.a, qb, st[2], 1u, i0, ROWS, p.queries, vec, hq);
    barrier();
    uint32_t qa[MT][KS][4];
#pragma unroll
    for (uint32_t mt = 0; mt < MT; mt++)
#pragma unroll
        for (uint32_t k = 0; k < KS; k++) frag_a(sh.x.a, r0 + 16u * mt, k, qa[mt][k]);
    barrier();          /* (the queries' tile free for the second stage) */
    /* tiles of keys and values into stage s: the next tile's copied while this one's products run, where both are
       kept as bfloat16 in rows of 16 bytes */
    const bool async = vec && hk && hv;
    auto fetch = [&](uint32_t j0, uint32_t s) {
        uint16_t *kt = s ? sh.x.s1 : sh.s0, *vt = kt + BT * SD;
        if (async) {
            fetch_tile(kt, kb, st[5], j0, keys);
            fetch_tile(vt, vb, st[8], j0, keys);
            copies_commit();
        } else {
            load_tile(p, kt, kb, st[5], st[20], j0, BT, keys, vec, hk);
            load_tile(p, vt, vb, st[8], 1u, j0, BT, keys, vec, hv);
        }
    };
    if (STAGES > 1u) fetch(cache && p.window && p0 + i0 + 1u > p.window ? (p0 + i0 + 1u - p.window) / BT * BT : 0u, 0u);
    float o[MT][DT][4], m_[MT][2], l_[MT][2];
#pragma unroll
    for (uint32_t mt = 0; mt < MT; mt++) {
#pragma unroll
        for (uint32_t j = 0; j < DT; j++) o[mt][j][0] = o[mt][j][1] = o[mt][j][2] = o[mt][j][3] = 0.0f;
        m_[mt][0] = m_[mt][1] = NEG;
        l_[mt][0] = l_[mt][1] = 0.0f;
    }
    uint32_t s = 0;
    /* (a prompt's block: the keys at its last query's position and before, from its first query's window on) */
    const uint32_t last_key = cache ? (p0 + i0 + ROWS < keys ? p0 + i0 + ROWS : keys) : keys;
    const uint32_t first_key = cache && p.window && p0 + i0 + 1u > p.window ? (p0 + i0 + 1u - p.window) / BT * BT : 0u;
    for (uint32_t j0 = first_key; j0 < last_key; j0 += BT, s ^= STAGES - 1u) {
        const bool more = j0 + BT < last_key;
        if (STAGES == 1u) fetch(j0, 0u);
        else if (more) fetch(j0 + BT, s ^ 1u);
        if (async) {
            if (STAGES > 1u && more) copies_wait<1>();
            else copies_wait<0>();
        }
        barrier();
        const uint16_t *kt = s ? sh.x.s1 : sh.s0, *vt = kt + BT * SD;
        float sc[MT][NT][4];
#pragma unroll
        for (uint32_t mt = 0; mt < MT; mt++)
#pragma unroll
            for (uint32_t j = 0; j < NT; j++) sc[mt][j][0] = sc[mt][j][1] = sc[mt][j][2] = sc[mt][j][3] = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < KS; k++)
#pragma unroll
            for (uint32_t j = 0; j < NT; j += 2u) {
                uint32_t bf[4];
                frag_b(kt, 8u * j, k, bf);
#pragma unroll
                for (uint32_t mt = 0; mt < MT; mt++) {
                    mma(sc[mt][j], qa[mt][k], bf[0], bf[1]);
                    mma(sc[mt][j + 1u], qa[mt][k], bf[2], bf[3]);
                }
            }
#pragma unroll
        for (uint32_t mt = 0; mt < MT; mt++) {
            /* the scale, then the masks (and keys past the end, queries past the end) on the tiles that need them:
               this thread's rows g and g + 8 of the tile */
            const uint32_t q0 = i0 + r0 + 16u * mt, ra = q0 + g, rb = ra + 8u;
            const bool edge = masked || cache || j0 + BT > keys || q0 + 16u > p.queries;
#pragma unroll
            for (uint32_t j = 0; j < NT; j++)
#pragma unroll
                for (uint32_t e = 0; e < 4u; e++) {
                    float v = sc[mt][j][e] * scale;
                    if (edge) {
                        const uint32_t key = j0 + 8u * j + 2u * t + (e & 1u), row = e < 2u ? ra : rb;
                        if (key >= keys || row >= p.queries || (cache && (key > p0 + row ||
                                                                        (p.window && key + p.window <= p0 + row)))) {
                            v = NEG;
                        } else {
                            if ((p.flags & SPG_GATTN_KEEP) &&
                                !((const uint8_t *)p.keep)[wbase + (uint64_t)row * st[18] + (uint64_t)key * st[19]])
                                v = p.fill;
                            if (p.flags & SPG_GATTN_MASK) v += F(p.mask)[mbase + (uint64_t)row * st[14] + (uint64_t)key * st[15]];
                            else if ((p.flags & SPG_GATTN_BOOL) &&
                                     !((const uint8_t *)p.mask)[mbase + (uint64_t)row * st[14] + (uint64_t)key * st[15]])
                                v = NEG;
                        }
                    }
                    sc[mt][j][e] = v;
                }
            float ta = NEG, tb = NEG;
#pragma unroll
            for (uint32_t j = 0; j < NT; j++) {
                ta = max_(ta, max_(sc[mt][j][0], sc[mt][j][1]));
                tb = max_(tb, max_(sc[mt][j][2], sc[mt][j][3]));
            }
            ta = quad_max(ta);
            tb = quad_max(tb);
            const float na = max_(m_[mt][0], ta), nb = max_(m_[mt][1], tb);
            /* e^(v - m) as 2^(v log2 e - m log2 e): one fused step a score; scores of -inf (only on edge tiles) 0 */
            const float L2E = 1.4426950408889634f, ma = na * L2E, mb = nb * L2E;
            const float aa = na == NEG ? 1.0f : __nvvm_ex2_approx_f(fma_(m_[mt][0], L2E, -ma));
            const float ab = nb == NEG ? 1.0f : __nvvm_ex2_approx_f(fma_(m_[mt][1], L2E, -mb));
            float suma = 0.0f, sumb = 0.0f;
            if (edge) {
#pragma unroll
                for (uint32_t j = 0; j < NT; j++)
#pragma unroll
                    for (uint32_t e = 0; e < 4u; e++)
                        sc[mt][j][e] = sc[mt][j][e] == NEG ? 0.0f
                                     : __nvvm_ex2_approx_f(fma_(sc[mt][j][e], L2E, e < 2u ? -ma : -mb));
            } else {
#pragma unroll
                for (uint32_t j = 0; j < NT; j++)
#pragma unroll
                    for (uint32_t e = 0; e < 4u; e++) sc[mt][j][e] = __nvvm_ex2_approx_f(fma_(sc[mt][j][e], L2E, e < 2u ? -ma : -mb));
            }
#pragma unroll
            for (uint32_t j = 0; j < NT; j++) {
                suma += sc[mt][j][0] + sc[mt][j][1];
                sumb += sc[mt][j][2] + sc[mt][j][3];
            }
            l_[mt][0] = l_[mt][0] * aa + suma;
            l_[mt][1] = l_[mt][1] * ab + sumb;
            m_[mt][0] = na;
            m_[mt][1] = nb;
#pragma unroll
            for (uint32_t j = 0; j < DT; j++) {
                o[mt][j][0] *= aa; o[mt][j][1] *= aa;
                o[mt][j][2] *= ab; o[mt][j][3] *= ab;
            }
        }
        /* O += P V: P's rows as A fragments, V's [key][d] tile transposed into B fragments */
#pragma unroll
        for (uint32_t k = 0; k < BT / 16u; k++) {
            uint32_t pa[MT][4];
#pragma unroll
            for (uint32_t mt = 0; mt < MT; mt++) scores_a(sc[mt], k, pa[mt]);
#pragma unroll
            for (uint32_t j = 0; j < DT; j += 2u) {
                uint32_t bf[4];
                frag_bt(vt, 8u * j, k, bf);
#pragma unroll
                for (uint32_t mt = 0; mt < MT; mt++) {
                    mma(o[mt][j], pa[mt], bf[0], bf[1]);
                    mma(o[mt][j + 1u], pa[mt], bf[2], bf[3]);
                }
            }
        }
        barrier();      /* (stage s free for the tile after next, or the next with one stage) */
    }
    /* y = O / sum (a row of no key: zeros with SAFE, else NaN), this thread's columns 8 j + 2 t, + 1 of rows g, g + 8 */
    const uint64_t yb = p.y + (hy ? 2ull : 4ull) * ((uint64_t)b * st[9] + (uint64_t)h * st[10]);
    const bool safe = (p.flags & SPG_GATTN_SAFE) != 0u;
#pragma unroll
    for (uint32_t mt = 0; mt < MT; mt++) {
        const uint32_t ra = i0 + r0 + 16u * mt + g, rb = ra + 8u;
        l_[mt][0] = quad_sum(l_[mt][0]);
        l_[mt][1] = quad_sum(l_[mt][1]);
#pragma unroll
        for (uint32_t hh = 0; hh < 2u; hh++) {
            const uint32_t row = hh ? rb : ra;
            const float sum = l_[mt][hh];
            const float inv = sum > 0.0f ? 1.0f / sum : safe ? 0.0f : __builtin_nanf("");
            if (row >= p.queries) continue;
            const uint64_t yr = (uint64_t)row * st[11];
#pragma unroll
            for (uint32_t j = 0; j < DT; j++) {
                const uint32_t c = 8u * j + 2u * t;
                const float v0 = sum > 0.0f || !safe ? o[mt][j][2u * hh] * inv : 0.0f;
                const float v1 = sum > 0.0f || !safe ? o[mt][j][2u * hh + 1u] * inv : 0.0f;
                if (hy) {
                    if (c + 1u < p.d && (yr + c) % 2u == 0u) ((uint32_t *)H(yb))[(yr + c) >> 1] = pack2(v0, v1);
                    else {
                        if (c < p.d) H(yb)[yr + c] = (uint16_t)to_bf16(v0);
                        if (c + 1u < p.d) H(yb)[yr + c + 1u] = (uint16_t)to_bf16(v1);
                    }
                } else {
                    if (c < p.d) F(yb)[yr + c] = v0;
                    if (c + 1u < p.d) F(yb)[yr + c + 1u] = v1;
                }
            }
        }
    }
}
