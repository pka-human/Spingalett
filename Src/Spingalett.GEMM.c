/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Native single-precision matrix multiplication, C = alpha * op(A) * op(B) + beta * C (row-major),
 * structured like BLIS/GotoBLAS: op(B) is packed in KC x NC blocks of NR-wide column panels,
 * op(A) in MC x KC blocks of MR-high row panels, and a register-blocked MR x NR micro-kernel
 * multiplies one panel pair. Packing absorbs the transposes (with SIMD 8 x 8 and 4 x 4 transposes
 * where AVX is available), so all four op() combinations run the same kernels. The micro-kernel
 * is chosen at compile time: AVX-512 (12 x 32), AVX/FMA (6 x 16) or portable C (4 x 8) that
 * compilers vectorize for SSE or NEON; the last row panel of a block runs a kernel with fewer rows
 * (8 or 4 of 12, 4 or 2 of 6) instead of computing zero rows.
 *
 * Threads split C by rows and columns only: every element is summed over k in the same order
 * whatever the thread count, so results do not depend on it. Large M splits row blocks (and
 * columns when there are few blocks), with op(B) packed once per K block for all threads; small M
 * (a mini-batch) packs op(A) once for all threads and gives each thread its own column panels,
 * which it packs and multiplies itself.
 *
 * x86-64 builds that are not tuned for the build machine (SPINGALETT_GEMM_DISPATCH) compile this
 * file twice more, through Kernels/Spingalett.GEMM.AVX2.c and Kernels/Spingalett.GEMM.AVX512.c
 * with those instruction sets enabled; spingalett_gemm_native() then runs the best kernels the
 * processor supports.
 */

#include "Spingalett.Private.h"
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#define MR 12
#define NR 32
#define MC 144
#elif defined(__AVX__)
#include <immintrin.h>
#define MR 6
#define NR 16
#define MC 120
#else
#define MR 4
#define NR 8
#define MC 128
#endif

#define KC 256
#define NC 3072
#define MC_MAX 144      /* the largest MC of any kernel set: one scratch serves them all */

/* Small M: at most this many row blocks run with op(A) shared between threads. */
#define SHARED_A_BLOCKS 2

struct SpingalettGemmScratch {
    float *b_pack;      /* KC x NC */
    float *a_pack;      /* threads x (MC_MAX x KC); with a shared op(A), one M x KC block */
    int threads;
};

#if !defined(SPINGALETT_GEMM_VARIANT)
SpingalettGemmScratch *spingalett_gemm_scratch_create(int threads) {
    if (threads < 1) threads = 1;
    SpingalettGemmScratch *s = (SpingalettGemmScratch *)calloc(1, sizeof(SpingalettGemmScratch));
    if (!s) return NULL;
    s->threads = threads;
    s->b_pack = (float *)spingalett_aligned_alloc((size_t)KC * NC * sizeof(float));
    s->a_pack = (float *)spingalett_aligned_alloc((size_t)threads * MC_MAX * KC * sizeof(float));
    if (!s->b_pack || !s->a_pack) {
        spingalett_gemm_scratch_free(s);
        return NULL;
    }
    return s;
}

void spingalett_gemm_scratch_free(SpingalettGemmScratch *s) {
    if (!s) return;
    spingalett_aligned_free(s->b_pack);
    spingalett_aligned_free(s->a_pack);
    free(s);
}
#endif

#if defined(__AVX__)
/* dst[k * ds + r] = src[r * ld + k] for an 8 x 8 block. */
static inline void transpose_8x8(const float *restrict src, size_t ld, float *restrict dst, size_t ds) {
    __m256 r0 = _mm256_loadu_ps(src),          r1 = _mm256_loadu_ps(src + ld);
    __m256 r2 = _mm256_loadu_ps(src + 2 * ld), r3 = _mm256_loadu_ps(src + 3 * ld);
    __m256 r4 = _mm256_loadu_ps(src + 4 * ld), r5 = _mm256_loadu_ps(src + 5 * ld);
    __m256 r6 = _mm256_loadu_ps(src + 6 * ld), r7 = _mm256_loadu_ps(src + 7 * ld);
    __m256 t0 = _mm256_unpacklo_ps(r0, r1), t1 = _mm256_unpackhi_ps(r0, r1);
    __m256 t2 = _mm256_unpacklo_ps(r2, r3), t3 = _mm256_unpackhi_ps(r2, r3);
    __m256 t4 = _mm256_unpacklo_ps(r4, r5), t5 = _mm256_unpackhi_ps(r4, r5);
    __m256 t6 = _mm256_unpacklo_ps(r6, r7), t7 = _mm256_unpackhi_ps(r6, r7);
    __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44), u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44), u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44), u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44), u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    _mm256_storeu_ps(dst,          _mm256_permute2f128_ps(u0, u4, 0x20));
    _mm256_storeu_ps(dst + ds,     _mm256_permute2f128_ps(u1, u5, 0x20));
    _mm256_storeu_ps(dst + 2 * ds, _mm256_permute2f128_ps(u2, u6, 0x20));
    _mm256_storeu_ps(dst + 3 * ds, _mm256_permute2f128_ps(u3, u7, 0x20));
    _mm256_storeu_ps(dst + 4 * ds, _mm256_permute2f128_ps(u0, u4, 0x31));
    _mm256_storeu_ps(dst + 5 * ds, _mm256_permute2f128_ps(u1, u5, 0x31));
    _mm256_storeu_ps(dst + 6 * ds, _mm256_permute2f128_ps(u2, u6, 0x31));
    _mm256_storeu_ps(dst + 7 * ds, _mm256_permute2f128_ps(u3, u7, 0x31));
}

/* The same for a 4 x 4 block. */
static inline void transpose_4x4(const float *restrict src, size_t ld, float *restrict dst, size_t ds) {
    __m128 r0 = _mm_loadu_ps(src), r1 = _mm_loadu_ps(src + ld);
    __m128 r2 = _mm_loadu_ps(src + 2 * ld), r3 = _mm_loadu_ps(src + 3 * ld);
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    _mm_storeu_ps(dst, r0);
    _mm_storeu_ps(dst + ds, r1);
    _mm_storeu_ps(dst + 2 * ds, r2);
    _mm_storeu_ps(dst + 3 * ds, r3);
}
#endif

/* Packs `rows` (<= width) rows of a row-major matrix, columns [k0, k0 + kc), transposed into a
   panel of `width` floats per k: dst[k * width + r] = src[r * ld + k0 + k]; rows past `rows`
   are zero. */
static void pack_transposed(float *restrict dst, const float *restrict src, size_t ld,
                            uint32_t k0, uint32_t kc, uint32_t rows, uint32_t width) {
    src += k0;
    uint32_t r = 0;
#if defined(__AVX__)
    uint32_t k8 = kc & ~7u, k4 = kc & ~3u;
    for (; r + 8 <= rows; r += 8)
        for (uint32_t k = 0; k < k8; k += 8)
            transpose_8x8(src + (size_t)r * ld + k, ld, dst + (size_t)k * width + r, width);
    for (; r + 4 <= rows; r += 4)
        for (uint32_t k = 0; k < k4; k += 4)
            transpose_4x4(src + (size_t)r * ld + k, ld, dst + (size_t)k * width + r, width);
    /* the k tail of the transposed row groups */
    for (uint32_t g = 0; g < r; g++) {
        uint32_t from = g < (rows & ~7u) ? k8 : k4;
        for (uint32_t k = from; k < kc; k++) dst[(size_t)k * width + g] = src[(size_t)g * ld + k];
    }
#endif
    for (; r < rows; r++)
        for (uint32_t k = 0; k < kc; k++) dst[(size_t)k * width + r] = src[(size_t)r * ld + k];
    for (; r < width; r++)
        for (uint32_t k = 0; k < kc; k++) dst[(size_t)k * width + r] = 0.0f;
}

/* Packs rows [0, rows) of an MR-row panel starting at row i of op(A), columns [k0, k0 + kc). */
static void pack_a_panel(float *restrict dst, const float *restrict A, size_t lda, bool trans,
                         uint32_t i, uint32_t rows, uint32_t k0, uint32_t kc) {
    if (!trans) {
        pack_transposed(dst, A + (size_t)i * lda, lda, k0, kc, rows, MR);
        return;
    }
    const float *src = A + (size_t)k0 * lda + i;
    for (uint32_t k = 0; k < kc; k++, src += lda, dst += MR) {
        for (uint32_t r = 0; r < rows; r++) dst[r] = src[r];
        for (uint32_t r = rows; r < MR; r++) dst[r] = 0.0f;
    }
}

/* Packs NR-column panel (columns [j, j + cols)) of op(B), rows [k0, k0 + kc). */
static void pack_b_panel(float *restrict dst, const float *restrict B, size_t ldb, bool trans,
                         uint32_t k0, uint32_t kc, uint32_t j, uint32_t cols) {
    if (trans) {
        pack_transposed(dst, B + (size_t)j * ldb, ldb, k0, kc, cols, NR);
        return;
    }
    const float *src = B + (size_t)k0 * ldb + j;
    for (uint32_t k = 0; k < kc; k++, src += ldb, dst += NR) {
        if (cols == NR) {
            memcpy(dst, src, NR * sizeof(float));
        } else {
            for (uint32_t c = 0; c < cols; c++) dst[c] = src[c];
            for (uint32_t c = cols; c < NR; c++) dst[c] = 0.0f;
        }
    }
}

/* Writes alpha * acc (+ beta * C) for a rows x cols tile; acc is MR x NR. */
static inline void store_tile(float *restrict c, size_t ldc, const float *restrict acc,
                              float alpha, float beta, uint32_t rows, uint32_t cols) {
    for (uint32_t r = 0; r < rows; r++) {
        float *cr = c + (size_t)r * ldc;
        const float *ar = acc + (size_t)r * NR;
        if (beta == 0.0f)
            for (uint32_t j = 0; j < cols; j++) cr[j] = alpha * ar[j];
        else
            for (uint32_t j = 0; j < cols; j++) cr[j] = alpha * ar[j] + beta * cr[j];
    }
}

#if defined(__clang__)
#define UNROLL_4 _Pragma("unroll 4")
#elif defined(__GNUC__)
#define UNROLL_4 _Pragma("GCC unroll 4")
#else
#define UNROLL_4
#endif

/* A micro-kernel multiplies an MR-row panel of op(A) (of which it uses the first `rows`, at most
   its own height) by an NR-column panel of op(B) over kc and stores the rows x cols tile. */
typedef void (*MicroKernel)(uint32_t kc, const float *restrict a, const float *restrict b,
                            float *restrict c, size_t ldc, float alpha, float beta,
                            uint32_t rows, uint32_t cols);

#if defined(__AVX512F__)

#define ROWS_12(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11)
#define ROWS_8(X)  X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#define ROWS_4(X)  X(0) X(1) X(2) X(3)

#define K_DECL(r) __m512 c##r##0 = _mm512_setzero_ps(), c##r##1 = _mm512_setzero_ps();
#define K_FMA(r) { __m512 ar = _mm512_set1_ps(a[r]); \
                   c##r##0 = _mm512_fmadd_ps(ar, b0, c##r##0); c##r##1 = _mm512_fmadd_ps(ar, b1, c##r##1); }
#define K_STORE(r) if ((r) < rows) { float *cr = c + (size_t)(r) * ldc; \
                   __m512 x0 = _mm512_mul_ps(va, c##r##0), x1 = _mm512_mul_ps(va, c##r##1); \
                   if (beta != 0.0f) { x0 = _mm512_fmadd_ps(vb, _mm512_loadu_ps(cr), x0); \
                                       x1 = _mm512_fmadd_ps(vb, _mm512_loadu_ps(cr + 16), x1); } \
                   _mm512_storeu_ps(cr, x0); _mm512_storeu_ps(cr + 16, x1); }
#define K_SPILL(r) _mm512_store_ps(acc + (r) * NR, c##r##0); _mm512_store_ps(acc + (r) * NR + 16, c##r##1);

#define DEFINE_KERNEL(name, ROWS) \
static void name(uint32_t kc, const float *restrict a, const float *restrict b, \
                 float *restrict c, size_t ldc, float alpha, float beta, uint32_t rows, uint32_t cols) { \
    ROWS(K_DECL) \
    UNROLL_4 for (uint32_t k = 0; k < kc; k++, a += MR, b += NR) { \
        __m512 b0 = _mm512_load_ps(b), b1 = _mm512_load_ps(b + 16); \
        ROWS(K_FMA) \
    } \
    if (cols == NR) { \
        __m512 va = _mm512_set1_ps(alpha), vb = _mm512_set1_ps(beta); \
        ROWS(K_STORE) \
    } else { \
        _Alignas(64) float acc[MR * NR]; \
        ROWS(K_SPILL) \
        store_tile(c, ldc, acc, alpha, beta, rows, cols); \
    } \
}

DEFINE_KERNEL(kernel_12, ROWS_12)
DEFINE_KERNEL(kernel_8, ROWS_8)
DEFINE_KERNEL(kernel_4, ROWS_4)

static inline MicroKernel kernel_for(uint32_t rows) {
    return rows > 8 ? kernel_12 : rows > 4 ? kernel_8 : kernel_4;
}

#elif defined(__AVX__)

#if defined(__FMA__)
#define GEMM_FMA(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#else
#define GEMM_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif
#define ROWS_6(X) X(0) X(1) X(2) X(3) X(4) X(5)
#define ROWS_4(X) X(0) X(1) X(2) X(3)
#define ROWS_2(X) X(0) X(1)

#define K_DECL(r) __m256 c##r##0 = _mm256_setzero_ps(), c##r##1 = _mm256_setzero_ps();
#define K_FMA(r) { __m256 ar = _mm256_broadcast_ss(a + (r)); \
                   c##r##0 = GEMM_FMA(ar, b0, c##r##0); c##r##1 = GEMM_FMA(ar, b1, c##r##1); }
#define K_STORE(r) if ((r) < rows) { float *cr = c + (size_t)(r) * ldc; \
                   __m256 x0 = _mm256_mul_ps(va, c##r##0), x1 = _mm256_mul_ps(va, c##r##1); \
                   if (beta != 0.0f) { x0 = GEMM_FMA(vb, _mm256_loadu_ps(cr), x0); \
                                       x1 = GEMM_FMA(vb, _mm256_loadu_ps(cr + 8), x1); } \
                   _mm256_storeu_ps(cr, x0); _mm256_storeu_ps(cr + 8, x1); }
#define K_SPILL(r) _mm256_store_ps(acc + (r) * NR, c##r##0); _mm256_store_ps(acc + (r) * NR + 8, c##r##1);

#define DEFINE_KERNEL(name, ROWS) \
static void name(uint32_t kc, const float *restrict a, const float *restrict b, \
                 float *restrict c, size_t ldc, float alpha, float beta, uint32_t rows, uint32_t cols) { \
    ROWS(K_DECL) \
    UNROLL_4 for (uint32_t k = 0; k < kc; k++, a += MR, b += NR) { \
        __m256 b0 = _mm256_load_ps(b), b1 = _mm256_load_ps(b + 8); \
        ROWS(K_FMA) \
    } \
    if (cols == NR) { \
        __m256 va = _mm256_set1_ps(alpha), vb = _mm256_set1_ps(beta); \
        ROWS(K_STORE) \
    } else { \
        _Alignas(32) float acc[MR * NR]; \
        ROWS(K_SPILL) \
        store_tile(c, ldc, acc, alpha, beta, rows, cols); \
    } \
}

DEFINE_KERNEL(kernel_6, ROWS_6)
DEFINE_KERNEL(kernel_4, ROWS_4)
DEFINE_KERNEL(kernel_2, ROWS_2)

static inline MicroKernel kernel_for(uint32_t rows) {
    return rows > 4 ? kernel_6 : rows > 2 ? kernel_4 : kernel_2;
}

#else

static void kernel_generic(uint32_t kc, const float *restrict a, const float *restrict b,
                           float *restrict c, size_t ldc, float alpha, float beta,
                           uint32_t rows, uint32_t cols) {
    float acc[MR * NR] = {0};
    for (uint32_t k = 0; k < kc; k++, a += MR, b += NR)
        for (uint32_t r = 0; r < MR; r++)
            for (uint32_t j = 0; j < NR; j++)
                acc[r * NR + j] += a[r] * b[j];
    store_tile(c, ldc, acc, alpha, beta, rows, cols);
}

static inline MicroKernel kernel_for(uint32_t rows) {
    (void)rows;
    return kernel_generic;
}

#endif

/* C rows [i0, i0 + mc) x NR-column panels [q_begin, q_end) of the current N block, from packed
   op(A) row panels (a, MR x kc each) and packed op(B) column panels (b_pack, NR x kc each). */
static void multiply_block(const float *a, const float *b_pack, uint32_t mc, uint32_t nc,
                           int64_t q_begin, int64_t q_end, uint32_t kc,
                           float *C, size_t ldc, float alpha, float beta) {
    for (int64_t q = q_begin; q < q_end; q++) {
        uint32_t j = (uint32_t)q * NR;
        uint32_t cols = (nc - j < NR) ? nc - j : NR;
        const float *b_panel = b_pack + (size_t)j * kc;
        for (uint32_t p = 0; p < mc; p += MR) {
            uint32_t rows = (mc - p < MR) ? mc - p : MR;
            kernel_for(rows)(kc, a + (size_t)p * kc, b_panel, C + (size_t)p * ldc + j, ldc, alpha, beta, rows, cols);
        }
    }
}

/* The multiplication with this translation unit's kernels (scratch is not NULL, M, N, K > 0). */
#if defined(SPINGALETT_GEMM_VARIANT)
void SPINGALETT_GEMM_VARIANT(SpingalettGemmScratch *scratch, bool trans_a, bool trans_b,
                             uint32_t M, uint32_t N, uint32_t K, float alpha,
                             const float *A, size_t lda, const float *B, size_t ldb,
                             float beta, float *C, size_t ldc, bool parallel) {
#elif defined(SPINGALETT_GEMM_DISPATCH)
void spingalett_gemm_baseline(SpingalettGemmScratch *scratch, bool trans_a, bool trans_b,
                              uint32_t M, uint32_t N, uint32_t K, float alpha,
                              const float *A, size_t lda, const float *B, size_t ldb,
                              float beta, float *C, size_t ldc, bool parallel) {
#else
static void spingalett_gemm_baseline(SpingalettGemmScratch *scratch, bool trans_a, bool trans_b,
                                     uint32_t M, uint32_t N, uint32_t K, float alpha,
                                     const float *A, size_t lda, const float *B, size_t ldb,
                                     float beta, float *C, size_t ldc, bool parallel) {
#endif
    int threads = parallel ? scratch->threads : 1;

    uint32_t m_panels = (M + MR - 1) / MR;
    int64_t mblocks = (int64_t)((M + MC - 1) / MC);
    bool shared_a = threads > 1 && mblocks <= SHARED_A_BLOCKS &&
                    (size_t)m_panels * MR <= (size_t)scratch->threads * MC_MAX;

#if defined(_OPENMP)
#pragma omp parallel num_threads(threads) if(threads > 1)
#endif
    {
        int tid = 0, team = 1;
#if defined(_OPENMP)
        tid = omp_get_thread_num();
        team = omp_get_num_threads();
#endif
        for (uint32_t j0 = 0; j0 < N; j0 += NC) {
            uint32_t nc = (N - j0 < NC) ? N - j0 : NC;
            int64_t panels = (int64_t)((nc + NR - 1) / NR);

            for (uint32_t k0 = 0; k0 < K; k0 += KC) {
                uint32_t kc = (K - k0 < KC) ? K - k0 : KC;
                float beta_k = (k0 == 0) ? beta : 1.0f;     /* later K blocks accumulate */

                if (shared_a) {
                    /* All row panels of op(A) are packed once for the team; each thread packs
                       and multiplies its own range of column panels. */
#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
                    for (int64_t p = 0; p < (int64_t)m_panels; p++) {
                        uint32_t i = (uint32_t)p * MR;
                        pack_a_panel(scratch->a_pack + (size_t)i * kc, A, lda, trans_a, i,
                                     (M - i < MR) ? M - i : MR, k0, kc);
                    }
                    int64_t per = (panels + team - 1) / team;
                    int64_t q_begin = (int64_t)tid * per;
                    int64_t q_end = q_begin + per < panels ? q_begin + per : panels;
                    for (int64_t q = q_begin; q < q_end; q++) {
                        uint32_t j = (uint32_t)q * NR;
                        pack_b_panel(scratch->b_pack + (size_t)j * kc, B, ldb, trans_b, k0, kc, j0 + j,
                                     (nc - j < NR) ? nc - j : NR);
                    }
#if defined(_OPENMP)
#pragma omp barrier
#endif
                    if (q_begin < q_end)
                        multiply_block(scratch->a_pack, scratch->b_pack, M, nc, q_begin, q_end, kc,
                                       C + j0, ldc, alpha, beta_k);
#if defined(_OPENMP)
#pragma omp barrier
#endif
                    continue;
                }

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (int64_t q = 0; q < panels; q++) {
                    uint32_t j = (uint32_t)q * NR;
                    pack_b_panel(scratch->b_pack + (size_t)j * kc, B, ldb, trans_b, k0, kc, j0 + j,
                                 (nc - j < NR) ? nc - j : NR);
                }

                /* Work items are (row block, column range) tiles; columns are split only as far
                   as needed to give every thread work when there are few row blocks. */
                int64_t nsplit = 1;
                if (team > 1 && mblocks < 2 * team) {
                    nsplit = (2 * team + mblocks - 1) / mblocks;
                    if (nsplit > panels) nsplit = panels;
                }
                int64_t panels_per_split = (panels + nsplit - 1) / nsplit;
                int64_t tiles = mblocks * nsplit;
                float *a_pack = scratch->a_pack + (size_t)tid * MC_MAX * KC;

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
                for (int64_t t = 0; t < tiles; t++) {
                    uint32_t i0 = (uint32_t)(t / nsplit) * MC;
                    uint32_t mc = (M - i0 < MC) ? M - i0 : MC;
                    int64_t q_begin = (t % nsplit) * panels_per_split;
                    int64_t q_end = q_begin + panels_per_split < panels ? q_begin + panels_per_split : panels;
                    if (q_begin >= q_end) continue;
                    for (uint32_t p = 0; p < mc; p += MR)
                        pack_a_panel(a_pack + (size_t)p * kc, A, lda, trans_a, i0 + p,
                                     (mc - p < MR) ? mc - p : MR, k0, kc);
                    multiply_block(a_pack, scratch->b_pack, mc, nc, q_begin, q_end, kc,
                                   C + (size_t)i0 * ldc + j0, ldc, alpha, beta_k);
                }
            }
        }
    }
}

#if !defined(SPINGALETT_GEMM_VARIANT)

#if defined(__AVX512F__)
#  define GEMM_KERNELS "AVX-512"
#elif defined(__AVX2__) && defined(__FMA__)
#  define GEMM_KERNELS "AVX2"
#elif defined(__AVX__)
#  define GEMM_KERNELS "AVX"
#elif defined(__SSE2__) || defined(_M_X64)
#  define GEMM_KERNELS "SSE2"
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#  define GEMM_KERNELS "NEON"
#else
#  define GEMM_KERNELS "C"
#endif

typedef void (*GemmFunction)(SpingalettGemmScratch *, bool, bool, uint32_t, uint32_t, uint32_t, float,
                             const float *, size_t, const float *, size_t, float, float *, size_t, bool);

/* The kernels for this processor: the run-time choice where there is one. */
static GemmFunction gemm_select(const char **name) {
#if defined(SPINGALETT_GEMM_DISPATCH)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) { *name = "AVX-512"; return spingalett_gemm_avx512; }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) { *name = "AVX2"; return spingalett_gemm_avx2; }
#endif
    *name = GEMM_KERNELS;
    return spingalett_gemm_baseline;
}

const char *spingalett_cpu_kernels(void) {
    const char *name;
    gemm_select(&name);
    return name;
}

void spingalett_gemm_native(SpingalettGemmScratch *scratch, bool trans_a, bool trans_b,
                            uint32_t M, uint32_t N, uint32_t K, float alpha,
                            const float *A, size_t lda, const float *B, size_t ldb,
                            float beta, float *C, size_t ldc, bool parallel) {
    if (M == 0 || N == 0) return;
    if (K == 0 || alpha == 0.0f) {      /* C = beta * C */
        for (uint32_t i = 0; i < M; i++) {
            float *cr = C + (size_t)i * ldc;
            if (beta == 0.0f) memset(cr, 0, N * sizeof(float));
            else for (uint32_t j = 0; j < N; j++) cr[j] *= beta;
        }
        return;
    }

    SpingalettGemmScratch *own = NULL;
    if (!scratch) {
        int threads = 1;
#if defined(_OPENMP)
        if (parallel) threads = omp_get_max_threads();
#endif
        scratch = own = spingalett_gemm_scratch_create(threads);
        if (!scratch) { set_error(SPINGALETT_ERR_ALLOC, "GEMM scratch allocation failed"); return; }
    }
    const char *name;
    gemm_select(&name)(scratch, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, parallel);
    spingalett_gemm_scratch_free(own);
}

#endif
