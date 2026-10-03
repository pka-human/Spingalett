/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Native single-precision matrix multiplication, C = alpha * op(A) * op(B) + beta * C (row-major),
 * structured like BLIS/GotoBLAS: op(B) is packed in KC x NC blocks of NR-wide column panels,
 * op(A) in MC x KC blocks of MR-high row panels, and a register-blocked MR x NR micro-kernel
 * multiplies one panel pair. Packing absorbs the transposes, so all four op() combinations run
 * the same kernel. The micro-kernel is chosen at compile time: AVX-512 (12 x 32), AVX/FMA
 * (6 x 16) or portable C (4 x 8) that compilers vectorize for SSE or NEON.
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

struct SpingalettGemmScratch {
    float *b_pack;      /* KC x NC */
    float *a_pack;      /* threads x (MC x KC) */
    int threads;
};

SpingalettGemmScratch *spingalett_gemm_scratch_create(int threads) {
    if (threads < 1) threads = 1;
    SpingalettGemmScratch *s = (SpingalettGemmScratch *)calloc(1, sizeof(SpingalettGemmScratch));
    if (!s) return NULL;
    s->threads = threads;
    s->b_pack = (float *)spingalett_aligned_alloc((size_t)KC * NC * sizeof(float));
    s->a_pack = (float *)spingalett_aligned_alloc((size_t)threads * MC * KC * sizeof(float));
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

/* Pack rows [i0, i0 + mc) x columns [k0, k0 + kc) of op(A) into MR-row panels laid out
   panel-major, then k, then row; rows past mc are zero. */
static void pack_a(float *restrict dst, const float *restrict A, size_t lda, bool trans,
                   uint32_t i0, uint32_t mc, uint32_t k0, uint32_t kc) {
    for (uint32_t p = 0; p < mc; p += MR) {
        uint32_t rows = (mc - p < MR) ? mc - p : MR;
        if (!trans) {
            const float *src = A + (size_t)(i0 + p) * lda + k0;
            for (uint32_t k = 0; k < kc; k++) {
                for (uint32_t r = 0; r < rows; r++) dst[r] = src[(size_t)r * lda + k];
                for (uint32_t r = rows; r < MR; r++) dst[r] = 0.0f;
                dst += MR;
            }
        } else {
            for (uint32_t k = 0; k < kc; k++) {
                const float *src = A + (size_t)(k0 + k) * lda + i0 + p;
                for (uint32_t r = 0; r < rows; r++) dst[r] = src[r];
                for (uint32_t r = rows; r < MR; r++) dst[r] = 0.0f;
                dst += MR;
            }
        }
    }
}

/* Pack NR-column panel q (columns [j0 + q, ...)) of op(B), rows [k0, k0 + kc). */
static void pack_b_panel(float *restrict dst, const float *restrict B, size_t ldb, bool trans,
                         uint32_t k0, uint32_t kc, uint32_t j, uint32_t cols) {
    if (!trans) {
        const float *src = B + (size_t)k0 * ldb + j;
        for (uint32_t k = 0; k < kc; k++, src += ldb, dst += NR) {
            if (cols == NR) {
                memcpy(dst, src, NR * sizeof(float));
            } else {
                for (uint32_t c = 0; c < cols; c++) dst[c] = src[c];
                for (uint32_t c = cols; c < NR; c++) dst[c] = 0.0f;
            }
        }
    } else {
        for (uint32_t k = 0; k < kc; k++, dst += NR) {
            const float *src = B + (size_t)j * ldb + k0 + k;
            for (uint32_t c = 0; c < cols; c++) dst[c] = src[(size_t)c * ldb];
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

#if defined(__AVX512F__)

#define K512_ROWS(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11)

static void micro_kernel(uint32_t kc, const float *restrict a, const float *restrict b,
                         float *restrict c, size_t ldc, float alpha, float beta,
                         uint32_t rows, uint32_t cols) {
#define DECL(r) __m512 c##r##0 = _mm512_setzero_ps(), c##r##1 = _mm512_setzero_ps();
    K512_ROWS(DECL)
#undef DECL
    for (uint32_t k = 0; k < kc; k++, a += MR, b += NR) {
        __m512 b0 = _mm512_load_ps(b), b1 = _mm512_load_ps(b + 16);
#define FMA(r) { __m512 ar = _mm512_set1_ps(a[r]); \
                 c##r##0 = _mm512_fmadd_ps(ar, b0, c##r##0); c##r##1 = _mm512_fmadd_ps(ar, b1, c##r##1); }
        K512_ROWS(FMA)
#undef FMA
    }
    if (rows == MR && cols == NR) {
        __m512 va = _mm512_set1_ps(alpha), vb = _mm512_set1_ps(beta);
#define STORE(r) { float *cr = c + (size_t)(r) * ldc; \
                   __m512 x0 = _mm512_mul_ps(va, c##r##0), x1 = _mm512_mul_ps(va, c##r##1); \
                   if (beta != 0.0f) { x0 = _mm512_fmadd_ps(vb, _mm512_loadu_ps(cr), x0); \
                                       x1 = _mm512_fmadd_ps(vb, _mm512_loadu_ps(cr + 16), x1); } \
                   _mm512_storeu_ps(cr, x0); _mm512_storeu_ps(cr + 16, x1); }
        K512_ROWS(STORE)
#undef STORE
    } else {
        _Alignas(64) float acc[MR * NR];
#define SPILL(r) _mm512_store_ps(acc + (r) * NR, c##r##0); _mm512_store_ps(acc + (r) * NR + 16, c##r##1);
        K512_ROWS(SPILL)
#undef SPILL
        store_tile(c, ldc, acc, alpha, beta, rows, cols);
    }
}

#elif defined(__AVX__)

#if defined(__FMA__)
#define GEMM_FMA(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#else
#define GEMM_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif
#define K256_ROWS(X) X(0) X(1) X(2) X(3) X(4) X(5)

static void micro_kernel(uint32_t kc, const float *restrict a, const float *restrict b,
                         float *restrict c, size_t ldc, float alpha, float beta,
                         uint32_t rows, uint32_t cols) {
#define DECL(r) __m256 c##r##0 = _mm256_setzero_ps(), c##r##1 = _mm256_setzero_ps();
    K256_ROWS(DECL)
#undef DECL
    for (uint32_t k = 0; k < kc; k++, a += MR, b += NR) {
        __m256 b0 = _mm256_load_ps(b), b1 = _mm256_load_ps(b + 8);
#define FMA(r) { __m256 ar = _mm256_broadcast_ss(a + (r)); \
                 c##r##0 = GEMM_FMA(ar, b0, c##r##0); c##r##1 = GEMM_FMA(ar, b1, c##r##1); }
        K256_ROWS(FMA)
#undef FMA
    }
    if (rows == MR && cols == NR) {
        __m256 va = _mm256_set1_ps(alpha), vb = _mm256_set1_ps(beta);
#define STORE(r) { float *cr = c + (size_t)(r) * ldc; \
                   __m256 x0 = _mm256_mul_ps(va, c##r##0), x1 = _mm256_mul_ps(va, c##r##1); \
                   if (beta != 0.0f) { x0 = GEMM_FMA(vb, _mm256_loadu_ps(cr), x0); \
                                       x1 = GEMM_FMA(vb, _mm256_loadu_ps(cr + 8), x1); } \
                   _mm256_storeu_ps(cr, x0); _mm256_storeu_ps(cr + 8, x1); }
        K256_ROWS(STORE)
#undef STORE
    } else {
        _Alignas(32) float acc[MR * NR];
#define SPILL(r) _mm256_store_ps(acc + (r) * NR, c##r##0); _mm256_store_ps(acc + (r) * NR + 8, c##r##1);
        K256_ROWS(SPILL)
#undef SPILL
        store_tile(c, ldc, acc, alpha, beta, rows, cols);
    }
}

#else

static void micro_kernel(uint32_t kc, const float *restrict a, const float *restrict b,
                         float *restrict c, size_t ldc, float alpha, float beta,
                         uint32_t rows, uint32_t cols) {
    float acc[MR * NR] = {0};
    for (uint32_t k = 0; k < kc; k++, a += MR, b += NR)
        for (uint32_t r = 0; r < MR; r++)
            for (uint32_t j = 0; j < NR; j++)
                acc[r * NR + j] += a[r] * b[j];
    store_tile(c, ldc, acc, alpha, beta, rows, cols);
}

#endif

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
    int threads = parallel ? scratch->threads : 1;

    for (uint32_t j0 = 0; j0 < N; j0 += NC) {
        uint32_t nc = (N - j0 < NC) ? N - j0 : NC;
        int64_t panels = (int64_t)((nc + NR - 1) / NR);

        for (uint32_t k0 = 0; k0 < K; k0 += KC) {
            uint32_t kc = (K - k0 < KC) ? K - k0 : KC;
            float beta_k = (k0 == 0) ? beta : 1.0f;     /* later K blocks accumulate */

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1 && panels > 1)
#endif
            for (int64_t q = 0; q < panels; q++) {
                uint32_t j = (uint32_t)q * NR;
                uint32_t cols = (nc - j < NR) ? nc - j : NR;
                pack_b_panel(scratch->b_pack + (size_t)j * kc, B, ldb, trans_b, k0, kc, j0 + j, cols);
            }

            /* Work items are (row block, column range) tiles; columns are split only as far as
               needed to give every thread work when there are few row blocks (small batches). */
            int64_t mblocks = (int64_t)((M + MC - 1) / MC);
            int64_t nsplit = 1;
            if (threads > 1 && mblocks < 2 * threads) {
                nsplit = (2 * threads + mblocks - 1) / mblocks;
                if (nsplit > panels) nsplit = panels;
            }
            int64_t panels_per_split = (panels + nsplit - 1) / nsplit;
            int64_t tiles = mblocks * nsplit;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic) num_threads(threads) if(threads > 1 && tiles > 1)
#endif
            for (int64_t t = 0; t < tiles; t++) {
                int tid = 0;
#if defined(_OPENMP)
                tid = omp_get_thread_num();
#endif
                float *a_pack = scratch->a_pack + (size_t)tid * MC * KC;
                uint32_t i0 = (uint32_t)(t / nsplit) * MC;
                uint32_t mc = (M - i0 < MC) ? M - i0 : MC;
                int64_t q_begin = (t % nsplit) * panels_per_split;
                int64_t q_end = q_begin + panels_per_split < panels ? q_begin + panels_per_split : panels;
                if (q_begin >= q_end) continue;

                pack_a(a_pack, A, lda, trans_a, i0, mc, k0, kc);
                for (int64_t q = q_begin; q < q_end; q++) {
                    uint32_t j = (uint32_t)q * NR;
                    uint32_t cols = (nc - j < NR) ? nc - j : NR;
                    const float *b_panel = scratch->b_pack + (size_t)j * kc;
                    for (uint32_t p = 0; p < mc; p += MR) {
                        uint32_t rows = (mc - p < MR) ? mc - p : MR;
                        micro_kernel(kc, a_pack + (size_t)p * kc, b_panel,
                                     C + (size_t)(i0 + p) * ldc + j0 + j, ldc, alpha, beta_k, rows, cols);
                    }
                }
            }
        }
    }

    spingalett_gemm_scratch_free(own);
}
