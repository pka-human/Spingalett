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
 * Threads split C by rows and columns: every element is summed over k in the same order whatever
 * the thread count, so results do not depend on it. Large M splits row blocks (and columns when
 * there are few blocks), with op(B) packed once per K block for all threads; small M (a mini-batch)
 * packs op(A) once for all threads and gives each thread its own column panels, which it packs and
 * multiplies itself. When all of op(B) fits the pack buffer (a convolution's filters), it is packed
 * once and each tile of C runs through all K blocks while it is in cache. A product with at most 16
 * columns from row-major operands (an output layer of a few units) is a set of dot products. A
 * small C with a long k range (a convolution's weight gradient: filters x
 * window, summed over every pixel of a batch) has too few tiles for the threads; its k range is cut
 * into slots fixed by the shape alone, each slot's partial product is one thread's work, and the
 * partial products are added in slot order, so the result still does not depend on the thread
 * count.
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

/* Products split along k: C at most SPLIT_M_MAX x SPLIT_N_MAX, at least SPLIT_K_MIN of k per slot,
   at most SPLIT_SLOTS slots and SPLIT_PARTIAL_FLOATS floats of partial products. */
#define SPLIT_M_MAX MC_MAX
#define SPLIT_N_MAX 512u
#define SPLIT_K_MIN 1024u
#define SPLIT_SLOTS 32u
#define SPLIT_PARTIAL_FLOATS (1u << 20)
#define NR_MAX 32u      /* the widest NR of any kernel set */

struct SpingalettGemmScratch {
    float *b_pack;      /* KC x NC */
    float *a_pack;      /* threads x (MC_MAX x KC); with a shared op(A), one M x KC block */
    int threads;
    float *split;       /* split products: the partial products, then a column-panel pack per thread */
    size_t split_floats;
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

int spingalett_gemm_scratch_threads(const SpingalettGemmScratch *s) {
    return s ? s->threads : 1;
}

size_t spingalett_gemm_scratch_bytes(int threads) {
    if (threads < 1) threads = 1;
    return ((size_t)KC * NC + (size_t)threads * MC_MAX * KC) * sizeof(float);
}

void spingalett_gemm_scratch_free(SpingalettGemmScratch *s) {
    if (!s) return;
    spingalett_aligned_free(s->b_pack);
    spingalett_aligned_free(s->a_pack);
    spingalett_aligned_free(s->split);
    free(s);
}
#endif

/* The number of k slots of an M x N x K product (1: not split); it depends on the shape only. */
static inline uint32_t split_slots(uint32_t M, uint32_t N, uint32_t K) {
    if (M > SPLIT_M_MAX || N > SPLIT_N_MAX || K < 2 * SPLIT_K_MIN) return 1;
    uint32_t slots = K / SPLIT_K_MIN, fit = SPLIT_PARTIAL_FLOATS / (M * N);
    if (slots > SPLIT_SLOTS) slots = SPLIT_SLOTS;
    if (slots > fit) slots = fit;
    return slots < 2 ? 1 : slots;
}

/* Partial products (16-float aligned), then a KC x N column-panel pack per thread. */
static inline size_t split_partial_floats(uint32_t M, uint32_t N, uint32_t slots) {
    return ((size_t)slots * M * N + 15u) & ~(size_t)15u;
}

static bool split_reserve(SpingalettGemmScratch *s, uint32_t M, uint32_t N, uint32_t slots, int threads) {
    size_t n_pad = (N + NR_MAX - 1) / NR_MAX * NR_MAX;
    size_t need = split_partial_floats(M, N, slots) + (size_t)threads * KC * n_pad;
    if (s->split_floats >= need) return true;
    spingalett_aligned_free(s->split);
    s->split = (float *)spingalett_aligned_alloc(need * sizeof(float));
    s->split_floats = s->split ? need : 0;
    return s->split != NULL;
}

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

/* Packs rows [0, rows) of an MR-row panel starting at row i of op(A), columns [k0, k0 + kc). A
   source (src) fills the operand's elements instead of A. */
static void pack_a_panel(float *restrict dst, const float *restrict A, size_t lda, bool trans,
                         const SpingalettGemmSource *src_op, uint32_t i, uint32_t rows, uint32_t k0, uint32_t kc) {
    if (src_op) {
        if (trans) {            /* A[k][i] is op(A)[i][k]: the source writes the panel itself */
            src_op->fill(src_op->ctx, k0, kc, i, rows, dst, MR);
            if (rows < MR)
                for (uint32_t k = 0; k < kc; k++)
                    for (uint32_t r = rows; r < MR; r++) dst[(size_t)k * MR + r] = 0.0f;
        } else {
            float tmp[MR * KC];
            src_op->fill(src_op->ctx, i, rows, k0, kc, tmp, kc);
            pack_transposed(dst, tmp, kc, 0, kc, rows, MR);
        }
        return;
    }
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

/* Packs NR-column panel (columns [j, j + cols)) of op(B), rows [k0, k0 + kc), from B or a source. */
static void pack_b_panel(float *restrict dst, const float *restrict B, size_t ldb, bool trans,
                         const SpingalettGemmSource *src_op, uint32_t k0, uint32_t kc, uint32_t j, uint32_t cols) {
    if (src_op) {
        if (!trans) {           /* B[k][j]: the source writes the panel itself */
            src_op->fill(src_op->ctx, k0, kc, j, cols, dst, NR);
            if (cols < NR)
                for (uint32_t k = 0; k < kc; k++)
                    for (uint32_t c = cols; c < NR; c++) dst[(size_t)k * NR + c] = 0.0f;
        } else {
            float tmp[NR * KC];
            src_op->fill(src_op->ctx, j, cols, k0, kc, tmp, kc);
            pack_transposed(dst, tmp, kc, 0, kc, cols, NR);
        }
        return;
    }
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

/* C = alpha * op(A) * op(B) + beta * C with k cut into `slots` slots (scratch->split reserved):
   slot s covers k [s * per, (s + 1) * per) and is multiplied by one thread into its own partial
   product; the partial products are then summed in slot order. */
static void gemm_split(SpingalettGemmScratch *scratch, int threads, uint32_t slots, bool trans_a, bool trans_b,
                       uint32_t M, uint32_t N, uint32_t K, float alpha,
                       const float *A, size_t lda, const SpingalettGemmSource *a_src,
                       const float *B, size_t ldb, const SpingalettGemmSource *b_src,
                       float beta, float *C, size_t ldc, const SpingalettGemmHooks *hooks) {
    const uint32_t per = (K + slots - 1) / slots, m_panels = (M + MR - 1) / MR, n_panels = (N + NR - 1) / NR;
    const size_t mn = (size_t)M * N, n_pad = (N + NR_MAX - 1) / NR_MAX * NR_MAX;
    float *partial = scratch->split, *b_area = partial + split_partial_floats(M, N, slots);
    (void)threads;

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) if(threads > 1)
#endif
    for (int64_t s = 0; s < (int64_t)slots; s++) {
        int tid = 0;
#if defined(_OPENMP)
        tid = omp_get_thread_num();
#endif
        float *a_pack = scratch->a_pack + (size_t)tid * MC_MAX * KC, *b_pack = b_area + (size_t)tid * KC * n_pad;
        float *P = partial + (size_t)s * mn;
        uint32_t k_begin = (uint32_t)s * per, k_end = K - k_begin < per ? K : k_begin + per;
        uint32_t len = k_end - k_begin, blocks = (len + KC - 1) / KC, step = (len + blocks - 1) / blocks;
        for (uint32_t k0 = k_begin; k0 < k_end; k0 += step) {
            uint32_t kc = k_end - k0 < step ? k_end - k0 : step;
            for (uint32_t p = 0; p < m_panels; p++) {
                uint32_t i = p * MR;
                pack_a_panel(a_pack + (size_t)i * kc, A, lda, trans_a, a_src, i, (M - i < MR) ? M - i : MR, k0, kc);
            }
            for (uint32_t q = 0; q < n_panels; q++) {
                uint32_t j = q * NR;
                pack_b_panel(b_pack + (size_t)j * kc, B, ldb, trans_b, b_src, k0, kc, j, (N - j < NR) ? N - j : NR);
            }
            multiply_block(a_pack, b_pack, M, N, 0, n_panels, kc, P, N, 1.0f, k0 == k_begin ? 0.0f : 1.0f);
        }
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1 && mn * slots >= (1u << 16))
#endif
    for (int64_t i = 0; i < (int64_t)M; i++) {
        float sum[SPLIT_N_MAX], *c = C + (size_t)i * ldc;
        memcpy(sum, partial + (size_t)i * N, N * sizeof(float));
        for (uint32_t s = 1; s < slots; s++) {
            const float *P = partial + (size_t)s * mn + (size_t)i * N;
            for (uint32_t j = 0; j < N; j++) sum[j] += P[j];
        }
        if (beta == 0.0f) for (uint32_t j = 0; j < N; j++) c[j] = alpha * sum[j];
        else              for (uint32_t j = 0; j < N; j++) c[j] = alpha * sum[j] + beta * c[j];
        if (hooks && hooks->epilogue) hooks->epilogue(hooks->epilogue_ctx, (uint32_t)i, 1, 0, N, c, ldc);
    }
}

/* Products with at most SMALL_N columns from two row-major operands (op(A) = A, op(B) = B^T: a
   dense layer's outputs with few units) leave most of an NR-wide panel idle; each element is
   instead a dot product of a row of A with a row of B, four rows of B per pass over A's row. The
   lanes of each dot product are summed in a fixed order, so results do not depend on the thread
   count. */
#define SMALL_N 16u
#define SMALL_N_MIN_K 64u       /* shorter dot products cost more in their reductions */

/* Rows of A per dot-product block: as many as the registers hold, with four rows of B. */
#if defined(__AVX512F__)
#  define DOT_RA 4
#elif defined(__AVX__)
#  define DOT_RA 2
#else
#  define DOT_RA 1
#endif

/* d[i][r] = row a_i . row b_r (n floats each) for i < DOT_RA, r < 4. */
static inline void dot_block(const float *const a[DOT_RA], const float *const b[4], uint32_t n, float d[DOT_RA][4]) {
    uint32_t k = 0;
#if defined(__AVX512F__)
    __m512 acc[DOT_RA][4];
    for (int i = 0; i < DOT_RA; i++) for (int r = 0; r < 4; r++) acc[i][r] = _mm512_setzero_ps();
    for (; k + 16 <= n; k += 16) {
        __m512 y0 = _mm512_loadu_ps(b[0] + k), y1 = _mm512_loadu_ps(b[1] + k);
        __m512 y2 = _mm512_loadu_ps(b[2] + k), y3 = _mm512_loadu_ps(b[3] + k);
        for (int i = 0; i < DOT_RA; i++) {
            __m512 x = _mm512_loadu_ps(a[i] + k);
            acc[i][0] = _mm512_fmadd_ps(x, y0, acc[i][0]);
            acc[i][1] = _mm512_fmadd_ps(x, y1, acc[i][1]);
            acc[i][2] = _mm512_fmadd_ps(x, y2, acc[i][2]);
            acc[i][3] = _mm512_fmadd_ps(x, y3, acc[i][3]);
        }
    }
    for (int i = 0; i < DOT_RA; i++) for (int r = 0; r < 4; r++) d[i][r] = _mm512_reduce_add_ps(acc[i][r]);
#elif defined(__AVX__)
    __m256 acc[DOT_RA][4];
    for (int i = 0; i < DOT_RA; i++) for (int r = 0; r < 4; r++) acc[i][r] = _mm256_setzero_ps();
    for (; k + 8 <= n; k += 8) {
        __m256 y0 = _mm256_loadu_ps(b[0] + k), y1 = _mm256_loadu_ps(b[1] + k);
        __m256 y2 = _mm256_loadu_ps(b[2] + k), y3 = _mm256_loadu_ps(b[3] + k);
        for (int i = 0; i < DOT_RA; i++) {
            __m256 x = _mm256_loadu_ps(a[i] + k);
#if defined(__FMA__)
            acc[i][0] = _mm256_fmadd_ps(x, y0, acc[i][0]);
            acc[i][1] = _mm256_fmadd_ps(x, y1, acc[i][1]);
            acc[i][2] = _mm256_fmadd_ps(x, y2, acc[i][2]);
            acc[i][3] = _mm256_fmadd_ps(x, y3, acc[i][3]);
#else
            acc[i][0] = _mm256_add_ps(acc[i][0], _mm256_mul_ps(x, y0));
            acc[i][1] = _mm256_add_ps(acc[i][1], _mm256_mul_ps(x, y1));
            acc[i][2] = _mm256_add_ps(acc[i][2], _mm256_mul_ps(x, y2));
            acc[i][3] = _mm256_add_ps(acc[i][3], _mm256_mul_ps(x, y3));
#endif
        }
    }
    for (int i = 0; i < DOT_RA; i++)
        for (int r = 0; r < 4; r++) {
            float t[8];
            _mm256_storeu_ps(t, acc[i][r]);
            d[i][r] = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
        }
#else
    for (int i = 0; i < DOT_RA; i++) for (int r = 0; r < 4; r++) d[i][r] = 0.0f;
#endif
    for (; k < n; k++)
        for (int i = 0; i < DOT_RA; i++) {
            float x = a[i][k];
            d[i][0] += x * b[0][k]; d[i][1] += x * b[1][k]; d[i][2] += x * b[2][k]; d[i][3] += x * b[3][k];
        }
}

#if DOT_RA > 1
/* d[r] = row a . row b_r for r < 4, computed exactly as dot_block computes one of its rows, so
   that a row's results do not depend on the rows it is computed with. */
static inline void dot_block1(const float *a, const float *const b[4], uint32_t n, float d[4]) {
    uint32_t k = 0;
#if defined(__AVX512F__)
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps(), acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    for (; k + 16 <= n; k += 16) {
        __m512 x = _mm512_loadu_ps(a + k);
        acc0 = _mm512_fmadd_ps(x, _mm512_loadu_ps(b[0] + k), acc0);
        acc1 = _mm512_fmadd_ps(x, _mm512_loadu_ps(b[1] + k), acc1);
        acc2 = _mm512_fmadd_ps(x, _mm512_loadu_ps(b[2] + k), acc2);
        acc3 = _mm512_fmadd_ps(x, _mm512_loadu_ps(b[3] + k), acc3);
    }
    d[0] = _mm512_reduce_add_ps(acc0);
    d[1] = _mm512_reduce_add_ps(acc1);
    d[2] = _mm512_reduce_add_ps(acc2);
    d[3] = _mm512_reduce_add_ps(acc3);
#else   /* AVX */
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (; k + 8 <= n; k += 8) {
        __m256 x = _mm256_loadu_ps(a + k);
        for (int r = 0; r < 4; r++) {
#if defined(__FMA__)
            acc[r] = _mm256_fmadd_ps(x, _mm256_loadu_ps(b[r] + k), acc[r]);
#else
            acc[r] = _mm256_add_ps(acc[r], _mm256_mul_ps(x, _mm256_loadu_ps(b[r] + k)));
#endif
        }
    }
    for (int r = 0; r < 4; r++) {
        float t[8];
        _mm256_storeu_ps(t, acc[r]);
        d[r] = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
    }
#endif
    for (; k < n; k++) {
        float x = a[k];
        d[0] += x * b[0][k]; d[1] += x * b[1][k]; d[2] += x * b[2][k]; d[3] += x * b[3][k];
    }
}
#else
static inline void dot_block1(const float *a, const float *const b[4], uint32_t n, float d[4]) {
    const float *rows[1] = {a};
    float t[1][4];
    dot_block(rows, b, n, t);
    memcpy(d, t[0], sizeof t[0]);
}
#endif

/* Products with at most SMALL_M rows from row-major operands (a dense layer run on one sample or
   a few) take the dot products as well, rather than packing all of op(B) for so few rows: each
   row of B is read once per DOT_RA rows of A, and short rows cost less in reductions than packing
   would. Threads split the columns. */
#define SMALL_M (2u * DOT_RA)

static void gemm_small_m(int threads, uint32_t M, uint32_t N, uint32_t K, float alpha, const float *A, size_t lda,
                         const float *B, size_t ldb, float beta, float *C, size_t ldc, const SpingalettGemmHooks *hooks) {
    const uint32_t span = 64;                           /* columns per work item and epilogue call */
    const int64_t items = (int64_t)((N + span - 1) / span);
    (void)threads;
    SPINGALETT_PARALLEL_FOR_THREADS(threads, threads > 1 && items > 1,
        for (int64_t it = 0; it < items; it++) {
            uint32_t j0 = (uint32_t)it * span, j1 = N - j0 < span ? N : j0 + span;
            for (uint32_t i = 0; i < M; i += DOT_RA) {
                uint32_t ra = M - i < DOT_RA ? M - i : DOT_RA;
                const float *a[DOT_RA];
                for (uint32_t t = 0; t < DOT_RA; t++) a[t] = A + (size_t)(i + (t < ra ? t : 0)) * lda;
                for (uint32_t j = j0; j < j1; j += 4) {
                    uint32_t rows = j1 - j < 4 ? j1 - j : 4;
                    const float *b[4];
                    for (uint32_t r = 0; r < 4; r++) b[r] = B + (size_t)(j + (r < rows ? r : 0)) * ldb;
                    float d[DOT_RA][4];
                    if (ra < DOT_RA)
                        for (uint32_t t = 0; t < ra; t++) dot_block1(a[t], b, K, d[t]);
                    else
                        dot_block(a, b, K, d);
                    for (uint32_t t = 0; t < ra; t++) {
                        float *c = C + (size_t)(i + t) * ldc + j;
                        for (uint32_t r = 0; r < rows; r++)
                            c[r] = alpha * d[t][r] + (beta == 0.0f ? 0.0f : beta * c[r]);
                    }
                }
            }
            if (hooks && hooks->epilogue)
                hooks->epilogue(hooks->epilogue_ctx, 0, M, j0, j1 - j0, C + j0, ldc);
        }
    );
}

static void gemm_small_n(int threads, uint32_t M, uint32_t N, uint32_t K, float alpha, const float *A, size_t lda,
                         const float *B, size_t ldb, float beta, float *C, size_t ldc, const SpingalettGemmHooks *hooks) {
    const uint32_t block = 16;                          /* rows per work item and epilogue call */
    const int64_t blocks = (int64_t)((M + block - 1) / block);
    (void)threads;
    SPINGALETT_PARALLEL_FOR_THREADS(threads, threads > 1 && blocks > 1,
        for (int64_t bi = 0; bi < blocks; bi++) {
            uint32_t i0 = (uint32_t)bi * block, i1 = M - i0 < block ? M : i0 + block;
            for (uint32_t i = i0; i < i1; i += DOT_RA) {
                uint32_t ra = i1 - i < DOT_RA ? i1 - i : DOT_RA;
                const float *a[DOT_RA];
                for (uint32_t t = 0; t < DOT_RA; t++) a[t] = A + (size_t)(i + (t < ra ? t : 0)) * lda;  /* rows past the block repeat */
                for (uint32_t j0 = 0; j0 < N; j0 += 4) {
                    uint32_t rows = N - j0 < 4 ? N - j0 : 4;
                    const float *b[4];
                    for (uint32_t r = 0; r < 4; r++) b[r] = B + (size_t)(j0 + (r < rows ? r : 0)) * ldb;
                    float d[DOT_RA][4];
                    dot_block(a, b, K, d);
                    for (uint32_t t = 0; t < ra; t++) {
                        float *c = C + (size_t)(i + t) * ldc + j0;
                        for (uint32_t r = 0; r < rows; r++)
                            c[r] = alpha * d[t][r] + (beta == 0.0f ? 0.0f : beta * c[r]);
                    }
                }
            }
            if (hooks && hooks->epilogue)
                hooks->epilogue(hooks->epilogue_ctx, i0, i1 - i0, 0, N, C + (size_t)i0 * ldc, ldc);
        }
    );
}

/* The multiplication with this translation unit's kernels (scratch is not NULL, M, N, K > 0). */
#if defined(SPINGALETT_GEMM_VARIANT)
void SPINGALETT_GEMM_VARIANT(SpingalettGemmScratch *scratch, bool trans_a, bool trans_b,
                             uint32_t M, uint32_t N, uint32_t K, float alpha,
                             const float *A, size_t lda, const float *B, size_t ldb,
                             float beta, float *C, size_t ldc, bool parallel, const SpingalettGemmHooks *hooks) {
#elif defined(SPINGALETT_GEMM_DISPATCH)
void spingalett_gemm_baseline(SpingalettGemmScratch *scratch, bool trans_a, bool trans_b,
                              uint32_t M, uint32_t N, uint32_t K, float alpha,
                              const float *A, size_t lda, const float *B, size_t ldb,
                              float beta, float *C, size_t ldc, bool parallel, const SpingalettGemmHooks *hooks) {
#else
static void spingalett_gemm_baseline(SpingalettGemmScratch *scratch, bool trans_a, bool trans_b,
                                     uint32_t M, uint32_t N, uint32_t K, float alpha,
                                     const float *A, size_t lda, const float *B, size_t ldb,
                                     float beta, float *C, size_t ldc, bool parallel,
                                     const SpingalettGemmHooks *hooks) {
#endif
    const SpingalettGemmSource *a_src = hooks ? hooks->a : NULL, *b_src = hooks ? hooks->b : NULL;
    void (*epilogue)(const void *, uint32_t, uint32_t, uint32_t, uint32_t, float *, size_t) =
        hooks ? hooks->epilogue : NULL;
    const uint64_t work = (uint64_t)M * N * K;
    if (N <= SMALL_N && K >= SMALL_N_MIN_K && !trans_a && trans_b && !a_src && !b_src) {
        /* one parallel loop without barriers: worth threads at a quarter of the usual work */
        gemm_small_n(parallel && work >= SPINGALETT_GEMM_PARALLEL_WORK / 4u ? scratch->threads : 1, M, N, K, alpha,
                     A, lda, B, ldb, beta, C, ldc, hooks);
        return;
    }
    if (M <= SMALL_M && !trans_a && trans_b && !a_src && !b_src) {
        gemm_small_m(parallel && work >= SPINGALETT_GEMM_PARALLEL_WORK / 4u ? scratch->threads : 1, M, N, K, alpha,
                     A, lda, B, ldb, beta, C, ldc, hooks);
        return;
    }
    int threads = parallel && work >= SPINGALETT_GEMM_PARALLEL_WORK ? scratch->threads : 1;

    uint32_t slots = split_slots(M, N, K);
    if (slots > 1 && split_reserve(scratch, M, N, slots, threads)) {
        gemm_split(scratch, threads, slots, trans_a, trans_b, M, N, K, alpha, A, lda, a_src, B, ldb, b_src,
                   beta, C, ldc, hooks);
        return;
    }

    /* K blocks of equal size (at most KC): 288 runs as 2 x 144, not 256 + 32 */
    uint32_t k_blocks = (K + KC - 1) / KC, kc_step = (K + k_blocks - 1) / k_blocks;
    uint32_t m_panels = (M + MR - 1) / MR;
    int64_t mblocks = (int64_t)((M + MC - 1) / MC);
    bool shared_a = threads > 1 && mblocks <= SHARED_A_BLOCKS &&
                    (size_t)m_panels * MR <= (size_t)scratch->threads * MC_MAX;
    /* All of op(B) fits the pack buffer (a convolution's filters): it is packed once, and each
       tile then runs every K block while its part of C stays in cache, the epilogue following */
    const int64_t n_panels = (int64_t)((N + NR - 1) / NR);
    const size_t b_block = (size_t)kc_step * (size_t)n_panels * NR;
    bool resident_b = !shared_a && N <= NC && (size_t)k_blocks * b_block <= (size_t)KC * NC;

#if defined(_OPENMP)
#pragma omp parallel num_threads(threads) if(threads > 1)
#endif
    {
        int tid = 0, team = 1;
#if defined(_OPENMP)
        tid = omp_get_thread_num();
        team = omp_get_num_threads();
#endif
        if (resident_b) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (int64_t w = 0; w < (int64_t)k_blocks * n_panels; w++) {
                uint32_t kb = (uint32_t)(w / n_panels), j = (uint32_t)(w % n_panels) * NR;
                uint32_t k0 = kb * kc_step, kc = (K - k0 < kc_step) ? K - k0 : kc_step;
                pack_b_panel(scratch->b_pack + kb * b_block + (size_t)j * kc, B, ldb, trans_b, b_src, k0, kc, j,
                             (N - j < NR) ? N - j : NR);
            }
            int64_t nsplit = 1;
            if (team > 1 && mblocks < 2 * team) {
                nsplit = (2 * team + mblocks - 1) / mblocks;
                if (nsplit > n_panels) nsplit = n_panels;
            }
            int64_t panels_per_split = (n_panels + nsplit - 1) / nsplit;
            float *a_pack = scratch->a_pack + (size_t)tid * MC_MAX * KC;
#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
            for (int64_t t = 0; t < mblocks * nsplit; t++) {
                uint32_t i0 = (uint32_t)(t / nsplit) * MC;
                uint32_t mc = (M - i0 < MC) ? M - i0 : MC;
                int64_t q_begin = (t % nsplit) * panels_per_split;
                int64_t q_end = q_begin + panels_per_split < n_panels ? q_begin + panels_per_split : n_panels;
                if (q_begin >= q_end) continue;
                for (uint32_t kb = 0; kb < k_blocks; kb++) {
                    uint32_t k0 = kb * kc_step, kc = (K - k0 < kc_step) ? K - k0 : kc_step;
                    for (uint32_t p = 0; p < mc; p += MR)
                        pack_a_panel(a_pack + (size_t)p * kc, A, lda, trans_a, a_src, i0 + p,
                                     (mc - p < MR) ? mc - p : MR, k0, kc);
                    multiply_block(a_pack, scratch->b_pack + kb * b_block, mc, N, q_begin, q_end, kc,
                                   C + (size_t)i0 * ldc, ldc, alpha, kb == 0 ? beta : 1.0f);
                }
                if (epilogue) {
                    uint32_t c0 = (uint32_t)q_begin * NR, c1 = (uint32_t)q_end * NR < N ? (uint32_t)q_end * NR : N;
                    epilogue(hooks->epilogue_ctx, i0, mc, c0, c1 - c0, C + (size_t)i0 * ldc + c0, ldc);
                }
            }
        }
        for (uint32_t j0 = 0; !resident_b && j0 < N; j0 += NC) {
            uint32_t nc = (N - j0 < NC) ? N - j0 : NC;
            int64_t panels = (int64_t)((nc + NR - 1) / NR);

            for (uint32_t k0 = 0; k0 < K; k0 += kc_step) {
                uint32_t kc = (K - k0 < kc_step) ? K - k0 : kc_step;
                float beta_k = (k0 == 0) ? beta : 1.0f;     /* later K blocks accumulate */

                if (shared_a) {
                    /* All row panels of op(A) are packed once for the team; each thread packs
                       and multiplies its own range of column panels. */
#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
                    for (int64_t p = 0; p < (int64_t)m_panels; p++) {
                        uint32_t i = (uint32_t)p * MR;
                        pack_a_panel(scratch->a_pack + (size_t)i * kc, A, lda, trans_a, a_src, i,
                                     (M - i < MR) ? M - i : MR, k0, kc);
                    }
                    int64_t per = (panels + team - 1) / team;
                    int64_t q_begin = (int64_t)tid * per;
                    int64_t q_end = q_begin + per < panels ? q_begin + per : panels;
                    for (int64_t q = q_begin; q < q_end; q++) {
                        uint32_t j = (uint32_t)q * NR;
                        pack_b_panel(scratch->b_pack + (size_t)j * kc, B, ldb, trans_b, b_src, k0, kc, j0 + j,
                                     (nc - j < NR) ? nc - j : NR);
                    }
#if defined(_OPENMP)
#pragma omp barrier
#endif
                    if (q_begin < q_end) {
                        multiply_block(scratch->a_pack, scratch->b_pack, M, nc, q_begin, q_end, kc,
                                       C + j0, ldc, alpha, beta_k);
                        if (epilogue && k0 + kc == K) {
                            uint32_t c0 = (uint32_t)q_begin * NR, c1 = (uint32_t)q_end * NR < nc ? (uint32_t)q_end * NR : nc;
                            epilogue(hooks->epilogue_ctx, 0, M, j0 + c0, c1 - c0, C + j0 + c0, ldc);
                        }
                    }
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
                    pack_b_panel(scratch->b_pack + (size_t)j * kc, B, ldb, trans_b, b_src, k0, kc, j0 + j,
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
                        pack_a_panel(a_pack + (size_t)p * kc, A, lda, trans_a, a_src, i0 + p,
                                     (mc - p < MR) ? mc - p : MR, k0, kc);
                    multiply_block(a_pack, scratch->b_pack, mc, nc, q_begin, q_end, kc,
                                   C + (size_t)i0 * ldc + j0, ldc, alpha, beta_k);
                    if (epilogue && k0 + kc == K) {
                        uint32_t c0 = (uint32_t)q_begin * NR, c1 = (uint32_t)q_end * NR < nc ? (uint32_t)q_end * NR : nc;
                        epilogue(hooks->epilogue_ctx, i0, mc, j0 + c0, c1 - c0, C + (size_t)i0 * ldc + j0 + c0, ldc);
                    }
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
                             const float *, size_t, const float *, size_t, float, float *, size_t, bool,
                             const SpingalettGemmHooks *);

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
    spingalett_gemm_hooked(scratch, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, parallel, NULL);
}

void spingalett_gemm_hooked(SpingalettGemmScratch *scratch, bool trans_a, bool trans_b,
                            uint32_t M, uint32_t N, uint32_t K, float alpha,
                            const float *A, size_t lda, const float *B, size_t ldb,
                            float beta, float *C, size_t ldc, bool parallel, const SpingalettGemmHooks *hooks) {
    if (M == 0 || N == 0) return;
    if (K == 0 || alpha == 0.0f) {      /* C = beta * C */
        for (uint32_t i = 0; i < M; i++) {
            float *cr = C + (size_t)i * ldc;
            if (beta == 0.0f) memset(cr, 0, N * sizeof(float));
            else for (uint32_t j = 0; j < N; j++) cr[j] *= beta;
        }
        if (hooks && hooks->epilogue) hooks->epilogue(hooks->epilogue_ctx, 0, M, 0, N, C, ldc);
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
    gemm_select(&name)(scratch, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, parallel, hooks);
    spingalett_gemm_scratch_free(own);
}

#endif
