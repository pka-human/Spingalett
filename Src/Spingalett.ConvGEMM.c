/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Convolutions as indirect matrix products (after Dukhan's indirect convolution algorithm): a
 * register-blocked micro-kernel multiplies MR rows by an NR-column panel of packed weights, reading
 * each row's window straight from the image through one pointer per kernel tap (a zero run where the
 * window leaves the image), so no window is ever gathered, copied or transposed.
 *
 *   forward          rows: output pixels; taps: (kh, kw); a tap's run: the group's input channels;
 *                    panels: filters
 *   data gradient    rows: input pixels; taps: the output pixels whose windows cover the input pixel
 *                    at (kh, kw); a tap's run: the group's filters; panels: input channels
 *   weight gradient  a tile of MR filters by NR window elements summed over the pixels: the
 *                    filters' gradients broadcast from dy, the window elements loaded from the image
 *                    through the pixels' tap pointers (NR elements in runs of a vector each)
 *
 * Every output element is summed over its taps and channels (the weight gradient: over the pixels)
 * in one fixed order, by one thread; the weight gradient cuts the pixels into slots fixed by the
 * shape and adds the slots' partial sums in order. So results do not depend on the thread count.
 *
 * x86-64 builds that are not tuned for the build machine compile this file again through
 * Kernels/Spingalett.ConvGEMM.AVX2.c and .AVX512.c, as Spingalett.GEMM.c.
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
#define VW 16
#elif defined(__AVX__)
#include <immintrin.h>
#define MR 6
#define NR 16
#define VW 8
#else
#define MR 4
#define NR 8
#define VW 8
#endif

/* The implementations' names: name_avx2 / name_avx512 in the variants, name_baseline in a build that
   chooses at run time, name_impl otherwise; the wrappers at the end pick one. */
#if defined(SPINGALETT_CONV_VARIANT)
#define CONV_NAME_(name, variant) name##_##variant
#define CONV_NAME__(name, variant) CONV_NAME_(name, variant)
#define CONV_NAME(name) CONV_NAME__(name, SPINGALETT_CONV_VARIANT)
#elif defined(SPINGALETT_GEMM_DISPATCH)
#define CONV_NAME(name) name##_baseline
#else
#define CONV_NAME(name) name##_impl
#endif

/* Pixels per block of the weight gradient: their tap pointers are made once for all tiles, and
   their rows of dy and windows stay in cache while every tile passes over them. */
#define WGRAD_BLOCK SPINGALETT_CONV_WGRAD_BLOCK

/* ------------------------------------------------------------------------- micro-kernels */

/* acc[r][0..NR) += sum over taps t and k < run of p[t * MR + r][k] * b[(t * run + k) * NR + 0..NR),
   then the rows x cols tile is stored at c (rows ldc apart). */
#if defined(__AVX512F__)

#define ROWS(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11)
#define I_DECL(r) __m512 c##r##0 = _mm512_setzero_ps(), c##r##1 = _mm512_setzero_ps(); const float *a##r;
#define I_PTR(r) a##r = p[r];
#define I_FMA(r) { __m512 ar = _mm512_set1_ps(a##r[k]); \
                   c##r##0 = _mm512_fmadd_ps(ar, b0, c##r##0); c##r##1 = _mm512_fmadd_ps(ar, b1, c##r##1); }
#define I_SPILL(r) _mm512_store_ps(acc + (r) * NR, c##r##0); _mm512_store_ps(acc + (r) * NR + 16, c##r##1);

static void ikernel(uint32_t taps, uint32_t run, const float *const *p, const float *b, float *c, size_t ldc,
                    uint32_t rows, uint32_t cols) {
    ROWS(I_DECL)
    for (uint32_t t = 0; t < taps; t++, p += MR) {
        ROWS(I_PTR)
        for (uint32_t k = 0; k < run; k++, b += NR) {
            __m512 b0 = _mm512_load_ps(b), b1 = _mm512_load_ps(b + 16);
            ROWS(I_FMA)
        }
    }
    _Alignas(64) float acc[MR * NR];
    ROWS(I_SPILL)
    for (uint32_t r = 0; r < rows; r++) memcpy(c + (size_t)r * ldc, acc + (size_t)r * NR, cols * sizeof(float));
}

#elif defined(__AVX__)

#if defined(__FMA__)
#define FMA256(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#else
#define FMA256(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif
#define ROWS(X) X(0) X(1) X(2) X(3) X(4) X(5)
#define I_DECL(r) __m256 c##r##0 = _mm256_setzero_ps(), c##r##1 = _mm256_setzero_ps(); const float *a##r;
#define I_PTR(r) a##r = p[r];
#define I_FMA(r) { __m256 ar = _mm256_broadcast_ss(a##r + k); \
                   c##r##0 = FMA256(ar, b0, c##r##0); c##r##1 = FMA256(ar, b1, c##r##1); }
#define I_STORE(r) if ((r) < rows) { float *cr = c + (size_t)(r) * ldc; \
                   if (cols == NR) { _mm256_storeu_ps(cr, c##r##0); _mm256_storeu_ps(cr + 8, c##r##1); } \
                   else { _Alignas(32) float t[NR]; _mm256_store_ps(t, c##r##0); _mm256_store_ps(t + 8, c##r##1); \
                          memcpy(cr, t, cols * sizeof(float)); } }

static void ikernel(uint32_t taps, uint32_t run, const float *const *p, const float *b, float *c, size_t ldc,
                    uint32_t rows, uint32_t cols) {
    ROWS(I_DECL)
    for (uint32_t t = 0; t < taps; t++, p += MR) {
        ROWS(I_PTR)
        for (uint32_t k = 0; k < run; k++, b += NR) {
            __m256 b0 = _mm256_load_ps(b), b1 = _mm256_load_ps(b + 8);
            ROWS(I_FMA)
        }
    }
    ROWS(I_STORE)
}

#else

static void ikernel(uint32_t taps, uint32_t run, const float *const *p, const float *b, float *c, size_t ldc,
                    uint32_t rows, uint32_t cols) {
    float acc[MR][NR] = {{0}};
    for (uint32_t t = 0; t < taps; t++, p += MR)
        for (uint32_t k = 0; k < run; k++, b += NR)
            for (uint32_t r = 0; r < MR; r++) {
                float a = p[r][k];
                for (uint32_t j = 0; j < NR; j++) acc[r][j] += a * b[j];
            }
    for (uint32_t r = 0; r < rows; r++) memcpy(c + (size_t)r * ldc, acc[r], cols * sizeof(float));
}

#endif

/* The weight gradient's tile: acc[r][h * VW + 0..VW) += sum over pixels i of a[i][r] * b_h[i][0..VW),
   a[i] the MR filters' gradients of pixel i (rows `lda` floats apart), b_h[i] the window run of
   half h of pixel i, through pointers q[h][i]. acc is loaded first and stored back (MR x NR). */
#if defined(__AVX512F__)

#define W_DECL(r) __m512 w##r##0 = _mm512_load_ps(acc + (r) * NR), w##r##1 = _mm512_load_ps(acc + (r) * NR + 16);
#define W_FMA(r) { __m512 ar = _mm512_set1_ps(a[r]); \
                   w##r##0 = _mm512_fmadd_ps(ar, b0, w##r##0); w##r##1 = _mm512_fmadd_ps(ar, b1, w##r##1); }
#define W_STORE(r) _mm512_store_ps(acc + (r) * NR, w##r##0); _mm512_store_ps(acc + (r) * NR + 16, w##r##1);

static void wkernel(uint32_t pixels, const float *a, size_t lda, const float *const *q0, const float *const *q1,
                    uint32_t off0, uint32_t off1, float *acc) {
    ROWS(W_DECL)
    for (uint32_t i = 0; i < pixels; i++, a += lda) {
        __m512 b0 = _mm512_loadu_ps(q0[i] + off0), b1 = _mm512_loadu_ps(q1[i] + off1);
        ROWS(W_FMA)
    }
    ROWS(W_STORE)
}

#elif defined(__AVX__)

#define W_DECL(r) __m256 w##r##0 = _mm256_load_ps(acc + (r) * NR), w##r##1 = _mm256_load_ps(acc + (r) * NR + 8);
#define W_FMA(r) { __m256 ar = _mm256_broadcast_ss(a + (r)); \
                   w##r##0 = FMA256(ar, b0, w##r##0); w##r##1 = FMA256(ar, b1, w##r##1); }
#define W_STORE(r) _mm256_store_ps(acc + (r) * NR, w##r##0); _mm256_store_ps(acc + (r) * NR + 8, w##r##1);

static void wkernel(uint32_t pixels, const float *a, size_t lda, const float *const *q0, const float *const *q1,
                    uint32_t off0, uint32_t off1, float *acc) {
    ROWS(W_DECL)
    for (uint32_t i = 0; i < pixels; i++, a += lda) {
        __m256 b0 = _mm256_loadu_ps(q0[i] + off0), b1 = _mm256_loadu_ps(q1[i] + off1);
        ROWS(W_FMA)
    }
    ROWS(W_STORE)
}

#else

/* NR = VW: one run per tile (q1 unused) */
static void wkernel(uint32_t pixels, const float *a, size_t lda, const float *const *q0, const float *const *q1,
                    uint32_t off0, uint32_t off1, float *acc) {
    (void)q1; (void)off1;
    for (uint32_t i = 0; i < pixels; i++, a += lda) {
        const float *b = q0[i] + off0;
        for (uint32_t r = 0; r < MR; r++)
            for (uint32_t j = 0; j < NR; j++) acc[r * NR + j] += a[r] * b[j];
    }
}

#endif

/* ------------------------------------------------------------------------- drivers */

/* Window positions of a convolution: rows are pixels of one map (output pixels for the forward
   pass, input pixels for the data gradient) over n samples. */
typedef struct {
    uint32_t taps, run, kw;
    uint32_t sh, sw, ph, pw;
    uint32_t rh, rw;            /* the rows' map */
    uint32_t th, tw, tc;        /* the map the taps read, its channels per pixel */
    uint32_t c0;                /* the first channel read (the group's) */
    bool transposed;            /* data gradient: the tap's pixel is the output pixel covering the row */
    const float *data;          /* the map the taps read */
    const float *zeros;         /* run zeros, for taps outside */
    int64_t off[SPINGALETT_CONV_DIRECT_TAPS];   /* tap t's offset from the window's first cell (forward)
                                                   or the covering pixel's offset back from the row's
                                                   own cell (data gradient, stride 1) */
} Taps;

static void taps_init(Taps *g) {
    for (uint32_t t = 0; t < g->taps; t++)
        g->off[t] = ((int64_t)(t / g->kw) * g->tw + (int64_t)(t % g->kw)) * g->tc;
}

/* The tap pointers of rows [row, row + rows) (MR of them, missing rows reading zeros), tap-major:
   p[t * MR + r]. A row whose window lies inside the map (most of them) takes its first cell plus the
   taps' offsets. */
static void row_taps(const Taps *g, uint64_t row, uint32_t rows, const float **p) {
    const uint64_t cells = (uint64_t)g->rh * g->rw;
    const uint32_t kh_n = g->taps / g->kw;
    uint64_t n = row / cells;
    uint32_t cell = (uint32_t)(row % cells), h = cell / g->rw, w = cell % g->rw;
    for (uint32_t r = 0; r < MR; r++) {
        if (r >= rows) {
            for (uint32_t t = 0; t < g->taps; t++) p[t * MR + r] = g->zeros;
            continue;
        }
        const float *map = g->data + n * (uint64_t)g->th * g->tw * g->tc + g->c0;
        if (!g->transposed) {
            int64_t y0 = (int64_t)h * g->sh - g->ph, x0 = (int64_t)w * g->sw - g->pw;
            if (y0 >= 0 && x0 >= 0 && y0 + kh_n <= g->th && x0 + g->kw <= g->tw) {
                const float *base = map + ((uint64_t)y0 * g->tw + (uint64_t)x0) * g->tc;
                for (uint32_t t = 0; t < g->taps; t++) p[t * MR + r] = base + g->off[t];
                if (++w == g->rw) { w = 0; if (++h == g->rh) { h = 0; n++; } }
                continue;
            }
        } else if (g->sh == 1 && g->sw == 1) {
            /* tap (kh, kw) reads the output cell (h + ph - kh, w + pw - kw) */
            int64_t y1 = (int64_t)h + g->ph, x1 = (int64_t)w + g->pw;
            if (y1 + 1 >= kh_n && x1 + 1 >= g->kw && y1 < g->th && x1 < g->tw) {
                const float *base = map + ((uint64_t)y1 * g->tw + (uint64_t)x1) * g->tc;
                for (uint32_t t = 0; t < g->taps; t++) p[t * MR + r] = base - g->off[t];
                if (++w == g->rw) { w = 0; if (++h == g->rh) { h = 0; n++; } }
                continue;
            }
        }
        for (uint32_t t = 0, kh = 0, kw = 0; t < g->taps; t++) {
            int64_t y, x;
            bool inside;
            if (!g->transposed) {
                y = (int64_t)h * g->sh - g->ph + kh;
                x = (int64_t)w * g->sw - g->pw + kw;
                inside = y >= 0 && y < g->th && x >= 0 && x < g->tw;
            } else {
                int64_t ty = (int64_t)h + g->ph - kh, tx = (int64_t)w + g->pw - kw;
                inside = ty >= 0 && tx >= 0 && ty % g->sh == 0 && tx % g->sw == 0;
                y = ty / g->sh;
                x = tx / g->sw;
                inside = inside && y < g->th && x < g->tw;
            }
            p[t * MR + r] = inside ? map + ((uint64_t)y * g->tw + (uint64_t)x) * g->tc : g->zeros;
            if (++kw == g->kw) { kw = 0; kh++; }
        }
        if (++w == g->rw) { w = 0; if (++h == g->rh) { h = 0; n++; } }
    }
}

/* Packs B[k][j] = src[j * ldj + k * ldk] for k < K, j < cols into NR-wide panels (zero-padded). */
static void pack_panels(float *dst, const float *src, size_t ldj, size_t ldk, uint32_t K, uint32_t cols) {
    for (uint32_t j0 = 0; j0 < cols; j0 += NR) {
        uint32_t width = cols - j0 < NR ? cols - j0 : NR;
        for (uint32_t k = 0; k < K; k++, dst += NR) {
            for (uint32_t j = 0; j < width; j++) dst[j] = src[(size_t)(j0 + j) * ldj + (size_t)k * ldk];
            for (uint32_t j = width; j < NR; j++) dst[j] = 0.0f;
        }
    }
}

/* rows x panels of one map: c[row][0..cols) for every row, then the epilogue on each tile of rows. */
static void indirect_product(const Taps *g, uint64_t rows_total, const float *packed, uint32_t cols, float *c, size_t ldc,
                             const SpingalettGemmHooks *hooks, bool parallel, int threads) {
    const uint32_t K = g->taps * g->run;
    const int64_t tiles = (int64_t)((rows_total + MR - 1) / MR);
    (void)threads; (void)parallel;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(threads) if(parallel && tiles > 1)
#endif
    for (int64_t t = 0; t < tiles; t++) {
        const float *p[MR * SPINGALETT_CONV_DIRECT_TAPS];
        uint64_t row = (uint64_t)t * MR;
        uint32_t rows = rows_total - row < MR ? (uint32_t)(rows_total - row) : MR;
        row_taps(g, row, rows, p);
        for (uint32_t j = 0; j < cols; j += NR)
            ikernel(g->taps, g->run, p, packed + (size_t)j * K, c + row * ldc + j, ldc, rows,
                    cols - j < NR ? cols - j : NR);
        if (hooks && hooks->epilogue) hooks->epilogue(hooks->epilogue_ctx, (uint32_t)row, rows, 0, cols, c + row * ldc, ldc);
    }
}

void CONV_NAME(spingalett_conv_direct_forward)(const LayerShape *in, const LayerShape *out, uint32_t g, const float *W,
                                               const float *x, float *y, uint32_t n, float *scratch,
                                               const SpingalettGemmHooks *epilogue, bool parallel, int threads) {
    const uint32_t G = out->groups ? out->groups : 1u, C = in->channels, CG = C / G, OC = out->channels, OG = OC / G;
    const uint32_t taps = out->kernel_h * out->kernel_w, K = taps * CG;
    float *packed = scratch, *zeros = scratch + spingalett_conv_direct_packed_floats(OG, K);
    memset(zeros, 0, ((size_t)(CG > OG ? CG : OG) + 32u) * sizeof(float));
    pack_panels(packed, W + (size_t)g * OG * K, K, 1, K, OG);
    Taps tp = {taps, CG, out->kernel_w, out->stride_h, out->stride_w, out->pad_h, out->pad_w, out->height, out->width,
               in->height, in->width, C, g * CG, false, x, zeros, {0}};
    taps_init(&tp);
    indirect_product(&tp, (uint64_t)n * out->height * out->width, packed, OG, y + (size_t)g * OG, OC, epilogue,
                     parallel, threads);
}

/* The data gradient of a strided convolution, a phase at a time: the input cells (h, w) with
   h = ph (mod stride_h) and w = pw (mod stride_w) are covered by the same taps, those with
   kh = ph + pad_h (mod stride_h) and kw alike, so each phase is a product over its own taps only
   (with stride 2, a quarter of a 3 x 3 kernel's on average); a phase no tap covers gets zeros. */
static void strided_backward_data(const LayerShape *in, const LayerShape *out, uint32_t g, const float *W,
                                  const float *dy, float *dx, uint32_t n, float *packed, const float *zeros,
                                  const SpingalettGemmHooks *epilogue, bool parallel, int threads) {
    const uint32_t G = out->groups ? out->groups : 1u, C = in->channels, CG = C / G, OC = out->channels, OG = OC / G;
    const uint32_t KH = out->kernel_h, KW = out->kernel_w, SH = out->stride_h, SW = out->stride_w;
    const uint32_t H = in->height, Wd = in->width, OH = out->height, OW = out->width, K = KH * KW * CG;
    const float *Wg = W + (size_t)g * OG * K;
    for (uint32_t ph = 0; ph < SH && ph < H; ph++)
        for (uint32_t pw = 0; pw < SW && pw < Wd; pw++) {
            /* the phase's taps */
            uint32_t th[SPINGALETT_CONV_DIRECT_TAPS], tw[SPINGALETT_CONV_DIRECT_TAPS], nh = 0, nw = 0;
            for (uint32_t kh = 0; kh < KH; kh++) if ((ph + out->pad_h + SH * KH - kh) % SH == 0) th[nh++] = kh;
            for (uint32_t kw = 0; kw < KW; kw++) if ((pw + out->pad_w + SW * KW - kw) % SW == 0) tw[nw++] = kw;
            const uint32_t taps = nh * nw, KR = taps * OG;
            const uint32_t rows_h = (H - ph + SH - 1) / SH, rows_w = (Wd - pw + SW - 1) / SW;
            /* B[(t, oc)][j] = W[g OG + oc][(kh_t, kw_t) CG + j] over the phase's taps */
            for (uint32_t j0 = 0; taps && j0 < CG; j0 += NR) {
                uint32_t width = CG - j0 < NR ? CG - j0 : NR;
                float *dst = packed + (size_t)j0 * KR;
                for (uint32_t a = 0; a < nh; a++)
                    for (uint32_t b = 0; b < nw; b++)
                        for (uint32_t oc = 0; oc < OG; oc++, dst += NR) {
                            const float *src = Wg + (size_t)oc * K + ((size_t)th[a] * KW + tw[b]) * CG + j0;
                            for (uint32_t j = 0; j < width; j++) dst[j] = src[j];
                            for (uint32_t j = width; j < NR; j++) dst[j] = 0.0f;
                        }
            }
            /* tiles of MR cells of one row of the phase */
            const uint32_t per_row = (rows_w + MR - 1) / MR;
            const int64_t tiles = (int64_t)n * rows_h * per_row;
            (void)threads; (void)parallel;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(threads) if(parallel && tiles > 1)
#endif
            for (int64_t t = 0; t < tiles; t++) {
                const float *p[MR * SPINGALETT_CONV_DIRECT_TAPS];
                const uint64_t row = (uint64_t)t / per_row, s = row / rows_h;
                const uint32_t i = (uint32_t)(row % rows_h), j0 = (uint32_t)((uint64_t)t % per_row) * MR;
                const uint32_t rows = rows_w - j0 < MR ? rows_w - j0 : MR, h = ph + i * SH;
                const uint64_t first = (s * H + h) * Wd + pw + (uint64_t)j0 * SW;    /* the tile's first cell */
                float *c = dx + first * C + (size_t)g * CG;
                if (taps == 0) {
                    for (uint32_t r = 0; r < rows; r++) memset(c + (size_t)r * SW * C, 0, CG * sizeof(float));
                    continue;
                }
                const float *map = dy + s * (uint64_t)OH * OW * OC + (size_t)g * OG;
                for (uint32_t r = 0; r < MR; r++) {
                    const uint32_t w = pw + (j0 + r) * SW;
                    for (uint32_t a = 0, k = 0; a < nh; a++)
                        for (uint32_t b = 0; b < nw; b++, k++) {
                            /* (h + pad - kh) and (w + pad - kw) are multiples of the strides */
                            int64_t oh = ((int64_t)h + out->pad_h - th[a]) / SH, ow = ((int64_t)w + out->pad_w - tw[b]) / SW;
                            bool inside = r < rows && (int64_t)h + out->pad_h >= th[a] && (int64_t)w + out->pad_w >= tw[b] &&
                                          oh < OH && ow < OW;
                            p[k * MR + r] = inside ? map + ((uint64_t)oh * OW + (uint64_t)ow) * OC : zeros;
                        }
                }
                for (uint32_t j = 0; j < CG; j += NR)
                    ikernel(taps, OG, p, packed + (size_t)j * KR, c + j, (size_t)SW * C, rows, CG - j < NR ? CG - j : NR);
                if (epilogue && epilogue->epilogue)
                    for (uint32_t r = 0; r < rows; r++)
                        epilogue->epilogue(epilogue->epilogue_ctx, (uint32_t)(first + (uint64_t)r * SW), 1, 0, CG,
                                           c + (size_t)r * SW * C, C);
            }
        }
}

void CONV_NAME(spingalett_conv_direct_backward_data)(const LayerShape *in, const LayerShape *out, uint32_t g,
                                                     const float *W, const float *dy, float *dx, uint32_t n,
                                                     float *scratch, const SpingalettGemmHooks *epilogue, bool parallel,
                                                     int threads) {
    const uint32_t G = out->groups ? out->groups : 1u, C = in->channels, CG = C / G, OC = out->channels, OG = OC / G;
    const uint32_t taps = out->kernel_h * out->kernel_w, K = taps * CG, KR = taps * OG;
    float *packed = scratch, *zeros = scratch + spingalett_conv_direct_packed_floats(CG, KR);
    memset(zeros, 0, ((size_t)(CG > OG ? CG : OG) + 32u) * sizeof(float));
    if (out->stride_h > 1 || out->stride_w > 1) {
        strided_backward_data(in, out, g, W, dy, dx, n, packed, zeros, epilogue, parallel, threads);
        return;
    }
    /* B[t * OG + oc][j] = W[g OG + oc][t CG + j] */
    const float *Wg = W + (size_t)g * OG * K;
    for (uint32_t j0 = 0; j0 < CG; j0 += NR) {
        uint32_t width = CG - j0 < NR ? CG - j0 : NR;
        float *dst = packed + (size_t)j0 * KR;
        for (uint32_t t = 0; t < taps; t++)
            for (uint32_t oc = 0; oc < OG; oc++, dst += NR) {
                const float *src = Wg + (size_t)oc * K + (size_t)t * CG + j0;
                for (uint32_t j = 0; j < width; j++) dst[j] = src[j];
                for (uint32_t j = width; j < NR; j++) dst[j] = 0.0f;
            }
    }
    Taps tp = {taps, OG, out->kernel_w, out->stride_h, out->stride_w, out->pad_h, out->pad_w, in->height, in->width,
               out->height, out->width, OC, g * OG, true, dy, zeros, {0}};
    taps_init(&tp);
    indirect_product(&tp, (uint64_t)n * in->height * in->width, packed, CG, dx + (size_t)g * CG, C, epilogue, parallel,
                     threads);
}

/* The pointers of tap t for `count` consecutive rows from `row` (window rows of a forward map). */
static void tap_column(const Taps *g, uint64_t row, uint32_t count, uint32_t t, const float **q) {
    const uint64_t cells = (uint64_t)g->rh * g->rw;
    const uint32_t kh = t / g->kw, kw = t % g->kw;
    uint64_t n = row / cells;
    uint32_t cell = (uint32_t)(row % cells), h = cell / g->rw, w = cell % g->rw;
    const uint64_t sample = (uint64_t)g->th * g->tw * g->tc;
    const float *map = g->data + n * sample + g->c0;
    for (uint32_t i = 0; i < count; i++) {
        int64_t y = (int64_t)h * g->sh - g->ph + kh, x = (int64_t)w * g->sw - g->pw + kw;
        q[i] = y >= 0 && y < g->th && x >= 0 && x < g->tw ? map + ((uint64_t)y * g->tw + (uint64_t)x) * g->tc : g->zeros;
        if (++w == g->rw) {
            w = 0;
            if (++h == g->rh) { h = 0; n++; map += sample; }
        }
    }
}

void CONV_NAME(spingalett_conv_direct_backward_weights)(const LayerShape *in, const LayerShape *out, uint32_t g,
                                                        const float *x, const float *dy, uint32_t n, float scale,
                                                        float beta, float *gW, float *scratch, bool parallel,
                                                        int threads) {
    const uint32_t G = out->groups ? out->groups : 1u, C = in->channels, CG = C / G, OC = out->channels, OG = OC / G;
    const uint32_t taps = out->kernel_h * out->kernel_w, K = taps * CG;
    const uint64_t pixels = (uint64_t)n * out->height * out->width;
    const uint32_t slots = spingalett_conv_wgrad_slots(pixels);
    const uint32_t mtiles = (OG + MR - 1) / MR, ntiles = (K + NR - 1) / NR;
    /* scratch: the slots' partial sums (tile by tile, MR x NR each), zeros, per-thread tap pointers */
    const size_t part = (size_t)mtiles * ntiles * MR * NR, tile = (size_t)MR * NR;
    float *partial = scratch, *zeros = partial + spingalett_conv_wgrad_partial_floats(OG, K, slots);
    memset(zeros, 0, (size_t)(CG + SPINGALETT_CONV_NR_MAX) * sizeof(float));
    const float **ptrs = (const float **)(void *)(zeros + spingalett_conv_wgrad_zero_floats(CG));
    const uint64_t per = (pixels + slots - 1) / slots;
    /* work items: a slot of pixels by a column of NR window elements: each reads its window elements
       of every pixel once (two runs of VW, through the pointers of their taps) and sums them with
       the gradients of every tile row of filters */
    const int64_t items = (int64_t)slots * ntiles;
    Taps tp = {taps, CG, out->kernel_w, out->stride_h, out->stride_w, out->pad_h, out->pad_w, out->height,
               out->width, in->height, in->width, C, g * CG, false, x, zeros, {0}};
    (void)threads; (void)parallel;
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) if(parallel && items > 1)
#endif
    for (int64_t it = 0; it < items; it++) {
        int tid = 0;
#if defined(_OPENMP)
        tid = omp_get_thread_num();
#endif
        const uint32_t s = (uint32_t)(it / ntiles), nt = (uint32_t)(it % ntiles);
        const float **q0 = ptrs + (size_t)tid * 2u * WGRAD_BLOCK, **q1 = q0 + WGRAD_BLOCK;
        float *P = partial + (size_t)s * part;
        for (uint32_t mt = 0; mt < mtiles; mt++) memset(P + ((size_t)mt * ntiles + nt) * tile, 0, tile * sizeof(float));
        const uint64_t p0 = (uint64_t)s * per, p1 = p0 + per < pixels ? p0 + per : pixels;
        /* window elements k0 .. k0 + NR: runs of VW, each within one tap */
        const uint32_t k0 = nt * NR, k1 = k0 + VW < K ? k0 + VW : k0, t0 = k0 / CG, t1 = k1 / CG;
        for (uint64_t b0 = p0; b0 < p1; b0 += WGRAD_BLOCK) {
            uint32_t count = p1 - b0 < WGRAD_BLOCK ? (uint32_t)(p1 - b0) : WGRAD_BLOCK;
            tap_column(&tp, b0, count, t0, q0);
            if (t1 != t0) tap_column(&tp, b0, count, t1, q1);
            for (uint32_t mt = 0; mt < mtiles; mt++) {
                /* rows past the group's filters: the tile row is shifted back, its first rows computed
                   twice (they are not read) */
                const uint32_t oc0 = mt * MR + MR <= OG ? mt * MR : OG - MR;
                wkernel(count, dy + b0 * OC + (size_t)g * OG + oc0, OC, q0, t1 != t0 ? q1 : q0, k0 - t0 * CG,
                        k1 - t1 * CG, P + ((size_t)mt * ntiles + nt) * tile);
            }
        }
    }
    /* the slots' sums in order, into the group's rows of gW */
    float *gWg = gW + (size_t)g * OG * K;
    SPINGALETT_PARALLEL_FOR_THREADS(threads, parallel && (uint64_t)OG * K * slots >= (1u << 16),
        for (int64_t oc = 0; oc < (int64_t)OG; oc++) {
            /* the tile row that holds filter oc (the last tile row is shifted back) */
            const uint32_t mt = (uint32_t)oc / MR;
            const uint32_t base = mt * MR + MR <= OG ? mt * MR : OG - MR, r = (uint32_t)oc - base;
            for (uint32_t k = 0; k < K; k++) {
                const size_t at = ((size_t)mt * ntiles + k / NR) * tile + (size_t)r * NR + k % NR;
                float sum = partial[at];
                for (uint32_t s = 1; s < slots; s++) sum += partial[(size_t)s * part + at];
                float *dst = gWg + (size_t)oc * K + k;
                *dst = beta == 0.0f ? scale * sum : scale * sum + beta * *dst;
            }
        }
    );
}

/* ------------------------------------------------------------------------- dispatch */

#if !defined(SPINGALETT_CONV_VARIANT)

#if defined(SPINGALETT_GEMM_DISPATCH)
#define CONV_VARIANTS(name) void name##_avx2 name##_ARGS; void name##_avx512 name##_ARGS;
#define spingalett_conv_direct_forward_ARGS (const LayerShape *, const LayerShape *, uint32_t, const float *, \
    const float *, float *, uint32_t, float *, const SpingalettGemmHooks *, bool, int)
#define spingalett_conv_direct_backward_data_ARGS spingalett_conv_direct_forward_ARGS
#define spingalett_conv_direct_backward_weights_ARGS (const LayerShape *, const LayerShape *, uint32_t, const float *, \
    const float *, uint32_t, float, float, float *, float *, bool, int)
CONV_VARIANTS(spingalett_conv_direct_forward)
CONV_VARIANTS(spingalett_conv_direct_backward_data)
CONV_VARIANTS(spingalett_conv_direct_backward_weights)

/* 0: baseline, 1: AVX2 and FMA, 2: AVX-512, as the GEMM chooses */
static int conv_level(void) {
    static int level = -1;
    if (level < 0) {
        __builtin_cpu_init();
        level = __builtin_cpu_supports("avx512f") ? 2
              : __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") ? 1 : 0;
    }
    return level;
}
#define DISPATCH(name, ...) do { switch (conv_level()) { \
        case 2: name##_avx512(__VA_ARGS__); break; case 1: name##_avx2(__VA_ARGS__); break; \
        default: name##_baseline(__VA_ARGS__); break; } } while (0)
#else
#define DISPATCH(name, ...) name##_impl(__VA_ARGS__)
#endif

void spingalett_conv_direct_forward(const LayerShape *in, const LayerShape *out, uint32_t g, const float *W,
                                    const float *x, float *y, uint32_t n, float *scratch,
                                    const SpingalettGemmHooks *epilogue, bool parallel, int threads) {
    DISPATCH(spingalett_conv_direct_forward, in, out, g, W, x, y, n, scratch, epilogue, parallel, threads);
}

void spingalett_conv_direct_backward_data(const LayerShape *in, const LayerShape *out, uint32_t g, const float *W,
                                          const float *dy, float *dx, uint32_t n, float *scratch,
                                          const SpingalettGemmHooks *epilogue, bool parallel, int threads) {
    DISPATCH(spingalett_conv_direct_backward_data, in, out, g, W, dy, dx, n, scratch, epilogue, parallel, threads);
}

void spingalett_conv_direct_backward_weights(const LayerShape *in, const LayerShape *out, uint32_t g, const float *x,
                                             const float *dy, uint32_t n, float scale, float beta, float *gW,
                                             float *scratch, bool parallel, int threads) {
    DISPATCH(spingalett_conv_direct_backward_weights, in, out, g, x, dy, n, scale, beta, gW, scratch, parallel,
             threads);
}

#endif
