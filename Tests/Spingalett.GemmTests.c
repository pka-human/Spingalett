/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* White-box tests of the native GEMM against a double-precision reference: every transpose
   combination, edge sizes around the micro-kernel and block dimensions, alpha/beta handling
   (beta = 0 must not read C), leading dimensions larger than the matrix, threading, operands
   produced by a source instead of read from memory, epilogues (each element visited once, after
   its final value), products split along k (and their results being the same on one thread and
   many); for every kernel set of the build that the processor can run. */

#include "Spingalett.Private.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

typedef void (*Gemm)(SpingalettGemmScratch *, bool, bool, uint32_t, uint32_t, uint32_t, float,
                     const float *, size_t, const float *, size_t, float, float *, size_t, bool,
                     const SpingalettGemmHooks *);

static int failures = 0;
static Gemm gemm = spingalett_gemm_hooked;     /* the kernel set under test */
static unsigned state = 1;
static float frand(void) { state = state * 1103515245u + 12345u; return (float)((state >> 8) & 0xFFFF) / 65536.0f - 0.5f; }

/* A source reading a stored matrix, checking that every request lies inside it. */
typedef struct { const float *data; size_t ld; uint32_t rows, cols; } Stored;

static void fill_stored(const void *ctx, uint32_t row, uint32_t rows, uint32_t col, uint32_t cols, float *dst, size_t ld) {
    const Stored *m = ctx;
    if (rows == 0 || cols == 0 || row + rows > m->rows || col + cols > m->cols) {
        failures++;
        printf("  FAIL source request rows [%u, +%u) cols [%u, +%u) outside %u x %u\n", row, rows, col, cols, m->rows, m->cols);
        return;
    }
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < cols; c++) dst[(size_t)r * ld + c] = m->data[(size_t)(row + r) * m->ld + col + c];
}

/* An epilogue mapping c to 2c + 1 that counts its calls per element (each must be visited once). */
typedef struct { unsigned char *visits; size_t ldc; float *C; } Visits;

static void epilogue_visit(const void *ctx, uint32_t row, uint32_t rows, uint32_t col, uint32_t cols, float *c, size_t ldc) {
    const Visits *v = ctx;
    if (ldc != v->ldc || c != v->C + (size_t)row * ldc + col) { failures++; printf("  FAIL epilogue block pointer\n"); return; }
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t j = 0; j < cols; j++) {
            c[(size_t)r * ldc + j] = 2.0f * c[(size_t)r * ldc + j] + 1.0f;
            v->visits[(size_t)(row + r) * ldc + col + j]++;
        }
}

/* sources: bit 0 produces A and bit 1 B through a source (the matrix pointer is then NULL); bit 2
   adds the epilogue. */
static void check_case(bool ta, bool tb, uint32_t M, uint32_t N, uint32_t K, float alpha, float beta,
                       uint32_t pad, bool parallel, SpingalettGemmScratch *scratch, int sources) {
    size_t lda = (ta ? M : K) + pad, ldb = (tb ? K : N) + pad, ldc = N + pad;
    size_t a_rows = ta ? K : M, b_rows = tb ? N : K;
    float *A = malloc(sizeof(float) * (a_rows * lda + 1));
    float *B = malloc(sizeof(float) * (b_rows * ldb + 1));
    float *C = malloc(sizeof(float) * (M * ldc + 1));
    double *R = malloc(sizeof(double) * M * N);
    for (size_t i = 0; i < a_rows * lda; i++) A[i] = frand();
    for (size_t i = 0; i < b_rows * ldb; i++) B[i] = frand();
    for (size_t i = 0; i < M * ldc; i++) C[i] = beta == 0.0f ? NAN : frand();   /* beta = 0 must ignore C */

    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++) {
            double s = 0;
            for (uint32_t k = 0; k < K; k++) {
                double a = ta ? A[(size_t)k * lda + i] : A[(size_t)i * lda + k];
                double b = tb ? B[(size_t)j * ldb + k] : B[(size_t)k * ldb + j];
                s += a * b;
            }
            R[(size_t)i * N + j] = alpha * s + (beta == 0.0f ? 0.0 : beta * C[(size_t)i * ldc + j]);
        }

    Stored a_stored = {A, lda, (uint32_t)a_rows, ta ? M : K}, b_stored = {B, ldb, (uint32_t)b_rows, tb ? K : N};
    SpingalettGemmSource a_src = {fill_stored, &a_stored}, b_src = {fill_stored, &b_stored};
    bool sa = sources & 1, sb = sources & 2, epi = sources & 4;
    unsigned char *visits = calloc(M * ldc + 1, 1);
    Visits v = {visits, ldc, C};
    SpingalettGemmHooks hooks = {sa ? &a_src : NULL, sb ? &b_src : NULL, epi ? epilogue_visit : NULL, &v};
    gemm(scratch, ta, tb, M, N, K, alpha, sa ? NULL : A, lda, sb ? NULL : B, ldb, beta, C, ldc, parallel, &hooks);
    if (epi) {
        for (uint32_t i = 0; i < M; i++)
            for (uint32_t j = 0; j < N; j++) {
                if (visits[(size_t)i * ldc + j] != 1) {
                    failures++;
                    printf("  FAIL epilogue visited C[%u][%u] %d times (M=%u N=%u K=%u)\n", i, j, visits[(size_t)i * ldc + j], M, N, K);
                    i = M;
                    break;
                }
                R[(size_t)i * N + j] = 2.0 * R[(size_t)i * N + j] + 1.0;
            }
    }
    free(visits);

    double worst = 0;
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++) {
            double err = fabs(C[(size_t)i * ldc + j] - R[(size_t)i * N + j]) / (1.0 + sqrt((double)K) * 0.25);
            if (!(err <= worst)) worst = err;      /* also catches NaN */
        }
    bool ok = worst < 1e-5;
    if (!ok) {
        failures++;
        printf("  FAIL ta=%d tb=%d M=%u N=%u K=%u alpha=%g beta=%g pad=%u parallel=%d sources=%d: err %.3e\n",
               ta, tb, M, N, K, alpha, beta, pad, parallel, sources, worst);
    }
    free(A); free(B); free(C); free(R);
}

/* A product split along k gives the same bits on one thread as on all of them. */
static void check_split_threads(bool ta, bool tb, uint32_t M, uint32_t N, uint32_t K, SpingalettGemmScratch *scratch) {
    size_t a_size = (size_t)M * K, b_size = (size_t)K * N, c_size = (size_t)M * N;
    float *A = malloc(sizeof(float) * a_size), *B = malloc(sizeof(float) * b_size);
    float *C1 = malloc(sizeof(float) * c_size), *C2 = malloc(sizeof(float) * c_size);
    for (size_t i = 0; i < a_size; i++) A[i] = frand();
    for (size_t i = 0; i < b_size; i++) B[i] = frand();
    gemm(scratch, ta, tb, M, N, K, 0.5f, A, ta ? M : K, B, tb ? K : N, 0.0f, C1, N, false, NULL);
    gemm(scratch, ta, tb, M, N, K, 0.5f, A, ta ? M : K, B, tb ? K : N, 0.0f, C2, N, true, NULL);
    if (memcmp(C1, C2, sizeof(float) * c_size) != 0) {
        failures++;
        printf("  FAIL split product differs between 1 and many threads: ta=%d tb=%d M=%u N=%u K=%u\n", ta, tb, M, N, K);
    }
    free(A); free(B); free(C1); free(C2);
}

int main(void) {
    int threads = 1;
#if defined(_OPENMP)
    threads = omp_get_max_threads() < 4 ? omp_get_max_threads() : 4;
#endif
    SpingalettGemmScratch *scratch = spingalett_gemm_scratch_create(threads);
    const uint32_t sizes[][3] = {
        {1, 1, 1}, {1, 17, 3}, {5, 1, 9}, {6, 16, 1}, {7, 33, 2}, {12, 32, 256}, {13, 31, 257},
        {121, 47, 300}, {150, 100, 513}, {37, 3100, 20}, {64, 1000, 64}, {300, 7, 600},
    };
    const float ab[][2] = {{1.0f, 0.0f}, {0.5f, 1.0f}, {-1.5f, 0.25f}};
    struct { const char *name; Gemm fn; bool supported; } sets[] = {
        {spingalett_cpu_kernels(), spingalett_gemm_hooked, true},
#if defined(SPINGALETT_GEMM_DISPATCH)
        {"baseline", spingalett_gemm_baseline, true},
        {"AVX2", spingalett_gemm_avx2, __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")},
        {"AVX-512", spingalett_gemm_avx512, __builtin_cpu_supports("avx512f")},
#endif
    };
    /* small C, long k: split into slots */
    const uint32_t split_sizes[][3] = {{9, 32, 6000}, {40, 96, 2200}, {140, 40, 2100}, {1, 1, 2049}, {12, 512, 2048}};
    int cases = 0;
    for (unsigned k = 0; k < sizeof sets / sizeof *sets; k++) {
        if (!sets[k].supported) { printf("%s kernels: not supported by this processor\n", sets[k].name); continue; }
        gemm = sets[k].fn;
        int before = failures;
        for (unsigned s = 0; s < sizeof sizes / sizeof *sizes; s++)
            for (int t = 0; t < 4; t++)
                for (unsigned c = 0; c < 3; c++)
                    for (int par = 0; par < 2; par++) {
                        /* the kernel sets themselves need a scratch; the entry point makes one */
                        SpingalettGemmScratch *sc = par || k > 0 ? scratch : NULL;
                        /* each transpose combination also once with sourced operands (cycling
                           A, B, both), so all four packing paths of a source run per size */
                        int sources = c == 2 ? 1 + (int)((s + (unsigned)par) % 3) : c == 1 ? 4 + (int)((s + (unsigned)t) % 4) : 0;
                        check_case(t & 1, t & 2, sizes[s][0], sizes[s][1], sizes[s][2], ab[c][0], ab[c][1],
                                   (s + t) % 3 == 0 ? 3 : 0, par, sc, sources);
                        cases++;
                    }
        for (unsigned s = 0; s < sizeof split_sizes / sizeof *split_sizes; s++)
            for (int t = 0; t < 4; t++)
                for (int par = 0; par < 2; par++) {
                    SpingalettGemmScratch *sc = par || k > 0 ? scratch : NULL;
                    check_case(t & 1, t & 2, split_sizes[s][0], split_sizes[s][1], split_sizes[s][2],
                               par ? 1.0f : -0.75f, par ? 0.0f : 0.5f, (s + t) % 2 ? 2 : 0, par, sc, (s + t + par) % 8);
                    cases++;
                }
        for (int t = 0; t < 4; t++) check_split_threads(t & 1, t & 2, 48, 200, 5000, scratch);
        printf("%s kernels%s: %d failures\n", sets[k].name, k == 0 ? " (selected)" : "", failures - before);
    }
    /* K = 0 / alpha = 0 reduce to C = beta * C */
    gemm = spingalett_gemm_hooked;
    check_case(false, false, 9, 9, 0, 1.0f, 0.5f, 0, false, NULL, 4);
    check_case(false, true, 9, 9, 5, 0.0f, 0.0f, 0, false, NULL, 7);
    spingalett_gemm_scratch_free(scratch);
    printf("%d GEMM cases, %d failures (threads %d)\n", cases + 2, failures, threads);
    return failures != 0;
}
