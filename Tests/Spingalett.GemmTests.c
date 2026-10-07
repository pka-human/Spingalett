/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* White-box tests of the native GEMM against a double-precision reference: every transpose
   combination, edge sizes around the micro-kernel and block dimensions, alpha/beta handling
   (beta = 0 must not read C), leading dimensions larger than the matrix, threading; for every
   kernel set of the build that the processor can run. */

#include "Spingalett.Private.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

typedef void (*Gemm)(SpingalettGemmScratch *, bool, bool, uint32_t, uint32_t, uint32_t, float,
                     const float *, size_t, const float *, size_t, float, float *, size_t, bool);

static int failures = 0;
static Gemm gemm = spingalett_gemm_native;      /* the kernel set under test */
static unsigned state = 1;
static float frand(void) { state = state * 1103515245u + 12345u; return (float)((state >> 8) & 0xFFFF) / 65536.0f - 0.5f; }

static void check_case(bool ta, bool tb, uint32_t M, uint32_t N, uint32_t K, float alpha, float beta,
                       uint32_t pad, bool parallel, SpingalettGemmScratch *scratch) {
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

    gemm(scratch, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, parallel);

    double worst = 0;
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++) {
            double err = fabs(C[(size_t)i * ldc + j] - R[(size_t)i * N + j]) / (1.0 + sqrt((double)K) * 0.25);
            if (!(err <= worst)) worst = err;      /* also catches NaN */
        }
    bool ok = worst < 1e-5;
    if (!ok) {
        failures++;
        printf("  FAIL ta=%d tb=%d M=%u N=%u K=%u alpha=%g beta=%g pad=%u parallel=%d: err %.3e\n",
               ta, tb, M, N, K, alpha, beta, pad, parallel, worst);
    }
    free(A); free(B); free(C); free(R);
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
        {spingalett_cpu_kernels(), spingalett_gemm_native, true},
#if defined(SPINGALETT_GEMM_DISPATCH)
        {"baseline", spingalett_gemm_baseline, true},
        {"AVX2", spingalett_gemm_avx2, __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")},
        {"AVX-512", spingalett_gemm_avx512, __builtin_cpu_supports("avx512f")},
#endif
    };
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
                        check_case(t & 1, t & 2, sizes[s][0], sizes[s][1], sizes[s][2], ab[c][0], ab[c][1],
                                   (s + t) % 3 == 0 ? 3 : 0, par, sc);
                        cases++;
                    }
        printf("%s kernels%s: %d failures\n", sets[k].name, k == 0 ? " (selected)" : "", failures - before);
    }
    /* K = 0 / alpha = 0 reduce to C = beta * C */
    gemm = spingalett_gemm_native;
    check_case(false, false, 9, 9, 0, 1.0f, 0.5f, 0, false, NULL);
    check_case(false, true, 9, 9, 5, 0.0f, 0.0f, 0, false, NULL);
    spingalett_gemm_scratch_free(scratch);
    printf("%d GEMM cases, %d failures (threads %d)\n", cases + 2, failures, threads);
    return failures != 0;
}
