/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Integer weight layers over batches (spingalett_model_predict): interleaved INT8 rows against
 * tiles of activation vectors, the pixels of a convolution or the samples of a dense layer.
 *
 * spingalett_i8_interleave stores blocks of 16 rows four bytes at a time: one 64-byte vector holds
 * bytes 4k .. 4k + 3 of the 16 rows, so one dpbusd (AVX-512 VNNI) adds four products to sixteen
 * sums, the four activation bytes broadcast. A tile of SPINGALETT_I8_TILE activation vectors runs
 * against 32 rows at a time, its 24 sums in registers and each weight vector loaded once for the
 * tile. The activations are taken as unsigned bytes x + 128 (dpbusd multiplies unsigned bytes by
 * signed ones), and 128 times each row's sum is taken off: the sums, exact modulo 2^32 like the
 * result, give it exactly.
 *
 * With AVX-VNNI (256-bit dpbusd, no AVX-512) a vector holds eight rows: a tile runs against 16
 * rows, six vectors at a time, in the 16 registers there are. On ARM with the dot product
 * instructions (sdot: signed bytes by signed bytes, no offset) a vector holds four rows, and a tile
 * runs against 16 rows, six vectors at a time.
 *
 * Builds that are not tuned for the build machine (SPINGALETT_INT8_DISPATCH: x86-64, and AArch64
 * Linux) compile the kernels once more each, through Kernels/Spingalett.Int8Tiles.AVX512.c,
 * .AVXVNNI.c and .DotProd.c with those instruction sets enabled, and run the best one the
 * processor has. Elsewhere spingalett_i8_interleaved() is false, and the callers take their other
 * kernels.
 */

#include "Spingalett.Private.h"
#include <string.h>

#if !defined(SPINGALETT_PORTABLE_KERNELS)
#  if defined(__AVX512VNNI__) && defined(__AVX512BW__)
#    include <immintrin.h>
#    define TILE_KERNEL 512
#  elif defined(__AVXVNNI__) && defined(__AVX2__)
#    include <immintrin.h>
#    define TILE_KERNEL 256
#  elif defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#    include <arm_neon.h>
#    define TILE_KERNEL 128
#  endif
#endif
#if defined(SPINGALETT_INT8_DISPATCH) && !defined(SPINGALETT_I8_TILE_VARIANT) && !defined(TILE_KERNEL)
#  if defined(__x86_64__) || defined(__i386__)
#    include <cpuid.h>
#    define TILE_DISPATCH_X86 1
#  elif defined(__aarch64__) && defined(__linux__)
#    include <sys/auxv.h>
#    if !defined(HWCAP_ASIMDDP)
#      define HWCAP_ASIMDDP (1ul << 20)
#    endif
#    define TILE_DISPATCH_ARM 1
#  endif
#endif

#if defined(TILE_KERNEL)
/* The sums live in named registers (arrays of them end up in memory). GCC's partial redundancy
   elimination has them copied between registers, or through memory, on every step (dpbusd adds
   into its first operand), and its first scheduling pass (on for AArch64) loads all of a step's
   activations at once, leaving too few registers for the sums: both are off for these loops. */
#if defined(__GNUC__) && !defined(__clang__)
#  define TILE_LOOP __attribute__((optimize("no-tree-pre", "no-schedule-insns"))) static void
#else
#  define TILE_LOOP static void
#endif

#if defined(SPINGALETT_I8_TILE_VARIANT)
#  define TILE_FUNCTION void SPINGALETT_I8_TILE_VARIANT
#else
#  define TILE_FUNCTION static void tile_native
#endif
#endif

#if defined(TILE_KERNEL) && TILE_KERNEL == 512

#define TILE_EACH(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9) M(10) M(11)
#define TILE_ZERO_ONE(p) __m512i a##p = _mm512_setzero_si512();
#define TILE_ZERO_TWO(p) __m512i a##p = _mm512_setzero_si512(), b##p = a##p;
#define TILE_BROADCAST(p) \
    int32_t q##p; \
    memcpy(&q##p, x + (size_t)(p) * xs + k, 4); \
    __m512i u##p = _mm512_set1_epi32(q##p);
#define TILE_ONE(p) TILE_BROADCAST(p) a##p = _mm512_dpbusd_epi32(a##p, u##p, v0);
#define TILE_TWO(p) \
    TILE_BROADCAST(p) \
    a##p = _mm512_dpbusd_epi32(a##p, u##p, v0); \
    b##p = _mm512_dpbusd_epi32(b##p, u##p, v1);
#define TILE_STORE_ONE(p) _mm512_storeu_si512((void *)(acc + (size_t)(p) * acc_stride), _mm512_sub_epi32(a##p, c0));
#define TILE_STORE_TWO(p) \
    TILE_STORE_ONE(p) \
    _mm512_storeu_si512((void *)(acc + (size_t)(p) * acc_stride + 16), _mm512_sub_epi32(b##p, c1));

/* 32 rows (two blocks, wstride bytes apart) against the tile */
TILE_LOOP tile_32(const int8_t *wp, size_t wstride, uint32_t len, const uint8_t *x, size_t xs, const int32_t *sums,
                  int32_t *acc, size_t acc_stride) {
    TILE_EACH(TILE_ZERO_TWO)
    for (uint32_t k = 0; k < len; k += 4u, wp += 64) {
        __m512i v0 = _mm512_loadu_si512((const void *)wp), v1 = _mm512_loadu_si512((const void *)(wp + wstride));
        TILE_EACH(TILE_TWO)
    }
    __m512i c0 = _mm512_slli_epi32(_mm512_loadu_si512((const void *)sums), 7);
    __m512i c1 = _mm512_slli_epi32(_mm512_loadu_si512((const void *)(sums + 16)), 7);
    TILE_EACH(TILE_STORE_TWO)
}

/* 16 rows against the tile */
TILE_LOOP tile_16(const int8_t *wp, uint32_t len, const uint8_t *x, size_t xs, const int32_t *sums, int32_t *acc,
                  size_t acc_stride) {
    TILE_EACH(TILE_ZERO_ONE)
    for (uint32_t k = 0; k < len; k += 4u, wp += 64) {
        __m512i v0 = _mm512_loadu_si512((const void *)wp);
        TILE_EACH(TILE_ONE)
    }
    __m512i c0 = _mm512_slli_epi32(_mm512_loadu_si512((const void *)sums), 7);
    TILE_EACH(TILE_STORE_ONE)
}

TILE_FUNCTION(const int8_t *weights, const int32_t *sums, uint32_t rows, uint32_t n, int8_t *x, size_t x_stride,
              int32_t *acc, size_t acc_stride) {
    const uint32_t R = spingalett_i8_interleaved_rows(rows), len = spingalett_i8_interleaved_len(n);
    const __m512i bias = _mm512_set1_epi8((char)0x80);
    for (uint32_t p = 0; p < SPINGALETT_I8_TILE; p++)
        for (uint32_t k = 0; k < len; k += 64u) {
            __mmask64 m = len - k >= 64u ? ~(__mmask64)0 : (((__mmask64)1 << (len - k)) - 1u);
            int8_t *d = x + (size_t)p * x_stride + k;
            _mm512_mask_storeu_epi8(d, m, _mm512_xor_si512(_mm512_maskz_loadu_epi8(m, d), bias));
        }
    const size_t wstride = (size_t)len * 16u;
    uint32_t j = 0;
    for (; j + 32u <= R; j += 32u)
        tile_32(weights + (size_t)(j / 16u) * wstride, wstride, len, (const uint8_t *)x, x_stride, sums + j, acc + j,
                acc_stride);
    if (j < R)
        tile_16(weights + (size_t)(j / 16u) * wstride, len, (const uint8_t *)x, x_stride, sums + j, acc + j, acc_stride);
}
#endif

#if defined(TILE_KERNEL) && TILE_KERNEL == 256
#define TILE6_EACH(M) M(0) M(1) M(2) M(3) M(4) M(5)
#define TILE6_ZERO(p) __m256i a##p = _mm256_setzero_si256(), b##p = a##p;
#define TILE6_STEP(p) \
    int32_t q##p; \
    memcpy(&q##p, x + (size_t)(p) * xs + k, 4); \
    __m256i u##p = _mm256_set1_epi32(q##p); \
    a##p = _mm256_dpbusd_avx_epi32(a##p, u##p, v0); \
    b##p = _mm256_dpbusd_avx_epi32(b##p, u##p, v1);
#define TILE6_STORE(p) \
    _mm256_storeu_si256((__m256i *)(void *)(acc + (size_t)(p) * acc_stride), _mm256_sub_epi32(a##p, c0)); \
    _mm256_storeu_si256((__m256i *)(void *)(acc + (size_t)(p) * acc_stride + 8), _mm256_sub_epi32(b##p, c1));

/* 16 rows (one block, two vectors of eight) against six vectors of the tile */
TILE_LOOP tile6_16(const int8_t *wp, uint32_t len, const uint8_t *x, size_t xs, const int32_t *sums, int32_t *acc,
                   size_t acc_stride) {
    TILE6_EACH(TILE6_ZERO)
    for (uint32_t k = 0; k < len; k += 4u, wp += 64) {
        __m256i v0 = _mm256_loadu_si256((const __m256i *)(const void *)wp);
        __m256i v1 = _mm256_loadu_si256((const __m256i *)(const void *)(wp + 32));
        TILE6_EACH(TILE6_STEP)
    }
    __m256i c0 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)(const void *)sums), 7);
    __m256i c1 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)(const void *)(sums + 8)), 7);
    TILE6_EACH(TILE6_STORE)
}

TILE_FUNCTION(const int8_t *weights, const int32_t *sums, uint32_t rows, uint32_t n, int8_t *x, size_t x_stride,
              int32_t *acc, size_t acc_stride) {
    const uint32_t R = spingalett_i8_interleaved_rows(rows), len = spingalett_i8_interleaved_len(n);
    const __m256i bias = _mm256_set1_epi8((char)0x80);
    for (uint32_t p = 0; p < SPINGALETT_I8_TILE; p++) {
        int8_t *d = x + (size_t)p * x_stride;
        uint32_t k = 0;
        for (; k + 32u <= len; k += 32u) {
            __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(d + k));
            _mm256_storeu_si256((__m256i *)(void *)(d + k), _mm256_xor_si256(v, bias));
        }
        for (; k < len; k++) d[k] = (int8_t)((uint8_t)d[k] ^ 0x80u);
    }
    const size_t wstride = (size_t)len * 16u;
    for (uint32_t j = 0; j < R; j += 16u)
        for (uint32_t p = 0; p < SPINGALETT_I8_TILE; p += 6u)
            tile6_16(weights + (size_t)(j / 16u) * wstride, len, (const uint8_t *)x + (size_t)p * x_stride, x_stride,
                     sums + j, acc + (size_t)p * acc_stride + j, acc_stride);
}
#endif

#if defined(TILE_KERNEL) && TILE_KERNEL == 128
#define TILEN_EACH(M) M(0) M(1) M(2) M(3) M(4) M(5)
#define TILEN_ZERO(p) int32x4_t a##p##0 = vdupq_n_s32(0), a##p##1 = a##p##0, a##p##2 = a##p##0, a##p##3 = a##p##0;
#define TILEN_STEP(p) \
    { \
        int32_t q; \
        memcpy(&q, x + (size_t)(p) * xs + k, 4); \
        int8x16_t u = vreinterpretq_s8_s32(vdupq_n_s32(q)); \
        a##p##0 = vdotq_s32(a##p##0, v0, u); \
        a##p##1 = vdotq_s32(a##p##1, v1, u); \
        a##p##2 = vdotq_s32(a##p##2, v2, u); \
        a##p##3 = vdotq_s32(a##p##3, v3, u); \
    }
#define TILEN_STORE(p) \
    vst1q_s32(acc + (size_t)(p) * acc_stride, a##p##0); \
    vst1q_s32(acc + (size_t)(p) * acc_stride + 4, a##p##1); \
    vst1q_s32(acc + (size_t)(p) * acc_stride + 8, a##p##2); \
    vst1q_s32(acc + (size_t)(p) * acc_stride + 12, a##p##3);

/* 16 rows (one block, four vectors of four) against six vectors of the tile */
TILE_LOOP tilen_16(const int8_t *wp, uint32_t len, const int8_t *x, size_t xs, int32_t *acc, size_t acc_stride) {
    TILEN_EACH(TILEN_ZERO)
    for (uint32_t k = 0; k < len; k += 4u, wp += 64) {
        int8x16_t v0 = vld1q_s8(wp), v1 = vld1q_s8(wp + 16), v2 = vld1q_s8(wp + 32), v3 = vld1q_s8(wp + 48);
        TILEN_EACH(TILEN_STEP)
    }
    TILEN_EACH(TILEN_STORE)
}

TILE_FUNCTION(const int8_t *weights, const int32_t *sums, uint32_t rows, uint32_t n, int8_t *x, size_t x_stride,
              int32_t *acc, size_t acc_stride) {
    (void)sums;
    const uint32_t R = spingalett_i8_interleaved_rows(rows), len = spingalett_i8_interleaved_len(n);
    const size_t wstride = (size_t)len * 16u;
    for (uint32_t j = 0; j < R; j += 16u)
        for (uint32_t p = 0; p < SPINGALETT_I8_TILE; p += 6u)
            tilen_16(weights + (size_t)(j / 16u) * wstride, len, x + (size_t)p * x_stride, x_stride,
                     acc + (size_t)p * acc_stride + j, acc_stride);
}
#endif

#if !defined(SPINGALETT_I8_TILE_VARIANT)
void spingalett_i8_interleave(const int8_t *w, size_t stride, uint32_t rows, uint32_t n, int8_t *out, int32_t *sums) {
    const uint32_t R = spingalett_i8_interleaved_rows(rows), len = spingalett_i8_interleaved_len(n);
    for (uint32_t j = 0; j < R; j++) {
        int8_t *d = out + (size_t)(j / 16u) * len * 16u + (j % 16u) * 4u;
        int32_t sum = 0;
        for (uint32_t k = 0; k < len; k++) {
            int8_t v = j < rows && k < n ? w[(size_t)j * stride + k] : 0;
            d[(size_t)(k / 4u) * 64u + k % 4u] = v;
            sum += v;
        }
        sums[j] = sum;
    }
}

#if defined(TILE_DISPATCH_X86)
/* The kernel this processor runs: 512 (AVX-512 VNNI), 256 (AVX-VNNI) or 0. (Compilers do not all
   know AVX-VNNI by name: CPUID leaf 7, subleaf 1, EAX bit 4; AVX2 says the system saves the 256-bit
   registers.) */
static int tile_kernel(void) {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512bw")) return 512;
    unsigned a, b, c, d;
    if (__builtin_cpu_supports("avx2") && __get_cpuid_count(7, 1, &a, &b, &c, &d) && (a & (1u << 4))) return 256;
    return 0;
}
#elif defined(TILE_DISPATCH_ARM)
/* 128 (the dot product instructions) or 0 */
static int tile_kernel(void) {
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) ? 128 : 0;
}
#endif

bool spingalett_i8_interleaved(void) {
#if defined(TILE_KERNEL)
    return true;
#elif defined(TILE_DISPATCH_X86) || defined(TILE_DISPATCH_ARM)
    return tile_kernel() != 0;
#else
    return false;
#endif
}

void spingalett_i8_interleaved_tile(const int8_t *weights, const int32_t *sums, uint32_t rows, uint32_t n, int8_t *x,
                                    size_t x_stride, int32_t *acc, size_t acc_stride) {
#if defined(TILE_KERNEL)
    tile_native(weights, sums, rows, n, x, x_stride, acc, acc_stride);
#elif defined(TILE_DISPATCH_X86)
    int kernel = tile_kernel();
    if (kernel == 512) spingalett_i8_tile_avx512(weights, sums, rows, n, x, x_stride, acc, acc_stride);
    else if (kernel == 256) spingalett_i8_tile_avxvnni(weights, sums, rows, n, x, x_stride, acc, acc_stride);
#elif defined(TILE_DISPATCH_ARM)
    if (tile_kernel()) spingalett_i8_tile_dotprod(weights, sums, rows, n, x, x_stride, acc, acc_stride);
#else
    (void)weights; (void)sums; (void)rows; (void)n; (void)x; (void)x_stride; (void)acc; (void)acc_stride;
#endif
}
#endif
