/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The inference engine (Spingalett.Inference.h) and the parts of the .slett format versions 3 to 6
 * the rest of the library shares with it (Spingalett.Engine.h). With -DSPINGALETT_INFERENCE_ONLY
 * this file builds on its own and calls nothing beyond memcpy, memset, memcmp, expf, tanhf, sqrtf
 * and lrintf.
 *
 * Kernels: AVX2 (with VNNI where the compiler targets it), AVX, SSE2, NEON (with the dot-product
 * extension where available), the Arm DSP extension (Cortex-M4, M7, M33) and portable C. Defining
 * SPINGALETT_PORTABLE_KERNELS selects the portable C ones only.
 */

#if defined(SPINGALETT_INFERENCE_ONLY)
#include "Spingalett.Engine.h"
#define ENGINE_FAIL(code, msg) (code)
#else
#include "Spingalett.Private.h"
#include <stdlib.h>
static int engine_fail(int code, const char *msg) { set_error(code, msg); return code; }
#define ENGINE_FAIL(code, msg) engine_fail((code), (msg))
#endif

#include <math.h>
#include <float.h>
#include <string.h>

#if !defined(SPINGALETT_PORTABLE_KERNELS)
#  if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#    include <immintrin.h>
#    define SPG_SSE2 1
#    if defined(__AVX__)
#      define SPG_AVX 1
#    endif
#    if defined(__AVX2__)
#      define SPG_AVX2 1
#    endif
#    if defined(__AVX__) && defined(__F16C__)
#      define SPG_F16C 1
#    endif
#    if defined(__AVX512VNNI__) && defined(__AVX512BW__)
#      define SPG_AVX512_VNNI 1
#    endif
#    if defined(__AVX2__) && defined(__AVXVNNI__)
#      define SPG_VNNI(acc, u, s) _mm256_dpbusd_avx_epi32((acc), (u), (s))
#      define SPG_VNNI128(acc, u, s) _mm_dpbusd_avx_epi32((acc), (u), (s))
#    elif defined(__AVX2__) && defined(__AVX512VNNI__) && defined(__AVX512VL__)
#      define SPG_VNNI(acc, u, s) _mm256_dpbusd_epi32((acc), (u), (s))
#      define SPG_VNNI128(acc, u, s) _mm_dpbusd_epi32((acc), (u), (s))
#    endif
#  elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#    include <arm_neon.h>
#    define SPG_NEON 1
#  elif defined(__ARM_FEATURE_SIMD32)
#    include <arm_acle.h>
#    define SPG_ARM_DSP 1
#  endif
#endif

/* Kernels whose loops carry many sums in registers: GCC's partial redundancy elimination has it
   copy them between registers, or through memory, on every step (dpbusd adds into its first
   operand), so it is off for these. Arrays of sums end up in memory too: kernels name them. */
#if defined(__GNUC__) && !defined(__clang__)
#  define SPG_REGISTER_SUMS __attribute__((optimize("no-tree-pre")))
#else
#  define SPG_REGISTER_SUMS
#endif

#if defined(SPG_AVX) && defined(__FMA__)
#  define SPG_FMADD256(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#elif defined(SPG_AVX)
#  define SPG_FMADD256(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif

/* ------------------------------------------------------------------------- shared helpers */

bool spingalett_host_is_little_endian(void) {
    const uint16_t one = 1;
    uint8_t first;
    memcpy(&first, &one, 1);
    return first == 1;
}

static const uint32_t crc_table[256] = {
    0x00000000u, 0x77073096u, 0xEE0E612Cu, 0x990951BAu, 0x076DC419u, 0x706AF48Fu,
    0xE963A535u, 0x9E6495A3u, 0x0EDB8832u, 0x79DCB8A4u, 0xE0D5E91Eu, 0x97D2D988u,
    0x09B64C2Bu, 0x7EB17CBDu, 0xE7B82D07u, 0x90BF1D91u, 0x1DB71064u, 0x6AB020F2u,
    0xF3B97148u, 0x84BE41DEu, 0x1ADAD47Du, 0x6DDDE4EBu, 0xF4D4B551u, 0x83D385C7u,
    0x136C9856u, 0x646BA8C0u, 0xFD62F97Au, 0x8A65C9ECu, 0x14015C4Fu, 0x63066CD9u,
    0xFA0F3D63u, 0x8D080DF5u, 0x3B6E20C8u, 0x4C69105Eu, 0xD56041E4u, 0xA2677172u,
    0x3C03E4D1u, 0x4B04D447u, 0xD20D85FDu, 0xA50AB56Bu, 0x35B5A8FAu, 0x42B2986Cu,
    0xDBBBC9D6u, 0xACBCF940u, 0x32D86CE3u, 0x45DF5C75u, 0xDCD60DCFu, 0xABD13D59u,
    0x26D930ACu, 0x51DE003Au, 0xC8D75180u, 0xBFD06116u, 0x21B4F4B5u, 0x56B3C423u,
    0xCFBA9599u, 0xB8BDA50Fu, 0x2802B89Eu, 0x5F058808u, 0xC60CD9B2u, 0xB10BE924u,
    0x2F6F7C87u, 0x58684C11u, 0xC1611DABu, 0xB6662D3Du, 0x76DC4190u, 0x01DB7106u,
    0x98D220BCu, 0xEFD5102Au, 0x71B18589u, 0x06B6B51Fu, 0x9FBFE4A5u, 0xE8B8D433u,
    0x7807C9A2u, 0x0F00F934u, 0x9609A88Eu, 0xE10E9818u, 0x7F6A0DBBu, 0x086D3D2Du,
    0x91646C97u, 0xE6635C01u, 0x6B6B51F4u, 0x1C6C6162u, 0x856530D8u, 0xF262004Eu,
    0x6C0695EDu, 0x1B01A57Bu, 0x8208F4C1u, 0xF50FC457u, 0x65B0D9C6u, 0x12B7E950u,
    0x8BBEB8EAu, 0xFCB9887Cu, 0x62DD1DDFu, 0x15DA2D49u, 0x8CD37CF3u, 0xFBD44C65u,
    0x4DB26158u, 0x3AB551CEu, 0xA3BC0074u, 0xD4BB30E2u, 0x4ADFA541u, 0x3DD895D7u,
    0xA4D1C46Du, 0xD3D6F4FBu, 0x4369E96Au, 0x346ED9FCu, 0xAD678846u, 0xDA60B8D0u,
    0x44042D73u, 0x33031DE5u, 0xAA0A4C5Fu, 0xDD0D7CC9u, 0x5005713Cu, 0x270241AAu,
    0xBE0B1010u, 0xC90C2086u, 0x5768B525u, 0x206F85B3u, 0xB966D409u, 0xCE61E49Fu,
    0x5EDEF90Eu, 0x29D9C998u, 0xB0D09822u, 0xC7D7A8B4u, 0x59B33D17u, 0x2EB40D81u,
    0xB7BD5C3Bu, 0xC0BA6CADu, 0xEDB88320u, 0x9ABFB3B6u, 0x03B6E20Cu, 0x74B1D29Au,
    0xEAD54739u, 0x9DD277AFu, 0x04DB2615u, 0x73DC1683u, 0xE3630B12u, 0x94643B84u,
    0x0D6D6A3Eu, 0x7A6A5AA8u, 0xE40ECF0Bu, 0x9309FF9Du, 0x0A00AE27u, 0x7D079EB1u,
    0xF00F9344u, 0x8708A3D2u, 0x1E01F268u, 0x6906C2FEu, 0xF762575Du, 0x806567CBu,
    0x196C3671u, 0x6E6B06E7u, 0xFED41B76u, 0x89D32BE0u, 0x10DA7A5Au, 0x67DD4ACCu,
    0xF9B9DF6Fu, 0x8EBEEFF9u, 0x17B7BE43u, 0x60B08ED5u, 0xD6D6A3E8u, 0xA1D1937Eu,
    0x38D8C2C4u, 0x4FDFF252u, 0xD1BB67F1u, 0xA6BC5767u, 0x3FB506DDu, 0x48B2364Bu,
    0xD80D2BDAu, 0xAF0A1B4Cu, 0x36034AF6u, 0x41047A60u, 0xDF60EFC3u, 0xA867DF55u,
    0x316E8EEFu, 0x4669BE79u, 0xCB61B38Cu, 0xBC66831Au, 0x256FD2A0u, 0x5268E236u,
    0xCC0C7795u, 0xBB0B4703u, 0x220216B9u, 0x5505262Fu, 0xC5BA3BBEu, 0xB2BD0B28u,
    0x2BB45A92u, 0x5CB36A04u, 0xC2D7FFA7u, 0xB5D0CF31u, 0x2CD99E8Bu, 0x5BDEAE1Du,
    0x9B64C2B0u, 0xEC63F226u, 0x756AA39Cu, 0x026D930Au, 0x9C0906A9u, 0xEB0E363Fu,
    0x72076785u, 0x05005713u, 0x95BF4A82u, 0xE2B87A14u, 0x7BB12BAEu, 0x0CB61B38u,
    0x92D28E9Bu, 0xE5D5BE0Du, 0x7CDCEFB7u, 0x0BDBDF21u, 0x86D3D2D4u, 0xF1D4E242u,
    0x68DDB3F8u, 0x1FDA836Eu, 0x81BE16CDu, 0xF6B9265Bu, 0x6FB077E1u, 0x18B74777u,
    0x88085AE6u, 0xFF0F6A70u, 0x66063BCAu, 0x11010B5Cu, 0x8F659EFFu, 0xF862AE69u,
    0x616BFFD3u, 0x166CCF45u, 0xA00AE278u, 0xD70DD2EEu, 0x4E048354u, 0x3903B3C2u,
    0xA7672661u, 0xD06016F7u, 0x4969474Du, 0x3E6E77DBu, 0xAED16A4Au, 0xD9D65ADCu,
    0x40DF0B66u, 0x37D83BF0u, 0xA9BCAE53u, 0xDEBB9EC5u, 0x47B2CF7Fu, 0x30B5FFE9u,
    0xBDBDF21Cu, 0xCABAC28Au, 0x53B39330u, 0x24B4A3A6u, 0xBAD03605u, 0xCDD70693u,
    0x54DE5729u, 0x23D967BFu, 0xB3667A2Eu, 0xC4614AB8u, 0x5D681B02u, 0x2A6F2B94u,
    0xB40BBE37u, 0xC30C8EA1u, 0x5A05DF1Bu, 0x2D02EF8Du,
};

#if defined(SPINGALETT_INFERENCE_ONLY)
uint32_t spingalett_crc32(uint32_t crc, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    crc = ~crc;
    for (size_t i = 0; i < n; i++) crc = crc_table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}
#else
/* The library checks whole model and data set files: eight bytes at a time ("slicing by 8"),
   with crc_slices[k][b] the CRC of byte b followed by k zero bytes, built from crc_table on first
   use (once, whichever thread gets there first). The engine alone keeps the byte-wise loop and its
   single table. */
#include <stdatomic.h>

static uint32_t crc_slices[8][256];
static atomic_int crc_slices_state;     /* 0: not built, 1: being built, 2: ready */

static void crc_slices_build(void) {
    if (atomic_load_explicit(&crc_slices_state, memory_order_acquire) == 2) return;
    int expected = 0;
    if (atomic_compare_exchange_strong(&crc_slices_state, &expected, 1)) {
        for (int b = 0; b < 256; b++) {
            uint32_t c = crc_table[b];
            crc_slices[0][b] = c;
            for (int k = 1; k < 8; k++) crc_slices[k][b] = c = crc_table[c & 0xFFu] ^ (c >> 8);
        }
        atomic_store_explicit(&crc_slices_state, 2, memory_order_release);
        return;
    }
    while (atomic_load_explicit(&crc_slices_state, memory_order_acquire) != 2) {}     /* microseconds */
}

uint32_t spingalett_crc32(uint32_t crc, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    crc = ~crc;
    if (n >= 64) {
        crc_slices_build();
        const uint32_t (*t)[256] = (const uint32_t (*)[256])crc_slices;
        for (; n >= 8; n -= 8, p += 8) {
            uint32_t a = crc ^ ((uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24);
            uint32_t b = (uint32_t)p[4] | (uint32_t)p[5] << 8 | (uint32_t)p[6] << 16 | (uint32_t)p[7] << 24;
            crc = t[7][a & 0xFFu] ^ t[6][(a >> 8) & 0xFFu] ^ t[5][(a >> 16) & 0xFFu] ^ t[4][a >> 24] ^
                  t[3][b & 0xFFu] ^ t[2][(b >> 8) & 0xFFu] ^ t[1][(b >> 16) & 0xFFu] ^ t[0][b >> 24];
        }
    }
    for (; n > 0; n--, p++) crc = crc_table[(crc ^ *p) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}
#endif

uint16_t spingalett_float_to_fp16(float x) {
    uint32_t f;
    memcpy(&f, &x, sizeof(float));

    uint16_t sign = (uint16_t)((f >> 16) & 0x8000u);
    uint32_t absf = f & 0x7FFFFFFFu;

    if (absf >= 0x7F800000u)                     /* Inf / NaN */
        return (uint16_t)(sign | (absf > 0x7F800000u ? 0x7E00u : 0x7C00u));
    if (absf >= 0x477FF000u)                     /* >= 65520 rounds to Inf */
        return (uint16_t)(sign | 0x7C00u);

    if (absf < 0x38800000u) {                    /* below 2^-14: FP16 subnormal or zero */
        if (absf < 0x33000000u)                  /* below 2^-25 rounds to zero */
            return sign;
        uint32_t shift = 126u - (absf >> 23);    /* 14..24 */
        uint32_t man   = (absf & 0x007FFFFFu) | 0x00800000u;
        uint32_t h     = man >> shift;
        uint32_t rem   = man & ((1u << shift) - 1u);
        uint32_t half  = 1u << (shift - 1u);
        if (rem > half || (rem == half && (h & 1u))) h++;
        return (uint16_t)(sign | h);
    }

    /* Normal: rebias the exponent, round the dropped 13 mantissa bits to nearest-even.
       A carry out of the mantissa correctly bumps the exponent. */
    uint32_t h   = (absf - 0x38000000u) >> 13;
    uint32_t rem = absf & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
    return (uint16_t)(sign | h);
}

float spingalett_fp16_to_float(uint16_t h) {
    uint32_t sign  = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t h_exp = (h >> 10) & 0x1Fu;
    uint32_t h_man = h & 0x03FFu;
    uint32_t f;

    if (h_exp == 0) {
        if (h_man == 0) {
            f = sign;
        } else {
            h_exp = 1;
            while (!(h_man & 0x0400u)) {
                h_man <<= 1;
                h_exp++;
            }
            h_man &= 0x03FFu;
            f = sign | ((uint32_t)(114 - h_exp) << 23) | (h_man << 13);
        }
    } else if (h_exp == 31) {
        f = (h_man == 0) ? (sign | 0x7F800000u) : (sign | 0x7FC00000u);
    } else {
        f = sign | ((h_exp + 112) << 23) | (h_man << 13);
    }

    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

uint16_t spingalett_float_to_bf16(float x) {
    uint32_t f;
    memcpy(&f, &x, sizeof(float));
    if ((f & 0x7FFFFFFFu) > 0x7F800000u)         /* NaN: keep it a (quiet) NaN */
        return (uint16_t)((f >> 16) | 0x0040u);
    f += 0x7FFFu + ((f >> 16) & 1u);             /* round to nearest-even */
    return (uint16_t)(f >> 16);
}

float spingalett_bf16_to_float(uint16_t h) {
    uint32_t f = ((uint32_t)h) << 16;
    float x;
    memcpy(&x, &f, sizeof(float));
    return x;
}

/* Signed values of the two INT4 codes (low nibble first) and four INT2 codes (low bits first) of
   every byte. */
static const int8_t int4_codes[256][2] = {
    {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0},
    {-8, 0}, {-7, 0}, {-6, 0}, {-5, 0}, {-4, 0}, {-3, 0}, {-2, 0}, {-1, 0},
    {0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}, {6, 1}, {7, 1},
    {-8, 1}, {-7, 1}, {-6, 1}, {-5, 1}, {-4, 1}, {-3, 1}, {-2, 1}, {-1, 1},
    {0, 2}, {1, 2}, {2, 2}, {3, 2}, {4, 2}, {5, 2}, {6, 2}, {7, 2},
    {-8, 2}, {-7, 2}, {-6, 2}, {-5, 2}, {-4, 2}, {-3, 2}, {-2, 2}, {-1, 2},
    {0, 3}, {1, 3}, {2, 3}, {3, 3}, {4, 3}, {5, 3}, {6, 3}, {7, 3},
    {-8, 3}, {-7, 3}, {-6, 3}, {-5, 3}, {-4, 3}, {-3, 3}, {-2, 3}, {-1, 3},
    {0, 4}, {1, 4}, {2, 4}, {3, 4}, {4, 4}, {5, 4}, {6, 4}, {7, 4},
    {-8, 4}, {-7, 4}, {-6, 4}, {-5, 4}, {-4, 4}, {-3, 4}, {-2, 4}, {-1, 4},
    {0, 5}, {1, 5}, {2, 5}, {3, 5}, {4, 5}, {5, 5}, {6, 5}, {7, 5},
    {-8, 5}, {-7, 5}, {-6, 5}, {-5, 5}, {-4, 5}, {-3, 5}, {-2, 5}, {-1, 5},
    {0, 6}, {1, 6}, {2, 6}, {3, 6}, {4, 6}, {5, 6}, {6, 6}, {7, 6},
    {-8, 6}, {-7, 6}, {-6, 6}, {-5, 6}, {-4, 6}, {-3, 6}, {-2, 6}, {-1, 6},
    {0, 7}, {1, 7}, {2, 7}, {3, 7}, {4, 7}, {5, 7}, {6, 7}, {7, 7},
    {-8, 7}, {-7, 7}, {-6, 7}, {-5, 7}, {-4, 7}, {-3, 7}, {-2, 7}, {-1, 7},
    {0, -8}, {1, -8}, {2, -8}, {3, -8}, {4, -8}, {5, -8}, {6, -8}, {7, -8},
    {-8, -8}, {-7, -8}, {-6, -8}, {-5, -8}, {-4, -8}, {-3, -8}, {-2, -8}, {-1, -8},
    {0, -7}, {1, -7}, {2, -7}, {3, -7}, {4, -7}, {5, -7}, {6, -7}, {7, -7},
    {-8, -7}, {-7, -7}, {-6, -7}, {-5, -7}, {-4, -7}, {-3, -7}, {-2, -7}, {-1, -7},
    {0, -6}, {1, -6}, {2, -6}, {3, -6}, {4, -6}, {5, -6}, {6, -6}, {7, -6},
    {-8, -6}, {-7, -6}, {-6, -6}, {-5, -6}, {-4, -6}, {-3, -6}, {-2, -6}, {-1, -6},
    {0, -5}, {1, -5}, {2, -5}, {3, -5}, {4, -5}, {5, -5}, {6, -5}, {7, -5},
    {-8, -5}, {-7, -5}, {-6, -5}, {-5, -5}, {-4, -5}, {-3, -5}, {-2, -5}, {-1, -5},
    {0, -4}, {1, -4}, {2, -4}, {3, -4}, {4, -4}, {5, -4}, {6, -4}, {7, -4},
    {-8, -4}, {-7, -4}, {-6, -4}, {-5, -4}, {-4, -4}, {-3, -4}, {-2, -4}, {-1, -4},
    {0, -3}, {1, -3}, {2, -3}, {3, -3}, {4, -3}, {5, -3}, {6, -3}, {7, -3},
    {-8, -3}, {-7, -3}, {-6, -3}, {-5, -3}, {-4, -3}, {-3, -3}, {-2, -3}, {-1, -3},
    {0, -2}, {1, -2}, {2, -2}, {3, -2}, {4, -2}, {5, -2}, {6, -2}, {7, -2},
    {-8, -2}, {-7, -2}, {-6, -2}, {-5, -2}, {-4, -2}, {-3, -2}, {-2, -2}, {-1, -2},
    {0, -1}, {1, -1}, {2, -1}, {3, -1}, {4, -1}, {5, -1}, {6, -1}, {7, -1},
    {-8, -1}, {-7, -1}, {-6, -1}, {-5, -1}, {-4, -1}, {-3, -1}, {-2, -1}, {-1, -1},
};
static const int8_t int2_codes[256][4] = {
    {0, 0, 0, 0}, {1, 0, 0, 0}, {-2, 0, 0, 0}, {-1, 0, 0, 0},
    {0, 1, 0, 0}, {1, 1, 0, 0}, {-2, 1, 0, 0}, {-1, 1, 0, 0},
    {0, -2, 0, 0}, {1, -2, 0, 0}, {-2, -2, 0, 0}, {-1, -2, 0, 0},
    {0, -1, 0, 0}, {1, -1, 0, 0}, {-2, -1, 0, 0}, {-1, -1, 0, 0},
    {0, 0, 1, 0}, {1, 0, 1, 0}, {-2, 0, 1, 0}, {-1, 0, 1, 0},
    {0, 1, 1, 0}, {1, 1, 1, 0}, {-2, 1, 1, 0}, {-1, 1, 1, 0},
    {0, -2, 1, 0}, {1, -2, 1, 0}, {-2, -2, 1, 0}, {-1, -2, 1, 0},
    {0, -1, 1, 0}, {1, -1, 1, 0}, {-2, -1, 1, 0}, {-1, -1, 1, 0},
    {0, 0, -2, 0}, {1, 0, -2, 0}, {-2, 0, -2, 0}, {-1, 0, -2, 0},
    {0, 1, -2, 0}, {1, 1, -2, 0}, {-2, 1, -2, 0}, {-1, 1, -2, 0},
    {0, -2, -2, 0}, {1, -2, -2, 0}, {-2, -2, -2, 0}, {-1, -2, -2, 0},
    {0, -1, -2, 0}, {1, -1, -2, 0}, {-2, -1, -2, 0}, {-1, -1, -2, 0},
    {0, 0, -1, 0}, {1, 0, -1, 0}, {-2, 0, -1, 0}, {-1, 0, -1, 0},
    {0, 1, -1, 0}, {1, 1, -1, 0}, {-2, 1, -1, 0}, {-1, 1, -1, 0},
    {0, -2, -1, 0}, {1, -2, -1, 0}, {-2, -2, -1, 0}, {-1, -2, -1, 0},
    {0, -1, -1, 0}, {1, -1, -1, 0}, {-2, -1, -1, 0}, {-1, -1, -1, 0},
    {0, 0, 0, 1}, {1, 0, 0, 1}, {-2, 0, 0, 1}, {-1, 0, 0, 1},
    {0, 1, 0, 1}, {1, 1, 0, 1}, {-2, 1, 0, 1}, {-1, 1, 0, 1},
    {0, -2, 0, 1}, {1, -2, 0, 1}, {-2, -2, 0, 1}, {-1, -2, 0, 1},
    {0, -1, 0, 1}, {1, -1, 0, 1}, {-2, -1, 0, 1}, {-1, -1, 0, 1},
    {0, 0, 1, 1}, {1, 0, 1, 1}, {-2, 0, 1, 1}, {-1, 0, 1, 1},
    {0, 1, 1, 1}, {1, 1, 1, 1}, {-2, 1, 1, 1}, {-1, 1, 1, 1},
    {0, -2, 1, 1}, {1, -2, 1, 1}, {-2, -2, 1, 1}, {-1, -2, 1, 1},
    {0, -1, 1, 1}, {1, -1, 1, 1}, {-2, -1, 1, 1}, {-1, -1, 1, 1},
    {0, 0, -2, 1}, {1, 0, -2, 1}, {-2, 0, -2, 1}, {-1, 0, -2, 1},
    {0, 1, -2, 1}, {1, 1, -2, 1}, {-2, 1, -2, 1}, {-1, 1, -2, 1},
    {0, -2, -2, 1}, {1, -2, -2, 1}, {-2, -2, -2, 1}, {-1, -2, -2, 1},
    {0, -1, -2, 1}, {1, -1, -2, 1}, {-2, -1, -2, 1}, {-1, -1, -2, 1},
    {0, 0, -1, 1}, {1, 0, -1, 1}, {-2, 0, -1, 1}, {-1, 0, -1, 1},
    {0, 1, -1, 1}, {1, 1, -1, 1}, {-2, 1, -1, 1}, {-1, 1, -1, 1},
    {0, -2, -1, 1}, {1, -2, -1, 1}, {-2, -2, -1, 1}, {-1, -2, -1, 1},
    {0, -1, -1, 1}, {1, -1, -1, 1}, {-2, -1, -1, 1}, {-1, -1, -1, 1},
    {0, 0, 0, -2}, {1, 0, 0, -2}, {-2, 0, 0, -2}, {-1, 0, 0, -2},
    {0, 1, 0, -2}, {1, 1, 0, -2}, {-2, 1, 0, -2}, {-1, 1, 0, -2},
    {0, -2, 0, -2}, {1, -2, 0, -2}, {-2, -2, 0, -2}, {-1, -2, 0, -2},
    {0, -1, 0, -2}, {1, -1, 0, -2}, {-2, -1, 0, -2}, {-1, -1, 0, -2},
    {0, 0, 1, -2}, {1, 0, 1, -2}, {-2, 0, 1, -2}, {-1, 0, 1, -2},
    {0, 1, 1, -2}, {1, 1, 1, -2}, {-2, 1, 1, -2}, {-1, 1, 1, -2},
    {0, -2, 1, -2}, {1, -2, 1, -2}, {-2, -2, 1, -2}, {-1, -2, 1, -2},
    {0, -1, 1, -2}, {1, -1, 1, -2}, {-2, -1, 1, -2}, {-1, -1, 1, -2},
    {0, 0, -2, -2}, {1, 0, -2, -2}, {-2, 0, -2, -2}, {-1, 0, -2, -2},
    {0, 1, -2, -2}, {1, 1, -2, -2}, {-2, 1, -2, -2}, {-1, 1, -2, -2},
    {0, -2, -2, -2}, {1, -2, -2, -2}, {-2, -2, -2, -2}, {-1, -2, -2, -2},
    {0, -1, -2, -2}, {1, -1, -2, -2}, {-2, -1, -2, -2}, {-1, -1, -2, -2},
    {0, 0, -1, -2}, {1, 0, -1, -2}, {-2, 0, -1, -2}, {-1, 0, -1, -2},
    {0, 1, -1, -2}, {1, 1, -1, -2}, {-2, 1, -1, -2}, {-1, 1, -1, -2},
    {0, -2, -1, -2}, {1, -2, -1, -2}, {-2, -2, -1, -2}, {-1, -2, -1, -2},
    {0, -1, -1, -2}, {1, -1, -1, -2}, {-2, -1, -1, -2}, {-1, -1, -1, -2},
    {0, 0, 0, -1}, {1, 0, 0, -1}, {-2, 0, 0, -1}, {-1, 0, 0, -1},
    {0, 1, 0, -1}, {1, 1, 0, -1}, {-2, 1, 0, -1}, {-1, 1, 0, -1},
    {0, -2, 0, -1}, {1, -2, 0, -1}, {-2, -2, 0, -1}, {-1, -2, 0, -1},
    {0, -1, 0, -1}, {1, -1, 0, -1}, {-2, -1, 0, -1}, {-1, -1, 0, -1},
    {0, 0, 1, -1}, {1, 0, 1, -1}, {-2, 0, 1, -1}, {-1, 0, 1, -1},
    {0, 1, 1, -1}, {1, 1, 1, -1}, {-2, 1, 1, -1}, {-1, 1, 1, -1},
    {0, -2, 1, -1}, {1, -2, 1, -1}, {-2, -2, 1, -1}, {-1, -2, 1, -1},
    {0, -1, 1, -1}, {1, -1, 1, -1}, {-2, -1, 1, -1}, {-1, -1, 1, -1},
    {0, 0, -2, -1}, {1, 0, -2, -1}, {-2, 0, -2, -1}, {-1, 0, -2, -1},
    {0, 1, -2, -1}, {1, 1, -2, -1}, {-2, 1, -2, -1}, {-1, 1, -2, -1},
    {0, -2, -2, -1}, {1, -2, -2, -1}, {-2, -2, -2, -1}, {-1, -2, -2, -1},
    {0, -1, -2, -1}, {1, -1, -2, -1}, {-2, -1, -2, -1}, {-1, -1, -2, -1},
    {0, 0, -1, -1}, {1, 0, -1, -1}, {-2, 0, -1, -1}, {-1, 0, -1, -1},
    {0, 1, -1, -1}, {1, 1, -1, -1}, {-2, 1, -1, -1}, {-1, 1, -1, -1},
    {0, -2, -1, -1}, {1, -2, -1, -1}, {-2, -2, -1, -1}, {-1, -2, -1, -1},
    {0, -1, -1, -1}, {1, -1, -1, -1}, {-2, -1, -1, -1}, {-1, -1, -1, -1},
};

void spingalett_unpack_int4(const uint8_t *src, int8_t *dst, uint32_t n) {
    uint32_t i = 0;
    for (; i + 2u <= n; i += 2u) memcpy(dst + i, int4_codes[src[i >> 1]], 2);
    if (i < n) dst[i] = int4_codes[src[i >> 1]][0];
}

void spingalett_unpack_int2(const uint8_t *src, int8_t *dst, uint32_t n) {
    uint32_t i = 0;
    for (; i + 4u <= n; i += 4u) memcpy(dst + i, int2_codes[src[i >> 2]], 4);
    for (; i < n; i++) dst[i] = int2_codes[src[i >> 2]][i & 3u];
}

/* Half to float without branches: the exponent and mantissa bits shifted into float position and
   scaled by 2^112 give normal and subnormal values exactly; Inf and NaN get the all-ones exponent. */
static inline float half_to_float_fast(uint16_t h) {
    uint32_t em = h & 0x7FFFu, bits = em << 13;
    float f;
    memcpy(&f, &bits, 4);
    f *= 5.192296858534828e33f;                  /* 2^112 */
    memcpy(&bits, &f, 4);
    if (em > 0x7BFFu) bits |= 0x7F800000u;
    bits |= (uint32_t)(h & 0x8000u) << 16;
    memcpy(&f, &bits, 4);
    return f;
}

uint64_t spingalett_slett_row_bytes(PrecisionMode precision, uint32_t inputs) {
    switch (precision) {
        case PRECISION_FLOAT32:  return (uint64_t)inputs * 4u;
        case PRECISION_FP16:
        case PRECISION_BFLOAT16: return (uint64_t)inputs * 2u;
        case PRECISION_INT8:     return inputs;
        case PRECISION_INT4:     return ((uint64_t)inputs + 1u) / 2u;
        case PRECISION_INT2:     return ((uint64_t)inputs + 3u) / 4u;
        default:                 return 0;
    }
}

/* ------------------------------------------------------------------------- kernels */

#if defined(SPG_NEON)
static inline int32x4_t neon_dot_i8(int32x4_t acc, int8x16_t a, int8x16_t b) {
#  if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, a, b);
#  else
    int16x8_t p = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    p = vmlal_s8(p, vget_high_s8(a), vget_high_s8(b));      /* two products stay in int16 */
    return vpadalq_s16(acc, p);
#  endif
}

static inline int32_t neon_hsum(int32x4_t v) {
#  if defined(__aarch64__)
    return vaddvq_s32(v);
#  else
    int32x2_t s2 = vadd_s32(vget_low_s32(v), vget_high_s32(v));
    return vget_lane_s32(vpadd_s32(s2, s2), 0);
#  endif
}

#endif

#if defined(SPG_AVX2)
/* acc += |x| * (w with the sign of x): maddubs multiplies unsigned by signed bytes, and pairs of
   products stay below 2 * 127 * 127, so its 16-bit sums cannot saturate. */
static inline __m256i madd_i8(__m256i acc, __m256i ax, __m256i w, __m256i x) {
#  if defined(SPG_VNNI)
    return SPG_VNNI(acc, ax, _mm256_sign_epi8(w, x));
#  else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(ax, _mm256_sign_epi8(w, x)), _mm256_set1_epi16(1)));
#  endif
}

/* The same for 16 bytes, for the tail of a row. */
static inline __m128i madd_i8_128(__m128i acc, __m128i w, __m128i x) {
    __m128i ax = _mm_sign_epi8(x, x);
#  if defined(SPG_VNNI128)
    return SPG_VNNI128(acc, ax, _mm_sign_epi8(w, x));
#  else
    return _mm_add_epi32(acc, _mm_madd_epi16(_mm_maddubs_epi16(ax, _mm_sign_epi8(w, x)), _mm_set1_epi16(1)));
#  endif
}

static inline int32_t hsum_epi32_128(__m128i s) {
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

/* The sums of four accumulators: lane r of the result is the total of a_r. */
static inline __m128i reduce_rows4(__m256i a0, __m256i a1, __m256i a2, __m256i a3) {
    __m256i t = _mm256_hadd_epi32(_mm256_hadd_epi32(a0, a1), _mm256_hadd_epi32(a2, a3));
    return _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
}
#endif

SPG_REGISTER_SUMS int32_t spingalett_dot_i8(const int8_t *a, const int8_t *b, uint32_t n) {
    uint32_t k = 0;
    int32_t sum = 0;
#if defined(SPG_AVX2)
    __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
    for (; k + 64u <= n; k += 64u) {
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + k)), b0 = _mm256_loadu_si256((const __m256i *)(b + k));
        __m256i a1 = _mm256_loadu_si256((const __m256i *)(a + k + 32)), b1 = _mm256_loadu_si256((const __m256i *)(b + k + 32));
        acc0 = madd_i8(acc0, _mm256_sign_epi8(b0, b0), a0, b0);
        acc1 = madd_i8(acc1, _mm256_sign_epi8(b1, b1), a1, b1);
    }
    for (; k + 32u <= n; k += 32u) {
        __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + k)), b0 = _mm256_loadu_si256((const __m256i *)(b + k));
        acc0 = madd_i8(acc0, _mm256_sign_epi8(b0, b0), a0, b0);
    }
    acc0 = _mm256_add_epi32(acc0, acc1);
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc0), _mm256_extracti128_si256(acc0, 1));
    if (k + 16u <= n) {
        s = madd_i8_128(s, _mm_loadu_si128((const __m128i *)(a + k)), _mm_loadu_si128((const __m128i *)(b + k)));
        k += 16u;
    }
    sum = hsum_epi32_128(s);
#elif defined(SPG_SSE2)
    __m128i acc = _mm_setzero_si128();
    for (; k + 16u <= n; k += 16u) {
        __m128i va = _mm_loadu_si128((const __m128i *)(a + k)), vb = _mm_loadu_si128((const __m128i *)(b + k));
        __m128i al = _mm_srai_epi16(_mm_unpacklo_epi8(va, va), 8), ah = _mm_srai_epi16(_mm_unpackhi_epi8(va, va), 8);
        __m128i bl = _mm_srai_epi16(_mm_unpacklo_epi8(vb, vb), 8), bh = _mm_srai_epi16(_mm_unpackhi_epi8(vb, vb), 8);
        acc = _mm_add_epi32(acc, _mm_add_epi32(_mm_madd_epi16(al, bl), _mm_madd_epi16(ah, bh)));
    }
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0x4E));
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, 0xB1));
    sum = _mm_cvtsi128_si32(acc);
#elif defined(SPG_NEON)
    int32x4_t acc = vdupq_n_s32(0);
    for (; k + 16u <= n; k += 16u) {
        int8x16_t va = vld1q_s8(a + k), vb = vld1q_s8(b + k);
#  if defined(__ARM_FEATURE_DOTPROD)
        acc = vdotq_s32(acc, va, vb);
#  else
        int16x8_t p = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        p = vmlal_s8(p, vget_high_s8(va), vget_high_s8(vb));      /* two products stay in int16 */
        acc = vpadalq_s16(acc, p);
#  endif
    }
#  if defined(__aarch64__)
    sum = vaddvq_s32(acc);
#  else
    int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    sum = vget_lane_s32(vpadd_s32(s2, s2), 0);
#  endif
#elif defined(SPG_ARM_DSP)
    /* SMLAD: two 16-bit multiply-accumulates per instruction on sign-extended byte pairs. */
    for (; k + 4u <= n; k += 4u) {
        uint32_t wa, wb;
        memcpy(&wa, a + k, 4);
        memcpy(&wb, b + k, 4);
        sum = __smlad(__sxtb16(wa), __sxtb16(wb), sum);
        sum = __smlad(__sxtb16((wa >> 8) | (wa << 24)), __sxtb16((wb >> 8) | (wb << 24)), sum);   /* bytes 1 and 3 */
    }
#endif
    for (; k < n; k++) sum += (int32_t)a[k] * (int32_t)b[k];
    return sum;
}

SPG_REGISTER_SUMS void spingalett_dot_i8_rows4(const int8_t *w, size_t stride, const int8_t *x, uint32_t n,
                                               int32_t acc[4]) {
#if defined(SPG_AVX2)
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    uint32_t k = 0;
    for (; k + 32u <= n; k += 32u) {
        __m256i vx = _mm256_loadu_si256((const __m256i *)(x + k)), ax = _mm256_sign_epi8(vx, vx);
        a0 = madd_i8(a0, ax, _mm256_loadu_si256((const __m256i *)(w + k)), vx);
        a1 = madd_i8(a1, ax, _mm256_loadu_si256((const __m256i *)(w + stride + k)), vx);
        a2 = madd_i8(a2, ax, _mm256_loadu_si256((const __m256i *)(w + 2 * stride + k)), vx);
        a3 = madd_i8(a3, ax, _mm256_loadu_si256((const __m256i *)(w + 3 * stride + k)), vx);
    }
    __m256i s = _mm256_hadd_epi32(_mm256_hadd_epi32(a0, a1), _mm256_hadd_epi32(a2, a3));
    __m128i t = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    if (k + 16u <= n) {
        __m128i vx = _mm_loadu_si128((const __m128i *)(x + k)), z = _mm_setzero_si128();
        __m128i b0 = madd_i8_128(z, _mm_loadu_si128((const __m128i *)(w + k)), vx);
        __m128i b1 = madd_i8_128(z, _mm_loadu_si128((const __m128i *)(w + stride + k)), vx);
        __m128i b2 = madd_i8_128(z, _mm_loadu_si128((const __m128i *)(w + 2 * stride + k)), vx);
        __m128i b3 = madd_i8_128(z, _mm_loadu_si128((const __m128i *)(w + 3 * stride + k)), vx);
        t = _mm_add_epi32(t, _mm_hadd_epi32(_mm_hadd_epi32(b0, b1), _mm_hadd_epi32(b2, b3)));
        k += 16u;
    }
    _mm_storeu_si128((__m128i *)(void *)acc, t);
    for (int r = 0; r < 4; r++)
        for (uint32_t i = k; i < n; i++) acc[r] += (int32_t)w[(size_t)r * stride + i] * (int32_t)x[i];
#else
    for (int r = 0; r < 4; r++) acc[r] = spingalett_dot_i8(w + (size_t)r * stride, x, n);
#endif
}

/* ---- four rows against four activation vectors: acc[4 r + p] = row r . x_p, each row and each
   vector loaded once per block. With VNNI the activations are offset to unsigned bytes, x + 128,
   and 128 times each row's sum is taken off: products need no sign handling, and the sums, exact
   modulo 2^32 like the result, give it exactly. */

#if defined(SPG_AVX512_VNNI)
/* The lane sums of 16 vectors as one vector, lane i the total of v[i]: pairs of 32-bit, then 64-bit
   unpacks add within 128-bit lanes, and two rounds of 128-bit shuffles add across them. */
static inline __m512i reduce16_epi32(const __m512i v[16]) {
    __m512i s[8], t[4];
    for (int i = 0; i < 8; i++)
        s[i] = _mm512_add_epi32(_mm512_unpacklo_epi32(v[2 * i], v[2 * i + 1]), _mm512_unpackhi_epi32(v[2 * i], v[2 * i + 1]));
    for (int j = 0; j < 4; j++)
        t[j] = _mm512_add_epi32(_mm512_unpacklo_epi64(s[2 * j], s[2 * j + 1]), _mm512_unpackhi_epi64(s[2 * j], s[2 * j + 1]));
    __m512i u01 = _mm512_add_epi32(_mm512_shuffle_i32x4(t[0], t[1], 0x44), _mm512_shuffle_i32x4(t[0], t[1], 0xEE));
    __m512i u23 = _mm512_add_epi32(_mm512_shuffle_i32x4(t[2], t[3], 0x44), _mm512_shuffle_i32x4(t[2], t[3], 0xEE));
    return _mm512_add_epi32(_mm512_shuffle_i32x4(u01, u23, 0x88), _mm512_shuffle_i32x4(u01, u23, 0xDD));
}
#endif

#if defined(SPG_AVX512_VNNI)
const bool spingalett_dot_i8_4x4_sums = true;
#else
const bool spingalett_dot_i8_4x4_sums = false;
#endif

int32_t spingalett_sum_i8(const int8_t *w, uint32_t n) {
    uint32_t k = 0;
    int32_t sum = 0;
#if defined(SPG_AVX512_VNNI)
    const __m512i ones = _mm512_set1_epi8(1);
    __m512i a = _mm512_setzero_si512();
    for (; k < n; k += 64u) {
        __mmask64 m = n - k >= 64u ? ~(__mmask64)0 : (((__mmask64)1 << (n - k)) - 1u);
        a = _mm512_dpbusd_epi32(a, ones, _mm512_maskz_loadu_epi8(m, w + k));
    }
    sum = _mm512_reduce_add_epi32(a);
#endif
    for (; k < n; k++) sum += w[k];
    return sum;
}

#if defined(SPG_AVX512_VNNI)
#define Q4_EACH(M) M(0) M(1) M(2) M(3)
#define Q4_ZERO(r) __m512i a##r##0 = _mm512_setzero_si512(), a##r##1 = a##r##0, a##r##2 = a##r##0, a##r##3 = a##r##0;
#define Q4_X(p) __m512i x##p = _mm512_xor_si512(_mm512_maskz_loadu_epi8(m, x + (size_t)(p) * x_stride + k), bias);
#define Q4_ROW(r) \
    { \
        __m512i wr = _mm512_maskz_loadu_epi8(m, w + (size_t)(r) * w_stride + k); \
        a##r##0 = _mm512_dpbusd_epi32(a##r##0, x0, wr); \
        a##r##1 = _mm512_dpbusd_epi32(a##r##1, x1, wr); \
        a##r##2 = _mm512_dpbusd_epi32(a##r##2, x2, wr); \
        a##r##3 = _mm512_dpbusd_epi32(a##r##3, x3, wr); \
    }

SPG_REGISTER_SUMS static void dot_i8_4x4_vnni(const int8_t *w, size_t w_stride, const int8_t *x, size_t x_stride,
                                              uint32_t n, const int32_t wsum[4], int32_t acc[16]) {
    const __m512i bias = _mm512_set1_epi8((char)0x80);
    Q4_EACH(Q4_ZERO)
    /* whole blocks of 64 bytes, then the rest as one block of masked loads (zeros add nothing:
       their offset activations meet zero weights) */
    for (uint32_t k = 0; k < n; k += 64u) {
        __mmask64 m = n - k >= 64u ? ~(__mmask64)0 : (((__mmask64)1 << (n - k)) - 1u);
        Q4_EACH(Q4_X)
        Q4_EACH(Q4_ROW)
    }
    const __m512i a[16] = {a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33};
    /* lane 4 r + p: the sum of row r and vector p, less 128 times the sum of row r */
    __m512i corr = _mm512_slli_epi32(_mm512_set_epi32(wsum[3], wsum[3], wsum[3], wsum[3], wsum[2], wsum[2], wsum[2],
                                                      wsum[2], wsum[1], wsum[1], wsum[1], wsum[1], wsum[0], wsum[0],
                                                      wsum[0], wsum[0]), 7);
    _mm512_storeu_si512((void *)acc, _mm512_sub_epi32(reduce16_epi32(a), corr));
}

#undef Q4_EACH
#undef Q4_ZERO
#undef Q4_X
#undef Q4_ROW
#endif

void spingalett_dot_i8_4x4(const int8_t *w, size_t w_stride, const int8_t *x, size_t x_stride, uint32_t n,
                           const int32_t wsum[4], int32_t acc[16]) {
#if defined(SPG_AVX512_VNNI)
    dot_i8_4x4_vnni(w, w_stride, x, x_stride, n, wsum, acc);
#else
    uint32_t k = 0;
    (void)wsum;
#  if defined(SPG_AVX2)
    /* two activation vectors at a time: four rows by two keep the registers of AVX2 */
    uint32_t kk = 0;
    for (int half = 0; half < 2; half++) {
        const int8_t *x0 = x + (size_t)(2 * half) * x_stride, *x1 = x0 + x_stride;
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
        kk = 0;
        for (; kk + 32u <= n; kk += 32u) {
            __m256i v0 = _mm256_loadu_si256((const __m256i *)(x0 + kk)), v1 = _mm256_loadu_si256((const __m256i *)(x1 + kk));
            __m256i u0 = _mm256_sign_epi8(v0, v0), u1 = _mm256_sign_epi8(v1, v1);
            __m256i w0 = _mm256_loadu_si256((const __m256i *)(w + kk));
            __m256i w1 = _mm256_loadu_si256((const __m256i *)(w + w_stride + kk));
            __m256i w2 = _mm256_loadu_si256((const __m256i *)(w + 2 * w_stride + kk));
            __m256i w3 = _mm256_loadu_si256((const __m256i *)(w + 3 * w_stride + kk));
            a0 = madd_i8(a0, u0, w0, v0); b0 = madd_i8(b0, u1, w0, v1);
            a1 = madd_i8(a1, u0, w1, v0); b1 = madd_i8(b1, u1, w1, v1);
            a2 = madd_i8(a2, u0, w2, v0); b2 = madd_i8(b2, u1, w2, v1);
            a3 = madd_i8(a3, u0, w3, v0); b3 = madd_i8(b3, u1, w3, v1);
        }
        __m128i ta = reduce_rows4(a0, a1, a2, a3), tb = reduce_rows4(b0, b1, b2, b3);
        if (kk + 16u <= n) {
            __m128i v0 = _mm_loadu_si128((const __m128i *)(x0 + kk)), v1 = _mm_loadu_si128((const __m128i *)(x1 + kk));
            __m128i z = _mm_setzero_si128(), c[4], d[4];
            for (int r = 0; r < 4; r++) {
                __m128i wr = _mm_loadu_si128((const __m128i *)(w + (size_t)r * w_stride + kk));
                c[r] = madd_i8_128(z, wr, v0);
                d[r] = madd_i8_128(z, wr, v1);
            }
            ta = _mm_add_epi32(ta, _mm_hadd_epi32(_mm_hadd_epi32(c[0], c[1]), _mm_hadd_epi32(c[2], c[3])));
            tb = _mm_add_epi32(tb, _mm_hadd_epi32(_mm_hadd_epi32(d[0], d[1]), _mm_hadd_epi32(d[2], d[3])));
            kk += 16u;
        }
        int32_t sa[4], sb[4];
        _mm_storeu_si128((__m128i *)(void *)sa, ta);
        _mm_storeu_si128((__m128i *)(void *)sb, tb);
        for (int r = 0; r < 4; r++) { acc[4 * r + 2 * half] = sa[r]; acc[4 * r + 2 * half + 1] = sb[r]; }
    }
    k = kk;
#  elif defined(SPG_NEON)
    int32x4_t a[16];
    for (int i = 0; i < 16; i++) a[i] = vdupq_n_s32(0);
    for (; k + 16u <= n; k += 16u) {
        int8x16_t xv[4];
        for (int p = 0; p < 4; p++) xv[p] = vld1q_s8(x + (size_t)p * x_stride + k);
        for (int r = 0; r < 4; r++) {
            int8x16_t wr = vld1q_s8(w + (size_t)r * w_stride + k);
            for (int p = 0; p < 4; p++) a[4 * r + p] = neon_dot_i8(a[4 * r + p], wr, xv[p]);
        }
    }
    for (int i = 0; i < 16; i++) acc[i] = neon_hsum(a[i]);
#  else
    for (int p = 0; p < 4; p++) {
        int32_t r4[4];
        spingalett_dot_i8_rows4(w, w_stride, x + (size_t)p * x_stride, n, r4);
        for (int r = 0; r < 4; r++) acc[4 * r + p] = r4[r];
    }
    k = n;
#  endif
    for (; k < n; k++)
        for (int r = 0; r < 4; r++)
            for (int p = 0; p < 4; p++)
                acc[4 * r + p] += (int32_t)w[(size_t)r * w_stride + k] * (int32_t)x[(size_t)p * x_stride + k];
#endif
}

/* ---- INT4 and INT2 rows, decoded in registers.
   The vector kernels take a row in blocks of B bytes (B = PACKED_BLOCK_BYTES, then halves of it
   down to PACKED_MIN_BLOCK, at most one block of each smaller size) and decode a block slot by
   slot: slot s holds code i * c + s of the block (c = 2 codes per byte for INT4, 4 for INT2) in
   byte i. permute_activations() stores the quantized activations of such a layer in that order,
   activation i * c + s of a block at s * B + i, so a slot meets its activations in one load. The
   codes after the last whole block, and all codes on targets without these kernels, are taken in
   row order. Integer sums do not depend on the order: every layout gives the same results. */

#if defined(SPG_AVX2)
#  define PACKED_BLOCK_BYTES 32u
#  define PACKED_MIN_BLOCK 4u
#elif defined(SPG_NEON)
#  define PACKED_BLOCK_BYTES 16u
#  define PACKED_MIN_BLOCK 16u
#endif

/* Returns the sum of the activations it moved (those of the whole blocks). */
static int32_t permute_activations(int8_t *q, uint32_t n, uint32_t per_byte) {
    int32_t sum = 0;
#if defined(PACKED_BLOCK_BYTES)
    int8_t block[4 * PACKED_BLOCK_BYTES];
    uint32_t k = 0;
    for (uint32_t bytes = PACKED_BLOCK_BYTES; bytes >= PACKED_MIN_BLOCK; bytes /= 2u) {
        uint32_t codes = bytes * per_byte;
        for (; n - k >= codes; k += codes) {
            memcpy(block, q + k, codes);
            for (uint32_t i = 0; i < bytes; i++)
                for (uint32_t s = 0; s < per_byte; s++) {
                    q[k + s * bytes + i] = block[i * per_byte + s];
                    sum += block[i * per_byte + s];
                }
        }
    }
#else
    (void)q; (void)n; (void)per_byte;
#endif
    return sum;
}

#if defined(PACKED_BLOCK_BYTES)
/* Signed value of code i of a stored INT4 (per_byte 2) or INT2 (4) row. */
static inline int32_t packed_code(const uint8_t *row, uint32_t i, uint32_t per_byte) {
    return per_byte == 2u ? int4_codes[row[i >> 1]][i & 1u] : int2_codes[row[i >> 2]][i & 3u];
}
#endif

#if defined(SPG_AVX2)
/* n (16, 8 or 4) bytes into the low bytes of a vector, the others 0. */
static inline __m128i load_bytes(const void *p, uint32_t n) {
    if (n == 16u) return _mm_loadu_si128((const __m128i *)p);
    if (n == 8u) return _mm_loadl_epi64((const __m128i *)p);
    int32_t v;
    memcpy(&v, p, 4);
    return _mm_cvtsi32_si128(v);
}

/* acc += u * s summed in fours, u unsigned and s signed bytes; u * s pairs stay far below the
   16-bit limit of maddubs here (u <= 15). */
static inline __m256i madd_u8s8(__m256i acc, __m256i u, __m256i s) {
#  if defined(SPG_VNNI)
    return SPG_VNNI(acc, u, s);
#  else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)));
#  endif
}

static inline __m128i madd_u8s8_128(__m128i acc, __m128i u, __m128i s) {
#  if defined(SPG_VNNI128)
    return SPG_VNNI128(acc, u, s);
#  else
    return _mm_add_epi32(acc, _mm_madd_epi16(_mm_maddubs_epi16(u, s), _mm_set1_epi16(1)));
#  endif
}

/* The codes are taken unsigned, offset by half their range: flipping the top bit of a two's
   complement code c gives c + 8 (INT4) or c + 2 (INT2), so a block contributes
   sum (c + offset) * x = sum c * x + offset * sum x, and the offset times the sum of the permuted
   activations (xsum) is subtracted at the end. */
static void dot_packed_rows4(const uint8_t *w, size_t stride, const int8_t *x, uint32_t n, uint32_t per_byte,
                             int32_t xsum, int32_t acc[4]) {
    const uint32_t bits = 8u / per_byte;
    const int32_t offset = per_byte == 2u ? 8 : 2;
    const __m128i flip128 = _mm_set1_epi8((char)(per_byte == 2u ? 0x88 : 0xAA));
    const __m128i mask128 = _mm_set1_epi8((char)((1u << bits) - 1u));
    const __m256i flip = _mm256_broadcastsi128_si256(flip128), mask = _mm256_broadcastsi128_si256(mask128);
    const uint8_t *w0 = w, *w1 = w + stride, *w2 = w + 2 * stride, *w3 = w + 3 * stride;
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    uint32_t k = 0;         /* activations done */
    size_t at = 0;          /* bytes of each row done */

#define PK_ROWS __m256i v0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(w0 + at)), flip); \
                __m256i v1 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(w1 + at)), flip); \
                __m256i v2 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(w2 + at)), flip); \
                __m256i v3 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(w3 + at)), flip);
#define PK_CODES(v, shift) _mm256_and_si256(_mm256_srli_epi16((v), (shift)), mask)
#define PK_SLOT(s, shift, A) { \
        __m256i vx = _mm256_loadu_si256((const __m256i *)(x + k + (s) * 32u)); \
        A##0 = madd_u8s8(A##0, PK_CODES(v0, shift), vx); A##1 = madd_u8s8(A##1, PK_CODES(v1, shift), vx); \
        A##2 = madd_u8s8(A##2, PK_CODES(v2, shift), vx); A##3 = madd_u8s8(A##3, PK_CODES(v3, shift), vx); }
    if (per_byte == 2u) {
        for (; n - k >= 64u; k += 64u, at += 32u) { PK_ROWS PK_SLOT(0, 0, a) PK_SLOT(1, 4, b) }
    } else {
        for (; n - k >= 128u; k += 128u, at += 32u) { PK_ROWS PK_SLOT(0, 0, a) PK_SLOT(1, 2, b) PK_SLOT(2, 4, a) PK_SLOT(3, 6, b) }
    }
#undef PK_ROWS
#undef PK_CODES
#undef PK_SLOT

    __m128i t0 = _mm_setzero_si128(), t1 = t0, t2 = t0, t3 = t0;
    for (uint32_t bytes = 16u; bytes >= PACKED_MIN_BLOCK; bytes /= 2u) {
        if (n - k < bytes * per_byte) continue;
        /* bytes past the block load as 0 and decode to the offset; their activations are 0 */
        __m128i u0 = _mm_xor_si128(load_bytes(w0 + at, bytes), flip128), u1 = _mm_xor_si128(load_bytes(w1 + at, bytes), flip128);
        __m128i u2 = _mm_xor_si128(load_bytes(w2 + at, bytes), flip128), u3 = _mm_xor_si128(load_bytes(w3 + at, bytes), flip128);
        for (uint32_t s = 0; s < per_byte; s++) {
            __m128i shift = _mm_cvtsi32_si128((int)(s * bits)), vx = load_bytes(x + k + s * bytes, bytes);
#define PK_CODES128(u) _mm_and_si128(_mm_srl_epi16((u), shift), mask128)
            t0 = madd_u8s8_128(t0, PK_CODES128(u0), vx);
            t1 = madd_u8s8_128(t1, PK_CODES128(u1), vx);
            t2 = madd_u8s8_128(t2, PK_CODES128(u2), vx);
            t3 = madd_u8s8_128(t3, PK_CODES128(u3), vx);
#undef PK_CODES128
        }
        k += bytes * per_byte;
        at += bytes;
    }
    __m128i t = reduce_rows4(_mm256_add_epi32(a0, b0), _mm256_add_epi32(a1, b1),
                             _mm256_add_epi32(a2, b2), _mm256_add_epi32(a3, b3));
    t = _mm_add_epi32(t, _mm_hadd_epi32(_mm_hadd_epi32(t0, t1), _mm_hadd_epi32(t2, t3)));
    t = _mm_sub_epi32(t, _mm_set1_epi32(offset * xsum));
    _mm_storeu_si128((__m128i *)(void *)acc, t);
    for (int r = 0; r < 4; r++)
        for (uint32_t i = k; i < n; i++) acc[r] += packed_code(w + (size_t)r * stride, i, per_byte) * (int32_t)x[i];
}

#elif defined(SPG_NEON)

/* Codes of 16 bytes as signed bytes: INT4 nibbles (two's complement) by xor and subtract, INT2
   codes c as (c & 1) - (c & 2). */
#define NEON_INT4(v) vsubq_s8(veorq_s8(vreinterpretq_s8_u8(v), eight), eight)
#define NEON_INT2(c) vsubq_s8(vreinterpretq_s8_u8(vandq_u8((c), one)), vreinterpretq_s8_u8(vandq_u8((c), two)))

static void dot_packed_rows4(const uint8_t *w, size_t stride, const int8_t *x, uint32_t n, uint32_t per_byte,
                             int32_t xsum, int32_t acc[4]) {
    (void)xsum;
    const int8x16_t eight = vdupq_n_s8(8);
    const uint8x16_t low = vdupq_n_u8(0x0F), one = vdupq_n_u8(1), two = vdupq_n_u8(2);
    int32x4_t s[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
    uint32_t k = 0;
    size_t at = 0;
    if (per_byte == 2u) {
        for (; n - k >= 32u; k += 32u, at += 16u) {
            int8x16_t x0 = vld1q_s8(x + k), x1 = vld1q_s8(x + k + 16);
            for (int r = 0; r < 4; r++) {
                uint8x16_t b = vld1q_u8(w + (size_t)r * stride + at);
                s[r] = neon_dot_i8(s[r], NEON_INT4(vandq_u8(b, low)), x0);
                s[r] = neon_dot_i8(s[r], NEON_INT4(vshrq_n_u8(b, 4)), x1);
            }
        }
    } else {
        for (; n - k >= 64u; k += 64u, at += 16u) {
            int8x16_t x0 = vld1q_s8(x + k), x1 = vld1q_s8(x + k + 16), x2 = vld1q_s8(x + k + 32), x3 = vld1q_s8(x + k + 48);
            for (int r = 0; r < 4; r++) {
                uint8x16_t b = vld1q_u8(w + (size_t)r * stride + at);
                s[r] = neon_dot_i8(s[r], NEON_INT2(b), x0);
                s[r] = neon_dot_i8(s[r], NEON_INT2(vshrq_n_u8(b, 2)), x1);
                s[r] = neon_dot_i8(s[r], NEON_INT2(vshrq_n_u8(b, 4)), x2);
                s[r] = neon_dot_i8(s[r], NEON_INT2(vshrq_n_u8(b, 6)), x3);
            }
        }
    }
    for (int r = 0; r < 4; r++) {
        int32_t sum = neon_hsum(s[r]);
        for (uint32_t i = k; i < n; i++) sum += packed_code(w + (size_t)r * stride, i, per_byte) * (int32_t)x[i];
        acc[r] = sum;
    }
}
#undef NEON_INT4
#undef NEON_INT2

#else

static void dot_packed_rows4(const uint8_t *w, size_t stride, const int8_t *x, uint32_t n, uint32_t per_byte,
                             int32_t xsum, int32_t acc[4]) {
    (void)xsum;
    int8_t block[256];
    for (int r = 0; r < 4; r++) {
        const uint8_t *row = w + (size_t)r * stride;
        int32_t sum = 0;
        for (uint32_t k = 0; k < n; k += 256u) {         /* 256 codes: whole bytes */
            uint32_t len = n - k < 256u ? n - k : 256u;
            if (per_byte == 2u) spingalett_unpack_int4(row + k / 2u, block, len);
            else spingalett_unpack_int2(row + k / 4u, block, len);
            sum += spingalett_dot_i8(block, x + k, len);
        }
        acc[r] = sum;
    }
}

#endif

#if defined(SPG_F16C) && defined(SPG_AVX2)
#  define SPG_F16_ROWS4 1       /* dot_f16_rows4 has a vector path of its own */
#endif

/* Single rows, for the targets without a four-row vector path. */
#if !defined(SPG_AVX) || !defined(SPG_F16_ROWS4)
static float dot_f32(const float *w, const float *x, uint32_t n) {
    uint32_t k = 0;
    float sum = 0.0f;
#if defined(SPG_AVX)
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    for (; k + 32u <= n; k += 32u) {
        s0 = SPG_FMADD256(_mm256_loadu_ps(w + k),      _mm256_loadu_ps(x + k),      s0);
        s1 = SPG_FMADD256(_mm256_loadu_ps(w + k + 8),  _mm256_loadu_ps(x + k + 8),  s1);
        s2 = SPG_FMADD256(_mm256_loadu_ps(w + k + 16), _mm256_loadu_ps(x + k + 16), s2);
        s3 = SPG_FMADD256(_mm256_loadu_ps(w + k + 24), _mm256_loadu_ps(x + k + 24), s3);
    }
    for (; k + 8u <= n; k += 8u)
        s0 = SPG_FMADD256(_mm256_loadu_ps(w + k), _mm256_loadu_ps(x + k), s0);
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 0x55));
    sum = _mm_cvtss_f32(h);
#elif defined(SPG_SSE2)
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
    for (; k + 8u <= n; k += 8u) {
        s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_loadu_ps(w + k), _mm_loadu_ps(x + k)));
        s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_loadu_ps(w + k + 4), _mm_loadu_ps(x + k + 4)));
    }
    __m128 h = _mm_add_ps(s0, s1);
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 0x55));
    sum = _mm_cvtss_f32(h);
#elif defined(SPG_NEON)
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    for (; k + 8u <= n; k += 8u) {
#  if defined(__aarch64__)
        s0 = vfmaq_f32(s0, vld1q_f32(w + k), vld1q_f32(x + k));
        s1 = vfmaq_f32(s1, vld1q_f32(w + k + 4), vld1q_f32(x + k + 4));
#  else
        s0 = vmlaq_f32(s0, vld1q_f32(w + k), vld1q_f32(x + k));
        s1 = vmlaq_f32(s1, vld1q_f32(w + k + 4), vld1q_f32(x + k + 4));
#  endif
    }
    float32x4_t s = vaddq_f32(s0, s1);
#  if defined(__aarch64__)
    sum = vaddvq_f32(s);
#  else
    float32x2_t s2 = vadd_f32(vget_low_f32(s), vget_high_f32(s));
    sum = vget_lane_f32(vpadd_f32(s2, s2), 0);
#  endif
#else
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (; k + 4u <= n; k += 4u) {
        s0 += w[k] * x[k];
        s1 += w[k + 1] * x[k + 1];
        s2 += w[k + 2] * x[k + 2];
        s3 += w[k + 3] * x[k + 3];
    }
    sum = (s0 + s1) + (s2 + s3);
#endif
    for (; k < n; k++) sum += w[k] * x[k];
    return sum;
}
#endif

/* Half and bfloat16 rows: converted in blocks to float, or with conversion instructions. */
#if !defined(SPG_F16_ROWS4)
static float dot_f16(const uint16_t *w, const float *x, uint32_t n, bool bf16) {
    uint32_t k = 0;
    float sum = 0.0f;
#if defined(SPG_F16C) || defined(SPG_AVX2)
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    for (; k + 16u <= n; k += 16u) {
        __m128i h0 = _mm_loadu_si128((const __m128i *)(w + k)), h1 = _mm_loadu_si128((const __m128i *)(w + k + 8));
        __m256 f0, f1;
        if (bf16) {
#  if defined(SPG_AVX2)
            f0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h0), 16));
            f1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h1), 16));
#  else
            break;
#  endif
        } else {
#  if defined(SPG_F16C)
            f0 = _mm256_cvtph_ps(h0);
            f1 = _mm256_cvtph_ps(h1);
#  else
            break;
#  endif
        }
        s0 = SPG_FMADD256(f0, _mm256_loadu_ps(x + k), s0);
        s1 = SPG_FMADD256(f1, _mm256_loadu_ps(x + k + 8), s1);
    }
    __m256 s = _mm256_add_ps(s0, s1);
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 0x55));
    sum = _mm_cvtss_f32(h);
#elif defined(SPG_NEON) && defined(__aarch64__)
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    for (; k + 8u <= n; k += 8u) {
        uint16x8_t h = vld1q_u16(w + k);
        float32x4_t f0, f1;
        if (bf16) {
            f0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
            f1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h), 16));
        } else {
            f0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h)));
            f1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h)));
        }
        s0 = vfmaq_f32(s0, f0, vld1q_f32(x + k));
        s1 = vfmaq_f32(s1, f1, vld1q_f32(x + k + 4));
    }
    sum = vaddvq_f32(vaddq_f32(s0, s1));
#elif defined(SPG_SSE2)
    /* the same conversions as half_to_float_fast and bf16, four lanes at a time */
    __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
    const __m128i zero = _mm_setzero_si128(), nosign = _mm_set1_epi32(0x7FFF), infnan = _mm_set1_epi32(0x7BFF);
    const __m128 magic = _mm_castsi128_ps(_mm_set1_epi32((127 + 112) << 23)), allones = _mm_castsi128_ps(_mm_set1_epi32(0x7F800000));
    for (; k + 8u <= n; k += 8u) {
        __m128i h = _mm_loadu_si128((const __m128i *)(w + k));
        __m128 f[2];
        for (int half = 0; half < 2; half++) {
            __m128i v = half ? _mm_unpackhi_epi16(h, zero) : _mm_unpacklo_epi16(h, zero);
            if (bf16) {
                f[half] = _mm_castsi128_ps(_mm_slli_epi32(v, 16));
            } else {
                __m128i em = _mm_and_si128(v, nosign);
                __m128 scaled = _mm_mul_ps(_mm_castsi128_ps(_mm_slli_epi32(em, 13)), magic);
                __m128 special = _mm_and_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(em, infnan)), allones);
                __m128 sign = _mm_castsi128_ps(_mm_slli_epi32(_mm_xor_si128(v, em), 16));
                f[half] = _mm_or_ps(_mm_or_ps(scaled, special), sign);
            }
        }
        s0 = _mm_add_ps(s0, _mm_mul_ps(f[0], _mm_loadu_ps(x + k)));
        s1 = _mm_add_ps(s1, _mm_mul_ps(f[1], _mm_loadu_ps(x + k + 4)));
    }
    __m128 hs = _mm_add_ps(s0, s1);
    hs = _mm_add_ps(hs, _mm_movehl_ps(hs, hs));
    hs = _mm_add_ss(hs, _mm_shuffle_ps(hs, hs, 0x55));
    sum = _mm_cvtss_f32(hs);
#endif
    float block[64];
    while (k < n) {
        uint32_t len = n - k < 64u ? n - k : 64u;
        for (uint32_t i = 0; i < len; i++)
            block[i] = bf16 ? spingalett_bf16_to_float(w[k + i]) : half_to_float_fast(w[k + i]);
        sum += dot_f32(block, x + k, len);
        k += len;
    }
    return sum;
}
#endif

#if defined(SPG_AVX)
static inline float hsum256(__m256 s) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 0x55));
    return _mm_cvtss_f32(h);
}
#endif

/* out[r] = dot of row r (rows `stride` floats apart) with x, for r = 0..3. */
static void dot_f32_rows4(const float *w, size_t stride, const float *x, uint32_t n, float out[4]) {
#if defined(SPG_AVX)
    const float *w0 = w, *w1 = w + stride, *w2 = w + 2 * stride, *w3 = w + 3 * stride;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    uint32_t k = 0;
    for (; k + 16u <= n; k += 16u) {
        __m256 x0 = _mm256_loadu_ps(x + k), x1 = _mm256_loadu_ps(x + k + 8);
        a0 = SPG_FMADD256(_mm256_loadu_ps(w0 + k), x0, a0);
        a1 = SPG_FMADD256(_mm256_loadu_ps(w1 + k), x0, a1);
        a2 = SPG_FMADD256(_mm256_loadu_ps(w2 + k), x0, a2);
        a3 = SPG_FMADD256(_mm256_loadu_ps(w3 + k), x0, a3);
        b0 = SPG_FMADD256(_mm256_loadu_ps(w0 + k + 8), x1, b0);
        b1 = SPG_FMADD256(_mm256_loadu_ps(w1 + k + 8), x1, b1);
        b2 = SPG_FMADD256(_mm256_loadu_ps(w2 + k + 8), x1, b2);
        b3 = SPG_FMADD256(_mm256_loadu_ps(w3 + k + 8), x1, b3);
    }
    if (k + 8u <= n) {
        __m256 x0 = _mm256_loadu_ps(x + k);
        a0 = SPG_FMADD256(_mm256_loadu_ps(w0 + k), x0, a0);
        a1 = SPG_FMADD256(_mm256_loadu_ps(w1 + k), x0, a1);
        a2 = SPG_FMADD256(_mm256_loadu_ps(w2 + k), x0, a2);
        a3 = SPG_FMADD256(_mm256_loadu_ps(w3 + k), x0, a3);
        k += 8u;
    }
    out[0] = hsum256(_mm256_add_ps(a0, b0));
    out[1] = hsum256(_mm256_add_ps(a1, b1));
    out[2] = hsum256(_mm256_add_ps(a2, b2));
    out[3] = hsum256(_mm256_add_ps(a3, b3));
    for (int r = 0; r < 4; r++)
        for (uint32_t i = k; i < n; i++) out[r] += w[(size_t)r * stride + i] * x[i];
#else
    for (int r = 0; r < 4; r++) out[r] = dot_f32(w + (size_t)r * stride, x, n);
#endif
}

/* The same for half or bfloat16 rows. */
static void dot_f16_rows4(const uint16_t *w, size_t stride, const float *x, uint32_t n, bool bf16, float out[4]) {
#if defined(SPG_F16C) && defined(SPG_AVX2)
    const uint16_t *w0 = w, *w1 = w + stride, *w2 = w + 2 * stride, *w3 = w + 3 * stride;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
    uint32_t k = 0;
#define F16_HALF(p) _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(p)))
#define F16_BF16(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16))
#define F16_ROWS4(LOAD) \
    for (; k + 16u <= n; k += 16u) { \
        __m256 x0 = _mm256_loadu_ps(x + k), x1 = _mm256_loadu_ps(x + k + 8); \
        a0 = SPG_FMADD256(LOAD(w0 + k), x0, a0); a1 = SPG_FMADD256(LOAD(w1 + k), x0, a1); \
        a2 = SPG_FMADD256(LOAD(w2 + k), x0, a2); a3 = SPG_FMADD256(LOAD(w3 + k), x0, a3); \
        b0 = SPG_FMADD256(LOAD(w0 + k + 8), x1, b0); b1 = SPG_FMADD256(LOAD(w1 + k + 8), x1, b1); \
        b2 = SPG_FMADD256(LOAD(w2 + k + 8), x1, b2); b3 = SPG_FMADD256(LOAD(w3 + k + 8), x1, b3); \
    } \
    if (k + 8u <= n) { \
        __m256 x0 = _mm256_loadu_ps(x + k); \
        a0 = SPG_FMADD256(LOAD(w0 + k), x0, a0); a1 = SPG_FMADD256(LOAD(w1 + k), x0, a1); \
        a2 = SPG_FMADD256(LOAD(w2 + k), x0, a2); a3 = SPG_FMADD256(LOAD(w3 + k), x0, a3); \
        k += 8u; \
    }
    if (bf16) { F16_ROWS4(F16_BF16) } else { F16_ROWS4(F16_HALF) }
#undef F16_ROWS4
#undef F16_HALF
#undef F16_BF16
    out[0] = hsum256(_mm256_add_ps(a0, b0));
    out[1] = hsum256(_mm256_add_ps(a1, b1));
    out[2] = hsum256(_mm256_add_ps(a2, b2));
    out[3] = hsum256(_mm256_add_ps(a3, b3));
    for (int r = 0; r < 4; r++)
        for (uint32_t i = k; i < n; i++) {
            uint16_t h = w[(size_t)r * stride + i];
            out[r] += (bf16 ? spingalett_bf16_to_float(h) : half_to_float_fast(h)) * x[i];
        }
#else
    for (int r = 0; r < 4; r++) out[r] = dot_f16(w + (size_t)r * stride, x, n, bf16);
#endif
}

float spingalett_quantize_activations(const float *x, uint32_t n, int8_t *q) {
    float amax = 0.0f;
    bool bad = false;
    uint32_t k = 0;
#if defined(SPG_AVX)
    __m256 vmax = _mm256_setzero_ps(), vbad = _mm256_setzero_ps();
    const __m256 sign = _mm256_set1_ps(-0.0f), big = _mm256_set1_ps(FLT_MAX);
    /* four maxima side by side, then together (the largest magnitude does not depend on the order) */
    __m256 m1 = vmax, m2 = vmax, m3 = vmax;
    for (; k + 32u <= n; k += 32u) {
        __m256 a0 = _mm256_andnot_ps(sign, _mm256_loadu_ps(x + k)), a1 = _mm256_andnot_ps(sign, _mm256_loadu_ps(x + k + 8));
        __m256 a2 = _mm256_andnot_ps(sign, _mm256_loadu_ps(x + k + 16)), a3 = _mm256_andnot_ps(sign, _mm256_loadu_ps(x + k + 24));
        vbad = _mm256_or_ps(vbad, _mm256_or_ps(_mm256_or_ps(_mm256_cmp_ps(a0, big, _CMP_NLE_UQ), _mm256_cmp_ps(a1, big, _CMP_NLE_UQ)),
                                               _mm256_or_ps(_mm256_cmp_ps(a2, big, _CMP_NLE_UQ), _mm256_cmp_ps(a3, big, _CMP_NLE_UQ))));
        vmax = _mm256_max_ps(vmax, a0);
        m1 = _mm256_max_ps(m1, a1);
        m2 = _mm256_max_ps(m2, a2);
        m3 = _mm256_max_ps(m3, a3);
    }
    vmax = _mm256_max_ps(_mm256_max_ps(vmax, m1), _mm256_max_ps(m2, m3));
    for (; k + 8u <= n; k += 8u) {
        __m256 a = _mm256_andnot_ps(sign, _mm256_loadu_ps(x + k));
        vbad = _mm256_or_ps(vbad, _mm256_cmp_ps(a, big, _CMP_NLE_UQ));     /* NaN or infinite */
        vmax = _mm256_max_ps(vmax, a);
    }
    bad = _mm256_movemask_ps(vbad) != 0;
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 0x55));
    amax = _mm_cvtss_f32(m);
#endif
    for (; k < n; k++) {
        float a = fabsf(x[k]);
        if (!(a <= FLT_MAX)) bad = true;
        else if (a > amax) amax = a;
    }
    if (bad || amax == 0.0f) {
        memset(q, 0, n);
        return bad ? NAN : 0.0f;
    }

    float inv = 127.0f / amax;
    k = 0;
#if defined(SPG_AVX2)
    const __m256 vinv = _mm256_set1_ps(inv);
    for (; k + 8u <= n; k += 8u) {
        __m256i i = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + k), vinv));   /* nearest even */
        __m128i p = _mm_packs_epi32(_mm256_castsi256_si128(i), _mm256_extracti128_si256(i, 1));
        _mm_storel_epi64((__m128i *)(q + k), _mm_packs_epi16(p, p));
    }
#elif defined(SPG_SSE2)
    const __m128 vinv = _mm_set1_ps(inv);
    for (; k + 8u <= n; k += 8u) {
        __m128i i0 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(x + k), vinv));
        __m128i i1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(x + k + 4), vinv));
        __m128i p = _mm_packs_epi32(i0, i1);
        _mm_storel_epi64((__m128i *)(q + k), _mm_packs_epi16(p, p));
    }
#elif defined(SPG_NEON) && defined(__aarch64__)
    const float32x4_t vinv = vdupq_n_f32(inv);
    for (; k + 8u <= n; k += 8u) {
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + k), vinv));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + k + 4), vinv));
        vst1_s8(q + k, vqmovn_s16(vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1))));
    }
#endif
    for (; k < n; k++) {
        long r = lrintf(x[k] * inv);                    /* nearest even, as the vector paths */
        q[k] = (int8_t)(r > 127 ? 127 : (r < -127 ? -127 : r));
    }
    return amax / 127.0f;
}

#if defined(SPINGALETT_INFERENCE_ONLY)
static float activate_one(float x, ActivationFunction act) {
    switch (act) {
        case ACT_RELU:       return x > 0.0f ? x : 0.0f;
        case ACT_LEAKY_RELU: return x > 0.0f ? x : 0.01f * x;
        case ACT_SIGMOID:    return 1.0f / (1.0f + expf(-x));
        case ACT_TANH:       return tanhf(x);
        case ACT_FOO52:      return x > 1.0f ? 1.0f + 0.01f * (x - 1.0f) : (x < 0.0f ? 0.01f * x : x);
        default:             return x;
    }
}
#endif

void spingalett_engine_activate(float *y, uint32_t n, ActivationFunction act) {
#if defined(SPINGALETT_INFERENCE_ONLY)
    if (act == ACT_SOFTMAX) {
        float m = -FLT_MAX, sum = 0.0f;
        for (uint32_t i = 0; i < n; i++) if (y[i] > m) m = y[i];
        for (uint32_t i = 0; i < n; i++) { y[i] = expf(y[i] - m); sum += y[i]; }
        if (sum > 0.0f) for (uint32_t i = 0; i < n; i++) y[i] /= sum;
    } else if (act != ACT_NONE) {
        for (uint32_t i = 0; i < n; i++) y[i] = activate_one(y[i], act);
    }
#else
    apply_activation_batch(y, n, act);
#endif
}

/* ------------------------------------------------------------------------- formats 3 to 6 */

static inline size_t entry_size(uint16_t version) {
    return version >= 6u ? SLETT_LAYER_ENTRY_SIZE_6 : version == 5u ? SLETT_LAYER_ENTRY_SIZE_5
         : version == 4u ? SLETT_LAYER_ENTRY_SIZE_4 : SLETT_LAYER_ENTRY_SIZE;
}

static inline const uint8_t *entry_at(const uint8_t *image, uint32_t index) {
    return image + SLETT_HEADER_SIZE + (size_t)index * entry_size(slett_get16(image + 6));
}

void spingalett_slett_layer(const uint8_t *image, uint32_t index, SlettLayer *layer) {
    const uint8_t *e = entry_at(image, index);
    memset(layer, 0, sizeof *layer);
    layer->inputs = slett_get32(e);
    layer->outputs = slett_get32(e + 4);
    layer->activation = slett_act(e[8]);
    layer->precision = (PrecisionMode)e[9];
    uint32_t d = slett_get32(e + 12);
    memcpy(&layer->dropout, &d, 4);
    layer->weights = slett_get64(e + 16);
    layer->scales = slett_get64(e + 24);
    layer->biases = slett_get64(e + 32);
    layer->optimizer = slett_get64(e + 40);
    layer->type = LAYER_DENSE;
    layer->in_h = layer->in_w = layer->out_h = layer->out_w = 1;
    uint16_t version = slett_get16(image + 6);
    if (version >= 4u) {
        layer->type = (LayerType)e[10];
        layer->out_h = slett_get16(e + 48);
        layer->out_w = slett_get16(e + 50);
        layer->kernel_h = slett_get16(e + 52);
        layer->kernel_w = slett_get16(e + 54);
        layer->stride_h = slett_get16(e + 56);
        layer->stride_w = slett_get16(e + 58);
        layer->pad_h = slett_get16(e + 60);
        layer->pad_w = slett_get16(e + 62);
        /* the input: layer `index` before version 6, the first one the entry names from version 6
           on (an entry naming a later layer keeps no shape, which validation rejects) */
        uint32_t from = index;
        layer->input_count = 1;
        layer->input0 = index;
        if (version >= 6u) {
            layer->input_count = slett_get32(e + 80);
            layer->input0 = slett_get32(e + 84);
            layer->input_list = slett_get64(e + 88);
            layer->act_offset = slett_get64(e + 96);
            from = layer->input0 <= index ? layer->input0 : UINT32_MAX;
        }
        if (from == 0) {            /* the input layer's shape is in the header */
            layer->in_h = slett_get32(image + 32);
            layer->in_w = slett_get32(image + 36);
        } else if (from != UINT32_MAX) {
            const uint8_t *prev = entry_at(image, from - 1);
            layer->in_h = slett_get16(prev + 48);
            layer->in_w = slett_get16(prev + 50);
        }
        if (layer->type == LAYER_CONV2D) layer->groups = 1;
    } else {
        layer->input_count = 1;
        layer->input0 = index;
    }
    if (version >= 5u) {
        layer->groups = slett_get32(e + 64);
        uint32_t f = slett_get32(e + 68);
        memcpy(&layer->eps, &f, 4);
        f = slett_get32(e + 72);
        memcpy(&layer->momentum, &f, 4);
    }
    if (version >= 7u) layer->mode = slett_get32(e + 76);
    uint64_t in_cells = (uint64_t)layer->in_h * layer->in_w, out_cells = (uint64_t)layer->out_h * layer->out_w;
    layer->in_c = in_cells ? (uint32_t)(layer->inputs / in_cells) : 0u;
    layer->out_c = out_cells ? (uint32_t)(layer->outputs / out_cells) : 0u;
    if (layer->type == LAYER_DENSE) {
        layer->rows = layer->outputs;
        layer->row_len = layer->inputs;
    } else if (layer->type == LAYER_CONV2D || layer->type == LAYER_CONV_TRANSPOSE2D) {
        uint64_t len = layer->groups ? (uint64_t)layer->kernel_h * layer->kernel_w * (layer->in_c / layer->groups) : 0u;
        layer->rows = layer->out_c;
        layer->row_len = len > UINT32_MAX ? 0u : (uint32_t)len;
    } else if (layer->type == LAYER_BATCH_NORM || layer->type == LAYER_LAYER_NORM) {
        layer->rows = layer->out_c;
        layer->row_len = 1;
    }
}

uint64_t spingalett_slett_act_offset(const uint8_t *image, uint32_t j) {
    return slett_get64(entry_at(image, j - 1) + 96);
}

void spingalett_slett_output_shape(const uint8_t *image, uint32_t j, uint32_t *height, uint32_t *width,
                                   uint32_t *units) {
    if (slett_get16(image + 6) < 4u) {
        *height = *width = 1;
    } else if (j == 0) {
        *height = slett_get32(image + 32);
        *width = slett_get32(image + 36);
    } else {
        *height = slett_get16(entry_at(image, j - 1) + 48);
        *width = slett_get16(entry_at(image, j - 1) + 50);
    }
    *units = j == 0 ? slett_get32(entry_at(image, 0)) : slett_get32(entry_at(image, j - 1) + 4);
}

/* [offset, offset + bytes) lies within size and offset is a multiple of `align`. */
static bool section_ok(uint64_t offset, uint64_t bytes, uint64_t size, uint64_t align) {
    return offset >= SLETT_HEADER_SIZE && offset % align == 0 && offset <= size && bytes <= size - offset;
}

/* Bytes of a layer's output among a workspace's activations. */
static inline uint64_t output_bytes(uint32_t units) {
    return slett_align((uint64_t)units * 4u);
}

/* The inputs of version 6 entry `index` (layer index + 1): earlier layers, in a list section when
   there are several; the layers an addition or concatenation reads agree with its shape. */
static bool inputs_ok(const uint8_t *p, uint32_t index, const SlettLayer *L, uint64_t file_size) {
    if (L->input_count == 0 || L->input_count > SPINGALETT_MAX_INPUTS || L->input0 > index) return false;
    bool combines = L->type == LAYER_ADD || L->type == LAYER_CONCAT;
    if (L->input_count == 1 ? L->input_list != 0
                            : !combines || !section_ok(L->input_list, 4u * (uint64_t)L->input_count, file_size, 4) ||
                              slett_get32(p + L->input_list) != L->input0)
        return false;
    if (!combines) return true;
    uint64_t channels = 0;
    for (uint32_t k = 0; k < L->input_count; k++) {
        uint32_t j = spingalett_slett_input(p, L, k), h, w, units;
        if (j > index) return false;
        spingalett_slett_output_shape(p, j, &h, &w, &units);
        if (h == 0 || w == 0 || h != L->out_h || w != L->out_w || (L->type == LAYER_ADD && units != L->outputs))
            return false;
        channels += units / ((uint64_t)h * w);
    }
    return L->type == LAYER_ADD || channels == L->out_c;
}

/* The output of version 6 entry `index` lies among the activations, apart from its inputs' (the last
   layer writes the caller's buffer instead). */
static bool act_offset_ok(const uint8_t *p, uint32_t index, const SlettLayer *L, uint32_t layers, uint64_t arena) {
    if (index + 2 == layers) return L->act_offset == 0;
    uint64_t at = L->act_offset, bytes = output_bytes(L->outputs);
    if (at % SLETT_SECTION_ALIGN != 0 || at > arena || bytes > arena - at) return false;
    for (uint32_t k = 0; k < L->input_count; k++) {
        uint32_t j = spingalett_slett_input(p, L, k), h, w, units;
        if (j == 0) continue;
        spingalett_slett_output_shape(p, j, &h, &w, &units);
        uint64_t other = spingalett_slett_act_offset(p, j);
        if (at < other + output_bytes(units) && other < at + bytes) return false;
    }
    return true;
}

/* The shape fields of a version 4 to 6 entry agree with its kind and its (first) input. */
static bool shape_ok(const SlettLayer *L, uint16_t version) {
    uint64_t in_cells = (uint64_t)L->in_h * L->in_w, out_cells = (uint64_t)L->out_h * L->out_w;
    if ((unsigned)L->type >= LAYER_TYPE_COUNT || in_cells == 0 || out_cells == 0 || L->in_c == 0 || L->out_c == 0 ||
        in_cells * L->in_c != L->inputs || out_cells * L->out_c != L->outputs ||
        L->in_h > SLETT_MAX_EXTENT || L->in_w > SLETT_MAX_EXTENT)
        return false;
    if ((L->type == LAYER_BATCH_NORM && version < 5u) || (L->type > LAYER_BATCH_NORM && version < 6u) ||
        (L->type > LAYER_GLOBAL_AVG_POOL && version < 7u))
        return false;
    /* version 7: a mode for upsampling only */
    if (L->type != LAYER_UPSAMPLE && L->mode != 0) return false;
    bool windowless = L->kernel_h == 0 && L->kernel_w == 0 && L->stride_h == 0 && L->stride_w == 0 &&
                      L->pad_h == 0 && L->pad_w == 0;
    bool parameterless = L->weights == 0 && L->scales == 0 && L->biases == 0 && L->optimizer == 0;
    if (L->type == LAYER_ADD || L->type == LAYER_CONCAT || L->type == LAYER_GLOBAL_AVG_POOL) {
        /* no window, no parameters; the shapes of all inputs are checked with them */
        if (!windowless || !parameterless || L->groups != 0 || L->eps != 0.0f || L->momentum != 0.0f) return false;
        if (L->type == LAYER_GLOBAL_AVG_POOL)
            return L->out_h == 1 && L->out_w == 1 && L->out_c == L->in_c && L->activation == ACT_NONE;
        return L->out_h == L->in_h && L->out_w == L->in_w && (L->type == LAYER_CONCAT || L->out_c == L->in_c);
    }
    if (L->type == LAYER_UPSAMPLE)       /* factors in the strides; no parameters, no activation */
        return L->kernel_h == 0 && L->kernel_w == 0 && L->pad_h == 0 && L->pad_w == 0 && L->stride_h != 0 &&
               L->stride_w != 0 && parameterless && L->groups == 0 && L->eps == 0.0f && L->momentum == 0.0f &&
               L->mode < UPSAMPLE_MODE_COUNT && L->activation == ACT_NONE && L->out_c == L->in_c &&
               (uint64_t)L->in_h * L->stride_h == L->out_h && (uint64_t)L->in_w * L->stride_w == L->out_w;
    /* version 5: groups for convolutions only, epsilon for normalizations, momentum for batch
       normalization only */
    const bool filters = L->type == LAYER_CONV2D || L->type == LAYER_CONV_TRANSPOSE2D;
    if (!filters && L->groups != 0) return false;
    if (L->type != LAYER_BATCH_NORM && L->type != LAYER_LAYER_NORM && L->eps != 0.0f) return false;
    if (L->type != LAYER_BATCH_NORM && L->momentum != 0.0f) return false;
    if (L->type == LAYER_DENSE)
        return L->out_h == 1 && L->out_w == 1 && windowless;
    if (L->type == LAYER_BATCH_NORM || L->type == LAYER_LAYER_NORM)
        return windowless && L->out_h == L->in_h && L->out_w == L->in_w && L->out_c == L->in_c &&
               L->precision == PRECISION_FLOAT32 && L->eps > 0.0f && L->eps < 1.0f &&
               L->momentum >= 0.0f && L->momentum <= 1.0f;
    if (L->type == LAYER_CONV_TRANSPOSE2D) {
        /* out = (in - 1) stride - 2 pad + kernel + output padding, the output padding below the stride */
        if (L->kernel_h == 0 || L->kernel_w == 0 || L->stride_h == 0 || L->stride_w == 0 ||
            L->pad_h >= L->kernel_h || L->pad_w >= L->kernel_w || L->groups == 0 || L->in_c % L->groups != 0 ||
            L->out_c % L->groups != 0 || L->row_len == 0)
            return false;
        int64_t oph = (int64_t)L->out_h - (((int64_t)L->in_h - 1) * L->stride_h - 2 * (int64_t)L->pad_h + L->kernel_h);
        int64_t opw = (int64_t)L->out_w - (((int64_t)L->in_w - 1) * L->stride_w - 2 * (int64_t)L->pad_w + L->kernel_w);
        return oph >= 0 && oph < (int64_t)L->stride_h && opw >= 0 && opw < (int64_t)L->stride_w;
    }
    if (L->kernel_h == 0 || L->kernel_w == 0 || L->stride_h == 0 || L->stride_w == 0 ||
        L->pad_h >= L->kernel_h || L->pad_w >= L->kernel_w ||
        L->out_h != slett_window_count(L->in_h, L->kernel_h, L->stride_h, L->pad_h) ||
        L->out_w != slett_window_count(L->in_w, L->kernel_w, L->stride_w, L->pad_w))
        return false;
    if (L->type == LAYER_CONV2D)
        return L->groups != 0 && L->in_c % L->groups == 0 && L->out_c % L->groups == 0 && L->row_len != 0;
    /* pooling: per channel, no activation, no parameters */
    return L->out_c == L->in_c && L->activation == ACT_NONE && parameterless;
}

int spingalett_slett_validate(const uint8_t *p, size_t size, SlettInfo *info) {
    memset(info, 0, sizeof *info);
    if (!spingalett_host_is_little_endian())
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: the .slett format needs a little-endian host");
    if (size < 8 || memcmp(p, SLETT_MAGIC, 6) != 0)
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: not a .slett format 3 to 7 image");
    uint16_t version = slett_get16(p + 6);
    if (version < 3 || version > 7)
        return ENGINE_FAIL(SPINGALETT_ERR_FORMAT_VERSION, "model: unsupported format version");
    if (size < SLETT_HEADER_SIZE)
        return ENGINE_FAIL(SPINGALETT_ERR_FILE_IO, "model: image is truncated");
    if (spingalett_crc32(0, p, 60) != slett_get32(p + 60))
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: header checksum mismatch");

    uint64_t file_size = slett_get64(p + 24);
    uint32_t layers = slett_get32(p + 8);
    if (file_size < SLETT_HEADER_SIZE || layers < 2 || layers > SLETT_MAX_LAYERS || p[12] >= LOSS_COUNT)
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: invalid header");
    if (file_size > size)
        return ENGINE_FAIL(SPINGALETT_ERR_FILE_IO, "model: image is truncated");
    uint64_t table_end = SLETT_HEADER_SIZE + (uint64_t)(layers - 1) * entry_size(version);
    if (table_end > file_size)
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: layer table exceeds the image");
    if (spingalett_crc32(0, p + SLETT_HEADER_SIZE, (size_t)(file_size - SLETT_HEADER_SIZE)) != slett_get32(p + 56))
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: checksum mismatch, the image is corrupt");

    info->version = version;
    info->layers = layers;
    info->loss = (LossFunction)p[12];
    info->flags = p[13];
    info->time_step = slett_get64(p + 16);
    info->size = file_size;
    uint64_t arena = version >= 6u ? slett_get64(p + 40) : 0u;

    uint32_t prev_out = 0;
    for (uint32_t i = 0; i + 1 < layers; i++) {
        SlettLayer L;
        spingalett_slett_layer(p, i, &L);
        /* the input's units: the previous entry's outputs, from version 6 on those of the first
           input (the network's input for the first entry) */
        bool linked = version >= 6u ? inputs_ok(p, i, &L, file_size) &&
                                      (L.input0 == 0 ? i == 0 || L.inputs == slett_get32(p + SLETT_HEADER_SIZE)
                                                     : L.inputs == slett_get32(entry_at(p, L.input0 - 1) + 4)) &&
                                      act_offset_ok(p, i, &L, layers, arena)
                                    : i == 0 || L.inputs == prev_out;
        if (L.inputs == 0 || L.outputs == 0 || !linked ||
            (unsigned)L.activation >= ACT_COUNT || (unsigned)L.precision >= PRECISION_COUNT ||
            !(L.dropout >= 0.0f && L.dropout < 1.0f) || (version >= 4 && !shape_ok(&L, version)))
            return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: invalid layer table entry");
        bool norm = L.type == LAYER_BATCH_NORM, layer_norm = L.type == LAYER_LAYER_NORM;
        bool weighted = L.type == LAYER_DENSE || L.type == LAYER_CONV2D || L.type == LAYER_CONV_TRANSPOSE2D || norm ||
                        layer_norm;
        bool is_int = weighted && !norm && !layer_norm && spingalett_precision_is_int(L.precision);
        if (is_int && L.row_len > SLETT_MAX_INT_INPUTS)
            return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: integer layer has too many inputs per output");
        if (weighted) {
            uint64_t row = spingalett_slett_row_bytes(L.precision, L.row_len);
            uint64_t elem = L.precision == PRECISION_FLOAT32 ? 4u : (L.precision <= PRECISION_BFLOAT16 ? 2u : 1u);
            /* batch normalization: gamma as the weights, the running mean and variance as the scales */
            bool ok = L.rows <= (file_size - SLETT_HEADER_SIZE) / row &&
                      section_ok(L.weights, row * L.rows, file_size, elem) &&
                      section_ok(L.biases, (uint64_t)L.rows * 4u, file_size, 4) &&
                      (!is_int || section_ok(L.scales, (uint64_t)L.rows * 4u, file_size, 4)) &&
                      (!norm || section_ok(L.scales, (uint64_t)L.rows * 8u, file_size, 4)) &&
                      (!layer_norm || L.scales == 0);
            if (ok && (info->flags & SLETT_FLAG_OPTIMIZER)) {
                uint64_t weights = (uint64_t)L.rows * L.row_len;
                ok = weights <= file_size / 8u &&
                     section_ok(L.optimizer, (2u * weights + 2u * L.rows) * 4u, file_size, 4);
            }
            if (!ok)
                return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: a section exceeds the image or is misaligned");
        }
        if (i + 2 < layers && L.outputs > info->max_width) info->max_width = L.outputs;
        if (is_int && L.inputs > info->max_int_inputs) info->max_int_inputs = L.inputs;
        if (slett_conv_scratch(&L) > info->conv_scratch) info->conv_scratch = slett_conv_scratch(&L);
        prev_out = L.outputs;
    }
    info->activations = version >= 6u ? arena : 2u * output_bytes(info->max_width);
    return SPINGALETT_OK;
}

/* ------------------------------------------------------------------------- the engine */

int spingalett_model_init(SpingalettModel *model, const void *image, size_t size) {
    if (!model || !image)
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: model or image is NULL");
    memset(model, 0, sizeof *model);
    if ((uintptr_t)image % 4u != 0)
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: the image address must be a multiple of 4");

    SlettInfo info;
    int rc = spingalett_slett_validate((const uint8_t *)image, size, &info);
    if (rc != SPINGALETT_OK) return rc;

    SlettLayer first, last;
    spingalett_slett_layer((const uint8_t *)image, 0, &first);
    spingalett_slett_layer((const uint8_t *)image, info.layers - 2, &last);
    /* the layers' outputs, the quantized input of an integer layer, convolution scratch */
    uint64_t workspace = info.activations + slett_align(info.max_int_inputs) + slett_align(info.conv_scratch);
    if (info.activations > SIZE_MAX || workspace > SIZE_MAX)
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: the workspace exceeds the address space");

    model->input_size = first.inputs;
    model->output_size = last.outputs;
    model->layer_count = info.layers - 1;
    model->loss = info.loss;
    model->workspace_size = workspace < 16u ? 16u : (size_t)workspace;
    model->image = image;
    model->image_size = (size_t)info.size;
    model->max_width_ = info.max_width;
    model->max_int_inputs_ = info.max_int_inputs;
    model->conv_scratch_ = (size_t)info.conv_scratch;
    model->activations_ = (size_t)info.activations;
    return SPINGALETT_OK;
}

bool spingalett_model_layer(const SpingalettModel *model, uint32_t index, SpingalettLayerInfo *info) {
    if (!model || !model->image || !info || index >= model->layer_count) return false;
    SlettLayer L;
    spingalett_slett_layer((const uint8_t *)model->image, index, &L);
    memset(info, 0, sizeof *info);
    info->type = L.type;
    info->inputs = L.inputs;
    info->outputs = L.outputs;
    info->activation = L.activation;
    info->precision = L.precision;
    info->in_height = L.in_h;
    info->in_width = L.in_w;
    info->in_channels = L.in_c;
    info->groups = L.groups;
    info->epsilon = L.eps;
    info->height = L.out_h;
    info->width = L.out_w;
    info->channels = L.out_c;
    info->kernel_h = L.kernel_h;
    info->kernel_w = L.kernel_w;
    info->stride_h = L.stride_h;
    info->stride_w = L.stride_w;
    info->padding_h = L.pad_h;
    info->padding_w = L.pad_w;
    info->upsample = (UpsampleMode)L.mode;
    info->input_count = L.input_count;
    for (uint32_t k = 0; k < L.input_count; k++)
        info->input_layers[k] = spingalett_slett_input((const uint8_t *)model->image, &L, k);
    return true;
}

/* Outputs j0 .. j0 + count - 1 of a weight layer: bias + the dot products of those weight rows with
   x (row_len values), in float; integer rows take x quantized instead (xq, with x_scale, and
   permuted with sum xsum for packed rows), the sums rescaled. */
SPG_REGISTER_SUMS static void weight_rows(const uint8_t *image, const SlettLayer *L, uint32_t j0, uint32_t count,
                                           const float *x, const int8_t *xq, float x_scale, int32_t xsum, float *y) {
    const float *bias = (const float *)(const void *)(image + L->biases);
    const uint8_t *weights = image + L->weights;
    uint32_t n = L->row_len;
    size_t row = (size_t)spingalett_slett_row_bytes(L->precision, n);
    /* rows in groups of four; the last count % 4 one at a time, as a group of one row repeated
       (stride 0) */
    for (uint32_t j = j0, rows, end = j0 + count; j < end; j += rows) {
        rows = end - j >= 4u ? 4u : 1u;
        size_t stride = rows == 4u ? row : 0u;
        const uint8_t *w = weights + (size_t)j * row;
        switch (L->precision) {
            case PRECISION_FLOAT32: {
                float dot[4];
                dot_f32_rows4((const float *)(const void *)w, stride / 4u, x, n, dot);
                for (uint32_t r = 0; r < rows; r++) y[j - j0 + r] = bias[j + r] + dot[r];
                break;
            }
            case PRECISION_FP16:
            case PRECISION_BFLOAT16: {
                float dot[4];
                dot_f16_rows4((const uint16_t *)(const void *)w, stride / 2u, x, n, L->precision == PRECISION_BFLOAT16, dot);
                for (uint32_t r = 0; r < rows; r++) y[j - j0 + r] = bias[j + r] + dot[r];
                break;
            }
            default: {
                const float *scale = (const float *)(const void *)(image + L->scales);
                int32_t acc[4];
                if (L->precision == PRECISION_INT8)
                    spingalett_dot_i8_rows4((const int8_t *)w, stride, xq, n, acc);
                else
                    dot_packed_rows4(w, stride, xq, n, L->precision == PRECISION_INT4 ? 2u : 4u, xsum, acc);
                for (uint32_t r = 0; r < rows; r++)
                    y[j - j0 + r] = spingalett_int_output(bias[j + r], scale[j + r], x_scale, acc[r]);
                break;
            }
        }
    }
}

void spingalett_gather_window(const void *x, const SlettLayer *L, uint32_t oh, uint32_t ow, void *window, size_t elem) {
    const uint32_t C = L->in_c, KW = L->kernel_w;
    const size_t run = (size_t)KW * C * elem;
    int64_t ih0 = (int64_t)oh * L->stride_h - L->pad_h, iw0 = (int64_t)ow * L->stride_w - L->pad_w;
    /* window columns [a, b) are inside the input */
    int64_t a = iw0 < 0 ? -iw0 : 0, b = iw0 + KW > (int64_t)L->in_w ? (int64_t)L->in_w - iw0 : (int64_t)KW;
    if (b < a) b = a;
    uint8_t *d = (uint8_t *)window;
    for (uint32_t kh = 0; kh < L->kernel_h; kh++, d += run) {
        int64_t ih = ih0 + kh;
        if (ih < 0 || ih >= (int64_t)L->in_h || a == b) { memset(d, 0, run); continue; }
        const uint8_t *src = (const uint8_t *)x + ((size_t)ih * L->in_w + (size_t)(iw0 + a)) * C * elem;
        if (a == 0 && b == (int64_t)KW) {       /* the whole row is inside: one copy */
            memcpy(d, src, run);
            continue;
        }
        memset(d, 0, (size_t)a * C * elem);
        memcpy(d + (size_t)a * C * elem, src, (size_t)(b - a) * C * elem);
        memset(d + (size_t)b * C * elem, 0, (size_t)(KW - b) * C * elem);
    }
}

void spingalett_conv_transpose_filters(const uint8_t *image, const SlettLayer *L, void *wt) {
    const uint32_t R = L->rows, K = L->row_len;
    const uint8_t *w = image + L->weights;
    size_t row = (size_t)spingalett_slett_row_bytes(L->precision, K);
    for (uint32_t j = 0; j < R; j++, w += row)
        for (uint32_t k = 0; k < K; k++) {
            size_t at = (size_t)k * R + j;
            switch (L->precision) {
                case PRECISION_FLOAT32:  { float f; memcpy(&f, w + 4u * k, 4); ((float *)wt)[at] = f; break; }
                case PRECISION_FP16:     ((float *)wt)[at] = spingalett_fp16_to_float(slett_get16(w + 2u * k)); break;
                case PRECISION_BFLOAT16: ((float *)wt)[at] = spingalett_bf16_to_float(slett_get16(w + 2u * k)); break;
                case PRECISION_INT8:     ((int8_t *)wt)[at] = (int8_t)w[k]; break;
                case PRECISION_INT4:     ((int8_t *)wt)[at] = int4_codes[w[k >> 1]][k & 1u]; break;
                default:                 ((int8_t *)wt)[at] = int2_codes[w[k >> 2]][k & 3u]; break;
            }
        }
}

/* Visits the window of output pixel (oh, ow) of convolution L cell by cell inside the input: body
   runs with `cell` the cell's offset in the input (in values) and `k0` the index of its first
   window weight. */
#define FOR_WINDOW_CELLS(L, oh, ow, body)                                                               \
    do {                                                                                                \
        int64_t ih0_ = (int64_t)(oh) * (L)->stride_h - (L)->pad_h, iw0_ = (int64_t)(ow) * (L)->stride_w - (L)->pad_w; \
        for (uint32_t kh_ = 0; kh_ < (L)->kernel_h; kh_++) {                                            \
            int64_t ih_ = ih0_ + kh_;                                                                   \
            if (ih_ < 0 || ih_ >= (int64_t)(L)->in_h) continue;                                         \
            for (uint32_t kw_ = 0; kw_ < (L)->kernel_w; kw_++) {                                        \
                int64_t iw_ = iw0_ + kw_;                                                               \
                if (iw_ < 0 || iw_ >= (int64_t)(L)->in_w) continue;                                     \
                size_t cell = ((size_t)ih_ * (L)->in_w + (size_t)iw_) * (L)->in_c;                      \
                size_t k0 = ((size_t)kh_ * (L)->kernel_w + kw_) * (L)->in_c;                            \
                body                                                                                    \
            }                                                                                           \
        }                                                                                               \
    } while (0)

/* Filters are taken 32, then 16 at a time with their sums in registers (any rest in memory); the
   window's cells (zeros skipped: ReLU outputs, image borders) each add one value times a block of
   weights. */
#define COLUMN_BLOCK(T, V, x, wt, acc, j0, n)                                                           \
    do {                                                                                                \
        T a_[n] = {0};                                                                                  \
        FOR_WINDOW_CELLS(L, oh, ow, {                                                                   \
            const V *w_ = (wt) + k0 * R + (j0);                                                         \
            for (uint32_t c_ = 0; c_ < C; c_++, w_ += R) {                                              \
                V v_ = (x)[cell + c_];                                                                  \
                if (v_ == 0) continue;                                                                  \
                for (uint32_t j_ = 0; j_ < (n); j_++) a_[j_] += COLUMN_PRODUCT(v_, w_[j_]);              \
            }                                                                                           \
        });                                                                                             \
        memcpy((acc) + (j0), a_, sizeof a_);                                                            \
    } while (0)

#define COLUMN_REST(T, V, x, wt, acc, j0)                                                               \
    do {                                                                                                \
        for (uint32_t j_ = (j0); j_ < R; j_++) (acc)[j_] = 0;                                           \
        FOR_WINDOW_CELLS(L, oh, ow, {                                                                   \
            const V *w_ = (wt) + k0 * R;                                                                \
            for (uint32_t c_ = 0; c_ < C; c_++, w_ += R) {                                              \
                V v_ = (x)[cell + c_];                                                                  \
                if (v_ == 0) continue;                                                                  \
                for (uint32_t j_ = (j0); j_ < R; j_++) (acc)[j_] += COLUMN_PRODUCT(v_, w_[j_]);          \
            }                                                                                           \
        });                                                                                             \
    } while (0)

/* |v w| <= 127^2: integer products fit 16 bits */
#define COLUMN_PRODUCT(v, w) (int16_t)((int16_t)(v) * (int16_t)(w))

void spingalett_conv_columns_i8(const int8_t *xq, const SlettLayer *L, uint32_t oh, uint32_t ow, const int8_t *wt,
                                int32_t *acc) {
    const uint32_t R = L->rows, C = L->in_c;
    uint32_t j0 = 0;
    for (; j0 + 32u <= R; j0 += 32u) COLUMN_BLOCK(int32_t, int8_t, xq, wt, acc, j0, 32u);
    if (j0 + 16u <= R) { COLUMN_BLOCK(int32_t, int8_t, xq, wt, acc, j0, 16u); j0 += 16u; }
    if (j0 < R) COLUMN_REST(int32_t, int8_t, xq, wt, acc, j0);
}

#undef COLUMN_PRODUCT
#define COLUMN_PRODUCT(v, w) ((v) * (w))

static void conv_columns_f32(const float *x, const SlettLayer *L, uint32_t oh, uint32_t ow, const float *wt,
                             float *acc) {
    const uint32_t R = L->rows, C = L->in_c;
    uint32_t j0 = 0;
    for (; j0 + 32u <= R; j0 += 32u) COLUMN_BLOCK(float, float, x, wt, acc, j0, 32u);
    if (j0 + 16u <= R) { COLUMN_BLOCK(float, float, x, wt, acc, j0, 16u); j0 += 16u; }
    if (j0 < R) COLUMN_REST(float, float, x, wt, acc, j0);
}

/* A convolution with a short window on one sample, filter-major (scratch: the transposed filters,
   then one pixel's sums). */
static void conv_forward_columns(const uint8_t *image, const SlettLayer *L, const float *x, float *y, int8_t *xq,
                                 void *scratch) {
    const uint32_t R = L->rows;
    const bool is_int = spingalett_precision_is_int(L->precision);
    const float *bias = (const float *)(const void *)(image + L->biases);
    void *sums = (uint8_t *)scratch + slett_align((uint64_t)L->row_len * R * (is_int ? 1u : 4u));
    spingalett_conv_transpose_filters(image, L, scratch);
    if (is_int) {
        const float *scale = (const float *)(const void *)(image + L->scales);
        float x_scale = spingalett_quantize_activations(x, L->inputs, xq);
        int32_t *acc = (int32_t *)sums;
        for (uint32_t oh = 0; oh < L->out_h; oh++)
            for (uint32_t ow = 0; ow < L->out_w; ow++) {
                float *o = y + ((size_t)oh * L->out_w + ow) * R;
                spingalett_conv_columns_i8(xq, L, oh, ow, (const int8_t *)scratch, acc);
                for (uint32_t j = 0; j < R; j++) o[j] = spingalett_int_output(bias[j], scale[j], x_scale, acc[j]);
            }
    } else {
        float *acc = (float *)sums;
        for (uint32_t oh = 0; oh < L->out_h; oh++)
            for (uint32_t ow = 0; ow < L->out_w; ow++) {
                float *o = y + ((size_t)oh * L->out_w + ow) * R;
                conv_columns_f32(x, L, oh, ow, (const float *)scratch, acc);
                for (uint32_t j = 0; j < R; j++) o[j] = bias[j] + acc[j];
            }
    }
}

void spingalett_gather_group_window(const void *x, const SlettLayer *L, uint32_t oh, uint32_t ow, uint32_t g,
                                    void *window, size_t elem) {
    const uint32_t C = L->in_c, CG = C / L->groups;
    const size_t run = (size_t)CG * elem;
    uint8_t *d = (uint8_t *)window;
    int64_t ih0 = (int64_t)oh * L->stride_h - L->pad_h, iw0 = (int64_t)ow * L->stride_w - L->pad_w;
    for (uint32_t kh = 0; kh < L->kernel_h; kh++)
        for (uint32_t kw = 0; kw < L->kernel_w; kw++, d += run) {
            int64_t ih = ih0 + kh, iw = iw0 + kw;
            if (ih < 0 || ih >= (int64_t)L->in_h || iw < 0 || iw >= (int64_t)L->in_w) { memset(d, 0, run); continue; }
            memcpy(d, (const uint8_t *)x + (((size_t)ih * L->in_w + (size_t)iw) * C + (size_t)g * CG) * elem, run);
        }
}

/* Each filter of a depthwise convolution reads channel j / m (m filters per channel). */
#define DEPTHWISE_SUMS(T, x, wt, acc)                                                                \
    do {                                                                                            \
        const uint32_t R = L->rows, C = L->in_c, m = R / C;                                         \
        for (uint32_t j = 0; j < R; j++) (acc)[j] = 0;                                              \
        int64_t ih0 = (int64_t)oh * L->stride_h - L->pad_h, iw0 = (int64_t)ow * L->stride_w - L->pad_w; \
        for (uint32_t kh = 0; kh < L->kernel_h; kh++) {                                             \
            int64_t ih = ih0 + kh;                                                                  \
            if (ih < 0 || ih >= (int64_t)L->in_h) continue;                                         \
            for (uint32_t kw = 0; kw < L->kernel_w; kw++) {                                         \
                int64_t iw = iw0 + kw;                                                              \
                if (iw < 0 || iw >= (int64_t)L->in_w) continue;                                     \
                const T *v = (x) + ((size_t)ih * L->in_w + (size_t)iw) * C;                         \
                const T *w = (wt) + ((size_t)kh * L->kernel_w + kw) * R;                            \
                if (m == 1) for (uint32_t c = 0; c < C; c++) (acc)[c] += DEPTHWISE_PRODUCT(v[c], w[c]); \
                else for (uint32_t c = 0; c < C; c++)                                               \
                    for (uint32_t i = 0; i < m; i++) (acc)[c * m + i] += DEPTHWISE_PRODUCT(v[c], w[c * m + i]); \
            }                                                                                       \
        }                                                                                           \
    } while (0)

#define DEPTHWISE_PRODUCT(v, w) (int32_t)((int16_t)(v) * (int16_t)(w))

void spingalett_conv_depthwise_i8(const int8_t *xq, const SlettLayer *L, uint32_t oh, uint32_t ow, const int8_t *wt,
                                  int32_t *acc) {
    DEPTHWISE_SUMS(int8_t, xq, wt, acc);
}

#undef DEPTHWISE_PRODUCT
#define DEPTHWISE_PRODUCT(v, w) ((v) * (w))

static void conv_depthwise_f32(const float *x, const SlettLayer *L, uint32_t oh, uint32_t ow, const float *wt,
                               float *acc) {
    DEPTHWISE_SUMS(float, x, wt, acc);
}

/* A depthwise convolution on one sample (scratch: the transposed filters, then one pixel's sums). */
static void conv_forward_depthwise(const uint8_t *image, const SlettLayer *L, const float *x, float *y, int8_t *xq,
                                   void *scratch) {
    const uint32_t R = L->rows;
    const bool is_int = spingalett_precision_is_int(L->precision);
    const float *bias = (const float *)(const void *)(image + L->biases);
    void *sums = (uint8_t *)scratch + slett_align((uint64_t)L->row_len * R * (is_int ? 1u : 4u));
    spingalett_conv_transpose_filters(image, L, scratch);
    const float *scale = is_int ? (const float *)(const void *)(image + L->scales) : NULL;
    float x_scale = is_int ? spingalett_quantize_activations(x, L->inputs, xq) : 0.0f;
    for (uint32_t oh = 0; oh < L->out_h; oh++)
        for (uint32_t ow = 0; ow < L->out_w; ow++) {
            float *o = y + ((size_t)oh * L->out_w + ow) * R;
            if (is_int) {
                int32_t *acc = (int32_t *)sums;
                spingalett_conv_depthwise_i8(xq, L, oh, ow, (const int8_t *)scratch, acc);
                for (uint32_t j = 0; j < R; j++) o[j] = spingalett_int_output(bias[j], scale[j], x_scale, acc[j]);
            } else {
                float *acc = (float *)sums;
                conv_depthwise_f32(x, L, oh, ow, (const float *)scratch, acc);
                for (uint32_t j = 0; j < R; j++) o[j] = bias[j] + acc[j];
            }
        }
}

/* A grouped convolution on one sample: each group's filters dotted with the window of its channels. */
static void conv_forward_grouped(const uint8_t *image, const SlettLayer *L, const float *x, float *y, int8_t *xq,
                                 void *window) {
    const uint32_t G = L->groups, OG = L->rows / G, K = L->row_len;
    const bool is_int = spingalett_precision_is_int(L->precision), packed = is_int && L->precision != PRECISION_INT8;
    const uint32_t per_byte = L->precision == PRECISION_INT4 ? 2u : 4u;
    float x_scale = is_int ? spingalett_quantize_activations(x, L->inputs, xq) : 0.0f;
    for (uint32_t oh = 0; oh < L->out_h; oh++)
        for (uint32_t ow = 0; ow < L->out_w; ow++) {
            float *o = y + ((size_t)oh * L->out_w + ow) * L->rows;
            for (uint32_t g = 0; g < G; g++) {
                spingalett_gather_group_window(is_int ? (const void *)xq : (const void *)x, L, oh, ow, g, window,
                                               is_int ? 1u : 4u);
                int32_t xsum = packed ? permute_activations((int8_t *)window, K, per_byte) : 0;
                weight_rows(image, L, g * OG, OG, (const float *)window, (const int8_t *)window, x_scale, xsum,
                            o + (size_t)g * OG);
            }
        }
}

/* The inputs that reach output pixel (oh, ow) of transposed convolution L through each tap of its
   window, over the channels of group g: input cell ((oh + pad - kh) / stride, ...) where the division
   is exact and the cell inside the input, zeros elsewhere. */
static void gather_transposed_window(const void *x, const SlettLayer *L, uint32_t oh, uint32_t ow, uint32_t g,
                                     void *window, size_t elem) {
    const uint32_t C = L->in_c, CG = C / L->groups;
    const size_t run = (size_t)CG * elem;
    uint8_t *d = (uint8_t *)window;
    for (uint32_t kh = 0; kh < L->kernel_h; kh++) {
        const int64_t nh = (int64_t)oh + L->pad_h - kh, ih = nh / (int64_t)L->stride_h;
        const bool row = nh >= 0 && nh % (int64_t)L->stride_h == 0 && ih < (int64_t)L->in_h;
        for (uint32_t kw = 0; kw < L->kernel_w; kw++, d += run) {
            const int64_t nw = (int64_t)ow + L->pad_w - kw, iw = nw / (int64_t)L->stride_w;
            if (!row || nw < 0 || nw % (int64_t)L->stride_w != 0 || iw >= (int64_t)L->in_w) { memset(d, 0, run); continue; }
            memcpy(d, (const uint8_t *)x + (((size_t)ih * L->in_w + (size_t)iw) * C + (size_t)g * CG) * elem, run);
        }
    }
}

void spingalett_engine_conv_transpose(const uint8_t *image, const SlettLayer *L, const float *x, float *y, int8_t *xq,
                                      void *window) {
    const uint32_t G = L->groups, OG = L->rows / G, K = L->row_len;
    const bool is_int = spingalett_precision_is_int(L->precision), packed = is_int && L->precision != PRECISION_INT8;
    const uint32_t per_byte = L->precision == PRECISION_INT4 ? 2u : 4u;
    float x_scale = is_int ? spingalett_quantize_activations(x, L->inputs, xq) : 0.0f;
    for (uint32_t oh = 0; oh < L->out_h; oh++)
        for (uint32_t ow = 0; ow < L->out_w; ow++) {
            float *o = y + ((size_t)oh * L->out_w + ow) * L->rows;
            for (uint32_t g = 0; g < G; g++) {
                gather_transposed_window(is_int ? (const void *)xq : (const void *)x, L, oh, ow, g, window,
                                         is_int ? 1u : 4u);
                int32_t xsum = packed ? permute_activations((int8_t *)window, K, per_byte) : 0;
                weight_rows(image, L, g * OG, OG, (const float *)window, (const int8_t *)window, x_scale, xsum,
                            o + (size_t)g * OG);
            }
        }
    spingalett_engine_activate(y, L->outputs, L->activation);
}

/* A convolution on one sample: each output pixel is its filters' dot products with its window. */
static void conv_forward(const uint8_t *image, const SlettLayer *L, const float *x, float *y, int8_t *xq, void *window) {
    if (slett_conv_columns(L)) {
        conv_forward_columns(image, L, x, y, xq, window);
        return;
    }
    if (slett_conv_depthwise(L)) {
        conv_forward_depthwise(image, L, x, y, xq, window);
        return;
    }
    if (L->groups > 1) {
        conv_forward_grouped(image, L, x, y, xq, window);
        return;
    }
    const uint32_t OC = L->rows, K = L->row_len;
    const bool is_int = spingalett_precision_is_int(L->precision), packed = is_int && L->precision != PRECISION_INT8;
    const bool pointwise = L->kernel_h == 1 && L->kernel_w == 1 && L->stride_h == 1 && L->stride_w == 1 &&
                           L->pad_h == 0 && L->pad_w == 0;
    const uint32_t per_byte = L->precision == PRECISION_INT4 ? 2u : 4u;
    float x_scale = is_int ? spingalett_quantize_activations(x, L->inputs, xq) : 0.0f;
    if (L->precision == PRECISION_INT8) {
        /* four pixels at a time (their windows fill the scratch of one float window), each filter
           read once for the four; the integer sums are those of one pixel at a time */
        const float *bias = (const float *)(const void *)(image + L->biases);
        const float *scale = (const float *)(const void *)(image + L->scales);
        const int8_t *weights = (const int8_t *)(image + L->weights);
        const uint32_t pixels = L->out_h * L->out_w;
        int8_t *windows = (int8_t *)window;
        int32_t *wsum = (int32_t *)(void *)((uint8_t *)window + slett_align((uint64_t)K * 4u));
        if (spingalett_dot_i8_4x4_sums && pixels >= 4u)
            for (uint32_t j = 0; j < OC; j++) wsum[j] = spingalett_sum_i8(weights + (size_t)j * K, K);
        for (uint32_t p0 = 0; p0 < pixels; p0 += 4u) {
            uint32_t count = pixels - p0 < 4u ? pixels - p0 : 4u;
            for (uint32_t i = 0; i < count; i++) {
                uint32_t p = p0 + i;
                if (pointwise) memcpy(windows + (size_t)i * K, xq + (size_t)p * L->in_c, K);
                else spingalett_gather_window(xq, L, p / L->out_w, p % L->out_w, windows + (size_t)i * K, 1);
            }
            for (uint32_t j = 0; j < OC; j += 4u) {
                uint32_t rows = OC - j < 4u ? OC - j : 4u;
                const int8_t *block = weights + (size_t)j * K;
                int32_t acc[16];
                if (rows == 4u && count == 4u) {
                    spingalett_dot_i8_4x4(block, K, windows, K, K, wsum + j, acc);
                } else {
                    for (uint32_t i = 0; i < count; i++)
                        for (uint32_t r = 0; r < rows; r++)
                            acc[4u * r + i] = spingalett_dot_i8(block + (size_t)r * K, windows + (size_t)i * K, K);
                }
                for (uint32_t i = 0; i < count; i++)
                    for (uint32_t r = 0; r < rows; r++)
                        y[(size_t)(p0 + i) * OC + j + r] = spingalett_int_output(bias[j + r], scale[j + r], x_scale, acc[4u * r + i]);
            }
        }
        return;
    }
    for (uint32_t oh = 0; oh < L->out_h; oh++)
        for (uint32_t ow = 0; ow < L->out_w; ow++) {
            size_t p = (size_t)oh * L->out_w + ow;
            const float *xw = x + p * L->in_c;
            const int8_t *qw = xq + p * L->in_c;
            int32_t xsum = 0;
            if (!pointwise || packed) {         /* packed rows permute their activations in place */
                if (is_int && pointwise) memcpy(window, qw, K);
                else spingalett_gather_window(is_int ? (const void *)xq : (const void *)x, L, oh, ow, window, is_int ? 1u : 4u);
                xw = (const float *)window;
                qw = (const int8_t *)window;
                if (packed) xsum = permute_activations((int8_t *)window, K, per_byte);
            }
            weight_rows(image, L, 0, OC, xw, qw, x_scale, xsum, y + p * OC);
        }
}

/* Pooling on one sample, cell by cell in the order the training kernels use, so results match. */
static void pool_forward(const SlettLayer *L, const float *x, float *y) {
    const uint32_t C = L->in_c, W = L->in_w;
    const bool max = L->type == LAYER_MAX_POOL2D;
    for (uint32_t oh = 0; oh < L->out_h; oh++)
        for (uint32_t ow = 0; ow < L->out_w; ow++) {
            int64_t hs = (int64_t)oh * L->stride_h - L->pad_h, ws = (int64_t)ow * L->stride_w - L->pad_w;
            int64_t he = hs + L->kernel_h, we = ws + L->kernel_w;
            uint32_t h0 = (uint32_t)(hs < 0 ? 0 : hs), h1 = (uint32_t)(he > (int64_t)L->in_h ? L->in_h : he);
            uint32_t w0 = (uint32_t)(ws < 0 ? 0 : ws), w1 = (uint32_t)(we > (int64_t)W ? W : we);
            float inv = max ? 1.0f : 1.0f / (float)((h1 - h0) * (w1 - w0));
            float *o = y + ((size_t)oh * L->out_w + ow) * C;
            for (uint32_t c = 0; c < C; c++) {
                float acc = x[((size_t)h0 * W + w0) * C + c];
                for (uint32_t h = h0; h < h1; h++)
                    for (uint32_t w = (h == h0 ? w0 + 1 : w0); w < w1; w++) {
                        float v = x[((size_t)h * W + w) * C + c];
                        if (max) { if (v > acc) acc = v; }
                        else acc += v;
                    }
                o[c] = acc * inv;
            }
        }
}

/* Batch normalization on one sample with the stored running statistics (coef: 2 x channels). */
static void norm_forward(const uint8_t *image, const SlettLayer *L, const float *x, float *y, float *coef) {
    const uint32_t C = L->out_c;
    const float *stats = (const float *)(const void *)(image + L->scales);
    spingalett_bn_coefficients((const float *)(const void *)(image + L->weights),
                               (const float *)(const void *)(image + L->biases), stats, stats + C, L->eps, C,
                               coef, coef + C);
    for (uint32_t i = 0; i < L->outputs; i += C)
        for (uint32_t c = 0; c < C; c++) y[i + c] = x[i + c] * coef[c] + coef[C + c];
}

void spingalett_engine_add(const float *const *x, uint32_t count, float *y, uint32_t n) {
    if (count == 1) {
        memcpy(y, x[0], (size_t)n * sizeof(float));
        return;
    }
    for (uint32_t i = 0; i < n; i++) y[i] = x[0][i] + x[1][i];
    for (uint32_t k = 2; k < count; k++)
        for (uint32_t i = 0; i < n; i++) y[i] += x[k][i];
}

void spingalett_engine_concat(const float *const *x, const uint32_t *channels, uint32_t count, float *y,
                              uint32_t cells) {
    uint32_t C = 0, c0 = 0;
    for (uint32_t k = 0; k < count; k++) C += channels[k];
    for (uint32_t k = 0; k < count; c0 += channels[k], k++)
        for (uint32_t p = 0; p < cells; p++)
            memcpy(y + (size_t)p * C + c0, x[k] + (size_t)p * channels[k], (size_t)channels[k] * sizeof(float));
}

void spingalett_engine_global_pool(const float *x, float *y, uint32_t cells, uint32_t channels) {
    const float inv = 1.0f / (float)cells;
    for (uint32_t c = 0; c < channels; c++) y[c] = x[c];
    for (uint32_t p = 1; p < cells; p++)
        for (uint32_t c = 0; c < channels; c++) y[c] += x[(size_t)p * channels + c];
    for (uint32_t c = 0; c < channels; c++) y[c] *= inv;
}

void spingalett_engine_bilinear(uint32_t o, uint32_t factor, uint32_t in, uint32_t *r0, uint32_t *r1, float *w) {
    /* source coordinate (o + 1/2) / factor - 1/2 = (2 o + 1 - factor) / (2 factor), at least 0 */
    const int64_t num = 2 * (int64_t)o + 1 - (int64_t)factor, den = 2 * (int64_t)factor;
    if (num <= 0) {
        *r0 = *r1 = 0;
        *w = 0.0f;
        return;
    }
    *r0 = (uint32_t)(num / den);
    *r1 = *r0 + 1 < in ? *r0 + 1 : *r0;
    *w = (float)(num % den) / (float)den;
}

void spingalett_engine_upsample(const float *x, uint32_t in_h, uint32_t in_w, uint32_t channels, uint32_t sh,
                                uint32_t sw, uint32_t mode, float *y) {
    const uint32_t H = in_h * sh, W = in_w * sw, C = channels;
    if (mode == UPSAMPLE_NEAREST) {
        for (uint32_t oy = 0; oy < H; oy++) {
            const float *row = x + (size_t)(oy / sh) * in_w * C;
            float *out = y + (size_t)oy * W * C;
            for (uint32_t ox = 0; ox < W; ox++) memcpy(out + (size_t)ox * C, row + (size_t)(ox / sw) * C, (size_t)C * sizeof(float));
        }
        return;
    }
    for (uint32_t oy = 0; oy < H; oy++) {
        uint32_t y0, y1, x0, x1;
        float ly, lx;
        spingalett_engine_bilinear(oy, sh, in_h, &y0, &y1, &ly);
        const float hy = 1.0f - ly;
        const float *r0 = x + (size_t)y0 * in_w * C, *r1 = x + (size_t)y1 * in_w * C;
        float *out = y + (size_t)oy * W * C;
        for (uint32_t ox = 0; ox < W; ox++) {
            spingalett_engine_bilinear(ox, sw, in_w, &x0, &x1, &lx);
            const float hx = 1.0f - lx;
            const float *a = r0 + (size_t)x0 * C, *b = r0 + (size_t)x1 * C, *c = r1 + (size_t)x0 * C, *d = r1 + (size_t)x1 * C;
            float *o = out + (size_t)ox * C;
            for (uint32_t k = 0; k < C; k++) {
                float top = hx * a[k] + lx * b[k], bottom = hx * c[k] + lx * d[k];
                o[k] = hy * top + ly * bottom;
            }
        }
    }
}

void spingalett_engine_layer_norm(const float *x, uint32_t cells, uint32_t channels, const float *gamma,
                                  const float *beta, float eps, float *y, float *stats) {
    const float inv = 1.0f / (float)channels;
    for (uint32_t p = 0; p < cells; p++) {
        const float *v = x + (size_t)p * channels;
        float *o = y + (size_t)p * channels;
        float sum = 0.0f;
        for (uint32_t c = 0; c < channels; c++) sum += v[c];
        const float mean = sum * inv;
        float var = 0.0f;
        for (uint32_t c = 0; c < channels; c++) {
            float d = v[c] - mean;
            var += d * d;
        }
        const float rstd = 1.0f / sqrtf(var * inv + eps);
        for (uint32_t c = 0; c < channels; c++) o[c] = (v[c] - mean) * rstd * gamma[c] + beta[c];
        if (stats) {
            stats[2u * p] = mean;
            stats[2u * p + 1u] = rstd;
        }
    }
}

/* One layer on one sample: x [inputs] -> y [outputs], activation included. */
static void layer_forward(const uint8_t *image, const SlettLayer *L, const float *x, float *y, int8_t *xq,
                          void *window) {
    switch (L->type) {
        case LAYER_CONV2D:
            conv_forward(image, L, x, y, xq, window);
            break;
        case LAYER_BATCH_NORM:
            norm_forward(image, L, x, y, (float *)window);
            break;
        case LAYER_MAX_POOL2D:
        case LAYER_AVG_POOL2D:
            pool_forward(L, x, y);
            break;
        case LAYER_CONV_TRANSPOSE2D:
            spingalett_engine_conv_transpose(image, L, x, y, xq, window);
            return;                     /* activation included */
        case LAYER_UPSAMPLE:
            spingalett_engine_upsample(x, L->in_h, L->in_w, L->in_c, L->stride_h, L->stride_w, L->mode, y);
            break;
        case LAYER_LAYER_NORM:
            spingalett_engine_layer_norm(x, L->out_h * L->out_w, L->out_c, (const float *)(const void *)(image + L->weights),
                                         (const float *)(const void *)(image + L->biases), L->eps, y, NULL);
            break;
        default: {
            bool is_int = spingalett_precision_is_int(L->precision);
            float x_scale = is_int ? spingalett_quantize_activations(x, L->inputs, xq) : 0.0f;
            int32_t xsum = is_int && L->precision != PRECISION_INT8
                         ? permute_activations(xq, L->inputs, L->precision == PRECISION_INT4 ? 2u : 4u) : 0;
            weight_rows(image, L, 0, L->outputs, x, xq, x_scale, xsum, y);
            break;
        }
    }
    spingalett_engine_activate(y, L->outputs, L->activation);
}

int spingalett_model_run(const SpingalettModel *model, const float *input, float *output, void *workspace) {
    if (!model || !model->image || !input || !output)
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model run: model, input or output is NULL");
    void *temp = NULL;
    if (!workspace) {
#if defined(SPINGALETT_INFERENCE_ONLY)
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model run: workspace is NULL");
#else
        temp = workspace = spingalett_aligned_alloc(model->workspace_size);
        if (!temp) return ENGINE_FAIL(SPINGALETT_ERR_ALLOC, "model run: workspace allocation failed");
#endif
    }
    if ((uintptr_t)workspace % 4u != 0)
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model run: the workspace address must be a multiple of 4");

    const uint8_t *image = (const uint8_t *)model->image;
    uint8_t *activations = (uint8_t *)workspace;
    int8_t *xq = (int8_t *)workspace + model->activations_;
    void *window = (uint8_t *)xq + slett_align(model->max_int_inputs_);

    if (slett_get16(image + 6) < 6u) {
        /* a chain: the outputs alternate between two buffers */
        float *buffers[2] = {(float *)workspace, (float *)(void *)(activations + model->activations_ / 2u)};
        const float *x = input;
        for (uint32_t i = 0; i < model->layer_count; i++) {
            SlettLayer L;
            spingalett_slett_layer(image, i, &L);
            float *y = i + 1 == model->layer_count ? output : buffers[i & 1u];
            layer_forward(image, &L, x, y, xq, window);
            x = y;
        }
    } else {
        /* a graph: every output in its place among the activations, until its last reader is done */
        for (uint32_t i = 0; i < model->layer_count; i++) {
            SlettLayer L;
            spingalett_slett_layer(image, i, &L);
            float *y = i + 1 == model->layer_count ? output : (float *)(void *)(activations + L.act_offset);
            const float *x[SPINGALETT_MAX_INPUTS];
            uint32_t channels[SPINGALETT_MAX_INPUTS];
            for (uint32_t k = 0; k < L.input_count; k++) {
                uint32_t j = spingalett_slett_input(image, &L, k), h, w, units;
                x[k] = j == 0 ? input : (const float *)(const void *)(activations + spingalett_slett_act_offset(image, j));
                spingalett_slett_output_shape(image, j, &h, &w, &units);
                channels[k] = units / (h * w);
            }
            if (L.type == LAYER_ADD) spingalett_engine_add(x, L.input_count, y, L.outputs);
            else if (L.type == LAYER_CONCAT) spingalett_engine_concat(x, channels, L.input_count, y, L.out_h * L.out_w);
            else if (L.type == LAYER_GLOBAL_AVG_POOL) spingalett_engine_global_pool(x[0], y, L.in_h * L.in_w, L.in_c);
            else {
                layer_forward(image, &L, x[0], y, xq, window);
                continue;
            }
            spingalett_engine_activate(y, L.outputs, L.activation);
        }
    }
#if !defined(SPINGALETT_INFERENCE_ONLY)
    spingalett_aligned_free(temp);
#else
    (void)temp;
#endif
    return SPINGALETT_OK;
}
