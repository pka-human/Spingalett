/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The inference engine (Spingalett.Inference.h) and the parts of the .slett format version 3 the
 * rest of the library shares with it (Spingalett.Engine.h). With -DSPINGALETT_INFERENCE_ONLY this
 * file builds on its own and calls nothing beyond memcpy, memset, memcmp, expf, tanhf and lrintf.
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

uint32_t spingalett_crc32(uint32_t crc, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    crc = ~crc;
    for (size_t i = 0; i < n; i++) crc = crc_table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}

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
#endif

int32_t spingalett_dot_i8(const int8_t *a, const int8_t *b, uint32_t n) {
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


void spingalett_dot_i8_rows4(const int8_t *w, size_t stride, const int8_t *x, uint32_t n, int32_t acc[4]) {
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

/* INT4 row times int8 activations, the codes unpacked in registers where possible. */
static int32_t dot_i4(const uint8_t *w, const int8_t *x, uint32_t n) {
    uint32_t k = 0;
    int32_t sum = 0;
#if defined(SPG_AVX2)
    const __m128i low = _mm_set1_epi8(0x0F);
    const __m256i eight = _mm256_set1_epi8(8);
    __m256i acc = _mm256_setzero_si256();
    for (; k + 32u <= n; k += 32u) {
        __m128i b = _mm_loadu_si128((const __m128i *)(w + k / 2u));
        __m128i lo = _mm_and_si128(b, low), hi = _mm_and_si128(_mm_srli_epi16(b, 4), low);
        __m256i v = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi8(lo, hi)), _mm_unpackhi_epi8(lo, hi), 1);
        v = _mm256_sub_epi8(_mm256_xor_si256(v, eight), eight);
        __m256i vx = _mm256_loadu_si256((const __m256i *)(x + k));
        acc = madd_i8(acc, _mm256_sign_epi8(vx, vx), v, vx);
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    sum = _mm_cvtsi128_si32(s);
#elif defined(SPG_NEON)
    int32x4_t acc = vdupq_n_s32(0);
    const int8x16_t eight = vdupq_n_s8(8);
    for (; k + 32u <= n; k += 32u) {
        uint8x16_t b = vld1q_u8(w + k / 2u);
        int8x16_t lo = vsubq_s8(veorq_s8(vreinterpretq_s8_u8(vandq_u8(b, vdupq_n_u8(0x0F))), eight), eight);
        int8x16_t hi = vsubq_s8(veorq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), eight), eight);
        int8x16x2_t v = vzipq_s8(lo, hi);
        int8x16_t x0 = vld1q_s8(x + k), x1 = vld1q_s8(x + k + 16);
#  if defined(__ARM_FEATURE_DOTPROD)
        acc = vdotq_s32(vdotq_s32(acc, v.val[0], x0), v.val[1], x1);
#  else
        int16x8_t p = vmull_s8(vget_low_s8(v.val[0]), vget_low_s8(x0));
        p = vmlal_s8(p, vget_high_s8(v.val[0]), vget_high_s8(x0));
        acc = vpadalq_s16(acc, p);
        p = vmull_s8(vget_low_s8(v.val[1]), vget_low_s8(x1));
        p = vmlal_s8(p, vget_high_s8(v.val[1]), vget_high_s8(x1));
        acc = vpadalq_s16(acc, p);
#  endif
    }
#  if defined(__aarch64__)
    sum = vaddvq_s32(acc);
#  else
    int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    sum = vget_lane_s32(vpadd_s32(s2, s2), 0);
#  endif
#endif
    int8_t block[256];
    while (k < n) {                               /* k is even: whole bytes */
        uint32_t len = n - k < 256u ? n - k : 256u;
        spingalett_unpack_int4(w + k / 2u, block, len);
        sum += spingalett_dot_i8(block, x + k, len);
        k += len;
    }
    return sum;
}

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

/* Half and bfloat16 rows: converted in blocks to float, or with conversion instructions. */
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

/* INT4 and INT2 rows; INT2 is unpacked in blocks of 256 (a multiple of 4, so blocks start on a byte). */
static int32_t dot_packed(const uint8_t *w, const int8_t *x, uint32_t n, PrecisionMode precision) {
    if (precision == PRECISION_INT4) return dot_i4(w, x, n);
    int8_t block[256];
    int32_t acc = 0;
    for (uint32_t k = 0; k < n; k += 256u) {
        uint32_t len = n - k < 256u ? n - k : 256u;
        spingalett_unpack_int2(w + k / 4u, block, len);
        acc += spingalett_dot_i8(block, x + k, len);
    }
    return acc;
}

float spingalett_quantize_activations(const float *x, uint32_t n, int8_t *q) {
    float amax = 0.0f;
    bool bad = false;
    uint32_t k = 0;
#if defined(SPG_AVX)
    __m256 vmax = _mm256_setzero_ps(), vbad = _mm256_setzero_ps();
    const __m256 sign = _mm256_set1_ps(-0.0f), big = _mm256_set1_ps(FLT_MAX);
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

/* ------------------------------------------------------------------------- format 3 */

void spingalett_slett_layer(const uint8_t *image, uint32_t index, SlettLayer *layer) {
    const uint8_t *e = image + SLETT_HEADER_SIZE + (size_t)index * SLETT_LAYER_ENTRY_SIZE;
    layer->inputs = slett_get32(e);
    layer->outputs = slett_get32(e + 4);
    layer->activation = (ActivationFunction)e[8];
    layer->precision = (PrecisionMode)e[9];
    uint32_t d = slett_get32(e + 12);
    memcpy(&layer->dropout, &d, 4);
    layer->weights = slett_get64(e + 16);
    layer->scales = slett_get64(e + 24);
    layer->biases = slett_get64(e + 32);
    layer->optimizer = slett_get64(e + 40);
}

/* [offset, offset + bytes) lies within size and offset is a multiple of `align`. */
static bool section_ok(uint64_t offset, uint64_t bytes, uint64_t size, uint64_t align) {
    return offset >= SLETT_HEADER_SIZE && offset % align == 0 && offset <= size && bytes <= size - offset;
}

int spingalett_slett_validate(const uint8_t *p, size_t size, SlettInfo *info) {
    memset(info, 0, sizeof *info);
    if (!spingalett_host_is_little_endian())
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: format 3 needs a little-endian host");
    if (size < 8 || memcmp(p, SLETT_MAGIC, 6) != 0)
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: not a .slett format 3 image");
    if (slett_get16(p + 6) != 3)
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
    uint64_t table_end = SLETT_HEADER_SIZE + (uint64_t)(layers - 1) * SLETT_LAYER_ENTRY_SIZE;
    if (table_end > file_size)
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: layer table exceeds the image");
    if (spingalett_crc32(0, p + SLETT_HEADER_SIZE, (size_t)(file_size - SLETT_HEADER_SIZE)) != slett_get32(p + 56))
        return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: checksum mismatch, the image is corrupt");

    info->layers = layers;
    info->loss = (LossFunction)p[12];
    info->flags = p[13];
    info->time_step = slett_get64(p + 16);
    info->size = file_size;

    uint32_t prev_out = 0;
    for (uint32_t i = 0; i + 1 < layers; i++) {
        SlettLayer L;
        spingalett_slett_layer(p, i, &L);
        if (L.inputs == 0 || L.outputs == 0 || (i > 0 && L.inputs != prev_out) ||
            (unsigned)L.activation >= ACT_COUNT || (unsigned)L.precision >= PRECISION_COUNT ||
            !(L.dropout >= 0.0f && L.dropout < 1.0f))
            return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: invalid layer table entry");
        bool is_int = spingalett_precision_is_int(L.precision);
        if (is_int && L.inputs > SLETT_MAX_INT_INPUTS)
            return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: integer layer has too many inputs");
        uint64_t row = spingalett_slett_row_bytes(L.precision, L.inputs);
        uint64_t elem = L.precision == PRECISION_FLOAT32 ? 4u : (L.precision <= PRECISION_BFLOAT16 ? 2u : 1u);
        bool ok = L.outputs <= (file_size - SLETT_HEADER_SIZE) / row &&
                  section_ok(L.weights, row * L.outputs, file_size, elem) &&
                  section_ok(L.biases, (uint64_t)L.outputs * 4u, file_size, 4) &&
                  (!is_int || section_ok(L.scales, (uint64_t)L.outputs * 4u, file_size, 4));
        if (ok && (info->flags & SLETT_FLAG_OPTIMIZER)) {
            uint64_t weights = (uint64_t)L.inputs * L.outputs;
            ok = weights <= file_size / 8u &&
                 section_ok(L.optimizer, (2u * weights + 2u * L.outputs) * 4u, file_size, 4);
        }
        if (!ok)
            return ENGINE_FAIL(SPINGALETT_ERR_INVALID, "model: a section exceeds the image or is misaligned");
        if (i + 2 < layers && L.outputs > info->max_width) info->max_width = L.outputs;
        if (is_int && L.inputs > info->max_int_inputs) info->max_int_inputs = L.inputs;
        prev_out = L.outputs;
    }
    return SPINGALETT_OK;
}

/* ------------------------------------------------------------------------- the engine */

static size_t workspace_float_bytes(uint32_t width) {
    return (size_t)slett_align((uint64_t)width * 4u);
}

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
    size_t workspace = 2u * workspace_float_bytes(info.max_width) + (size_t)slett_align(info.max_int_inputs);

    model->input_size = first.inputs;
    model->output_size = last.outputs;
    model->layer_count = info.layers - 1;
    model->loss = info.loss;
    model->workspace_size = workspace < 16u ? 16u : workspace;
    model->image = image;
    model->image_size = (size_t)info.size;
    model->max_width_ = info.max_width;
    return SPINGALETT_OK;
}

bool spingalett_model_layer(const SpingalettModel *model, uint32_t index, SpingalettLayerInfo *info) {
    if (!model || !model->image || !info || index >= model->layer_count) return false;
    SlettLayer L;
    spingalett_slett_layer((const uint8_t *)model->image, index, &L);
    info->inputs = L.inputs;
    info->outputs = L.outputs;
    info->activation = L.activation;
    info->precision = L.precision;
    return true;
}

/* One weight layer on one sample: x [inputs] -> y [outputs], activation included. */
static void layer_forward(const uint8_t *image, const SlettLayer *L, const float *x, float *y, int8_t *xq) {
    const float *bias = (const float *)(const void *)(image + L->biases);
    const uint8_t *weights = image + L->weights;
    uint32_t in = L->inputs, out = L->outputs;

    switch (L->precision) {
        case PRECISION_FLOAT32: {
            const float *W = (const float *)(const void *)weights;
            for (uint32_t j = 0; j < out; j++) y[j] = bias[j] + dot_f32(W + (size_t)j * in, x, in);
            break;
        }
        case PRECISION_FP16:
        case PRECISION_BFLOAT16: {
            const uint16_t *W = (const uint16_t *)(const void *)weights;
            bool bf16 = L->precision == PRECISION_BFLOAT16;
            for (uint32_t j = 0; j < out; j++) y[j] = bias[j] + dot_f16(W + (size_t)j * in, x, in, bf16);
            break;
        }
        default: {
            const float *scale = (const float *)(const void *)(image + L->scales);
            float x_scale = spingalett_quantize_activations(x, in, xq);
            size_t row = (size_t)spingalett_slett_row_bytes(L->precision, in);
            uint32_t j = 0;
            if (L->precision == PRECISION_INT8)
                for (; j + 4u <= out; j += 4u) {
                    int32_t acc[4];
                    spingalett_dot_i8_rows4((const int8_t *)(weights + (size_t)j * row), row, xq, in, acc);
                    for (uint32_t r = 0; r < 4; r++) y[j + r] = spingalett_int_output(bias[j + r], scale[j + r], x_scale, acc[r]);
                }
            for (; j < out; j++) {
                const uint8_t *w = weights + (size_t)j * row;
                int32_t acc = L->precision == PRECISION_INT8 ? spingalett_dot_i8((const int8_t *)w, xq, in)
                                                             : dot_packed(w, xq, in, L->precision);
                y[j] = spingalett_int_output(bias[j], scale[j], x_scale, acc);
            }
            break;
        }
    }
    spingalett_engine_activate(y, out, L->activation);
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
    size_t width_bytes = workspace_float_bytes(model->max_width_);
    float *buffers[2] = {(float *)workspace, (float *)(void *)((uint8_t *)workspace + width_bytes)};
    int8_t *xq = (int8_t *)workspace + 2u * width_bytes;

    const float *x = input;
    for (uint32_t i = 0; i < model->layer_count; i++) {
        SlettLayer L;
        spingalett_slett_layer(image, i, &L);
        float *y = i + 1 == model->layer_count ? output : buffers[i & 1u];
        layer_forward(image, &L, x, y, xq);
        x = y;
    }
#if !defined(SPINGALETT_INFERENCE_ONLY)
    spingalett_aligned_free(temp);
#else
    (void)temp;
#endif
    return SPINGALETT_OK;
}
