/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Weights quantized in the block precisions (GGUF's types) when a network is saved in one: the codes and
 * scales llama.cpp's reference quantizers choose (ggml-quants.c, MIT: quantize_row_q4_0_ref() and the others,
 * make_qx_quants(), make_q3_quants(), make_qkx2_quants(), and quantize_row_iq4_nl_impl() without an importance
 * matrix), so that a model quantized here is the model llama.cpp's tools make of the same weights. Every
 * scale is searched for the least (weighted) error of its block; nothing is random.
 */

#include "Spingalett.Private.h"
#include <float.h>
#include <math.h>
#include <string.h>

#define GROUP_MAX_EPS 1e-15f

static inline int nearest_int(float x) { return (int)lrintf(x); }  /* (nearest even, as llama.cpp's) */
static inline int imin(int a, int b) { return a < b ? a : b; }
static inline int imax(int a, int b) { return a > b ? a : b; }
static inline void put_half(uint8_t *p, float x) { slett_put16(p, spingalett_float_to_fp16(x)); }
static inline float get_half(const uint8_t *p) { return spingalett_fp16_to_float(slett_get16(p)); }

/* The values of IQ4_NL's and IQ4_XS's codes (kvalues_iq4nl). */
static const int8_t iq4_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

/* ------------------------------------------------------------------------- the types of 32 */

/* The value of largest magnitude of n (with its sign). */
static float signed_max(const float *x, uint32_t n, float *amax) {
    float max = 0.0f;
    *amax = 0.0f;
    for (uint32_t j = 0; j < n; j++)
        if (*amax < fabsf(x[j])) {
            *amax = fabsf(x[j]);
            max = x[j];
        }
    return max;
}

static void quantize_q4_0(const float *x, uint8_t *y) {
    float amax;
    const float d = signed_max(x, 32, &amax) / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    put_half(y, d);
    for (uint32_t j = 0; j < 16u; j++) {
        const int a = imin(15, (int8_t)(x[j] * id + 8.5f)), b = imin(15, (int8_t)(x[16 + j] * id + 8.5f));
        y[2 + j] = (uint8_t)(a | (b << 4));
    }
}

static void min_max(const float *x, uint32_t n, float *min, float *max) {
    *min = FLT_MAX;
    *max = -FLT_MAX;
    for (uint32_t j = 0; j < n; j++) {
        if (x[j] < *min) *min = x[j];
        if (x[j] > *max) *max = x[j];
    }
}

static void quantize_q4_1(const float *x, uint8_t *y) {
    float min, max;
    min_max(x, 32, &min, &max);
    const float d = (max - min) / 15.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    put_half(y, d);
    put_half(y + 2, min);
    for (uint32_t j = 0; j < 16u; j++) {
        const int a = imin(15, (int8_t)((x[j] - min) * id + 0.5f)), b = imin(15, (int8_t)((x[16 + j] - min) * id + 0.5f));
        y[4 + j] = (uint8_t)(a | (b << 4));
    }
}

static void quantize_q5_0(const float *x, uint8_t *y) {
    float amax;
    const float d = signed_max(x, 32, &amax) / -16.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    put_half(y, d);
    uint32_t qh = 0;
    for (uint32_t j = 0; j < 16u; j++) {
        const int a = imin(31, (int8_t)(x[j] * id + 16.5f)), b = imin(31, (int8_t)(x[16 + j] * id + 16.5f));
        y[6 + j] = (uint8_t)((a & 0x0F) | ((b & 0x0F) << 4));
        qh |= (uint32_t)((a & 0x10) >> 4) << j;
        qh |= (uint32_t)((b & 0x10) >> 4) << (j + 16u);
    }
    slett_put32(y + 2, qh);
}

static void quantize_q5_1(const float *x, uint8_t *y) {
    float min, max;
    min_max(x, 32, &min, &max);
    const float d = (max - min) / 31.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    put_half(y, d);
    put_half(y + 2, min);
    uint32_t qh = 0;
    for (uint32_t j = 0; j < 16u; j++) {
        const int a = imin(31, (int)(uint8_t)((x[j] - min) * id + 0.5f));
        const int b = imin(31, (int)(uint8_t)((x[16 + j] - min) * id + 0.5f));
        y[8 + j] = (uint8_t)((a & 0x0F) | ((b & 0x0F) << 4));
        qh |= (uint32_t)((a & 0x10) >> 4) << j;
        qh |= (uint32_t)((b & 0x10) >> 4) << (j + 16u);
    }
    slett_put32(y + 4, qh);
}

static void quantize_q8_0(const float *x, uint8_t *y) {
    float amax;
    (void)signed_max(x, 32, &amax);
    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    put_half(y, d);
    for (uint32_t j = 0; j < 32u; j++) y[2 + j] = (uint8_t)(int8_t)roundf(x[j] * id);
}

/* ------------------------------------------------------------------------- the searches of the k-quants */

/* A symmetric scale for n values of codes -nmax .. nmax - 1 (stored in L as code + nmax), searched around amax /
   nmax for the least error weighted by x^2 (rmse_type 1, make_qx_quants()). */
static float make_qx_quants(uint32_t n, int nmax, const float *x, int8_t *L) {
    float amax;
    const float max = signed_max(x, n, &amax);
    if (amax < GROUP_MAX_EPS) {
        memset(L, 0, n);
        return 0.0f;
    }
    float iscale = -(float)nmax / max, sumlx = 0.0f, suml2 = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        const int l = imax(-nmax, imin(nmax - 1, nearest_int(iscale * x[i])));
        L[i] = (int8_t)(l + nmax);
        const float w = x[i] * x[i];
        sumlx += w * x[i] * (float)l;
        suml2 += w * (float)l * (float)l;
    }
    float scale = suml2 != 0.0f ? sumlx / suml2 : 0.0f, best = scale * sumlx;
    for (int is = -9; is <= 9; is++) {
        if (is == 0) continue;
        iscale = -((float)nmax + 0.1f * (float)is) / max;
        sumlx = suml2 = 0.0f;
        for (uint32_t i = 0; i < n; i++) {
            const int l = imax(-nmax, imin(nmax - 1, nearest_int(iscale * x[i])));
            const float w = x[i] * x[i];
            sumlx += w * x[i] * (float)l;
            suml2 += w * (float)l * (float)l;
        }
        if (suml2 > 0.0f && sumlx * sumlx > best * suml2) {
            for (uint32_t i = 0; i < n; i++)
                L[i] = (int8_t)(nmax + imax(-nmax, imin(nmax - 1, nearest_int(iscale * x[i]))));
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

/* Q3_K's: a symmetric scale for codes -nmax .. nmax - 1, then each code moved while that lowers the error
   weighted by x^2 (make_q3_quants() with do_rmse). */
static float make_q3_quants(uint32_t n, int nmax, const float *x, int8_t *L) {
    float amax;
    const float max = signed_max(x, n, &amax);
    if (amax < GROUP_MAX_EPS) {
        memset(L, 0, n);
        return 0.0f;
    }
    const float iscale = -(float)nmax / max;
    float sumlx = 0.0f, suml2 = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        const int l = imax(-nmax, imin(nmax - 1, nearest_int(iscale * x[i])));
        L[i] = (int8_t)l;
        const float w = x[i] * x[i];
        sumlx += w * x[i] * (float)l;
        suml2 += w * (float)l * (float)l;
    }
    for (int itry = 0; itry < 5; itry++) {
        int changed = 0;
        for (uint32_t i = 0; i < n; i++) {
            const float w = x[i] * x[i];
            float slx = sumlx - w * x[i] * (float)L[i];
            if (slx > 0.0f) {
                float sl2 = suml2 - w * (float)L[i] * (float)L[i];
                const int l = imax(-nmax, imin(nmax - 1, nearest_int(x[i] * sl2 / slx)));
                if (l != L[i]) {
                    slx += w * x[i] * (float)l;
                    sl2 += w * (float)l * (float)l;
                    if (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = (int8_t)l;
                        sumlx = slx;
                        suml2 = sl2;
                        changed++;
                    }
                }
            }
        }
        if (!changed) break;
    }
    for (uint32_t i = 0; i < n; i++) L[i] = (int8_t)(L[i] + nmax);
    return sumlx / suml2;
}

/* A scale and a minimum for n values of codes 0 .. nmax, searched over nstep + 1 steps of the scale for the least
   weighted error (absolute with use_mad, else squared); *the_min receives minus the minimum (make_qkx2_quants()). */
static float make_qkx2_quants(uint32_t n, int nmax, const float *x, const float *weights, uint8_t *L, float *the_min,
                              uint8_t *Laux, float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0], max = x[0], sum_w = weights[0], sum_x = sum_w * x[0];
    for (uint32_t i = 1; i < n; i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        const float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0.0f) min = 0.0f;
    if (max == min) {
        memset(L, 0, n);
        *the_min = -min;
        return 0.0f;
    }
    float iscale = (float)nmax / (max - min), scale = 1.0f / iscale, best_error = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        L[i] = (uint8_t)imax(0, imin(nmax, nearest_int(iscale * (x[i] - min))));
        float diff = scale * (float)L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        best_error += weights[i] * diff;
    }
    for (int is = 0; is <= nstep; is++) {
        iscale = (rmin + rdelta * (float)is + (float)nmax) / (max - min);
        float sum_l = 0.0f, sum_l2 = 0.0f, sum_xl = 0.0f;
        for (uint32_t i = 0; i < n; i++) {
            const int l = imax(0, imin(nmax, nearest_int(iscale * (x[i] - min))));
            Laux[i] = (uint8_t)l;
            const float w = weights[i];
            sum_l += w * (float)l;
            sum_l2 += w * (float)l * (float)l;
            sum_xl += w * (float)l * x[i];
        }
        const float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0.0f) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D, this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0.0f) {
                this_min = 0.0f;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0.0f;
            for (uint32_t i = 0; i < n; i++) {
                float diff = this_scale * (float)Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                cur_error += weights[i] * diff;
            }
            if (cur_error < best_error) {
                memcpy(L, Laux, n);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

/* The 6-bit scale and minimum j of Q4_K's and Q5_K's twelve bytes. */
static void scale_min_k4(uint32_t j, const uint8_t *q, int *d, int *m) {
    if (j < 4u) {
        *d = q[j] & 63;
        *m = q[j + 4u] & 63;
    } else {
        *d = (q[j + 4u] & 0xF) | ((q[j - 4u] >> 6) << 4);
        *m = (q[j + 4u] >> 4) | ((q[j] >> 6) << 4);
    }
}

/* ------------------------------------------------------------------------- the k-quants */

static void quantize_q2_k(const float *x, uint8_t *y) {
    uint8_t L[256], Laux[16], *sc = y, *qs = y + 16;
    float weights[16], mins[16], scales[16], max_scale = 0.0f, max_min = 0.0f;
    for (uint32_t j = 0; j < 16u; j++) {
        for (uint32_t l = 0; l < 16u; l++) weights[l] = fabsf(x[16u * j + l]);
        scales[j] = make_qkx2_quants(16, 3, x + 16u * j, weights, L + 16u * j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }
    memset(sc, 0, 16);
    if (max_scale > 0.0f) {
        const float iscale = 15.0f / max_scale;
        for (uint32_t j = 0; j < 16u; j++) sc[j] = (uint8_t)nearest_int(iscale * scales[j]);
        put_half(y + 80, max_scale / 15.0f);
    } else {
        put_half(y + 80, 0.0f);
    }
    if (max_min > 0.0f) {
        const float iscale = 15.0f / max_min;
        for (uint32_t j = 0; j < 16u; j++) sc[j] |= (uint8_t)(nearest_int(iscale * mins[j]) << 4);
        put_half(y + 82, max_min / 15.0f);
    } else {
        put_half(y + 82, 0.0f);
    }
    for (uint32_t j = 0; j < 16u; j++) {
        const float d = get_half(y + 80) * (float)(sc[j] & 0xF);
        if (d == 0.0f) continue;
        const float dm = get_half(y + 82) * (float)(sc[j] >> 4);
        for (uint32_t i = 0; i < 16u; i++)
            L[16u * j + i] = (uint8_t)imax(0, imin(3, nearest_int((x[16u * j + i] + dm) / d)));
    }
    for (uint32_t j = 0; j < 256u; j += 128u)
        for (uint32_t l = 0; l < 32u; l++)
            qs[j / 4u + l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6));
}

static void quantize_q3_k(const float *x, uint8_t *y) {
    int8_t L[256];
    uint8_t *hmask = y, *qs = y + 32, *sc = y + 96;
    float scales[16], max_scale = 0.0f, amax = 0.0f;
    for (uint32_t j = 0; j < 16u; j++) {
        scales[j] = make_q3_quants(16, 4, x + 16u * j, L + 16u * j);
        if (fabsf(scales[j]) > amax) {
            amax = fabsf(scales[j]);
            max_scale = scales[j];
        }
    }
    memset(sc, 0, 12);
    if (max_scale != 0.0f) {
        const float iscale = -32.0f / max_scale;
        for (uint32_t j = 0; j < 16u; j++) {
            int l = imax(-32, imin(31, nearest_int(iscale * scales[j]))) + 32;
            if (j < 8u) sc[j] = (uint8_t)(l & 0xF);
            else sc[j - 8u] |= (uint8_t)((l & 0xF) << 4);
            l >>= 4;
            sc[j % 4u + 8u] |= (uint8_t)(l << (2u * (j / 4u)));
        }
        put_half(y + 108, 1.0f / iscale);
    } else {
        put_half(y + 108, 0.0f);
    }
    for (uint32_t j = 0; j < 16u; j++) {
        int s = j < 8u ? sc[j] & 0xF : sc[j - 8u] >> 4;
        s = (s | (((sc[8u + j % 4u] >> (2u * (j / 4u))) & 3) << 4)) - 32;
        const float d = get_half(y + 108) * (float)s;
        if (d == 0.0f) continue;
        for (uint32_t i = 0; i < 16u; i++)
            L[16u * j + i] = (int8_t)(imax(-4, imin(3, nearest_int(x[16u * j + i] / d))) + 4);
    }
    memset(hmask, 0, 32);
    uint32_t m = 0;
    uint8_t hm = 1;
    for (uint32_t j = 0; j < 256u; j++) {
        if (L[j] > 3) {
            hmask[m] |= hm;
            L[j] = (int8_t)(L[j] - 4);
        }
        if (++m == 32u) {
            m = 0;
            hm = (uint8_t)(hm << 1);
        }
    }
    for (uint32_t j = 0; j < 256u; j += 128u)
        for (uint32_t l = 0; l < 32u; l++)
            qs[j / 4u + l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6));
}

/* Q4_K and Q5_K: eight sub-blocks of 32 with a scale and a minimum each, then the codes of 4 bits (and fifth bits). */
static void quantize_q45_k(const float *x, uint8_t *y, bool five) {
    uint8_t L[256], Laux[32], *sc = y + 4;
    float weights[32], mins[8], scales[8], max_scale = 0.0f, max_min = 0.0f;
    const int nmax = five ? 31 : 15;
    for (uint32_t j = 0; j < 8u; j++) {
        float sum_x2 = 0.0f;
        for (uint32_t l = 0; l < 32u; l++) sum_x2 += x[32u * j + l] * x[32u * j + l];
        const float av_x = sqrtf(sum_x2 / 32.0f);
        for (uint32_t l = 0; l < 32u; l++) weights[l] = av_x + fabsf(x[32u * j + l]);
        scales[j] = make_qkx2_quants(32, nmax, x + 32u * j, weights, L + 32u * j, &mins[j], Laux,
                                     five ? -0.5f : -1.0f, 0.1f, five ? 15 : 20, false);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }
    const float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
    const float inv_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;
    memset(sc, 0, 12);
    for (uint32_t j = 0; j < 8u; j++) {
        const uint8_t ls = (uint8_t)imin(63, nearest_int(inv_scale * scales[j]));
        const uint8_t lm = (uint8_t)imin(63, nearest_int(inv_min * mins[j]));
        if (j < 4u) {
            sc[j] = ls;
            sc[j + 4u] = lm;
        } else {
            sc[j + 4u] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
            sc[j - 4u] |= (uint8_t)((ls >> 4) << 6);
            sc[j] |= (uint8_t)((lm >> 4) << 6);
        }
    }
    put_half(y, max_scale / 63.0f);
    put_half(y + 2, max_min / 63.0f);
    for (uint32_t j = 0; j < 8u; j++) {
        int s, mn;
        scale_min_k4(j, sc, &s, &mn);
        const float d = get_half(y) * (float)s;
        if (d == 0.0f) continue;
        const float dm = get_half(y + 2) * (float)mn;
        for (uint32_t i = 0; i < 32u; i++)
            L[32u * j + i] = (uint8_t)imax(0, imin(nmax, nearest_int((x[32u * j + i] + dm) / d)));
    }
    if (!five) {
        uint8_t *q = y + 16;
        for (uint32_t j = 0; j < 256u; j += 64u, q += 32)
            for (uint32_t l = 0; l < 32u; l++) q[l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 4));
        return;
    }
    uint8_t *qh = y + 16, *ql = y + 48, m1 = 1, m2 = 2;
    memset(qh, 0, 32);
    for (uint32_t n = 0; n < 256u; n += 64u, ql += 32) {
        for (uint32_t j = 0; j < 32u; j++) {
            int l1 = L[n + j], l2 = L[n + j + 32];
            if (l1 > 15) {
                l1 -= 16;
                qh[j] |= m1;
            }
            if (l2 > 15) {
                l2 -= 16;
                qh[j] |= m2;
            }
            ql[j] = (uint8_t)(l1 | (l2 << 4));
        }
        m1 = (uint8_t)(m1 << 2);
        m2 = (uint8_t)(m2 << 2);
    }
}

static void quantize_q6_k(const float *x, uint8_t *y) {
    int8_t L[256];
    uint8_t *ql = y, *qh = y + 128;
    int8_t *sc = (int8_t *)(y + 192);
    float scales[16], max_scale = 0.0f, max_abs_scale = 0.0f;
    for (uint32_t ib = 0; ib < 16u; ib++) {
        scales[ib] = make_qx_quants(16, 32, x + 16u * ib, L + 16u * ib);
        if (fabsf(scales[ib]) > max_abs_scale) {
            max_abs_scale = fabsf(scales[ib]);
            max_scale = scales[ib];
        }
    }
    if (max_abs_scale < GROUP_MAX_EPS) {
        memset(y, 0, 210);
        return;
    }
    const float iscale = -128.0f / max_scale;
    put_half(y + 208, 1.0f / iscale);
    for (uint32_t ib = 0; ib < 16u; ib++) sc[ib] = (int8_t)imin(127, nearest_int(iscale * scales[ib]));
    for (uint32_t j = 0; j < 16u; j++) {
        const float d = get_half(y + 208) * (float)sc[j];
        if (d == 0.0f) continue;
        for (uint32_t i = 0; i < 16u; i++)
            L[16u * j + i] = (int8_t)(imax(-32, imin(31, nearest_int(x[16u * j + i] / d))) + 32);
    }
    for (uint32_t j = 0; j < 256u; j += 128u, ql += 64, qh += 32)
        for (uint32_t l = 0; l < 32u; l++) {
            const int q1 = L[j + l] & 0xF, q2 = L[j + l + 32] & 0xF, q3 = L[j + l + 64] & 0xF, q4 = L[j + l + 96] & 0xF;
            ql[l] = (uint8_t)(q1 | (q3 << 4));
            ql[l + 32] = (uint8_t)(q2 | (q4 << 4));
            qh[l] = (uint8_t)((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) |
                              ((L[j + l + 96] >> 4) << 6));
        }
}

/* ------------------------------------------------------------------------- IQ4_NL and IQ4_XS */

/* The code whose value of the table lies nearest to x (best_index_int8()). */
static int best_index(float x) {
    if (x <= iq4_values[0]) return 0;
    if (x >= iq4_values[15]) return 15;
    int ml = 0, mu = 15;
    while (mu - ml > 1) {
        const int mav = (ml + mu) / 2;
        if (x < iq4_values[mav]) mu = mav;
        else ml = mav;
    }
    return x - iq4_values[mu - 1] < iq4_values[mu] - x ? mu - 1 : mu;
}

/* A super-block of `size` values in blocks of 32 (IQ4_NL: one; IQ4_XS: eight with 6-bit scales), scales tried in
   2 ntry + 1 steps (quantize_row_iq4_nl_impl() without an importance matrix: weights x^2). */
static void quantize_iq4(const float *x, uint32_t size, uint8_t *dh, uint8_t *q4, uint8_t *scales_h, uint8_t *scales_l,
                         int ntry) {
    uint8_t L[256];
    float scales[8], weight[32], max_scale = 0.0f, amax_scale = 0.0f;
    const uint32_t blocks = size / 32u;
    memset(q4, 0, size / 2u);
    memset(L, 0, sizeof L);
    put_half(dh, 0.0f);
    for (uint32_t ib = 0; ib < blocks; ib++) {
        const float *xb = x + 32u * ib;
        uint8_t *Lb = L + 32u * ib;
        for (uint32_t j = 0; j < 32u; j++) weight[j] = xb[j] * xb[j];
        float amax;
        const float max = signed_max(xb, 32, &amax);
        if (amax < GROUP_MAX_EPS) {
            scales[ib] = 0.0f;
            continue;
        }
        float d = ntry > 0 ? -max / (float)iq4_values[0] : max / (float)iq4_values[0], id = 1.0f / d;
        float sumqx = 0.0f, sumq2 = 0.0f;
        for (uint32_t j = 0; j < 32u; j++) {
            const int l = best_index(id * xb[j]);
            Lb[j] = (uint8_t)l;
            const float q = (float)iq4_values[l], w = weight[j];
            sumqx += w * q * xb[j];
            sumq2 += w * q * q;
        }
        d = sumqx / sumq2;
        float best = d * sumqx;
        for (int itry = -ntry; itry <= ntry; itry++) {
            id = (float)(itry + iq4_values[0]) / max;
            sumqx = sumq2 = 0.0f;
            for (uint32_t j = 0; j < 32u; j++) {
                const float q = (float)iq4_values[best_index(id * xb[j])], w = weight[j];
                sumqx += w * q * xb[j];
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f && sumqx * sumqx > best * sumq2) {
                d = sumqx / sumq2;
                best = d * sumqx;
            }
        }
        scales[ib] = d;
        if (fabsf(d) > amax_scale) {
            amax_scale = fabsf(d);
            max_scale = d;
        }
    }
    if (blocks > 1u) {
        const float d = -max_scale / 32.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint16_t high = 0;
        put_half(dh, d);
        memset(scales_l, 0, blocks / 2u);
        for (uint32_t ib = 0; ib < blocks; ib++) {
            int l = imax(-32, imin(31, nearest_int(id * scales[ib])));
            const float dl = d * (float)l, idl = dl != 0.0f ? 1.0f / dl : 0.0f;
            for (uint32_t j = 0; j < 32u; j++) L[32u * ib + j] = (uint8_t)best_index(idl * x[32u * ib + j]);
            l += 32;
            if (ib % 2u == 0) scales_l[ib / 2u] = (uint8_t)(l & 0xF);
            else scales_l[ib / 2u] |= (uint8_t)((l & 0xF) << 4);
            high |= (uint16_t)((l >> 4) << (2u * ib));
        }
        slett_put16(scales_h, high);
    } else {
        put_half(dh, scales[0]);
        const float id = scales[0] != 0.0f ? 1.0f / scales[0] : 0.0f;
        for (uint32_t j = 0; j < size; j++) L[j] = (uint8_t)best_index(id * x[j]);
    }
    for (uint32_t i = 0; i < size / 32u; i++)
        for (uint32_t j = 0; j < 16u; j++) q4[16u * i + j] = (uint8_t)(L[32u * i + j] | (L[32u * i + 16u + j] << 4));
}

/* ------------------------------------------------------------------------- rows */

void spingalett_block_quantize(PrecisionMode p, const float *x, uint8_t *dst, uint32_t n) {
    const uint32_t values = slett_block_values(p), bytes = slett_block_bytes(p);
    for (uint32_t i = 0; i < n; i += values, x += values, dst += bytes) {
        switch (p) {
            case PRECISION_Q8_0: quantize_q8_0(x, dst); break;
            case PRECISION_Q4_0: quantize_q4_0(x, dst); break;
            case PRECISION_Q4_1: quantize_q4_1(x, dst); break;
            case PRECISION_Q5_0: quantize_q5_0(x, dst); break;
            case PRECISION_Q5_1: quantize_q5_1(x, dst); break;
            case PRECISION_Q2_K: quantize_q2_k(x, dst); break;
            case PRECISION_Q3_K: quantize_q3_k(x, dst); break;
            case PRECISION_Q4_K: quantize_q45_k(x, dst, false); break;
            case PRECISION_Q5_K: quantize_q45_k(x, dst, true); break;
            case PRECISION_Q6_K: quantize_q6_k(x, dst); break;
            case PRECISION_IQ4_NL: quantize_iq4(x, 32, dst, dst + 2, NULL, NULL, 7); break;
            case PRECISION_IQ4_XS: quantize_iq4(x, 256, dst, dst + 8, dst + 2, dst + 4, 7); break;
            default: break;
        }
    }
}
