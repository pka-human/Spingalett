/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * What the importers share (Onnx.c, Torch.c): the conversion of a file's tensors into a network's
 * parameters in one pass, decoded from their type and reordered to channels-last as they are copied
 * (the files mapped into memory by Spingalett.Memory.c).
 */

#include "Spingalett.Private.h"
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- tensors */

size_t spingalett_filters_scratch(uint32_t CG, uint32_t KH, uint32_t KW) { return (size_t)CG * KH * KW; }

void spingalett_import_filters(float *dst, const uint8_t *src, int dtype, uint32_t OC, uint32_t CG, uint32_t KH,
                               uint32_t KW, float *scratch) {
    const size_t taps = (size_t)KH * KW, filter = (size_t)CG * taps, esize = spingalett_dtype_size(dtype);
    if (CG == 1 || taps == 1) {             /* the orders agree */
        spingalett_decode(dst, src, dtype, (size_t)OC * filter);
        return;
    }
    for (uint32_t o = 0; o < OC; o++) {
        spingalett_decode(scratch, src + (size_t)o * filter * esize, dtype, filter);
        float *d = dst + (size_t)o * filter;
        for (size_t t = 0; t < taps; t++)
            for (uint32_t c = 0; c < CG; c++) d[t * CG + c] = scratch[(size_t)c * taps + t];
    }
}

size_t spingalett_transposed_filters_scratch(uint32_t OG, uint32_t KH, uint32_t KW) {
    return (size_t)OG * KH * KW;
}

void spingalett_import_transposed_filters(float *dst, const uint8_t *src, int dtype, uint32_t IC, uint32_t OG,
                                          uint32_t G, uint32_t KH, uint32_t KW, float *scratch) {
    const size_t taps = (size_t)KH * KW, row = (size_t)OG * taps, esize = spingalett_dtype_size(dtype);
    const uint32_t IG = IC / G;
    if (IG == 1) {                          /* one input channel a group: the orders agree */
        spingalett_decode(dst, src, dtype, (size_t)IC * row);
        return;
    }
    /* input channel g IG + i holds its group's OG filters' taps: each goes to row g OG + o, column (t, i) */
    for (uint32_t ic = 0; ic < IC; ic++) {
        const uint32_t g = ic / IG, i = ic % IG;
        spingalett_decode(scratch, src + (size_t)ic * row * esize, dtype, row);
        float *d = dst + (size_t)g * OG * taps * IG + i;
        for (size_t k = 0; k < row; k++) d[k * IG] = scratch[k];
    }
}

/* Source rows k of a transposed product taken at once: a block of at most 1 MB. */
static size_t dense_block(uint32_t out) {
    size_t rows = (size_t)(262144u / (out ? out : 1u));
    return rows < 1 ? 1 : rows > 64 ? 64 : rows;
}

size_t spingalett_dense_scratch(uint32_t out, uint32_t in, bool transposed) {
    return transposed ? dense_block(out) * out : in;
}

void spingalett_import_dense(float *dst, const uint8_t *src, int dtype, uint32_t out, uint32_t in, uint32_t C,
                             uint32_t HW, bool transposed, float alpha, float *scratch) {
    const size_t esize = spingalett_dtype_size(dtype);
    if (!transposed) {
        for (uint32_t o = 0; o < out; o++) {
            float *d = dst + (size_t)o * in;
            const uint8_t *row = src + (size_t)o * in * esize;
            if (HW <= 1) {
                spingalett_decode(d, row, dtype, in);
            } else {
                /* column (c, p) of the map read flat goes to (p, c) */
                spingalett_decode(scratch, row, dtype, in);
                for (uint32_t p = 0; p < HW; p++)
                    for (uint32_t c = 0; c < C; c++) d[(size_t)p * C + c] = scratch[(size_t)c * HW + p];
            }
            if (alpha != 1.0f)
                for (uint32_t k = 0; k < in; k++) d[k] *= alpha;
        }
        return;
    }
    /* rows k of [in][out]: a block of them decoded, then written down the columns of dst */
    const size_t block = dense_block(out);
    for (uint32_t k0 = 0; k0 < in; k0 += (uint32_t)block) {
        const uint32_t kn = in - k0 < block ? in - k0 : (uint32_t)block;
        spingalett_decode(scratch, src + (size_t)k0 * out * esize, dtype, (size_t)kn * out);
        uint32_t col[64];
        for (uint32_t j = 0; j < kn; j++) {
            uint32_t k = k0 + j;
            col[j] = HW > 1 ? (k % HW) * C + k / HW : k;
        }
        for (uint32_t o = 0; o < out; o++) {
            float *d = dst + (size_t)o * in;
            for (uint32_t j = 0; j < kn; j++) d[col[j]] = alpha * scratch[(size_t)j * out + o];
        }
    }
}
