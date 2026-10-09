/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* colsum.comp: per-column sums of R rows of C floats over slices of slice_rows rows, block (x, y)
   summing columns [x COLS, + COLS) of slice y, each thread every (256 / COLS)-th row, the threads'
   sums added in a fixed tree. Spec: COLS, MODE, HALF (words: x 0, dy 1, k 2). 256 threads. */

#include "common.cuh"

#define SUM     0u
#define SHIFTED 1u
#define DY      2u
#define LN      3u

KERNEL(colsum, SpgColsumPush) {
    const uint32_t COLS = spec.v[0], MODE = spec.v[1], HALF = spec.v[2], ROWS = 256u / COLS;
    __shared__ float s1[256], s2[256];
    const uint32_t tid = thread_x(), lc = tid % COLS, lr = tid / COLS;
    const uint32_t c = block_x() * COLS + lc, slice = block_y();
    const uint32_t r0 = slice * p.slice_rows, r1 = umin(p.R, r0 + p.slice_rows);
    float t1 = 0.0f, t2 = 0.0f;
    if (c < p.C) {
        const float shift = MODE == SHIFTED || MODE == DY ? ld(p.k, c, 2u, HALF) : 0.0f;
        for (uint32_t r = r0 + lr; r < r1; r += ROWS) {
            const float v = ld(p.x, r * p.C + c, 0u, HALF);
            if (MODE == SUM) {
                t1 += v;
            } else if (MODE == SHIFTED) {
                const float d = v - shift;
                t1 += d;
                t2 = fma_(d, d, t2);
            } else if (MODE == DY) {
                const float g = ld(p.dy, r * p.C + c, 1u, HALF);
                t1 += g;
                t2 = fma_(g, v - shift, t2);
            } else {
                const float g = ld(p.dy, r * p.C + c, 1u, HALF);
                t1 += g;
                t2 = fma_(g, (v - F(p.k)[2u * r]) * F(p.k)[2u * r + 1u], t2);
            }
        }
    }
    s1[tid] = t1;
    s2[tid] = t2;
    barrier();
    for (uint32_t half_rows = ROWS / 2u; half_rows > 0u; half_rows /= 2u) {
        if (lr < half_rows) {
            s1[tid] += s1[tid + half_rows * COLS];
            s2[tid] += s2[tid + half_rows * COLS];
        }
        barrier();
    }
    if (lr == 0u && c < p.C) {
        const uint32_t width = MODE == SUM ? p.C : 2u * p.C;
        F(p.part)[slice * width + c] = s1[tid];
        if (MODE != SUM) F(p.part)[slice * width + p.C + c] = s2[tid];
    }
}
