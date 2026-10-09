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

/* Row r's contribution to the sums (v of x, g of dy). */
template <uint32_t MODE>
DEVICE void add_row(const SpgColsumPush &p, uint32_t r, float v, float g, float shift, float &t1, float &t2) {
    if (MODE == SUM) {
        t1 += v;
    } else if (MODE == SHIFTED) {
        const float d = v - shift;
        t1 += d;
        t2 = fma_(d, d, t2);
    } else if (MODE == DY) {
        t1 += g;
        t2 = fma_(g, v - shift, t2);
    } else {
        t1 += g;
        t2 = fma_(g, (v - F(p.k)[2u * r]) * F(p.k)[2u * r + 1u], t2);
    }
}

/* The sums of a thread's rows, eight rows at a time: their loads first, then their sums in the order of
   the rows (the sums of one row at a time, which the loads of the next would wait for otherwise). */
template <uint32_t MODE>
DEVICE void column(const SpgColsumPush &p, uint32_t HALF, uint32_t c, uint32_t r, uint32_t r1, uint32_t ROWS,
                   float &t1, float &t2) {
    const float shift = MODE == SHIFTED || MODE == DY ? ld(p.k, c, 2u, HALF) : 0.0f;
    constexpr uint32_t U = 8u;
    for (; r + (U - 1u) * ROWS < r1; r += U * ROWS) {
        float v[U], g[U] = {};
#pragma unroll
        for (uint32_t j = 0; j < U; j++) {
            v[j] = ld(p.x, (r + j * ROWS) * p.C + c, 0u, HALF);
            if (MODE == DY || MODE == LN) g[j] = ld(p.dy, (r + j * ROWS) * p.C + c, 1u, HALF);
        }
#pragma unroll
        for (uint32_t j = 0; j < U; j++) add_row<MODE>(p, r + j * ROWS, v[j], g[j], shift, t1, t2);
    }
    for (; r < r1; r += ROWS)
        add_row<MODE>(p, r, ld(p.x, r * p.C + c, 0u, HALF),
                      MODE == DY || MODE == LN ? ld(p.dy, r * p.C + c, 1u, HALF) : 0.0f, shift, t1, t2);
}

KERNEL(colsum, SpgColsumPush) {
    const uint32_t COLS = spec.v[0], MODE = spec.v[1], HALF = spec.v[2], ROWS = 256u / COLS;
    __shared__ float s1[256], s2[256];
    const uint32_t tid = thread_x(), lc = tid % COLS, lr = tid / COLS;
    const uint32_t c = block_x() * COLS + lc, slice = block_y();
    const uint32_t r0 = slice * p.slice_rows, r1 = umin(p.R, r0 + p.slice_rows);
    float t1 = 0.0f, t2 = 0.0f;
    if (c < p.C) {
        if (MODE == SUM) column<SUM>(p, HALF, c, r0 + lr, r1, ROWS, t1, t2);
        else if (MODE == SHIFTED) column<SHIFTED>(p, HALF, c, r0 + lr, r1, ROWS, t1, t2);
        else if (MODE == DY) column<DY>(p, HALF, c, r0 + lr, r1, ROWS, t1, t2);
        else column<LN>(p, HALF, c, r0 + lr, r1, ROWS, t1, t2);
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
