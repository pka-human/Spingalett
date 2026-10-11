/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* output.comp: the output layer's softmax, loss and gradients, a thread a row or (WIDE) a block of 64 a
   row with its sums in a fixed tree; the sparse cross-entropy's passes (COUNT, LOSS, SUM). Spec: OP, ACT,
   LOSS, WIDE. 64 threads. */

#include "common.cuh"

#define SOFTMAX 0u
#define LOSS_OP 1u
#define GRADS   2u
#define COUNT   3u
#define SUM     4u
#define LOSS_MSE 0u
#define LOSS_CROSS_ENTROPY 1u
#define LOSS_SPARSE 2u
#define FLAG_ACCURACY 1u

/* WIDE: the sum (or the maximum) of every thread's v, in a fixed tree. */
DEVICE float combine(float *red, uint32_t wide, uint32_t tid, float v, bool maximum) {
    if (wide == 0u) return v;
    red[tid] = v;
    barrier();
    for (uint32_t h = 32u; h > 0u; h /= 2u) {
        if (tid < h) red[tid] = maximum ? max_(red[tid], red[tid + h]) : red[tid] + red[tid + h];
        barrier();
    }
    const float r = red[0];
    barrier();
    return r;
}

KERNEL(output, SpgOutputPush) {
    const uint32_t OP = spec.v[0], ACT = spec.v[1], LOSS = spec.v[2], WIDE = spec.v[3];
    __shared__ float red[64];
    const uint32_t STEP = WIDE != 0u ? 64u : 1u;
    const uint32_t tid = WIDE != 0u ? thread_x() : 0u, n = p.n;
    const uint32_t first = WIDE != 0u ? block_x() : global_x();
    const uint32_t stride = WIDE != 0u ? blocks_x() : blocks_x() * 64u;
    float *y = F(p.y), *t = F(p.t), *delta = F(p.delta), *part = F(p.part);
    if (OP == COUNT || OP == SUM) {     /* the sparse cross-entropy's sums, a thread a sample (output.comp) */
        const uint32_t R = p.rows * p.cells;
        for (uint32_t s = global_x(); s < p.rows; s += blocks_x() * 64u) {
            if (OP == COUNT) {
                uint32_t count = 0;
                for (uint32_t c = 0; c < p.cells; c++) {
                    const float v = t[s * p.cells + c];
                    count += v >= 0.0f && v < (float)n ? 1u : 0u;
                }
                part[2u * R + s] = count > 0u ? 1.0f / (float)count : 0.0f;
                continue;
            }
            float loss = 0.0f, hits = 0.0f;
            for (uint32_t c = 0; c < p.cells; c++) {
                loss += part[s * p.cells + c];
                hits += part[R + s * p.cells + c];
            }
            const float inv = part[2u * R + s];
            F(p.loss)[s] = loss * inv;
            if (p.flags & FLAG_ACCURACY) F(p.loss)[p.rows + s] = hits * inv;
        }
        return;
    }
    for (uint32_t s = first; s < p.rows; s += stride) {
        const uint32_t base = s * n;
        if (OP == LOSS_OP && LOSS == LOSS_SPARSE) {
            /* a cell: the log of its softmax's sum, its loss against the smoothed target, its gradient */
            const uint32_t R = p.rows;
            const float tt = t[s], inv = part[2u * R + s / p.cells];
            const bool live = tt >= 0.0f && tt < (float)n;
            const uint32_t target = live ? (uint32_t)tt : 0u;
            float top = -3.402823466e38f, sum = 0.0f, all = 0.0f, later = 0.0f;
            for (uint32_t k = tid; k < n; k += STEP) top = max_(top, y[base + k]);
            top = combine(red, WIDE, tid, top, true);
            for (uint32_t k = tid; k < n; k += STEP) {
                const float z = y[base + k];
                sum += exp_(z - top);
                all += z;
                if (z == top) later = max_(later, (float)(n - k));      /* n - the first of the largest */
            }
            sum = combine(red, WIDE, tid, sum, false);
            all = combine(red, WIDE, tid, all, false);
            later = combine(red, WIDE, tid, later, true);
            const float lse = top + log_(sum), keep = 1.0f - p.smoothing, share = p.smoothing / (float)n;
            for (uint32_t k = tid; k < n; k += STEP) {
                float d = live ? (exp_(y[base + k] - lse) - share) * inv : 0.0f;
                if (live && k == target) d -= keep * inv;
                delta[base + k] = d;
            }
            if (tid == 0u) {
                const float loss = keep * (lse - y[base + target]) + (share > 0.0f ? share * ((float)n * lse - all) : 0.0f);
                part[s] = live ? loss : 0.0f;
                part[R + s] = live && (uint32_t)((float)n - later) == target ? 1.0f : 0.0f;
            }
            continue;
        }
        if (OP == SOFTMAX) {
            float top = -3.402823466e38f;
            for (uint32_t k = tid; k < n; k += STEP) top = max_(top, y[base + k]);
            top = combine(red, WIDE, tid, top, true);
            float sum = 0.0f;
            for (uint32_t k = tid; k < n; k += STEP) {
                const float e = exp_(y[base + k] - top);
                y[base + k] = e;
                sum += e;
            }
            sum = combine(red, WIDE, tid, sum, false);
            if (sum > 0.0f) {
                const float inv = 1.0f / sum;
                for (uint32_t k = tid; k < n; k += STEP) y[base + k] *= inv;
            }
            continue;
        }
        if (OP == GRADS) {
            if (ACT == ACT_SOFTMAX) {
                float dot = 0.0f;
                for (uint32_t k = tid; k < n; k += STEP) dot += t[base + k] * y[base + k];
                dot = combine(red, WIDE, tid, dot, false);
                for (uint32_t k = tid; k < n; k += STEP) delta[base + k] = y[base + k] * (t[base + k] - dot);
            } else {
                for (uint32_t k = tid; k < n; k += STEP) delta[base + k] = t[base + k] * derivative(y[base + k], ACT);
            }
            continue;
        }
        const float eps = 1e-9f;
        float loss = 0.0f, dot = 0.0f;
        for (uint32_t k = tid; k < n; k += STEP) {
            const float o = y[base + k], tt = t[base + k];
            if (LOSS == LOSS_MSE) {
                const float d = o - tt;
                loss += d * d;
                dot += d * o;
            } else if (ACT == ACT_SIGMOID) {
                loss -= tt * log_(max_(o, eps)) + (1.0f - tt) * log_(max_(1.0f - o, eps));
            } else {
                loss -= tt * log_(max_(o, eps));
            }
        }
        loss = combine(red, WIDE, tid, loss, false);
        if (LOSS == LOSS_MSE && ACT == ACT_SOFTMAX) dot = combine(red, WIDE, tid, dot, false);
        if (tid == 0u) F(p.loss)[s] = loss;
        for (uint32_t k = tid; k < n; k += STEP) {
            const float o = y[base + k];
            float d = o - t[base + k];
            if (LOSS == LOSS_MSE && ACT == ACT_SOFTMAX)
                d = o * (d - dot);
            else if (LOSS == LOSS_MSE || (ACT != ACT_SOFTMAX && ACT != ACT_SIGMOID))
                d *= derivative(o, ACT);
            delta[base + k] = d;
        }
    }
}
