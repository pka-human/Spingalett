/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* grow.comp: operators of graphs over rows (reductions, ArgMax and ArgMin, Softmax, LogSoftmax, Hardmax, LayerNorm,
   RMSNorm, GroupNorm), TPR threads a row (spec 1), each its values in order, the partial results combined through
   shared memory in a fixed tree: the same bits on every run. Means and variances by Welford's updates, combined by
   Chan's. 256 threads, workgroups looping over blocks of rows. */

#include "common.cuh"

#define THREADS 256u

struct Part {
    float a, b, c;      /* sums, Welford's count, mean and M2, or a value and its index */
};

/* The address of value l of row (o, i): rows of len values inner apart; with channels last (GROUPNORM_LAST), row o is
   (batch, group) and value l the pixel l / cpg's channel group cpg + l % cpg */
DEVICE uint64_t at_of(const SpgRowPush &p, uint32_t op, uint32_t o, uint32_t i, uint32_t l) {
    if (op == SPG_ROW_GROUPNORM_LAST) {
        const uint32_t b = o / p.groups, g = o % p.groups;
        return ((uint64_t)b * p.pix + l / p.cpg) * (uint64_t)(p.groups * p.cpg) + g * p.cpg + l % p.cpg;
    }
    return ((uint64_t)o * p.len + l) * p.inner + i;
}
DEVICE float x_at(const SpgRowPush &p, uint32_t o, uint32_t i, uint32_t l) {
    return F(p.x)[((uint64_t)o * p.len + l) * p.inner + i];
}

/* combines partial results b into a (in place), for the operator */
DEVICE Part combine(uint32_t op, Part a, Part b, bool last) {
    switch (op) {
        case SPG_ROW_MAX: return Part{isnan_(a.a) ? a.a : isnan_(b.a) || b.a > a.a ? b.a : a.a, 0.0f, 0.0f};
        case SPG_ROW_MIN: return Part{isnan_(a.a) ? a.a : isnan_(b.a) || b.a < a.a ? b.a : a.a, 0.0f, 0.0f};
        case SPG_ROW_PROD: return Part{a.a * b.a, 0.0f, 0.0f};
        case SPG_ROW_ARGMAX: case SPG_ROW_ARGMIN: {
            /* (b's indices follow a's: a NaN first wins, else the better value, equal ones the first, or the last) */
            if (b.b < 0.0f) return a;
            if (a.b < 0.0f) return b;
            if (isnan_(a.a)) return a;
            if (isnan_(b.a)) return b;
            const bool better = op == SPG_ROW_ARGMAX ? b.a > a.a : b.a < a.a;
            return better || (b.a == a.a && last) ? b : a;
        }
        case SPG_ROW_LAYERNORM: case SPG_ROW_GROUPNORM: {
            /* Chan's combination of (count, mean, M2) */
            const float n = a.a + b.a;
            if (n == 0.0f) return a;
            const float d = b.b - a.b, wb = b.a / n;
            return Part{n, a.b + d * wb, a.c + b.c + d * d * a.a * wb};
        }
        default: return Part{a.a + b.a, 0.0f, 0.0f};
    }
}

DEVICE Part reduce_row(Part v, Part *sh, uint32_t t, uint32_t lane, uint32_t TPR, uint32_t op, bool last) {
    sh[t] = v;
    for (uint32_t s = TPR / 2u; s > 0u; s >>= 1) {
        barrier();
        if (lane < s) sh[t] = combine(op, sh[t], sh[t + s], last);
    }
    barrier();
    const Part r = sh[t - lane];
    barrier();
    return r;
}

KERNEL(grow, SpgRowPush) {
    __shared__ Part sh[THREADS];
    const uint32_t op = spec.v[0], TPR = spec.v[1] ? spec.v[1] : 1u, per = THREADS / TPR;
    const uint32_t t = thread_x(), lane = t % TPR;
    const bool last = (p.flags & SPG_ROW_LAST) != 0u;
    for (uint32_t first = block_x() * per; first < p.rows; first += blocks_x() * per) {
        const uint32_t q = first + t / TPR;
        const bool live = q < p.rows;
        const uint32_t o = live ? q / p.inner : 0u, i = live ? q % p.inner : 0u;
        const uint32_t len = live ? p.len : 0u;
        if (op < SPG_ROW_SOFTMAX) {
            /* reductions: one value a row */
            float mx = -__builtin_inff();
            if (op == SPG_ROW_LOGSUMEXP) {
                Part m = {-__builtin_inff(), 0.0f, 0.0f};
                for (uint32_t l = lane; l < len; l += TPR) m = combine(SPG_ROW_MAX, m, Part{x_at(p, o, i, l), 0.0f, 0.0f}, false);
                mx = reduce_row(m, sh, t, lane, TPR, SPG_ROW_MAX, false).a;
            }
            const bool arg = op == SPG_ROW_ARGMAX || op == SPG_ROW_ARGMIN;
            Part acc = {op == SPG_ROW_PROD ? 1.0f : op == SPG_ROW_MAX ? -__builtin_inff() : op == SPG_ROW_MIN
                        ? __builtin_inff() : 0.0f, arg ? -1.0f : 0.0f, 0.0f};
            for (uint32_t l = lane; l < len; l += TPR) {
                const float v = x_at(p, o, i, l);
                Part e;
                switch (op) {
                    case SPG_ROW_L1: e = Part{abs_(v), 0.0f, 0.0f}; break;
                    case SPG_ROW_L2: case SPG_ROW_SUMSQ: e = Part{v * v, 0.0f, 0.0f}; break;
                    case SPG_ROW_LOGSUMEXP: e = Part{exp_(v - mx), 0.0f, 0.0f}; break;
                    case SPG_ROW_ARGMAX: case SPG_ROW_ARGMIN: e = Part{v, (float)l, 0.0f}; break;
                    default: e = Part{v, 0.0f, 0.0f}; break;
                }
                acc = combine(op == SPG_ROW_L1 || op == SPG_ROW_L2 || op == SPG_ROW_SUMSQ || op == SPG_ROW_LOGSUMEXP ||
                              op == SPG_ROW_MEAN || op == SPG_ROW_LOGSUM ? SPG_ROW_SUM : op, acc, e, last);
            }
            const uint32_t cop = op == SPG_ROW_L1 || op == SPG_ROW_L2 || op == SPG_ROW_SUMSQ || op == SPG_ROW_LOGSUMEXP ||
                                 op == SPG_ROW_MEAN || op == SPG_ROW_LOGSUM ? SPG_ROW_SUM : op;
            const Part r = TPR > 1u ? reduce_row(acc, sh, t, lane, TPR, cop, last) : acc;
            if (live && lane == 0u) {
                if (arg) {
                    ((int64_t *)p.y)[q] = (int64_t)(r.b < 0.0f ? 0.0f : r.b);
                } else {
                    float v = r.a;
                    if (op == SPG_ROW_MEAN) v = v / (float)p.len;
                    else if (op == SPG_ROW_L2) v = sqrt_(v);
                    else if (op == SPG_ROW_LOGSUMEXP) v = isinf_(mx) && mx < 0.0f ? mx : log_(v) + mx;
                    else if (op == SPG_ROW_LOGSUM) v = log_(v);
                    F(p.y)[q] = v;
                }
            }
            continue;
        }
        if (op <= SPG_ROW_HARDMAX) {
            Part m = {-__builtin_inff(), op == SPG_ROW_HARDMAX ? -1.0f : 0.0f, 0.0f};
            for (uint32_t l = lane; l < len; l += TPR)
                m = combine(op == SPG_ROW_HARDMAX ? SPG_ROW_ARGMAX : SPG_ROW_MAX, m, Part{x_at(p, o, i, l), (float)l, 0.0f},
                            false);
            const Part top = TPR > 1u ? reduce_row(m, sh, t, lane, TPR, op == SPG_ROW_HARDMAX ? SPG_ROW_ARGMAX : SPG_ROW_MAX,
                                                   false) : m;
            float *y = F(p.y);
            if (op == SPG_ROW_HARDMAX) {
                for (uint32_t l = lane; l < len; l += TPR)
                    y[((uint64_t)o * p.len + l) * p.inner + i] = (float)l == top.b ? 1.0f : 0.0f;
                continue;
            }
            Part s = {0.0f, 0.0f, 0.0f};
            for (uint32_t l = lane; l < len; l += TPR) s.a += exp_(x_at(p, o, i, l) - top.a);
            const float sum = TPR > 1u ? reduce_row(s, sh, t, lane, TPR, SPG_ROW_SUM, false).a : s.a;
            const float inv = 1.0f / sum, ls = log_(sum);
            for (uint32_t l = lane; l < len; l += TPR) {
                const float v = x_at(p, o, i, l) - top.a;
                y[((uint64_t)o * p.len + l) * p.inner + i] = op == SPG_ROW_LOGSOFTMAX ? v - ls : exp_(v) * inv;
            }
            continue;
        }
        /* normalizations: Welford's count, mean and M2 (RMS: the sum of squares) */
        Part w = {0.0f, 0.0f, 0.0f};
        for (uint32_t l = lane; l < len; l += TPR) {
            const float v = F(p.x)[at_of(p, op, o, i, l)];
            if (op == SPG_ROW_RMSNORM) {
                w.a += v * v;
            } else {
                w.a += 1.0f;
                const float d = v - w.b;
                w.b += d / w.a;
                w.c += d * (v - w.b);
            }
        }
        const Part r = TPR > 1u ? reduce_row(w, sh, t, lane, TPR, op == SPG_ROW_RMSNORM ? SPG_ROW_SUM : SPG_ROW_LAYERNORM,
                                             false) : w;
        const float mean = op == SPG_ROW_RMSNORM ? 0.0f : r.b;
        const float var = (op == SPG_ROW_RMSNORM ? r.a : r.c) / (float)(p.len ? p.len : 1u);
        const float inv = 1.0f / sqrt_(var + p.eps);
        if (live && lane == 0u && (p.flags & SPG_ROW_STATS)) {
            F(p.aux)[q] = mean;
            F(p.aux)[p.rows + q] = inv;
        }
        const bool grouped = op == SPG_ROW_GROUPNORM || op == SPG_ROW_GROUPNORM_LAST;
        const uint32_t g = grouped ? o % p.groups : 0u;
        for (uint32_t l = lane; l < len; l += TPR) {
            const uint64_t at = at_of(p, op, o, i, l);
            float sa, sb;
            if (grouped) {
                const uint32_t k = (p.flags & SPG_ROW_BY_GROUP) ? g : g * p.cpg + (op == SPG_ROW_GROUPNORM ? l / p.pix : l % p.cpg);
                sa = F(p.a)[k];
                sb = F(p.b)[k];
            } else {
                sa = F(p.a)[(p.flags & SPG_ROW_A_ONE) ? 0u : l];
                sb = (p.flags & SPG_ROW_B) ? F(p.b)[(p.flags & SPG_ROW_B_ONE) ? 0u : l] : 0.0f;
            }
            float v = (F(p.x)[at] - mean) * inv * sa + sb;
            if (p.act) v = activate(v, p.act);
            if (p.flags & SPG_ROW_HALF_Y) H(p.y)[at] = (uint16_t)to_bf16(v);
            else F(p.y)[at] = v;
        }
    }
}
