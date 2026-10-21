/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The operators of graphs (Spingalett.IR.h, Spingalett.IROps.def): each kernel makes its outputs' shapes from its
 * inputs (and the values of inputs that hold shapes), allocates them (ir_output()) and computes them, with ONNX's
 * semantics. Floats compute in single precision, integers in 64 bits stored in their type, booleans as bytes.
 *
 * Every output value is computed by one thread in one fixed order, so that the results are the same on any number
 * of threads: element-wise operators split their outputs between threads, reductions and normalizations each sum a
 * row in its order, products and convolutions run on the library's matrix kernels (whose sums are fixed by the
 * shape, never by the threads).
 */

#include "Spingalett.IR.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define F32 SPINGALETT_DTYPE_FLOAT32
#define I64 SPINGALETT_DTYPE_INT64
#define I32 SPINGALETT_DTYPE_INT32
#define BOOL SPINGALETT_DTYPE_BOOL
#define OP(r) ((IrOp)(r)->node->op)

/* Work below which loops run on one thread. */
#define PARALLEL_WORK ((uint64_t)1 << 16)

static bool is_int(uint8_t dtype) {
    return dtype == I64 || dtype == I32 || dtype == SPINGALETT_DTYPE_INT8 || dtype == SPINGALETT_DTYPE_UINT8;
}

/* Value i of t stored as an int64 into u of dtype. */
static void put_int(IrTensor *t, uint64_t i, int64_t x) {
    switch (t->dtype) {
        case I64: ((int64_t *)t->data)[i] = x; break;
        case I32: ((int32_t *)t->data)[i] = (int32_t)x; break;
        case SPINGALETT_DTYPE_INT8: ((int8_t *)t->data)[i] = (int8_t)x; break;
        case SPINGALETT_DTYPE_UINT8: case BOOL: ((uint8_t *)t->data)[i] = (uint8_t)x; break;
        case F32: ((float *)t->data)[i] = (float)x; break;
        default: break;
    }
}

static float *fdata(const IrTensor *t) {
    return (float *)t->data;
}

/* An axis of a tensor of `rank` axes, negatives from the end; false when out of range. */
static bool axis_of(IrRun *r, int64_t axis, uint32_t rank, uint32_t *out) {
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= (int64_t)rank) return ir_fail(r, "the axis %lld of a tensor of %u axes", (long long)axis, rank);
    *out = (uint32_t)axis;
    return true;
}

/* An integer list from an attribute or, in later opsets, an input (NULL: none); count -1 when neither gives it. */
static int64_t int_list(const IrNode *n, const char *attr, const IrTensor *input, int64_t *out, int64_t most) {
    if (input) {
        const int64_t c = (int64_t)input->count < most ? (int64_t)input->count : most;
        for (int64_t i = 0; i < c; i++) out[i] = ir_get_int(input, (uint64_t)i);
        return c;
    }
    const IrAttr *a = attr ? ir_attr(n, attr) : NULL;
    if (!a || a->type != ATTR_INTS) return -1;
    const int64_t c = (int64_t)a->count < most ? (int64_t)a->count : most;
    for (int64_t i = 0; i < c; i++) out[i] = a->ints[i];
    return c;
}

/* ------------------------------------------------------------------------- broadcasting */

/* The shape of numpy's broadcast of n tensors. */
static bool broadcast_shape(IrRun *r, IrTensor *const *in, uint32_t n, uint32_t *rank, int64_t *dims) {
    uint32_t R = 0;
    for (uint32_t k = 0; k < n; k++)
        if (in[k] && in[k]->rank > R) R = in[k]->rank;
    for (uint32_t d = 0; d < R; d++) dims[d] = 1;
    for (uint32_t k = 0; k < n; k++) {
        if (!in[k]) continue;
        const uint32_t off = R - in[k]->rank;
        for (uint32_t d = 0; d < in[k]->rank; d++) {
            const int64_t s = in[k]->dims[d];
            if (s == dims[off + d] || s == 1) continue;
            if (dims[off + d] != 1) return ir_fail(r, "shapes that do not broadcast (%lld against %lld on an axis)",
                                                  (long long)s, (long long)dims[off + d]);
            dims[off + d] = s;
        }
    }
    *rank = R;
    return true;
}

/* An input's strides over the output's axes (0 where it broadcasts). */
static void broadcast_strides(const IrTensor *t, uint32_t R, const int64_t *dims, int64_t *strides) {
    int64_t s = 1;
    for (uint32_t d = R; d-- > 0;) {
        const int32_t td = (int32_t)d - (int32_t)(R - t->rank);
        const int64_t size = td >= 0 ? t->dims[td] : 1;
        strides[d] = size == 1 && dims[d] != 1 ? 0 : s;
        if (td >= 0) s *= size;
    }
}

/* The offsets into up to three inputs of output values [begin, end): fn(i, offsets) for each, the offsets kept by
   an odometer over the output's axes. */
typedef struct {
    uint32_t rank, n;
    int64_t dims[IR_MAX_RANK];
    int64_t strides[3][IR_MAX_RANK];
} Walk;

static void walk_init(Walk *w, IrTensor *const *in, uint32_t n, uint32_t rank, const int64_t *dims) {
    w->rank = rank;
    w->n = n;
    for (uint32_t d = 0; d < rank; d++) w->dims[d] = dims[d];
    for (uint32_t k = 0; k < n; k++) broadcast_strides(in[k], rank, dims, w->strides[k]);
}

/* The offsets of output value i. */
static void walk_offsets(const Walk *w, uint64_t i, int64_t *idx, int64_t *off) {
    for (uint32_t k = 0; k < w->n; k++) off[k] = 0;
    for (uint32_t d = w->rank; d-- > 0;) {
        idx[d] = (int64_t)(i % (uint64_t)w->dims[d]);
        i /= (uint64_t)w->dims[d];
        for (uint32_t k = 0; k < w->n; k++) off[k] += idx[d] * w->strides[k][d];
    }
}

static void walk_step(const Walk *w, int64_t *idx, int64_t *off) {
    for (uint32_t d = w->rank; d-- > 0;) {
        for (uint32_t k = 0; k < w->n; k++) off[k] += w->strides[k][d];
        if (++idx[d] < w->dims[d]) return;
        for (uint32_t k = 0; k < w->n; k++) off[k] -= w->strides[k][d] * w->dims[d];
        idx[d] = 0;
    }
}

/* Splits [0, count) between threads in pieces; body(begin, end) for each. */
#define FOR_PIECES(r, count, begin, end, ...) do { \
        const uint64_t total_ = (count); \
        const int64_t pieces_ = (r)->parallel && total_ >= PARALLEL_WORK ? 64 : 1; \
        SPINGALETT_PARALLEL_FOR(pieces_ > 1, \
            for (int64_t p_ = 0; p_ < pieces_; p_++) { \
                const uint64_t begin = total_ * (uint64_t)p_ / (uint64_t)pieces_; \
                const uint64_t end = total_ * (uint64_t)(p_ + 1) / (uint64_t)pieces_; \
                __VA_ARGS__ \
            } \
        ); \
    } while (0)

/* Splits rows [0, rows) of `work` values in all between threads; body(begin, end) for each piece. */
#define FOR_ROWS(r, rows, work, begin, end, ...) do { \
        const uint64_t rows_ = (rows); \
        const int64_t pieces_ = (r)->parallel && (work) >= PARALLEL_WORK ? (rows_ < 64 ? (int64_t)rows_ : 64) : 1; \
        SPINGALETT_PARALLEL_FOR(pieces_ > 1, \
            for (int64_t p_ = 0; p_ < pieces_; p_++) { \
                const uint64_t begin = rows_ * (uint64_t)p_ / (uint64_t)pieces_; \
                const uint64_t end = rows_ * (uint64_t)(p_ + 1) / (uint64_t)pieces_; \
                __VA_ARGS__ \
            } \
        ); \
    } while (0)

/* ------------------------------------------------------------------------- element-wise */

/* Python's (and numpy's) remainder: the divisor's sign. */
static float python_mod(float a, float b) {
    float m = fmodf(a, b);
    if (b == 0.0f) return m;
    if (m != 0.0f) {
        if ((b < 0.0f) != (m < 0.0f)) m += b;
    } else {
        m = copysignf(0.0f, b);
    }
    return m;
}

static float binary_f(IrOp op, float a, float b, bool fmod_mode) {
    switch (op) {
        case IR_Add: case IR_Sum: return a + b;
        case IR_Sub: return a - b;
        case IR_Mul: return a * b;
        case IR_Div: return a / b;
        case IR_Pow: return powf(a, b);
        case IR_Mod: return fmod_mode ? fmodf(a, b) : python_mod(a, b);
        case IR_Max: return isnan(a) || isnan(b) ? NAN : a > b ? a : b;
        case IR_Min: return isnan(a) || isnan(b) ? NAN : a < b ? a : b;
        case IR_PRelu: return a < 0.0f ? a * b : a;
        default: return 0.0f;
    }
}

static int64_t binary_i(IrOp op, int64_t a, int64_t b, bool fmod_mode) {
    switch (op) {
        case IR_Add: case IR_Sum: return (int64_t)((uint64_t)a + (uint64_t)b);
        case IR_Sub: return (int64_t)((uint64_t)a - (uint64_t)b);
        case IR_Mul: return (int64_t)((uint64_t)a * (uint64_t)b);
        case IR_Div: return b ? (a == INT64_MIN && b == -1 ? a : a / b) : 0;
        case IR_Pow: {
            int64_t p = 1;
            for (int64_t k = 0; k < b && k < 64; k++) p *= a;
            return b < 0 ? 0 : p;
        }
        case IR_Mod: {
            if (!b) return 0;
            int64_t m = a % b;
            if (!fmod_mode && m != 0 && ((m < 0) != (b < 0))) m += b;      /* (Python's sign of the divisor) */
            return m;
        }
        case IR_Max: return a > b ? a : b;
        case IR_Min: return a < b ? a : b;
        case IR_PRelu: return a < 0 ? a * b : a;
        default: return 0;
    }
}

/* y = a op b with broadcasting, of a's type (Pow: a's). */
static bool binary_into(IrRun *r, IrOp op, IrTensor *a, IrTensor *b, IrTensor *y) {
    if (!a || !b) return ir_fail(r, "an input is missing");
    const bool floats = a->dtype == F32;
    if (op != IR_Pow && a->dtype != b->dtype)
        return ir_fail(r, "inputs of %s and %s", ir_dtype_name(a->dtype), ir_dtype_name(b->dtype));
    if (!floats && !is_int(a->dtype)) return ir_fail(r, "inputs of %s", ir_dtype_name(a->dtype));
    uint32_t R;
    int64_t dims[IR_MAX_RANK];
    IrTensor *in[2] = {a, b};
    if (!broadcast_shape(r, in, 2, &R, dims)) return false;
    /* (the output may be an input itself: Sum and friends fold into their first) */
    IrTensor out;
    memset(&out, 0, sizeof out);
    if (!ir_output(r, &out, a->dtype, R, dims)) return false;
    const bool fmod_mode = ir_attr_int(r->node, "fmod", 0) != 0;
    const bool same = a->count == out.count && b->count == out.count;
    if (floats && b->dtype == F32) {
        const float *x = fdata(a), *z = fdata(b);
        float *o = fdata(&out);
        if (same) {
            FOR_PIECES(r, out.count, begin, end,
                switch (op) {
                    case IR_Add: case IR_Sum: for (uint64_t i = begin; i < end; i++) o[i] = x[i] + z[i]; break;
                    case IR_Sub: for (uint64_t i = begin; i < end; i++) o[i] = x[i] - z[i]; break;
                    case IR_Mul: for (uint64_t i = begin; i < end; i++) o[i] = x[i] * z[i]; break;
                    case IR_Div: for (uint64_t i = begin; i < end; i++) o[i] = x[i] / z[i]; break;
                    default: for (uint64_t i = begin; i < end; i++) o[i] = binary_f(op, x[i], z[i], fmod_mode); break;
                }
            );
        } else if (b->count == 1 && a->count == out.count) {
            const float s = z[0];
            FOR_PIECES(r, out.count, begin, end,
                switch (op) {
                    case IR_Add: case IR_Sum: for (uint64_t i = begin; i < end; i++) o[i] = x[i] + s; break;
                    case IR_Sub: for (uint64_t i = begin; i < end; i++) o[i] = x[i] - s; break;
                    case IR_Mul: for (uint64_t i = begin; i < end; i++) o[i] = x[i] * s; break;
                    case IR_Div: for (uint64_t i = begin; i < end; i++) o[i] = x[i] / s; break;
                    default: for (uint64_t i = begin; i < end; i++) o[i] = binary_f(op, x[i], s, fmod_mode); break;
                }
            );
        } else {
            /* the longest run of trailing axes over which each operand is either contiguous or one value: rows of
               the output computed by the loops of the fast paths, the operands' offsets walked once a row */
            Walk w;
            walk_init(&w, in, 2, R, dims);
            uint32_t split = R;
            uint64_t inner = 1;
            int mode[2] = {0, 0};               /* over the run: 0 not yet known, 1 contiguous, 2 one value */
            for (uint32_t d = R; d-- > 0;) {
                bool ok = true;
                int next[2] = {mode[0], mode[1]};
                for (int k = 0; k < 2 && dims[d] != 1; k++) {
                    const int here = w.strides[k][d] == (int64_t)inner ? 1 : w.strides[k][d] == 0 ? 2 : -1;
                    if (here < 0 || (mode[k] && mode[k] != here)) ok = false;
                    next[k] = here;
                }
                if (!ok) break;
                mode[0] = next[0];
                mode[1] = next[1];
                split = d;
                inner *= (uint64_t)dims[d];
            }
            if (split < R && inner >= 16) {
                const uint64_t rows = out.count / inner;
                const bool sa = mode[0] == 1, sb = mode[1] == 1;
                FOR_ROWS(r, rows, out.count, begin, end,
                    for (uint64_t q = begin; q < end; q++) {
                        int64_t idx[IR_MAX_RANK], off[3];
                        walk_offsets(&w, q * inner, idx, off);
                        const float *xa = x + off[0], *zb = z + off[1];
                        float *oq = o + q * inner;
                        if (sa && sb) {
                            switch (op) {
                                case IR_Add: case IR_Sum: for (uint64_t i = 0; i < inner; i++) oq[i] = xa[i] + zb[i]; break;
                                case IR_Sub: for (uint64_t i = 0; i < inner; i++) oq[i] = xa[i] - zb[i]; break;
                                case IR_Mul: for (uint64_t i = 0; i < inner; i++) oq[i] = xa[i] * zb[i]; break;
                                case IR_Div: for (uint64_t i = 0; i < inner; i++) oq[i] = xa[i] / zb[i]; break;
                                default: for (uint64_t i = 0; i < inner; i++) oq[i] = binary_f(op, xa[i], zb[i], fmod_mode);
                            }
                        } else if (sa) {
                            const float c = zb[0];
                            switch (op) {
                                case IR_Add: case IR_Sum: for (uint64_t i = 0; i < inner; i++) oq[i] = xa[i] + c; break;
                                case IR_Sub: for (uint64_t i = 0; i < inner; i++) oq[i] = xa[i] - c; break;
                                case IR_Mul: for (uint64_t i = 0; i < inner; i++) oq[i] = xa[i] * c; break;
                                case IR_Div: for (uint64_t i = 0; i < inner; i++) oq[i] = xa[i] / c; break;
                                default: for (uint64_t i = 0; i < inner; i++) oq[i] = binary_f(op, xa[i], c, fmod_mode);
                            }
                        } else if (sb) {
                            const float c = xa[0];
                            switch (op) {
                                case IR_Add: case IR_Sum: for (uint64_t i = 0; i < inner; i++) oq[i] = c + zb[i]; break;
                                case IR_Sub: for (uint64_t i = 0; i < inner; i++) oq[i] = c - zb[i]; break;
                                case IR_Mul: for (uint64_t i = 0; i < inner; i++) oq[i] = c * zb[i]; break;
                                case IR_Div: for (uint64_t i = 0; i < inner; i++) oq[i] = c / zb[i]; break;
                                default: for (uint64_t i = 0; i < inner; i++) oq[i] = binary_f(op, c, zb[i], fmod_mode);
                            }
                        } else {
                            const float c = binary_f(op, xa[0], zb[0], fmod_mode);
                            for (uint64_t i = 0; i < inner; i++) oq[i] = c;
                        }
                    }
                );
            } else {
                FOR_PIECES(r, out.count, begin, end,
                    int64_t idx[IR_MAX_RANK], off[3];
                    if (begin < end) walk_offsets(&w, begin, idx, off);
                    for (uint64_t i = begin; i < end; i++) {
                        o[i] = binary_f(op, x[off[0]], z[off[1]], fmod_mode);
                        walk_step(&w, idx, off);
                    }
                );
            }
        }
    } else if (floats) {
        /* Pow of a float by an integer exponent */
        Walk w;
        walk_init(&w, in, 2, R, dims);
        float *o = fdata(&out);
        int64_t idx[IR_MAX_RANK], off[3];
        if (out.count) walk_offsets(&w, 0, idx, off);
        for (uint64_t i = 0; i < out.count; i++) {
            o[i] = powf(fdata(a)[off[0]], (float)ir_get_float(b, (uint64_t)off[1]));
            walk_step(&w, idx, off);
        }
    } else {
        Walk w;
        walk_init(&w, in, 2, R, dims);
        int64_t idx[IR_MAX_RANK], off[3];
        if (out.count) walk_offsets(&w, 0, idx, off);
        for (uint64_t i = 0; i < out.count; i++) {
            put_int(&out, i, binary_i(op, ir_get_int(a, (uint64_t)off[0]), ir_get_int(b, (uint64_t)off[1]), fmod_mode));
            walk_step(&w, idx, off);
        }
    }
    ir_release(r->g, y);
    *y = out;
    return true;
}

static bool k_binary(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1) return ir_fail(r, "two inputs and an output are needed");
    return binary_into(r, OP(r), in[0], in[1], out[0]);
}

/* Max, Min, Sum, Mean of any number of inputs, folded left to right. */
static bool k_variadic(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    const IrOp op = OP(r) == IR_Mean ? IR_Sum : OP(r);
    if (n == 1) {
        if (!ir_output(r, out[0], in[0]->dtype, in[0]->rank, in[0]->dims)) return false;
        memcpy(out[0]->data, in[0]->data, (size_t)in[0]->count * ir_dtype_size(in[0]->dtype));
    } else {
        if (!binary_into(r, op, in[0], in[1], out[0])) return false;
        for (uint32_t k = 2; k < n; k++)
            if (!binary_into(r, op, out[0], in[k], out[0])) return false;
    }
    if (OP(r) == IR_Mean) {
        if (out[0]->dtype != F32) return ir_fail(r, "Mean of integers");
        float *o = fdata(out[0]);
        for (uint64_t i = 0; i < out[0]->count; i++) o[i] /= (float)n;
    }
    return true;
}

static bool k_compare(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "two inputs and an output are needed");
    IrTensor *a = in[0], *b = in[1];
    if (a->dtype != b->dtype) return ir_fail(r, "inputs of %s and %s", ir_dtype_name(a->dtype), ir_dtype_name(b->dtype));
    uint32_t R;
    int64_t dims[IR_MAX_RANK];
    IrTensor *pair[2] = {a, b};
    if (!broadcast_shape(r, pair, 2, &R, dims) || !ir_output(r, out[0], BOOL, R, dims)) return false;
    Walk w;
    walk_init(&w, pair, 2, R, dims);
    uint8_t *o = (uint8_t *)out[0]->data;
    const IrOp op = OP(r);
    const bool floats = a->dtype == F32;
    FOR_PIECES(r, out[0]->count, begin, end,
        int64_t idx[IR_MAX_RANK], off[3];
        if (begin < end) walk_offsets(&w, begin, idx, off);
        for (uint64_t i = begin; i < end; i++) {
            bool v;
            if (floats) {
                const float x = fdata(a)[off[0]], z = fdata(b)[off[1]];
                v = op == IR_Equal ? x == z : op == IR_Less ? x < z : op == IR_LessOrEqual ? x <= z
                  : op == IR_Greater ? x > z : x >= z;
            } else {
                const int64_t x = ir_get_int(a, (uint64_t)off[0]), z = ir_get_int(b, (uint64_t)off[1]);
                v = op == IR_Equal ? x == z : op == IR_Less ? x < z : op == IR_LessOrEqual ? x <= z
                  : op == IR_Greater ? x > z : x >= z;
            }
            o[i] = v;
            walk_step(&w, idx, off);
        }
    );
    return true;
}

static bool k_logic(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "two inputs and an output are needed");
    uint32_t R;
    int64_t dims[IR_MAX_RANK];
    IrTensor *pair[2] = {in[0], in[1]};
    if (!broadcast_shape(r, pair, 2, &R, dims) || !ir_output(r, out[0], BOOL, R, dims)) return false;
    Walk w;
    walk_init(&w, pair, 2, R, dims);
    uint8_t *o = (uint8_t *)out[0]->data;
    int64_t idx[IR_MAX_RANK], off[3];
    if (out[0]->count) walk_offsets(&w, 0, idx, off);
    for (uint64_t i = 0; i < out[0]->count; i++) {
        const bool x = ir_get_int(in[0], (uint64_t)off[0]) != 0, z = ir_get_int(in[1], (uint64_t)off[1]) != 0;
        o[i] = OP(r) == IR_And ? x && z : OP(r) == IR_Or ? x || z : x != z;
        walk_step(&w, idx, off);
    }
    return true;
}

static bool k_where(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 3 || m < 1 || !in[0] || !in[1] || !in[2]) return ir_fail(r, "three inputs and an output are needed");
    if (in[1]->dtype != in[2]->dtype) return ir_fail(r, "values of %s and %s", ir_dtype_name(in[1]->dtype),
                                                    ir_dtype_name(in[2]->dtype));
    uint32_t R;
    int64_t dims[IR_MAX_RANK];
    if (!broadcast_shape(r, in, 3, &R, dims) || !ir_output(r, out[0], in[1]->dtype, R, dims)) return false;
    Walk w;
    walk_init(&w, in, 3, R, dims);
    const size_t size = ir_dtype_size(in[1]->dtype);
    const uint8_t *c = (const uint8_t *)in[0]->data;
    const size_t csize = ir_dtype_size(in[0]->dtype);
    FOR_PIECES(r, out[0]->count, begin, end,
        int64_t idx[IR_MAX_RANK], off[3];
        if (begin < end) walk_offsets(&w, begin, idx, off);
        for (uint64_t i = begin; i < end; i++) {
            bool cond = false;
            for (size_t b = 0; b < csize; b++) cond = cond || c[(size_t)off[0] * csize + b];
            const uint8_t *src = cond ? (const uint8_t *)in[1]->data + (size_t)off[1] * size
                                      : (const uint8_t *)in[2]->data + (size_t)off[2] * size;
            memcpy((uint8_t *)out[0]->data + i * size, src, size);
            walk_step(&w, idx, off);
        }
    );
    return true;
}

static bool k_unary(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    const IrOp op = OP(r);
    IrTensor *x = in[0];
    const bool to_bool = op == IR_IsNaN || op == IR_IsInf;
    if (op == IR_Not) {
        if (!ir_output(r, out[0], x->dtype, x->rank, x->dims)) return false;
        for (uint64_t i = 0; i < x->count; i++) put_int(out[0], i, ir_get_int(x, i) == 0);
        return true;
    }
    if (x->dtype != F32) {
        /* integers: Abs, Neg, Sign, Relu and identities */
        if (!is_int(x->dtype) || !ir_output(r, out[0], x->dtype, x->rank, x->dims))
            return x->dtype == BOOL || !is_int(x->dtype) ? ir_fail(r, "an input of %s", ir_dtype_name(x->dtype)) : false;
        for (uint64_t i = 0; i < x->count; i++) {
            const int64_t v = ir_get_int(x, i);
            int64_t y = v;
            if (op == IR_Abs) y = v < 0 ? -v : v;
            else if (op == IR_Neg) y = -v;
            else if (op == IR_Sign) y = (v > 0) - (v < 0);
            else if (op == IR_Relu) y = v > 0 ? v : 0;
            else if (op == IR_BitwiseNot) y = ~v;
            else if (op != IR_Floor && op != IR_Ceil && op != IR_Round) return ir_fail(r, "an input of %s",
                                                                                      ir_dtype_name(x->dtype));
            put_int(out[0], i, y);
        }
        return true;
    }
    if (!ir_output(r, out[0], to_bool ? BOOL : F32, x->rank, x->dims)) return false;
    const float *a = fdata(x);
    const float alpha = ir_attr_float(r->node, "alpha", op == IR_LeakyRelu ? 0.01f : op == IR_HardSigmoid ? 0.2f
                                                     : op == IR_Selu ? 1.67326319217681884765625f : 1.0f);
    const float beta = ir_attr_float(r->node, "beta", 0.5f), bias = ir_attr_float(r->node, "bias", 0.0f);
    const float lambd = ir_attr_float(r->node, "lambd", 0.5f);
    const float gamma = ir_attr_float(r->node, "gamma", 1.05070102214813232421875f);
    const IrAttr *approx = ir_attr(r->node, "approximate");
    const bool tanh_form = approx && approx->type == ATTR_STRING && !strcmp(approx->s, "tanh");
    if (to_bool) {
        uint8_t *o = (uint8_t *)out[0]->data;
        const int64_t pos = ir_attr_int(r->node, "detect_positive", 1), neg = ir_attr_int(r->node, "detect_negative", 1);
        for (uint64_t i = 0; i < x->count; i++)
            o[i] = op == IR_IsNaN ? isnan(a[i]) != 0 : isinf(a[i]) && ((a[i] > 0 && pos) || (a[i] < 0 && neg));
        return true;
    }
    float *o = fdata(out[0]);
    FOR_PIECES(r, x->count, begin, end,
        switch (op) {
            case IR_Abs: for (uint64_t i = begin; i < end; i++) o[i] = fabsf(a[i]); break;
            case IR_Neg: for (uint64_t i = begin; i < end; i++) o[i] = -a[i]; break;
            case IR_Exp: for (uint64_t i = begin; i < end; i++) o[i] = expf(a[i]); break;
            case IR_Log: for (uint64_t i = begin; i < end; i++) o[i] = logf(a[i]); break;
            case IR_Sqrt: for (uint64_t i = begin; i < end; i++) o[i] = sqrtf(a[i]); break;
            case IR_Reciprocal: for (uint64_t i = begin; i < end; i++) o[i] = 1.0f / a[i]; break;
            case IR_Sin: for (uint64_t i = begin; i < end; i++) o[i] = sinf(a[i]); break;
            case IR_Cos: for (uint64_t i = begin; i < end; i++) o[i] = cosf(a[i]); break;
            case IR_Tanh: for (uint64_t i = begin; i < end; i++) o[i] = tanhf(a[i]); break;
            case IR_Sigmoid: for (uint64_t i = begin; i < end; i++) o[i] = 1.0f / (1.0f + expf(-a[i])); break;
            case IR_Erf: for (uint64_t i = begin; i < end; i++) o[i] = erff(a[i]); break;
            case IR_Relu: for (uint64_t i = begin; i < end; i++) o[i] = a[i] > 0.0f ? a[i] : 0.0f; break;
            case IR_LeakyRelu: for (uint64_t i = begin; i < end; i++) o[i] = a[i] >= 0.0f ? a[i] : alpha * a[i]; break;
            case IR_Elu: for (uint64_t i = begin; i < end; i++) o[i] = a[i] >= 0.0f ? a[i] : alpha * expm1f(a[i]); break;
            case IR_Selu:
                for (uint64_t i = begin; i < end; i++) o[i] = gamma * (a[i] > 0.0f ? a[i] : alpha * (expf(a[i]) - 1.0f));
                break;
            case IR_Gelu:
                memcpy(o + begin, a + begin, (size_t)(end - begin) * sizeof(float));
                spingalett_activate_training(o + begin, end - begin, tanh_form ? ACT_GELU_TANH : ACT_GELU);
                break;
            case IR_HardSigmoid:
                for (uint64_t i = begin; i < end; i++) o[i] = fmaxf(0.0f, fminf(1.0f, alpha * a[i] + beta));
                break;
            case IR_HardSwish:
                for (uint64_t i = begin; i < end; i++) o[i] = a[i] * fmaxf(0.0f, fminf(1.0f, a[i] / 6.0f + 0.5f));
                break;
            case IR_Softplus: for (uint64_t i = begin; i < end; i++) o[i] = log1pf(expf(a[i])); break;
            case IR_Softsign: for (uint64_t i = begin; i < end; i++) o[i] = a[i] / (1.0f + fabsf(a[i])); break;
            case IR_Mish: for (uint64_t i = begin; i < end; i++) o[i] = a[i] * tanhf(log1pf(expf(a[i]))); break;
            case IR_Floor: for (uint64_t i = begin; i < end; i++) o[i] = floorf(a[i]); break;
            case IR_Ceil: for (uint64_t i = begin; i < end; i++) o[i] = ceilf(a[i]); break;
            case IR_Round: for (uint64_t i = begin; i < end; i++) o[i] = nearbyintf(a[i]); break;
            case IR_Sign: for (uint64_t i = begin; i < end; i++) o[i] = (float)((a[i] > 0.0f) - (a[i] < 0.0f)); break;
            case IR_Tan: for (uint64_t i = begin; i < end; i++) o[i] = tanf(a[i]); break;
            case IR_Asin: for (uint64_t i = begin; i < end; i++) o[i] = asinf(a[i]); break;
            case IR_Acos: for (uint64_t i = begin; i < end; i++) o[i] = acosf(a[i]); break;
            case IR_Atan: for (uint64_t i = begin; i < end; i++) o[i] = atanf(a[i]); break;
            case IR_Sinh: for (uint64_t i = begin; i < end; i++) o[i] = sinhf(a[i]); break;
            case IR_Cosh: for (uint64_t i = begin; i < end; i++) o[i] = coshf(a[i]); break;
            case IR_Asinh: for (uint64_t i = begin; i < end; i++) o[i] = asinhf(a[i]); break;
            case IR_Acosh: for (uint64_t i = begin; i < end; i++) o[i] = acoshf(a[i]); break;
            case IR_Atanh: for (uint64_t i = begin; i < end; i++) o[i] = atanhf(a[i]); break;
            case IR_Celu:
                for (uint64_t i = begin; i < end; i++) o[i] = fmaxf(0.0f, a[i]) + fminf(0.0f, alpha * expm1f(a[i] / alpha));
                break;
            case IR_ThresholdedRelu: for (uint64_t i = begin; i < end; i++) o[i] = a[i] > alpha ? a[i] : 0.0f; break;
            case IR_Shrink:
                for (uint64_t i = begin; i < end; i++) o[i] = a[i] < -lambd ? a[i] + bias : a[i] > lambd ? a[i] - bias : 0.0f;
                break;
            case IR_Swish:
                if (alpha == 1.0f) {
                    memcpy(o + begin, a + begin, (size_t)(end - begin) * sizeof(float));
                    spingalett_activate_training(o + begin, end - begin, ACT_SILU);
                } else {
                    for (uint64_t i = begin; i < end; i++) o[i] = a[i] / (1.0f + expf(-alpha * a[i]));
                }
                break;
            default: break;
        }
    );
    return true;
}

static bool k_clip(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    IrTensor *x = in[0];
    if (!ir_output(r, out[0], x->dtype, x->rank, x->dims)) return false;
    if (x->dtype == F32) {
        float lo = ir_attr_float(r->node, "min", -FLT_MAX), hi = ir_attr_float(r->node, "max", FLT_MAX);
        if (n > 1 && in[1] && in[1]->count) lo = (float)ir_get_float(in[1], 0);
        if (n > 2 && in[2] && in[2]->count) hi = (float)ir_get_float(in[2], 0);
        const float *a = fdata(x);
        float *o = fdata(out[0]);
        for (uint64_t i = 0; i < x->count; i++) {
            const float v = a[i] < lo ? lo : a[i];          /* (numpy's order: a min above the max gives the max) */
            o[i] = v > hi ? hi : v;
        }
        return true;
    }
    int64_t lo = INT64_MIN, hi = INT64_MAX;
    if (n > 1 && in[1] && in[1]->count) lo = ir_get_int(in[1], 0);
    if (n > 2 && in[2] && in[2]->count) hi = ir_get_int(in[2], 0);
    for (uint64_t i = 0; i < x->count; i++) {
        const int64_t v = ir_get_int(x, i);
        const int64_t c = v < lo ? lo : v;
        put_int(out[0], i, c > hi ? hi : c);
    }
    return true;
}

/* Cast and CastLike: floats to integers truncate toward zero (ONNX's saturate is for its 8-bit floats); to float16
   and bfloat16 the values are rounded to those types and kept as float32. */
static bool k_cast(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    IrTensor *x = in[0];
    int64_t to;
    if (OP(r) == IR_CastLike) {
        if (n < 2 || !in[1]) return ir_fail(r, "CastLike needs a second input");
        to = in[1]->dtype;
    } else {
        to = ir_attr_int(r->node, "to", 0);
        /* ONNX's codes: those graphs hold, others as their nearest */
        if (to == 11) to = F32;
        else if (to == 4 || to == 5) to = I32;
        else if (to == 12 || to == 13) to = I64;
    }
    const bool narrow = to == SPINGALETT_DTYPE_FLOAT16 || to == SPINGALETT_DTYPE_BFLOAT16;
    const uint8_t dtype = narrow ? F32 : (uint8_t)to;
    if (!ir_dtype_size(dtype)) return ir_fail(r, "a cast to ONNX's type %lld", (long long)to);
    if (!ir_output(r, out[0], dtype, x->rank, x->dims)) return false;
    for (uint64_t i = 0; i < x->count; i++) {
        if (dtype == F32) {
            float v = (float)ir_get_float(x, i);
            if (to == SPINGALETT_DTYPE_FLOAT16) v = spingalett_fp16_to_float(spingalett_float_to_fp16(v));
            else if (to == SPINGALETT_DTYPE_BFLOAT16) v = spingalett_bf16_to_float(spingalett_float_to_bf16(v));
            fdata(out[0])[i] = v;
        } else if (dtype == BOOL) {
            ((uint8_t *)out[0]->data)[i] = x->dtype == F32 ? fdata(x)[i] != 0.0f : ir_get_int(x, i) != 0;
        } else if (x->dtype == F32) {
            const float v = fdata(x)[i];
            put_int(out[0], i, isnan(v) ? 0 : v >= 9.2233720368547758e18f ? INT64_MAX : v <= -9.2233720368547758e18f
                                                  ? INT64_MIN : (int64_t)v);
        } else {
            put_int(out[0], i, ir_get_int(x, i));
        }
    }
    return true;
}

static bool k_identity(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    ir_view(r, out[0], in[0], in[0]->rank, in[0]->dims);
    if (OP(r) == IR_Dropout && m > 1 && out[1]) {      /* the mask: all kept */
        if (!ir_output(r, out[1], BOOL, in[0]->rank, in[0]->dims)) return false;
        memset(out[1]->data, 1, (size_t)out[1]->count);
    }
    return true;
}

/* ------------------------------------------------------------------------- constants and shapes as data */

static bool k_constant(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    (void)in;
    (void)n;
    if (m < 1) return ir_fail(r, "no output");
    const IrNode *node = r->node;
    for (uint32_t a = 0; a < node->attr_count; a++) {
        const IrAttr *at = &node->attrs[a];
        if (!strcmp(at->name, "value") && at->type == ATTR_TENSOR) {
            const IrValue *v = &r->g->values[at->tensor];
            if (!ir_output(r, out[0], v->dtype, v->rank, v->dims)) return false;
            memcpy(out[0]->data, v->data, (size_t)out[0]->count * ir_dtype_size(v->dtype));
            return true;
        }
        if (!strcmp(at->name, "value_float") || !strcmp(at->name, "value_int")) {
            const bool f = !strcmp(at->name, "value_float");
            if (!ir_output(r, out[0], f ? F32 : I64, 0, NULL)) return false;
            if (f) fdata(out[0])[0] = at->f;
            else ((int64_t *)out[0]->data)[0] = at->i;
            return true;
        }
        if (!strcmp(at->name, "value_floats") || !strcmp(at->name, "value_ints")) {
            const bool f = !strcmp(at->name, "value_floats");
            const int64_t dims[1] = {at->count};
            if (!ir_output(r, out[0], f ? F32 : I64, 1, dims)) return false;
            if (f) memcpy(out[0]->data, at->floats, at->count * sizeof(float));
            else memcpy(out[0]->data, at->ints, at->count * sizeof(int64_t));
            return true;
        }
    }
    return ir_fail(r, "a constant without a value (sparse and string constants are not supported)");
}

static bool k_shape(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    const int64_t R = in[0]->rank;
    int64_t start = ir_attr_int(r->node, "start", 0), end = ir_attr_int(r->node, "end", R);
    if (start < 0) start += R;
    if (end < 0) end += R;
    start = start < 0 ? 0 : start > R ? R : start;
    end = end < 0 ? 0 : end > R ? R : end;
    const int64_t len = end > start ? end - start : 0;
    if (!ir_output(r, out[0], I64, 1, &len)) return false;
    for (int64_t k = 0; k < len; k++) ((int64_t *)out[0]->data)[k] = in[0]->dims[start + k];
    return true;
}

static bool k_size(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    if (!ir_output(r, out[0], I64, 0, NULL)) return false;
    ((int64_t *)out[0]->data)[0] = (int64_t)in[0]->count;
    return true;
}

static bool k_constant_of_shape(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    int64_t dims[IR_MAX_RANK];
    if (in[0]->count > IR_MAX_RANK) return ir_fail(r, "a shape of more than eight axes");
    for (uint64_t k = 0; k < in[0]->count; k++) dims[k] = ir_get_int(in[0], k);
    const IrAttr *v = ir_attr(r->node, "value");
    const IrValue *fill = v && v->type == ATTR_TENSOR ? &r->g->values[v->tensor] : NULL;
    const uint8_t dtype = fill ? fill->dtype : F32;
    if (!ir_output(r, out[0], dtype, (uint32_t)in[0]->count, dims)) return false;
    if (fill) {
        const size_t size = ir_dtype_size(dtype);
        for (uint64_t i = 0; i < out[0]->count; i++) memcpy((uint8_t *)out[0]->data + i * size, fill->data, size);
    }
    return true;
}

static bool k_range(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 3 || m < 1 || !in[0] || !in[1] || !in[2]) return ir_fail(r, "three inputs and an output are needed");
    const uint8_t dtype = in[0]->dtype;
    if (dtype == F32) {
        const double start = ir_get_float(in[0], 0), limit = ir_get_float(in[1], 0), delta = ir_get_float(in[2], 0);
        if (delta == 0.0) return ir_fail(r, "a delta of 0");
        double c = ceil((limit - start) / delta);
        const int64_t len = c > 0.0 ? (int64_t)c : 0;
        if (!ir_output(r, out[0], F32, 1, &len)) return false;
        for (int64_t k = 0; k < len; k++) fdata(out[0])[k] = (float)(start + (double)k * delta);
        return true;
    }
    const int64_t start = ir_get_int(in[0], 0), limit = ir_get_int(in[1], 0), delta = ir_get_int(in[2], 0);
    if (!delta) return ir_fail(r, "a delta of 0");
    int64_t len = (limit - start + delta + (delta > 0 ? -1 : 1)) / delta;
    if (len < 0) len = 0;
    if (!ir_output(r, out[0], dtype, 1, &len)) return false;
    for (int64_t k = 0; k < len; k++) put_int(out[0], (uint64_t)k, start + k * delta);
    return true;
}

/* ------------------------------------------------------------------------- shapes */

/* y: x's values in another shape, a view of them (no copy). */
static bool copy_as(IrRun *r, const IrTensor *x, IrTensor *y, uint32_t rank, const int64_t *dims) {
    uint64_t count = 1;
    for (uint32_t k = 0; k < rank; k++) count *= (uint64_t)dims[k];
    if (count != x->count) return ir_fail(r, "%llu values into a shape of %llu", (unsigned long long)x->count,
                                          (unsigned long long)count);
    if (rank > IR_MAX_RANK) return ir_fail(r, "more than eight axes");
    ir_view(r, y, x, rank, dims);
    return true;
}

/* Reshape, Flatten, Squeeze, Unsqueeze: the values as they are, a new shape. */
static bool k_reshape(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    const IrTensor *x = in[0];
    int64_t dims[IR_MAX_RANK];
    uint32_t rank = 0;
    switch (OP(r)) {
        case IR_Reshape: {
            if (n < 2 || !in[1]) return ir_fail(r, "Reshape needs a shape");
            if (in[1]->count > IR_MAX_RANK) return ir_fail(r, "a shape of more than eight axes");
            const bool allowzero = ir_attr_int(r->node, "allowzero", 0) != 0;
            rank = (uint32_t)in[1]->count;
            int64_t known = 1, infer = -1;
            for (uint32_t k = 0; k < rank; k++) {
                int64_t d = ir_get_int(in[1], k);
                if (d == 0 && !allowzero) {
                    if (k >= x->rank) return ir_fail(r, "a size 0 that copies an axis the input lacks");
                    d = x->dims[k];
                }
                if (d == -1) {
                    if (infer >= 0) return ir_fail(r, "two sizes of -1");
                    infer = k;
                    d = 1;
                } else if (d < 0) return ir_fail(r, "a size of %lld", (long long)d);
                dims[k] = d;
                known *= d;
            }
            if (infer >= 0) {
                if (!known || x->count % (uint64_t)known) return ir_fail(r, "%llu values do not divide into the shape",
                                                                        (unsigned long long)x->count);
                dims[infer] = (int64_t)(x->count / (uint64_t)known);
            }
            break;
        }
        case IR_Flatten: {
            int64_t axis = ir_attr_int(r->node, "axis", 1);
            if (axis < 0) axis += x->rank;
            if (axis < 0 || axis > x->rank) return ir_fail(r, "the axis %lld", (long long)axis);
            int64_t a = 1;
            for (int64_t k = 0; k < axis; k++) a *= x->dims[k];
            dims[0] = a;
            dims[1] = a ? (int64_t)(x->count / (uint64_t)a) : 0;
            if (!a) {
                int64_t b = 1;
                for (uint32_t k = (uint32_t)axis; k < x->rank; k++) b *= x->dims[k];
                dims[1] = b;
            }
            rank = 2;
            break;
        }
        case IR_Squeeze: {
            int64_t axes[IR_MAX_RANK];
            const int64_t na = int_list(r->node, "axes", n > 1 ? in[1] : NULL, axes, IR_MAX_RANK);
            bool drop[IR_MAX_RANK] = {false};
            for (int64_t k = 0; k < na; k++) {
                uint32_t a;
                if (!axis_of(r, axes[k], x->rank, &a)) return false;
                if (x->dims[a] != 1) return ir_fail(r, "an axis of size %lld squeezed", (long long)x->dims[a]);
                drop[a] = true;
            }
            for (uint32_t k = 0; k < x->rank; k++)
                if (!(na < 0 ? x->dims[k] == 1 : drop[k])) dims[rank++] = x->dims[k];
            break;
        }
        default: {                  /* Unsqueeze */
            int64_t axes[IR_MAX_RANK];
            const int64_t na = int_list(r->node, "axes", n > 1 ? in[1] : NULL, axes, IR_MAX_RANK);
            if (na < 0) return ir_fail(r, "Unsqueeze needs axes");
            rank = x->rank + (uint32_t)na;
            if (rank > IR_MAX_RANK) return ir_fail(r, "more than eight axes");
            bool add[IR_MAX_RANK] = {false};
            for (int64_t k = 0; k < na; k++) {
                uint32_t a;
                if (!axis_of(r, axes[k], rank, &a)) return false;
                add[a] = true;
            }
            for (uint32_t k = 0, j = 0; k < rank; k++) dims[k] = add[k] ? 1 : x->dims[j++];
            break;
        }
    }
    return copy_as(r, x, out[0], rank, dims);
}

/* y made of x's values through a permutation of its axes: y[i...] = x[p(i...)]. */
static bool permute(IrRun *r, const IrTensor *x, const int64_t *perm, IrTensor *y) {
    const uint32_t R = x->rank;
    int64_t dims[IR_MAX_RANK], xstride[IR_MAX_RANK], stride[IR_MAX_RANK];
    bool seen[IR_MAX_RANK] = {false};
    for (uint32_t k = 0; k < R; k++) {
        if (perm[k] < 0 || perm[k] >= (int64_t)R || seen[perm[k]]) return ir_fail(r, "perm is no permutation");
        seen[perm[k]] = true;
        dims[k] = x->dims[perm[k]];
    }
    if (!ir_output_raw(r, y, x->dtype, R, dims)) return false;
    int64_t s = 1;
    for (uint32_t k = R; k-- > 0;) { xstride[k] = s; s *= x->dims[k]; }
    for (uint32_t k = 0; k < R; k++) stride[k] = xstride[perm[k]];
    const size_t size = ir_dtype_size(x->dtype);
    const uint8_t *src = (const uint8_t *)x->data;
    uint8_t *dst = (uint8_t *)y->data;
    /* trailing axes the permutation keeps in place: runs of values copied whole */
    uint32_t kept = R;
    uint64_t run = 1;
    while (kept > 0 && perm[kept - 1] == (int64_t)(kept - 1) && stride[kept - 1] == (int64_t)run) {
        run *= (uint64_t)dims[kept - 1];
        kept--;
    }
    if (kept < R && run >= 8 && y->count) {
        const uint64_t runs = y->count / run;
        FOR_ROWS(r, runs, y->count, begin, end,
            int64_t idx[IR_MAX_RANK], off = 0;
            uint64_t i = begin;
            for (uint32_t k = kept; k-- > 0;) {
                idx[k] = (int64_t)(i % (uint64_t)dims[k]);
                i /= (uint64_t)dims[k];
                off += idx[k] * stride[k];
            }
            for (uint64_t j = begin; j < end; j++) {
                memcpy(dst + j * run * size, src + (size_t)off * size, (size_t)run * size);
                for (uint32_t k = kept; k-- > 0;) {
                    off += stride[k];
                    if (++idx[k] < dims[k]) break;
                    off -= stride[k] * dims[k];
                    idx[k] = 0;
                }
            }
        );
        return true;
    }
    FOR_PIECES(r, y->count, begin, end,
        int64_t idx[IR_MAX_RANK], off = 0;
        uint64_t i = begin;
        for (uint32_t k = R; k-- > 0;) {
            idx[k] = (int64_t)(i % (uint64_t)dims[k]);
            i /= (uint64_t)dims[k];
            off += idx[k] * stride[k];
        }
        for (uint64_t j = begin; j < end; j++) {
            if (size == 4) memcpy(dst + 4 * j, src + 4 * off, 4);
            else memcpy(dst + size * j, src + size * (size_t)off, size);
            for (uint32_t k = R; k-- > 0;) {
                off += stride[k];
                if (++idx[k] < dims[k]) break;
                off -= stride[k] * dims[k];
                idx[k] = 0;
            }
        }
    );
    return true;
}

static bool k_transpose(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    int64_t perm[IR_MAX_RANK];
    if (int_list(r->node, "perm", NULL, perm, IR_MAX_RANK) != (int64_t)in[0]->rank)
        for (uint32_t k = 0; k < in[0]->rank; k++) perm[k] = in[0]->rank - 1 - k;
    return permute(r, in[0], perm, out[0]);
}

static bool k_concat(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1) return ir_fail(r, "inputs and an output are needed");
    const IrTensor *first = NULL;
    for (uint32_t k = 0; k < n && !first; k++) if (in[k]) first = in[k];
    if (!first) return ir_fail(r, "no inputs");
    uint32_t axis;
    if (!axis_of(r, ir_attr_int(r->node, "axis", 0), first->rank, &axis)) return false;
    int64_t dims[IR_MAX_RANK];
    for (uint32_t d = 0; d < first->rank; d++) dims[d] = first->dims[d];
    dims[axis] = 0;
    for (uint32_t k = 0; k < n; k++) {
        if (!in[k]) continue;
        if (in[k]->rank != first->rank || in[k]->dtype != first->dtype) return ir_fail(r, "inputs of different ranks or types");
        for (uint32_t d = 0; d < first->rank; d++)
            if (d != axis && in[k]->dims[d] != first->dims[d]) return ir_fail(r, "inputs of different shapes");
        dims[axis] += in[k]->dims[axis];
    }
    if (!ir_output(r, out[0], first->dtype, first->rank, dims)) return false;
    uint64_t outer = 1, inner = 1;
    for (uint32_t d = 0; d < axis; d++) outer *= (uint64_t)dims[d];
    for (uint32_t d = axis + 1; d < first->rank; d++) inner *= (uint64_t)dims[d];
    const size_t size = ir_dtype_size(first->dtype);
    const uint64_t row = (uint64_t)dims[axis] * inner * size;
    uint64_t at = 0;
    for (uint32_t k = 0; k < n; k++) {
        if (!in[k]) continue;
        const uint64_t part = (uint64_t)in[k]->dims[axis] * inner * size;
        for (uint64_t o = 0; o < outer; o++)
            if (part) memcpy((uint8_t *)out[0]->data + o * row + at, (const uint8_t *)in[k]->data + o * part, (size_t)part);
        at += part;
    }
    return true;
}

static bool k_split(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and outputs are needed");
    const IrTensor *x = in[0];
    uint32_t axis;
    if (!axis_of(r, ir_attr_int(r->node, "axis", 0), x->rank, &axis)) return false;
    int64_t sizes[64];
    int64_t ns = int_list(r->node, "split", n > 1 ? in[1] : NULL, sizes, 64);
    const int64_t total = x->dims[axis];
    if (ns < 0) {
        /* equal parts (opset 18: num_outputs, the last smaller) */
        const int64_t parts = (int64_t)m, each = (total + parts - 1) / parts;
        for (int64_t k = 0; k < parts; k++) sizes[k] = k + 1 < parts ? each : total - each * (parts - 1);
        ns = parts;
    }
    if (ns != (int64_t)m) return ir_fail(r, "%lld sizes for %u outputs", (long long)ns, m);
    int64_t sum = 0;
    for (int64_t k = 0; k < ns; k++) { if (sizes[k] < 0) return ir_fail(r, "a negative size"); sum += sizes[k]; }
    if (sum != total) return ir_fail(r, "sizes that add up to %lld of %lld", (long long)sum, (long long)total);
    uint64_t outer = 1, inner = 1;
    for (uint32_t d = 0; d < axis; d++) outer *= (uint64_t)x->dims[d];
    for (uint32_t d = axis + 1; d < x->rank; d++) inner *= (uint64_t)x->dims[d];
    const size_t size = ir_dtype_size(x->dtype);
    const uint64_t row = (uint64_t)total * inner * size;
    uint64_t at = 0;
    for (uint32_t k = 0; k < m; k++) {
        int64_t dims[IR_MAX_RANK];
        for (uint32_t d = 0; d < x->rank; d++) dims[d] = x->dims[d];
        dims[axis] = sizes[k];
        const uint64_t part = (uint64_t)sizes[k] * inner * size;
        if (out[k]) {
            if (!ir_output(r, out[k], x->dtype, x->rank, dims)) return false;
            for (uint64_t o = 0; o < outer; o++)
                if (part) memcpy((uint8_t *)out[k]->data + o * part, (const uint8_t *)x->data + o * row + at, (size_t)part);
        }
        at += part;
    }
    return true;
}

static bool k_slice(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    const IrTensor *x = in[0];
    const uint32_t R = x->rank;
    int64_t starts[IR_MAX_RANK], ends[IR_MAX_RANK], axes[IR_MAX_RANK], steps[IR_MAX_RANK];
    const int64_t ns = int_list(r->node, "starts", n > 1 ? in[1] : NULL, starts, IR_MAX_RANK);
    const int64_t ne = int_list(r->node, "ends", n > 2 ? in[2] : NULL, ends, IR_MAX_RANK);
    int64_t na = int_list(r->node, "axes", n > 3 ? in[3] : NULL, axes, IR_MAX_RANK);
    const int64_t nt = int_list(r->node, "steps", n > 4 ? in[4] : NULL, steps, IR_MAX_RANK);
    if (ns < 0 || ne != ns) return ir_fail(r, "starts and ends of different lengths");
    if (na < 0) { for (int64_t k = 0; k < ns; k++) axes[k] = k; na = ns; }
    if (na != ns || (nt >= 0 && nt != ns)) return ir_fail(r, "axes or steps of another length");
    int64_t first[IR_MAX_RANK], step[IR_MAX_RANK], dims[IR_MAX_RANK];
    for (uint32_t d = 0; d < R; d++) { first[d] = 0; step[d] = 1; dims[d] = x->dims[d]; }
    for (int64_t k = 0; k < ns; k++) {
        uint32_t a;
        if (!axis_of(r, axes[k], R, &a)) return false;
        const int64_t size = x->dims[a], st = nt >= 0 ? steps[k] : 1;
        if (!st) return ir_fail(r, "a step of 0");
        int64_t s = starts[k], e = ends[k];
        if (s < 0) s += size;
        if (e < 0) e += size;
        if (st > 0) {
            s = s < 0 ? 0 : s > size ? size : s;
            e = e < 0 ? 0 : e > size ? size : e;
            dims[a] = e > s ? (e - s + st - 1) / st : 0;
        } else {
            s = s < 0 ? -1 : s > size - 1 ? size - 1 : s;
            e = e < -1 ? -1 : e > size - 1 ? size - 1 : e;
            dims[a] = s > e ? (s - e - st - 1) / -st : 0;
        }
        first[a] = s;
        step[a] = st;
    }
    if (!ir_output(r, out[0], x->dtype, R, dims)) return false;
    int64_t xstride[IR_MAX_RANK], s = 1;
    for (uint32_t d = R; d-- > 0;) { xstride[d] = s; s *= x->dims[d]; }
    const size_t size = ir_dtype_size(x->dtype);
    int64_t base = 0, stride[IR_MAX_RANK];
    for (uint32_t d = 0; d < R; d++) { base += first[d] * xstride[d]; stride[d] = step[d] * xstride[d]; }
    const uint8_t *src = (const uint8_t *)x->data;
    uint8_t *dst = (uint8_t *)out[0]->data;
    /* rows of the last axis at once where its step is 1 */
    const bool rows = R > 0 && step[R - 1] == 1;
    const uint64_t row = rows ? (uint64_t)dims[R - 1] : 1u, nrows = row ? out[0]->count / row : 0;
    const uint32_t outer_rank = rows ? R - 1 : R;
    for (uint64_t q = 0; q < nrows; q++) {
        uint64_t i = q;
        int64_t off = base;
        for (uint32_t d = outer_rank; d-- > 0;) {
            off += (int64_t)(i % (uint64_t)dims[d]) * stride[d];
            i /= (uint64_t)dims[d];
        }
        if (rows) memcpy(dst + q * row * size, src + (size_t)off * size, (size_t)row * size);
        else memcpy(dst + q * size, src + (size_t)off * size, size);
    }
    return true;
}

static bool k_gather(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "two inputs and an output are needed");
    const IrTensor *x = in[0], *idx = in[1];
    uint32_t axis;
    if (!axis_of(r, ir_attr_int(r->node, "axis", 0), x->rank, &axis)) return false;
    const uint32_t R = x->rank - 1 + idx->rank;
    if (R > IR_MAX_RANK) return ir_fail(r, "more than eight axes");
    int64_t dims[IR_MAX_RANK];
    uint32_t k = 0;
    for (uint32_t d = 0; d < axis; d++) dims[k++] = x->dims[d];
    for (uint32_t d = 0; d < idx->rank; d++) dims[k++] = idx->dims[d];
    for (uint32_t d = axis + 1; d < x->rank; d++) dims[k++] = x->dims[d];
    if (!ir_output(r, out[0], x->dtype, R, dims)) return false;
    uint64_t outer = 1, inner = 1;
    for (uint32_t d = 0; d < axis; d++) outer *= (uint64_t)x->dims[d];
    for (uint32_t d = axis + 1; d < x->rank; d++) inner *= (uint64_t)x->dims[d];
    const int64_t size_axis = x->dims[axis];
    const size_t size = ir_dtype_size(x->dtype), chunk = (size_t)inner * size;
    for (uint64_t j = 0; j < idx->count; j++) {
        int64_t i = ir_get_int(idx, j);
        if (i < 0) i += size_axis;
        if (i < 0 || i >= size_axis) return ir_fail(r, "an index %lld of an axis of %lld", (long long)ir_get_int(idx, j),
                                                   (long long)size_axis);
    }
    for (uint64_t o = 0; o < outer; o++)
        for (uint64_t j = 0; j < idx->count; j++) {
            int64_t i = ir_get_int(idx, j);
            if (i < 0) i += size_axis;
            memcpy((uint8_t *)out[0]->data + (o * idx->count + j) * chunk,
                   (const uint8_t *)x->data + (o * (uint64_t)size_axis + (uint64_t)i) * chunk, chunk);
        }
    return true;
}

static bool k_gather_elements(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "two inputs and an output are needed");
    const IrTensor *x = in[0], *idx = in[1];
    uint32_t axis;
    if (!axis_of(r, ir_attr_int(r->node, "axis", 0), x->rank, &axis) || idx->rank != x->rank)
        return r->error[0] ? false : ir_fail(r, "indices of another rank");
    if (!ir_output(r, out[0], x->dtype, idx->rank, idx->dims)) return false;
    int64_t xs[IR_MAX_RANK], s = 1;
    for (uint32_t d = x->rank; d-- > 0;) { xs[d] = s; s *= x->dims[d]; }
    const size_t size = ir_dtype_size(x->dtype);
    for (uint64_t i = 0; i < idx->count; i++) {
        uint64_t q = i;
        int64_t off = 0;
        for (uint32_t d = idx->rank; d-- > 0;) {
            int64_t c = (int64_t)(q % (uint64_t)idx->dims[d]);
            q /= (uint64_t)idx->dims[d];
            if (d == axis) {
                c = ir_get_int(idx, i);
                if (c < 0) c += x->dims[axis];
                if (c < 0 || c >= x->dims[axis]) return ir_fail(r, "an index out of range");
            }
            off += c * xs[d];
        }
        memcpy((uint8_t *)out[0]->data + i * size, (const uint8_t *)x->data + (size_t)off * size, size);
    }
    return true;
}

/* The value a scatter's reduction leaves at a place: the update, or it combined with what is there. */
typedef enum { REDUCE_NONE, REDUCE_ADD, REDUCE_MUL, REDUCE_MAX, REDUCE_MIN } Reduction;

static bool reduction_of(IrRun *r, Reduction *red) {
    const IrAttr *a = ir_attr(r->node, "reduction");
    const char *s = a && a->type == ATTR_STRING ? a->s : "none";
    *red = !strcmp(s, "none") ? REDUCE_NONE : !strcmp(s, "add") ? REDUCE_ADD : !strcmp(s, "mul") ? REDUCE_MUL
         : !strcmp(s, "max") ? REDUCE_MAX : !strcmp(s, "min") ? REDUCE_MIN : (Reduction)-1;
    return (int)*red >= 0 ? true : ir_fail(r, "the reduction '%s'", s);
}

/* y[at] = upd[from], or combined by the reduction. */
static void scatter_one(IrTensor *y, uint64_t at, const IrTensor *upd, uint64_t from, Reduction red, size_t size) {
    if (red == REDUCE_NONE) {
        memcpy((uint8_t *)y->data + at * size, (const uint8_t *)upd->data + from * size, size);
    } else if (y->dtype == F32) {
        float *o = fdata(y) + at;
        const float u = fdata(upd)[from];
        *o = red == REDUCE_ADD ? *o + u : red == REDUCE_MUL ? *o * u : red == REDUCE_MAX ? (u > *o ? u : *o)
                                                                                         : (u < *o ? u : *o);
    } else {
        const int64_t o = ir_get_int(y, at), u = ir_get_int(upd, from);
        put_int(y, at, red == REDUCE_ADD ? o + u : red == REDUCE_MUL ? o * u : red == REDUCE_MAX ? (u > o ? u : o)
                                                                                                 : (u < o ? u : o));
    }
}

static bool k_gather_nd(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "two inputs and an output are needed");
    const IrTensor *x = in[0], *idx = in[1];
    const int64_t b = ir_attr_int(r->node, "batch_dims", 0);
    if (!idx->rank) return ir_fail(r, "indices of no axes");
    const uint32_t q = (uint32_t)idx->dims[idx->rank - 1];
    if (b < 0 || b >= (int64_t)idx->rank || (uint64_t)b + q > x->rank) return ir_fail(r, "batch_dims or indices out of range");
    for (int64_t d = 0; d < b; d++) if (x->dims[d] != idx->dims[d]) return ir_fail(r, "batches of other sizes");
    const uint32_t R = idx->rank - 1 + x->rank - q - (uint32_t)b;
    if (R > IR_MAX_RANK) return ir_fail(r, "more than eight axes");
    int64_t dims[IR_MAX_RANK];
    uint32_t k = 0;
    for (uint32_t d = 0; d + 1 < idx->rank; d++) dims[k++] = idx->dims[d];
    for (uint32_t d = (uint32_t)b + q; d < x->rank; d++) dims[k++] = x->dims[d];
    if (!ir_output(r, out[0], x->dtype, R, dims)) return false;
    int64_t xs[IR_MAX_RANK], s = 1;
    for (uint32_t d = x->rank; d-- > 0;) { xs[d] = s; s *= x->dims[d]; }
    uint64_t inner = 1, batches = 1;
    for (uint32_t d = (uint32_t)b + q; d < x->rank; d++) inner *= (uint64_t)x->dims[d];
    for (int64_t d = 0; d < b; d++) batches *= (uint64_t)x->dims[d];
    const size_t size = ir_dtype_size(x->dtype);
    const uint64_t points = q ? idx->count / q : idx->count, per = batches ? points / batches : 0;
    const uint64_t batch_size = batches ? x->count / batches : 0;
    for (uint64_t p = 0; p < points; p++) {
        int64_t off = (int64_t)((per ? p / per : 0) * batch_size);
        for (uint32_t d = 0; d < q; d++) {
            const int64_t size_d = x->dims[b + d];
            int64_t c = ir_get_int(idx, p * q + d);
            if (c < 0) c += size_d;
            if (c < 0 || c >= size_d) return ir_fail(r, "an index out of range");
            off += c * xs[b + d];
        }
        memcpy((uint8_t *)out[0]->data + p * inner * size, (const uint8_t *)x->data + (size_t)off * size, (size_t)(inner * size));
    }
    return true;
}

static bool k_scatter_nd(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 3 || m < 1 || !in[0] || !in[1] || !in[2]) return ir_fail(r, "three inputs and an output are needed");
    const IrTensor *x = in[0], *idx = in[1], *upd = in[2];
    Reduction red;
    if (!reduction_of(r, &red)) return false;
    if (!idx->rank) return ir_fail(r, "indices of no axes");
    const uint32_t q = (uint32_t)idx->dims[idx->rank - 1];
    if (q > x->rank || upd->dtype != x->dtype) return ir_fail(r, "indices longer than the data's rank, or updates of another type");
    if (!ir_output(r, out[0], x->dtype, x->rank, x->dims)) return false;
    const size_t size = ir_dtype_size(x->dtype);
    memcpy(out[0]->data, x->data, (size_t)x->count * size);
    int64_t xs[IR_MAX_RANK], s = 1;
    for (uint32_t d = x->rank; d-- > 0;) { xs[d] = s; s *= x->dims[d]; }
    uint64_t inner = 1;
    for (uint32_t d = q; d < x->rank; d++) inner *= (uint64_t)x->dims[d];
    const uint64_t points = q ? idx->count / q : 0;
    if (upd->count != points * inner) return ir_fail(r, "updates of another size");
    for (uint64_t p = 0; p < points; p++) {
        int64_t off = 0;
        for (uint32_t d = 0; d < q; d++) {
            int64_t c = ir_get_int(idx, p * q + d);
            if (c < 0) c += x->dims[d];
            if (c < 0 || c >= x->dims[d]) return ir_fail(r, "an index out of range");
            off += c * xs[d];
        }
        for (uint64_t i = 0; i < inner; i++) scatter_one(out[0], (uint64_t)off + i, upd, p * inner + i, red, size);
    }
    return true;
}

/* ScatterElements (and Scatter, its name before opset 11): each update to the place its index names on the axis. */
static bool k_scatter_elements(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 3 || m < 1 || !in[0] || !in[1] || !in[2]) return ir_fail(r, "three inputs and an output are needed");
    const IrTensor *x = in[0], *idx = in[1], *upd = in[2];
    Reduction red;
    if (!reduction_of(r, &red)) return false;
    uint32_t axis;
    if (!axis_of(r, ir_attr_int(r->node, "axis", 0), x->rank, &axis)) return false;
    if (idx->rank != x->rank || upd->rank != x->rank || upd->count != idx->count || upd->dtype != x->dtype)
        return ir_fail(r, "indices or updates of another shape or type");
    if (!ir_output(r, out[0], x->dtype, x->rank, x->dims)) return false;
    const size_t size = ir_dtype_size(x->dtype);
    memcpy(out[0]->data, x->data, (size_t)x->count * size);
    int64_t xs[IR_MAX_RANK], s = 1;
    for (uint32_t d = x->rank; d-- > 0;) { xs[d] = s; s *= x->dims[d]; }
    for (uint64_t i = 0; i < idx->count; i++) {
        uint64_t t = i;
        int64_t off = 0;
        for (uint32_t d = idx->rank; d-- > 0;) {
            int64_t c = (int64_t)(t % (uint64_t)idx->dims[d]);
            t /= (uint64_t)idx->dims[d];
            if (d == axis) {
                c = ir_get_int(idx, i);
                if (c < 0) c += x->dims[d];
                if (c < 0 || c >= x->dims[d]) return ir_fail(r, "an index out of range");
            }
            off += c * xs[d];
        }
        scatter_one(out[0], (uint64_t)off, upd, i, red, size);
    }
    return true;
}

/* TensorScatter (opset 24): a cache with an update written along an axis from each batch's index (circularly or
   not). */
static bool k_tensor_scatter(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "a cache, an update and an output are needed");
    const IrTensor *x = in[0], *upd = in[1], *wi = n > 2 ? in[2] : NULL;
    uint32_t axis;
    if (!axis_of(r, ir_attr_int(r->node, "axis", -2), x->rank, &axis)) return false;
    if (!axis) return ir_fail(r, "the batch's axis as the sequence's");
    const IrAttr *ma = ir_attr(r->node, "mode");
    const bool circular = ma && ma->type == ATTR_STRING && !strcmp(ma->s, "circular");
    if (upd->rank != x->rank || upd->dtype != x->dtype) return ir_fail(r, "an update of another rank or type");
    for (uint32_t d = 0; d < x->rank; d++)
        if (d != axis && upd->dims[d] != x->dims[d]) return ir_fail(r, "an update of another shape");
    if (upd->dims[axis] > x->dims[axis]) return ir_fail(r, "an update longer than the cache");
    if (wi && wi->count != (uint64_t)x->dims[0]) return ir_fail(r, "write_indices of another size than the batch");
    if (!ir_output(r, out[0], x->dtype, x->rank, x->dims)) return false;
    const size_t size = ir_dtype_size(x->dtype);
    memcpy(out[0]->data, x->data, (size_t)x->count * size);
    uint64_t outer = 1, inner = 1;
    for (uint32_t d = 0; d < axis; d++) outer *= (uint64_t)x->dims[d];
    for (uint32_t d = axis + 1; d < x->rank; d++) inner *= (uint64_t)x->dims[d];
    const int64_t L = x->dims[axis], S = upd->dims[axis];
    const uint64_t per_batch = x->dims[0] ? outer / (uint64_t)x->dims[0] : 0;
    for (uint64_t o = 0; o < outer; o++) {
        const int64_t start = wi ? ir_get_int(wi, per_batch ? o / per_batch : 0) : 0;
        for (int64_t t = 0; t < S; t++) {
            int64_t at = start + t;
            if (circular) at = ((at % L) + L) % L;
            if (at < 0 || at >= L) return ir_fail(r, "a write beyond the cache");
            memcpy((uint8_t *)out[0]->data + ((o * (uint64_t)L + (uint64_t)at) * inner) * size,
                   (const uint8_t *)upd->data + ((o * (uint64_t)S + (uint64_t)t) * inner) * size, (size_t)inner * size);
        }
    }
    return true;
}

static bool k_expand(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "two inputs and an output are needed");
    if (in[1]->count > IR_MAX_RANK) return ir_fail(r, "a shape of more than eight axes");
    IrTensor shape;
    memset(&shape, 0, sizeof shape);
    shape.rank = (uint8_t)in[1]->count;
    for (uint32_t k = 0; k < shape.rank; k++) shape.dims[k] = ir_get_int(in[1], k);
    IrTensor *pair[2] = {in[0], &shape};
    uint32_t R;
    int64_t dims[IR_MAX_RANK];
    if (!broadcast_shape(r, pair, 2, &R, dims) || !ir_output(r, out[0], in[0]->dtype, R, dims)) return false;
    Walk w;
    walk_init(&w, pair, 1, R, dims);
    const size_t size = ir_dtype_size(in[0]->dtype);
    FOR_PIECES(r, out[0]->count, begin, end,
        int64_t idx[IR_MAX_RANK], off[3];
        if (begin < end) walk_offsets(&w, begin, idx, off);
        for (uint64_t i = begin; i < end; i++) {
            memcpy((uint8_t *)out[0]->data + i * size, (const uint8_t *)in[0]->data + (size_t)off[0] * size, size);
            walk_step(&w, idx, off);
        }
    );
    return true;
}

static bool k_tile(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "two inputs and an output are needed");
    const IrTensor *x = in[0];
    if (in[1]->count != x->rank) return ir_fail(r, "repeats of another length than the rank");
    int64_t dims[IR_MAX_RANK];
    for (uint32_t d = 0; d < x->rank; d++) dims[d] = x->dims[d] * ir_get_int(in[1], d);
    if (!ir_output(r, out[0], x->dtype, x->rank, dims)) return false;
    const size_t size = ir_dtype_size(x->dtype);
    int64_t xs[IR_MAX_RANK], s = 1;
    for (uint32_t d = x->rank; d-- > 0;) { xs[d] = s; s *= x->dims[d]; }
    for (uint64_t i = 0; i < out[0]->count; i++) {
        uint64_t q = i;
        int64_t off = 0;
        for (uint32_t d = x->rank; d-- > 0;) {
            off += (int64_t)((q % (uint64_t)dims[d]) % (uint64_t)x->dims[d]) * xs[d];
            q /= (uint64_t)dims[d];
        }
        memcpy((uint8_t *)out[0]->data + i * size, (const uint8_t *)x->data + (size_t)off * size, size);
    }
    return true;
}

static bool k_pad(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    const IrTensor *x = in[0];
    const uint32_t R = x->rank;
    int64_t pads[2 * IR_MAX_RANK], axes[IR_MAX_RANK];
    const int64_t np = int_list(r->node, "pads", n > 1 ? in[1] : NULL, pads, 2 * IR_MAX_RANK);
    int64_t na = int_list(r->node, "axes", n > 3 ? in[3] : NULL, axes, IR_MAX_RANK);
    if (na < 0) { for (uint32_t k = 0; k < R; k++) axes[k] = k; na = R; }
    if (np != 2 * na) return ir_fail(r, "%lld pads for %lld axes", (long long)np, (long long)na);
    int64_t before[IR_MAX_RANK] = {0}, after[IR_MAX_RANK] = {0}, dims[IR_MAX_RANK];
    for (int64_t k = 0; k < na; k++) {
        uint32_t a;
        if (!axis_of(r, axes[k], R, &a)) return false;
        before[a] = pads[k];
        after[a] = pads[na + k];
    }
    for (uint32_t d = 0; d < R; d++) {
        dims[d] = x->dims[d] + before[d] + after[d];
        if (dims[d] < 0) return ir_fail(r, "pads that remove more than the axis holds");
    }
    const IrAttr *mode_attr = ir_attr(r->node, "mode");
    const char *mode = mode_attr && mode_attr->type == ATTR_STRING ? mode_attr->s : "constant";
    const int kind = !strcmp(mode, "constant") ? 0 : !strcmp(mode, "reflect") ? 1 : !strcmp(mode, "edge") ? 2
                   : !strcmp(mode, "wrap") ? 3 : -1;
    if (kind < 0) return ir_fail(r, "the mode '%s'", mode);
    if (!ir_output(r, out[0], x->dtype, R, dims)) return false;
    const size_t size = ir_dtype_size(x->dtype);
    uint8_t fill[8] = {0};
    if (kind == 0 && n > 2 && in[2] && in[2]->count) memcpy(fill, in[2]->data, size);
    int64_t xs[IR_MAX_RANK], s = 1;
    for (uint32_t d = R; d-- > 0;) { xs[d] = s; s *= x->dims[d]; }
    FOR_PIECES(r, out[0]->count, begin, end,
        for (uint64_t i = begin; i < end; i++) {
            uint64_t q = i;
            int64_t off = 0;
            bool inside = true;
            for (uint32_t d = R; d-- > 0;) {
                int64_t c = (int64_t)(q % (uint64_t)dims[d]) - before[d];
                q /= (uint64_t)dims[d];
                const int64_t size_d = x->dims[d];
                if (c < 0 || c >= size_d) {
                    if (kind == 0 || size_d == 0) { inside = false; continue; }
                    if (kind == 2) c = c < 0 ? 0 : size_d - 1;
                    else if (kind == 3) c = ((c % size_d) + size_d) % size_d;
                    else {
                        const int64_t period = size_d > 1 ? 2 * (size_d - 1) : 1;
                        c = ((c % period) + period) % period;
                        if (c >= size_d) c = period - c;
                    }
                }
                off += c * xs[d];
            }
            memcpy((uint8_t *)out[0]->data + i * size, inside ? (const uint8_t *)x->data + (size_t)off * size : fill, size);
        }
    );
    return true;
}

static bool k_trilu(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    const IrTensor *x = in[0];
    if (x->rank < 2) return ir_fail(r, "an input of fewer than two axes");
    const int64_t k = n > 1 && in[1] && in[1]->count ? ir_get_int(in[1], 0) : 0;
    const bool upper = ir_attr_int(r->node, "upper", 1) != 0;
    if (!ir_output(r, out[0], x->dtype, x->rank, x->dims)) return false;
    const int64_t rows = x->dims[x->rank - 2], cols = x->dims[x->rank - 1];
    const size_t size = ir_dtype_size(x->dtype);
    for (uint64_t i = 0; i < x->count; i++) {
        const int64_t c = (int64_t)(i % (uint64_t)cols), row = (int64_t)((i / (uint64_t)cols) % (uint64_t)rows);
        const bool keep = upper ? c - row >= k : c - row <= k;
        if (keep) memcpy((uint8_t *)out[0]->data + i * size, (const uint8_t *)x->data + i * size, size);
    }
    return true;
}

static bool k_depth_space(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    const IrTensor *x = in[0];
    if (x->rank != 4) return ir_fail(r, "an input of %u axes, not four", x->rank);
    const int64_t b = ir_attr_int(r->node, "blocksize", 0);
    if (b < 1) return ir_fail(r, "a block size of %lld", (long long)b);
    const IrAttr *mode_attr = ir_attr(r->node, "mode");
    const bool crd = mode_attr && mode_attr->type == ATTR_STRING && !strcmp(mode_attr->s, "CRD");
    const int64_t N = x->dims[0], C = x->dims[1], H = x->dims[2], W = x->dims[3];
    const bool to_space = OP(r) == IR_DepthToSpace;
    int64_t dims[4];
    if (to_space) {
        if (C % (b * b)) return ir_fail(r, "channels not divisible by the block size's square");
        dims[0] = N; dims[1] = C / (b * b); dims[2] = H * b; dims[3] = W * b;
    } else {
        if (H % b || W % b) return ir_fail(r, "sizes not divisible by the block size");
        dims[0] = N; dims[1] = C * b * b; dims[2] = H / b; dims[3] = W / b;
    }
    if (!ir_output(r, out[0], x->dtype, 4, dims)) return false;
    const size_t size = ir_dtype_size(x->dtype);
    const int64_t OC = dims[1], OH = dims[2], OW = dims[3];
    for (int64_t nn = 0; nn < N; nn++)
        for (int64_t c = 0; c < OC; c++)
            for (int64_t h = 0; h < OH; h++)
                for (int64_t w = 0; w < OW; w++) {
                    int64_t sc, sh, sw;
                    if (to_space) {
                        const int64_t bh = h % b, bw = w % b;
                        sc = crd ? c * b * b + bh * b + bw : (bh * b + bw) * OC + c;
                        sh = h / b;
                        sw = w / b;
                    } else {
                        const int64_t cc = crd ? c / (b * b) : c % C, blk = crd ? c % (b * b) : c / C;
                        sc = cc;
                        sh = h * b + blk / b;
                        sw = w * b + blk % b;
                    }
                    memcpy((uint8_t *)out[0]->data + (size_t)(((nn * OC + c) * OH + h) * OW + w) * size,
                           (const uint8_t *)x->data + (size_t)(((nn * C + sc) * H + sh) * W + sw) * size, size);
                }
    return true;
}

static bool k_nonzero(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    const IrTensor *x = in[0];
    uint64_t count = 0;
    for (uint64_t i = 0; i < x->count; i++) count += ir_get_float(x, i) != 0.0;
    const int64_t dims[2] = {x->rank, (int64_t)count};
    if (!ir_output(r, out[0], I64, 2, dims)) return false;
    uint64_t j = 0;
    for (uint64_t i = 0; i < x->count; i++) {
        if (ir_get_float(x, i) == 0.0) continue;
        uint64_t q = i;
        for (uint32_t d = x->rank; d-- > 0;) {
            ((int64_t *)out[0]->data)[d * count + j] = (int64_t)(q % (uint64_t)x->dims[d]);
            q /= (uint64_t)x->dims[d];
        }
        j++;
    }
    return true;
}

static bool k_onehot(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 3 || m < 1 || !in[0] || !in[1] || !in[2]) return ir_fail(r, "three inputs and an output are needed");
    const IrTensor *idx = in[0];
    const int64_t depth = (int64_t)ir_get_float(in[1], 0);
    if (depth < 1 || in[2]->count != 2) return ir_fail(r, "a depth below 1 or values not two");
    const uint32_t R = idx->rank + 1;
    if (R > IR_MAX_RANK) return ir_fail(r, "more than eight axes");
    uint32_t axis;
    if (!axis_of(r, ir_attr_int(r->node, "axis", -1), R, &axis)) return false;
    int64_t dims[IR_MAX_RANK];
    for (uint32_t d = 0, j = 0; d < R; d++) dims[d] = d == axis ? depth : idx->dims[j++];
    if (!ir_output(r, out[0], in[2]->dtype, R, dims)) return false;
    const size_t size = ir_dtype_size(in[2]->dtype);
    uint64_t inner = 1;
    for (uint32_t d = axis + 1; d < R; d++) inner *= (uint64_t)dims[d];
    for (uint64_t i = 0; i < out[0]->count; i++) {
        const uint64_t c = (i / inner) % (uint64_t)depth, rest = (i / inner / (uint64_t)depth) * inner + i % inner;
        int64_t v = ir_get_int(idx, rest);
        if (v < 0) v += depth;
        memcpy((uint8_t *)out[0]->data + i * size, (const uint8_t *)in[2]->data + ((int64_t)c == v ? size : 0), size);
    }
    return true;
}

/* ------------------------------------------------------------------------- products */

/* Filters packed once for spingalett_conv_rows(): the N columns of a constant B (K x N, or N x K transposed) as
   N filters of K weights. */
typedef struct {
    uint32_t K, N;
    float *panels;
    float *zeros;                   /* N: the bias of an epilogue that only activates */
} Packed;

static void packed_free(void *p) {
    if (!p) return;
    spingalett_aligned_free(((Packed *)p)->panels);
    free(((Packed *)p)->zeros);
    free(p);
}

static const Packed *packed_of(IrRun *r, const IrTensor *B, bool trans) {
    Packed *p = (Packed *)ir_prepared(r);
    if (p) return p;
    const uint32_t K = (uint32_t)(trans ? B->dims[1] : B->dims[0]), N = (uint32_t)(trans ? B->dims[0] : B->dims[1]);
    p = (Packed *)calloc(1, sizeof *p);
    float *W = trans ? NULL : (float *)malloc((size_t)K * N * sizeof(float) + 1);
    if (p) {
        p->panels = (float *)spingalett_aligned_alloc(spingalett_conv_direct_packed_floats(N, K) * sizeof(float) + 64);
        p->zeros = (float *)calloc((size_t)N + 1, sizeof(float));
    }
    if (!p || !p->panels || !p->zeros || (!trans && !W)) {
        free(W);
        packed_free(p);
        ir_fail(r, "out of memory for packed weights");
        return NULL;
    }
    if (!trans)         /* (in tiles of 32 x 32: the transpose's reads and writes both in cache) */
        for (uint32_t k0 = 0; k0 < K; k0 += 32)
            for (uint32_t n0 = 0; n0 < N; n0 += 32)
                for (uint32_t n = n0; n < n0 + 32 && n < N; n++)
                    for (uint32_t k = k0; k < k0 + 32 && k < K; k++) W[(size_t)n * K + k] = fdata(B)[(size_t)k * N + n];
    spingalett_conv_pack(trans ? fdata(B) : W, K, N, p->panels);
    free(W);
    p->K = K;
    p->N = N;
    ir_prepare(r, p, packed_free);
    return p;
}

/* The epilogue of Linear: the bias, then the activation (GELU, its tanh form and SiLU in vectors). */
static void linear_epilogue(const void *ctx, uint32_t row, uint32_t rows, uint32_t col, uint32_t cols, float *c,
                            size_t ldc) {
    const SpingalettBiasActivation *e = (const SpingalettBiasActivation *)ctx;
    const float *bias = e->bias + col;
    const bool vector = e->act == ACT_GELU || e->act == ACT_GELU_TANH || e->act == ACT_SILU;
    for (uint32_t r = 0; r < rows; r++) {
        float *cr = c + (size_t)r * ldc;
        for (uint32_t j = 0; j < cols; j++) cr[j] += bias[j];
        if (vector) spingalett_activate_training(cr, cols, e->act);
        else if (e->act != ACT_NONE) apply_activation_bulk(cr, cols, e->act);
    }
    (void)row;
}

/* Y (rows x N) = A (rows x K) times a constant B, its filters packed once; a bias added per column when given, then
   an activation. */
static void product_rows(IrRun *r, const Packed *p, const float *a, uint64_t rows, float *y, const float *bias,
                         ActivationFunction act) {
    if (!rows || !p->N) return;
    if (!bias && act != ACT_NONE) bias = p->zeros;
    SpingalettBiasActivation epilogue = {bias, act, NULL};
    SpingalettGemmHooks hooks = {NULL, NULL, linear_epilogue, &epilogue};
    const SpingalettGemmScratch *scratch = ir_scratch(r);
    const int threads = scratch ? spingalett_gemm_scratch_threads(scratch) : 1;
    /* (in pieces of fewer than 2^31 rows) */
    for (uint64_t r0 = 0; r0 < rows; r0 += (uint64_t)1 << 30) {
        const uint32_t n = (uint32_t)(rows - r0 < ((uint64_t)1 << 30) ? rows - r0 : (uint64_t)1 << 30);
        spingalett_conv_rows(a + r0 * p->K, p->K, n, p->panels, p->K, p->N, y + r0 * p->N, p->N, bias ? &hooks : NULL,
                             r->parallel && (uint64_t)n * p->N * p->K >= SPINGALETT_GEMM_PARALLEL_WORK, threads);
    }
}

/* numpy's matmul: Y = A B over the operands' broadcast batches. */
static bool matmul_into(IrRun *r, const IrTensor *A, const IrTensor *B, IrTensor *Y) {
    if (A->dtype != F32 || B->dtype != F32) return ir_fail(r, "products of %s and %s (float32 only)", ir_dtype_name(A->dtype),
                                                          ir_dtype_name(B->dtype));
    if (!A->rank || !B->rank) return ir_fail(r, "a scalar operand");
    const int64_t M = A->rank >= 2 ? A->dims[A->rank - 2] : 1, K = A->dims[A->rank - 1];
    const int64_t Kb = B->rank >= 2 ? B->dims[B->rank - 2] : B->dims[0], N = B->rank >= 2 ? B->dims[B->rank - 1] : 1;
    if (K != Kb) return ir_fail(r, "operands of %lld and %lld columns and rows", (long long)K, (long long)Kb);
    /* the batches' shape: the broadcast of the operands' leading axes */
    IrTensor ba, bb;
    memset(&ba, 0, sizeof ba);
    memset(&bb, 0, sizeof bb);
    ba.rank = A->rank >= 2 ? A->rank - 2 : 0;
    bb.rank = B->rank >= 2 ? B->rank - 2 : 0;
    for (uint32_t d = 0; d < ba.rank; d++) ba.dims[d] = A->dims[d];
    for (uint32_t d = 0; d < bb.rank; d++) bb.dims[d] = B->dims[d];
    IrTensor *pair[2] = {&ba, &bb};
    uint32_t R;
    int64_t dims[IR_MAX_RANK];
    if (!broadcast_shape(r, pair, 2, &R, dims)) return false;
    uint64_t batches = 1;
    for (uint32_t d = 0; d < R; d++) batches *= (uint64_t)dims[d];
    int64_t odims[IR_MAX_RANK];
    uint32_t orank = R;
    for (uint32_t d = 0; d < R; d++) odims[d] = dims[d];
    if (A->rank >= 2) odims[orank++] = M;
    if (B->rank >= 2) odims[orank++] = N;
    if (orank > IR_MAX_RANK) return ir_fail(r, "more than eight axes");
    if (B->constant && B->rank == 2 && K && N) {
        /* a constant B: its filters packed once, every row of A's batches a row of one product */
        const Packed *p = packed_of(r, B, false);
        if (!p || !ir_output_raw(r, Y, F32, orank, odims)) return false;
        product_rows(r, p, fdata(A), A->count / (uint64_t)K, fdata(Y), NULL, ACT_NONE);
        return true;
    }
    if (!ir_output(r, Y, F32, orank, odims)) return false;
    if (!M || !N) return true;
    int64_t sa[IR_MAX_RANK], sb[IR_MAX_RANK];
    broadcast_strides(&ba, R, dims, sa);
    broadcast_strides(&bb, R, dims, sb);
    const float *a = fdata(A), *b = fdata(B);
    float *c = fdata(Y);
    const uint64_t work = (uint64_t)M * (uint64_t)N * (uint64_t)K;
    /* many small products side by side on one thread each; few large ones each on all the threads */
    const bool across = r->parallel && batches > 1 && work < ((uint64_t)1 << 22);
    SpingalettGemmScratch *main = ir_scratch(r);
    if (!main) return false;
    SPINGALETT_PARALLEL_FOR(across,
        for (int64_t q = 0; q < (int64_t)batches; q++) {
            int64_t oa = 0, ob = 0;
            uint64_t i = (uint64_t)q;
            for (uint32_t d = R; d-- > 0;) {
                const int64_t idx = (int64_t)(i % (uint64_t)dims[d]);
                i /= (uint64_t)dims[d];
                oa += idx * sa[d];
                ob += idx * sb[d];
            }
            /* (a thread beyond the scratches' count, after the threads were raised, takes a temporary one) */
            SpingalettGemmScratch *s = across ? ir_thread_scratch(r, spingalett_thread_num()) : main;
            spingalett_gemm_native(s, false, false, (uint32_t)M, (uint32_t)N, (uint32_t)K, 1.0f,
                                   a + (size_t)oa * (size_t)(M * K), (size_t)K, b + (size_t)ob * (size_t)(K * N), (size_t)N,
                                   0.0f, c + (size_t)q * (size_t)(M * N), (size_t)N,
                                   !across && r->parallel);
        }
    );
    return true;
}

static bool k_matmul(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "two inputs and an output are needed");
    return matmul_into(r, in[0], in[1], out[0]);
}

static bool k_gemm(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "two inputs and an output are needed");
    const IrTensor *A = in[0], *B = in[1], *C = n > 2 ? in[2] : NULL;
    if (A->dtype != F32 || B->dtype != F32 || A->rank != 2 || B->rank != 2)
        return ir_fail(r, "operands other than matrices of float32");
    const bool ta = ir_attr_int(r->node, "transA", 0) != 0, tb = ir_attr_int(r->node, "transB", 0) != 0;
    const float alpha = ir_attr_float(r->node, "alpha", 1.0f), beta = ir_attr_float(r->node, "beta", 1.0f);
    const int64_t M = ta ? A->dims[1] : A->dims[0], K = ta ? A->dims[0] : A->dims[1];
    const int64_t Kb = tb ? B->dims[1] : B->dims[0], N = tb ? B->dims[0] : B->dims[1];
    if (K != Kb) return ir_fail(r, "operands of %lld and %lld columns and rows", (long long)K, (long long)Kb);
    const int64_t dims[2] = {M, N};
    if (B->constant && !ta && K && N) {
        /* a constant B: its filters packed once; a C of one value a column added in the products' epilogue */
        const Packed *p = packed_of(r, B, tb);
        if (!p || !ir_output_raw(r, out[0], F32, 2, dims)) return false;
        const bool fused = C && C->dtype == F32 && alpha == 1.0f && beta == 1.0f && C->count == (uint64_t)N &&
                           (C->rank == 1 || (C->rank == 2 && C->dims[0] == 1));
        product_rows(r, p, fdata(A), (uint64_t)M, fdata(out[0]), fused ? fdata(C) : NULL, ACT_NONE);
        if (alpha != 1.0f)
            for (uint64_t i = 0; i < out[0]->count; i++) fdata(out[0])[i] *= alpha;
        if (fused) return true;
    } else {
        if (!ir_output(r, out[0], F32, 2, dims)) return false;
        SpingalettGemmScratch *s = ir_scratch(r);
        if (!s) return false;
        if (M && N)
            spingalett_gemm_native(s, ta, tb, (uint32_t)M, (uint32_t)N, (uint32_t)K, alpha, fdata(A), (size_t)A->dims[1],
                                   fdata(B), (size_t)B->dims[1], 0.0f, fdata(out[0]), (size_t)N, r->parallel);
    }
    if (C && beta != 0.0f) {
        if (C->dtype != F32) return ir_fail(r, "C of %s", ir_dtype_name(C->dtype));
        IrTensor *pair[2] = {out[0], (IrTensor *)C};
        uint32_t R;
        int64_t bd[IR_MAX_RANK];
        if (!broadcast_shape(r, pair, 2, &R, bd) || R != 2 || bd[0] != M || bd[1] != N) return ir_fail(r, "a C that does not broadcast to the product");
        Walk w;
        IrTensor *one[1] = {(IrTensor *)C};
        walk_init(&w, one, 1, 2, dims);
        int64_t idx[IR_MAX_RANK], off[3];
        float *y = fdata(out[0]);
        if (out[0]->count) walk_offsets(&w, 0, idx, off);
        for (uint64_t i = 0; i < out[0]->count; i++) {
            y[i] += beta * fdata(C)[off[0]];
            walk_step(&w, idx, off);
        }
    }
    return true;
}

/* ------------------------------------------------------------------------- convolutions and pooling */

/* The geometry of a convolution or pooling over one to three spatial axes, held as three (leading axes of size 1). */
typedef struct {
    uint32_t spatial;               /* the input's spatial axes */
    int64_t in[3], k[3], s[3], pb[3], pe[3], dil[3], out[3];
} Geometry;

/* kernel_shape (or the weights'), strides, dilations, pads or auto_pad; outputs by ceil_mode for pooling. */
static bool geometry(IrRun *r, const IrTensor *x, const int64_t *kernel, Geometry *g, bool ceil_mode) {
    const uint32_t S = x->rank - 2;
    if (x->rank < 3 || S > 3) return ir_fail(r, "an input of %u axes (one to three spatial ones expected)", x->rank);
    memset(g, 0, sizeof *g);
    g->spatial = S;
    int64_t strides[3], dils[3], pads[6];
    const int64_t ns = int_list(r->node, "strides", NULL, strides, 3), nd = int_list(r->node, "dilations", NULL, dils, 3);
    const int64_t np = int_list(r->node, "pads", NULL, pads, 6);
    const IrAttr *ap = ir_attr(r->node, "auto_pad");
    const char *auto_pad = ap && ap->type == ATTR_STRING ? ap->s : "NOTSET";
    for (uint32_t i = 0; i < 3; i++) {
        const int32_t a = (int32_t)i - (int32_t)(3 - S);
        g->in[i] = a >= 0 ? x->dims[2 + a] : 1;
        g->k[i] = a >= 0 ? kernel[a] : 1;
        g->s[i] = a >= 0 && ns == (int64_t)S ? strides[a] : 1;
        g->dil[i] = a >= 0 && nd == (int64_t)S ? dils[a] : 1;
        g->pb[i] = a >= 0 && np == 2 * (int64_t)S ? pads[a] : 0;
        g->pe[i] = a >= 0 && np == 2 * (int64_t)S ? pads[S + a] : 0;
        if (g->k[i] < 1 || g->s[i] < 1 || g->dil[i] < 1 || g->pb[i] < 0 || g->pe[i] < 0)
            return ir_fail(r, "a kernel, stride, dilation or pad out of range");
        const int64_t span = g->dil[i] * (g->k[i] - 1) + 1;
        if (a >= 0 && (!strcmp(auto_pad, "SAME_UPPER") || !strcmp(auto_pad, "SAME_LOWER"))) {
            const int64_t o = (g->in[i] + g->s[i] - 1) / g->s[i];
            int64_t total = (o - 1) * g->s[i] + span - g->in[i];
            if (total < 0) total = 0;
            g->pb[i] = !strcmp(auto_pad, "SAME_UPPER") ? total / 2 : total - total / 2;
            g->pe[i] = total - g->pb[i];
        } else if (a >= 0 && !strcmp(auto_pad, "VALID")) {
            g->pb[i] = g->pe[i] = 0;
        }
        const int64_t room = g->in[i] + g->pb[i] + g->pe[i] - span;
        if (room < 0) return ir_fail(r, "a kernel larger than its padded input");
        g->out[i] = (ceil_mode ? (room + g->s[i] - 1) / g->s[i] : room / g->s[i]) + 1;
        /* (a last window that starts in the end's padding is left out, as PyTorch does) */
        if (ceil_mode && (g->out[i] - 1) * g->s[i] >= g->in[i] + g->pb[i]) g->out[i]--;
    }
    return true;
}

/* dst[b][p][c] = src[b][c][p] (channels last, transposed: dst[b][c][p] = src[b][p][c]) in tiles of 32 x 32. */
static void to_channels(IrRun *r, const float *src, float *dst, int64_t batch, int64_t C, int64_t P, bool last) {
    const int64_t tc = (C + 31) / 32, tp = (P + 31) / 32, tiles = batch * tc * tp;
    SPINGALETT_PARALLEL_FOR(r->parallel && (uint64_t)batch * C * P >= PARALLEL_WORK,
        for (int64_t t = 0; t < tiles; t++) {
            const int64_t b = t / (tc * tp), c0 = (t / tp) % tc * 32, p0 = t % tp * 32;
            const int64_t c1 = c0 + 32 < C ? c0 + 32 : C, p1 = p0 + 32 < P ? p0 + 32 : P;
            const float *s = src + b * C * P;
            float *d = dst + b * C * P;
            if (last) {
                for (int64_t p = p0; p < p1; p++)
                    for (int64_t c = c0; c < c1; c++) d[p * C + c] = s[c * P + p];
            } else {
                for (int64_t c = c0; c < c1; c++)
                    for (int64_t p = p0; p < p1; p++) d[c * P + p] = s[p * C + c];
            }
        }
    );
}

/* Filters of a convolution on the library's kernels: (OC, KH, KW, C / G), the bias (zeros without one). */
typedef struct {
    float *filters, *bias;
} ConvFilters;

static void conv_filters_free(void *p) {
    if (!p) return;
    free(((ConvFilters *)p)->filters);
    free(((ConvFilters *)p)->bias);
    free(p);
}

/* A convolution of one or two spatial axes on the library's kernels (Spingalett.Conv.c: indirect products,
   depthwise ones directly), which compute channels last: the input transposed to (N, H, W, C), the filters once to
   (OC, KH, KW, C / G), the output transposed back, the bias and an activation in the products' epilogue. It takes
   convolutions without dilation whose padding at the end adds nothing beyond the padding at the start that the
   windows read; false (nothing done, no error) for others. */
static bool conv_channels_last(IrRun *r, const IrTensor *X, const IrTensor *W, const IrTensor *B, const Geometry *g,
                               int64_t group, ActivationFunction act, IrTensor *Y, bool *done) {
    *done = false;
    if (X->rank > 4 || g->in[0] != 1 || g->k[0] != 1 || g->dil[1] != 1 || g->dil[2] != 1 || !W->constant ||
        (B && (!B->constant || B->dtype != F32)))
        return true;
    for (int a = 1; a < 3; a++)
        if (g->pb[a] > g->pe[a] || (g->in[a] + 2 * g->pb[a] - g->k[a]) / g->s[a] + 1 != g->out[a] || g->in[a] + 2 * g->pb[a] < g->k[a])
            return true;
    const int64_t N = X->dims[0], C = X->dims[1], OC = W->dims[0], Cg = C / group, taps = g->k[1] * g->k[2];
    const int64_t ipix = g->in[1] * g->in[2], opix = g->out[1] * g->out[2];
    if (N > UINT32_MAX || C > 65535 || OC > 65535 || ipix > ((int64_t)1 << 30)) return true;
    ConvFilters *f = (ConvFilters *)ir_prepared(r);
    if (!f) {
        f = (ConvFilters *)calloc(1, sizeof *f);
        if (f) {
            f->filters = (float *)malloc((size_t)(OC * taps * Cg) * sizeof(float) + 1);
            f->bias = (float *)calloc((size_t)OC + 1, sizeof(float));
        }
        if (!f || !f->filters || !f->bias) {
            conv_filters_free(f);
            return ir_fail(r, "out of memory for filters");
        }
        for (int64_t o = 0; o < OC; o++)
            for (int64_t c = 0; c < Cg; c++)
                for (int64_t t = 0; t < taps; t++) f->filters[(o * taps + t) * Cg + c] = fdata(W)[(o * Cg + c) * taps + t];
        if (B) memcpy(f->bias, B->data, (size_t)OC * sizeof(float));
        ir_prepare(r, f, conv_filters_free);
    }
    LayerShape is = {.type = LAYER_DENSE, .height = (uint32_t)g->in[1], .width = (uint32_t)g->in[2], .channels = (uint32_t)C};
    LayerShape os = {.type = LAYER_CONV2D, .height = (uint32_t)g->out[1], .width = (uint32_t)g->out[2],
                     .channels = (uint32_t)OC, .kernel_h = (uint32_t)g->k[1], .kernel_w = (uint32_t)g->k[2],
                     .stride_h = (uint32_t)g->s[1], .stride_w = (uint32_t)g->s[2], .pad_h = (uint32_t)g->pb[1],
                     .pad_w = (uint32_t)g->pb[2], .groups = (uint32_t)group};
    const ComputeMode mode = r->parallel ? COMPUTE_OPENMP : COMPUTE_SINGLE_THREADED;
    SpingalettGemmScratch *gemm = ir_scratch(r);
    if (!gemm) return false;
    const size_t scratch_floats = spingalett_conv_forward_scratch(&is, &os, (uint32_t)N, mode);
    IrTensor xt, yt;
    memset(&xt, 0, sizeof xt);
    memset(&yt, 0, sizeof yt);
    const int64_t xd[1] = {(int64_t)X->count}, yd[1] = {N * OC * opix}, sd[1] = {(int64_t)scratch_floats + 16};
    IrTensor scratch;
    memset(&scratch, 0, sizeof scratch);
    int64_t dims[IR_MAX_RANK] = {N, OC};
    for (uint32_t a = 0; a + 2 < X->rank; a++) dims[2 + a] = g->out[3 - (X->rank - 2) + a];
    bool ok = ir_output_raw(r, &xt, F32, 1, xd) && ir_output_raw(r, &yt, F32, 1, yd) &&
              ir_output_raw(r, &scratch, F32, 1, sd) && ir_output_raw(r, Y, F32, X->rank, dims);
    if (ok) {
        to_channels(r, fdata(X), fdata(&xt), N, C, ipix, true);
        spingalett_conv_forward_shapes(&is, &os, f->filters, f->bias, fdata(&xt), fdata(&yt), (uint32_t)N, act,
                                       fdata(&scratch), gemm, mode);
        to_channels(r, fdata(&yt), fdata(Y), N, OC, opix, false);
        *done = true;
    }
    ir_release(r->g, &xt);
    ir_release(r->g, &yt);
    ir_release(r->g, &scratch);
    return ok;
}

static bool k_conv(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "inputs and an output are needed");
    const IrTensor *X = in[0], *W = in[1], *B = n > 2 ? in[2] : NULL;
    if (X->dtype != F32 || W->dtype != F32 || W->rank != X->rank) return ir_fail(r, "weights of another rank, or not float32");
    const int64_t group = ir_attr_int(r->node, "group", 1), N = X->dims[0], C = X->dims[1], OC = W->dims[0];
    if (group < 1 || C % group || OC % group || W->dims[1] != C / group) return ir_fail(r, "channels that do not fit the groups");
    Geometry g;
    if (!geometry(r, X, W->dims + 2, &g, false)) return false;
    const int64_t act = ir_attr_int(r->node, "activation", ACT_NONE);
    if (act < 0 || act >= ACT_COUNT || act == ACT_SOFTMAX) return ir_fail(r, "the activation %lld", (long long)act);
    if (B && (B->dtype != F32 || B->count != (uint64_t)OC)) return ir_fail(r, "a bias of %llu values for %lld channels",
                                                                          (unsigned long long)B->count, (long long)OC);
    bool done;
    if (!conv_channels_last(r, X, W, B, &g, group, (ActivationFunction)act, out[0], &done)) return false;
    if (done) return true;
    int64_t dims[IR_MAX_RANK] = {N, OC};
    for (uint32_t a = 0; a < g.spatial; a++) dims[2 + a] = g.out[3 - g.spatial + a];
    if (!ir_output(r, out[0], F32, X->rank, dims)) return false;
    const int64_t Cg = C / group, OCg = OC / group, kprod = g.k[0] * g.k[1] * g.k[2];
    const int64_t ipix = g.in[0] * g.in[1] * g.in[2], opix = g.out[0] * g.out[1] * g.out[2], KK = Cg * kprod;
    const float *x = fdata(X), *w = fdata(W);
    float *y = fdata(out[0]);
    if (Cg == 1 && OCg == 1 && group > 1) {
        /* depthwise: each output a window's sum of its channel, in the window's order */
        SPINGALETT_PARALLEL_FOR(r->parallel && (uint64_t)N * C * opix * kprod >= PARALLEL_WORK,
            for (int64_t q = 0; q < N * C; q++) {
                const int64_t c = q % C;
                const float *xc = x + (size_t)q * (size_t)ipix, *wc = w + (size_t)c * (size_t)kprod;
                float *yc = y + (size_t)q * (size_t)opix;
                for (int64_t o0 = 0; o0 < g.out[0]; o0++)
                    for (int64_t o1 = 0; o1 < g.out[1]; o1++)
                        for (int64_t o2 = 0; o2 < g.out[2]; o2++) {
                            float sum = 0.0f;
                            for (int64_t k0 = 0; k0 < g.k[0]; k0++) {
                                const int64_t i0 = o0 * g.s[0] - g.pb[0] + k0 * g.dil[0];
                                if (i0 < 0 || i0 >= g.in[0]) continue;
                                for (int64_t k1 = 0; k1 < g.k[1]; k1++) {
                                    const int64_t i1 = o1 * g.s[1] - g.pb[1] + k1 * g.dil[1];
                                    if (i1 < 0 || i1 >= g.in[1]) continue;
                                    const float *row = xc + (size_t)((i0 * g.in[1] + i1) * g.in[2]);
                                    const float *wr = wc + (size_t)((k0 * g.k[1] + k1) * g.k[2]);
                                    for (int64_t k2 = 0; k2 < g.k[2]; k2++) {
                                        const int64_t i2 = o2 * g.s[2] - g.pb[2] + k2 * g.dil[2];
                                        if (i2 >= 0 && i2 < g.in[2]) sum += row[i2] * wr[k2];
                                    }
                                }
                            }
                            yc[(o0 * g.out[1] + o1) * g.out[2] + o2] = sum + (B ? fdata(B)[c] : 0.0f);
                        }
            }
        );
        if (act != ACT_NONE) apply_activation_bulk(y, out[0]->count, (ActivationFunction)act);
        return true;
    }
    /* windows gathered into columns a block of output pixels at a time, then the group's weights times them */
    const bool direct = kprod == 1 && g.s[0] * g.s[1] * g.s[2] == 1 && !g.pb[0] && !g.pb[1] && !g.pb[2] && opix == ipix;
    int64_t block = direct ? opix : ((int64_t)1 << 22) / (KK ? KK : 1);
    if (block < 64) block = 64;
    if (block > opix) block = opix;
    float *cols = direct ? NULL : (float *)spingalett_aligned_alloc((size_t)(KK * block) * sizeof(float) + 64);
    SpingalettGemmScratch *scratch = ir_scratch(r);
    if ((!direct && !cols) || !scratch) {
        spingalett_aligned_free(cols);
        return r->error[0] ? false : ir_fail(r, "out of memory for the windows");
    }
    for (int64_t b = 0; b < N; b++)
        for (int64_t gr = 0; gr < group; gr++) {
            const float *xg = x + (size_t)(b * C + gr * Cg) * (size_t)ipix;
            float *yg = y + (size_t)(b * OC + gr * OCg) * (size_t)opix;
            const float *wg = w + (size_t)(gr * OCg) * (size_t)KK;
            for (int64_t p0 = 0; p0 < opix; p0 += block) {
                const int64_t P = opix - p0 < block ? opix - p0 : block;
                const float *src = xg;
                size_t ldb = (size_t)opix;
                if (!direct) {
                    SPINGALETT_PARALLEL_FOR(r->parallel && (uint64_t)KK * P >= PARALLEL_WORK,
                        for (int64_t row = 0; row < KK; row++) {
                            const int64_t c = row / kprod, kk = row % kprod;
                            const int64_t k0 = kk / (g.k[1] * g.k[2]), k1 = (kk / g.k[2]) % g.k[1], k2 = kk % g.k[2];
                            float *dst = cols + (size_t)row * (size_t)P;
                            const float *xc = xg + (size_t)c * (size_t)ipix;
                            for (int64_t p = 0; p < P; p++) {
                                const int64_t o = p0 + p, o2 = o % g.out[2], o1 = (o / g.out[2]) % g.out[1];
                                const int64_t o0 = o / (g.out[1] * g.out[2]);
                                const int64_t i0 = o0 * g.s[0] - g.pb[0] + k0 * g.dil[0];
                                const int64_t i1 = o1 * g.s[1] - g.pb[1] + k1 * g.dil[1];
                                const int64_t i2 = o2 * g.s[2] - g.pb[2] + k2 * g.dil[2];
                                dst[p] = i0 >= 0 && i0 < g.in[0] && i1 >= 0 && i1 < g.in[1] && i2 >= 0 && i2 < g.in[2]
                                       ? xc[(i0 * g.in[1] + i1) * g.in[2] + i2] : 0.0f;
                            }
                        }
                    );
                    src = cols;
                    ldb = (size_t)P;
                }
                spingalett_gemm_native(scratch, false, false, (uint32_t)OCg, (uint32_t)P, (uint32_t)KK, 1.0f, wg,
                                       (size_t)KK, direct ? src + p0 : src, ldb, 0.0f, yg + p0, (size_t)opix,
                                       r->parallel);
            }
        }
    spingalett_aligned_free(cols);
    if (B) {
        SPINGALETT_PARALLEL_FOR(r->parallel && (uint64_t)N * OC * opix >= PARALLEL_WORK,
            for (int64_t q = 0; q < N * OC; q++) {
                const float bias = fdata(B)[q % OC];
                float *yc = y + (size_t)q * (size_t)opix;
                for (int64_t p = 0; p < opix; p++) yc[p] += bias;
            }
        );
    }
    if (act != ACT_NONE) apply_activation_bulk(y, out[0]->count, (ActivationFunction)act);
    return true;
}

static bool k_conv_transpose(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "inputs and an output are needed");
    const IrTensor *X = in[0], *W = in[1], *B = n > 2 ? in[2] : NULL;
    if (X->dtype != F32 || W->dtype != F32 || W->rank != X->rank || X->rank < 3 || X->rank > 5)
        return ir_fail(r, "weights of another rank, or not float32");
    const uint32_t S = X->rank - 2;
    const int64_t group = ir_attr_int(r->node, "group", 1), N = X->dims[0], C = X->dims[1];
    if (group < 1 || C % group || W->dims[0] != C) return ir_fail(r, "channels that do not fit the groups");
    const int64_t Cg = C / group, OCg = W->dims[1], OC = OCg * group;
    int64_t strides[3], dils[3], pads[6], opad[3], oshape[3];
    const int64_t ns = int_list(r->node, "strides", NULL, strides, 3), nd = int_list(r->node, "dilations", NULL, dils, 3);
    const int64_t np = int_list(r->node, "pads", NULL, pads, 6), no = int_list(r->node, "output_padding", NULL, opad, 3);
    const int64_t nsh = int_list(r->node, "output_shape", NULL, oshape, 3);
    const IrAttr *ap = ir_attr(r->node, "auto_pad");
    const char *auto_pad = ap && ap->type == ATTR_STRING ? ap->s : "NOTSET";
    int64_t in3[3], k3[3], s3[3], d3[3], pb3[3], out3[3];
    for (uint32_t i = 0; i < 3; i++) {
        const int32_t a = (int32_t)i - (int32_t)(3 - S);
        in3[i] = a >= 0 ? X->dims[2 + a] : 1;
        k3[i] = a >= 0 ? W->dims[2 + a] : 1;
        s3[i] = a >= 0 && ns == (int64_t)S ? strides[a] : 1;
        d3[i] = a >= 0 && nd == (int64_t)S ? dils[a] : 1;
        const int64_t op = a >= 0 && no == (int64_t)S ? opad[a] : 0;
        int64_t pb = a >= 0 && np == 2 * (int64_t)S ? pads[a] : 0, pe = a >= 0 && np == 2 * (int64_t)S ? pads[S + a] : 0;
        const int64_t full = s3[i] * (in3[i] - 1) + op + (k3[i] - 1) * d3[i] + 1;
        if (a >= 0 && (nsh == (int64_t)S || (strcmp(auto_pad, "NOTSET") && strcmp(auto_pad, "VALID")))) {
            const int64_t want = nsh == (int64_t)S ? oshape[a] : in3[i] * s3[i];
            const int64_t total = full - want;
            const bool upper = strcmp(auto_pad, "SAME_LOWER") != 0;
            pb = upper ? total / 2 : total - total / 2;
            pe = total - pb;
        }
        pb3[i] = pb;
        out3[i] = full - pb - pe;
        if (out3[i] < 1 || s3[i] < 1 || d3[i] < 1) return ir_fail(r, "a stride, dilation or pad out of range");
    }
    int64_t dims[IR_MAX_RANK] = {N, OC};
    for (uint32_t a = 0; a < S; a++) dims[2 + a] = out3[3 - S + a];
    if (!ir_output(r, out[0], F32, X->rank, dims)) return false;
    const int64_t kprod = k3[0] * k3[1] * k3[2], ipix = in3[0] * in3[1] * in3[2], opix = out3[0] * out3[1] * out3[2];
    const int64_t MM = OCg * kprod;
    int64_t block = ((int64_t)1 << 22) / (MM ? MM : 1);
    if (block < 64) block = 64;
    if (block > ipix) block = ipix;
    float *cols = (float *)spingalett_aligned_alloc((size_t)(MM * block) * sizeof(float) + 64);
    SpingalettGemmScratch *scratch = ir_scratch(r);
    if (!cols || !scratch) {
        spingalett_aligned_free(cols);
        return r->error[0] ? false : ir_fail(r, "out of memory for the windows");
    }
    const float *x = fdata(X), *w = fdata(W);
    float *y = fdata(out[0]);
    for (int64_t b = 0; b < N; b++)
        for (int64_t gr = 0; gr < group; gr++) {
            const float *xg = x + (size_t)(b * C + gr * Cg) * (size_t)ipix;
            const float *wg = w + (size_t)(gr * Cg) * (size_t)MM;
            float *yg = y + (size_t)(b * OC + gr * OCg) * (size_t)opix;
            for (int64_t p0 = 0; p0 < ipix; p0 += block) {
                const int64_t P = ipix - p0 < block ? ipix - p0 : block;
                /* cols [MM x P] = wg^T [MM x Cg] times xg [Cg x P] */
                spingalett_gemm_native(scratch, true, false, (uint32_t)MM, (uint32_t)P, (uint32_t)Cg, 1.0f, wg, (size_t)MM,
                                       xg + p0, (size_t)ipix, 0.0f, cols, (size_t)P,
                                       r->parallel);
                /* each output channel's columns added into its pixels, in the order of the kernel's taps and pixels */
                SPINGALETT_PARALLEL_FOR(r->parallel && (uint64_t)MM * P >= PARALLEL_WORK,
                    for (int64_t oc = 0; oc < OCg; oc++) {
                        float *yc = yg + (size_t)oc * (size_t)opix;
                        for (int64_t kk = 0; kk < kprod; kk++) {
                            const int64_t k0 = kk / (k3[1] * k3[2]), k1 = (kk / k3[2]) % k3[1], k2 = kk % k3[2];
                            const float *src = cols + (size_t)(oc * kprod + kk) * (size_t)P;
                            for (int64_t p = 0; p < P; p++) {
                                const int64_t q = p0 + p, i2 = q % in3[2], i1 = (q / in3[2]) % in3[1];
                                const int64_t i0 = q / (in3[1] * in3[2]);
                                const int64_t o0 = i0 * s3[0] - pb3[0] + k0 * d3[0], o1 = i1 * s3[1] - pb3[1] + k1 * d3[1];
                                const int64_t o2 = i2 * s3[2] - pb3[2] + k2 * d3[2];
                                if (o0 >= 0 && o0 < out3[0] && o1 >= 0 && o1 < out3[1] && o2 >= 0 && o2 < out3[2])
                                    yc[(o0 * out3[1] + o1) * out3[2] + o2] += src[p];
                            }
                        }
                    }
                );
            }
        }
    spingalett_aligned_free(cols);
    if (B) {
        if (B->dtype != F32 || B->count != (uint64_t)OC) return ir_fail(r, "a bias of another size");
        for (int64_t q = 0; q < N * OC; q++) {
            const float bias = fdata(B)[q % OC];
            float *yc = y + (size_t)q * (size_t)opix;
            for (int64_t p = 0; p < opix; p++) yc[p] += bias;
        }
    }
    return true;
}

static bool k_pool(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    const IrTensor *X = in[0];
    const IrOp op = OP(r);
    const bool max = op == IR_MaxPool, lp = op == IR_LpPool;
    if (X->dtype != F32 && !(max && is_int(X->dtype))) return ir_fail(r, "an input of %s", ir_dtype_name(X->dtype));
    int64_t kernel[3];
    if (int_list(r->node, "kernel_shape", NULL, kernel, 3) != (int64_t)X->rank - 2) return ir_fail(r, "no kernel_shape");
    Geometry g;
    if (!geometry(r, X, kernel, &g, ir_attr_int(r->node, "ceil_mode", 0) != 0)) return false;
    const bool include = ir_attr_int(r->node, "count_include_pad", 0) != 0;
    const float p = lp ? (float)ir_attr_int(r->node, "p", 2) : 1.0f;
    const int64_t N = X->dims[0], C = X->dims[1];
    int64_t dims[IR_MAX_RANK] = {N, C};
    for (uint32_t a = 0; a < g.spatial; a++) dims[2 + a] = g.out[3 - g.spatial + a];
    if (!ir_output(r, out[0], X->dtype, X->rank, dims)) return false;
    int64_t *indices = NULL;
    if (max && m > 1 && out[1]) {
        if (ir_attr_int(r->node, "storage_order", 0)) return ir_fail(r, "indices in column-major order");
        if (!ir_output(r, out[1], I64, X->rank, dims)) return false;
        indices = (int64_t *)out[1]->data;
    }
    const int64_t ipix = g.in[0] * g.in[1] * g.in[2], opix = g.out[0] * g.out[1] * g.out[2];
    const bool floats = X->dtype == F32;
    const float *x = fdata(X);
    SPINGALETT_PARALLEL_FOR(r->parallel && (uint64_t)N * C * opix >= PARALLEL_WORK / 4,
        for (int64_t q = 0; q < N * C; q++) {
            for (int64_t o = 0; o < opix; o++) {
                const int64_t o2 = o % g.out[2], o1 = (o / g.out[2]) % g.out[1], o0 = o / (g.out[1] * g.out[2]);
                float acc = max ? -INFINITY : 0.0f;
                int64_t count = 0, padded = 0, at = -1;
                for (int64_t k0 = 0; k0 < g.k[0]; k0++) {
                    const int64_t i0 = o0 * g.s[0] - g.pb[0] + k0 * g.dil[0];
                    for (int64_t k1 = 0; k1 < g.k[1]; k1++) {
                        const int64_t i1 = o1 * g.s[1] - g.pb[1] + k1 * g.dil[1];
                        for (int64_t k2 = 0; k2 < g.k[2]; k2++) {
                            const int64_t i2 = o2 * g.s[2] - g.pb[2] + k2 * g.dil[2];
                            const bool within_pad = i0 < g.in[0] + g.pe[0] && i1 < g.in[1] + g.pe[1] && i2 < g.in[2] + g.pe[2];
                            padded += within_pad;
                            if (i0 < 0 || i0 >= g.in[0] || i1 < 0 || i1 >= g.in[1] || i2 < 0 || i2 >= g.in[2]) continue;
                            const int64_t src = q * ipix + (i0 * g.in[1] + i1) * g.in[2] + i2;
                            const float v = floats ? x[src] : (float)ir_get_int(X, (uint64_t)src);
                            if (max) {
                                if (at < 0 || v > acc || (isnan(v) && !isnan(acc))) { acc = v; at = src; }
                            } else {
                                acc += lp ? powf(fabsf(v), p) : v;
                            }
                            count++;
                        }
                    }
                }
                const float div = (float)(include ? padded : count ? count : 1);
                const float y = max ? acc : lp ? powf(acc, 1.0f / p) : acc / div;
                if (floats) fdata(out[0])[q * opix + o] = y;
                else put_int(out[0], (uint64_t)(q * opix + o), (int64_t)y);
                if (indices) indices[q * opix + o] = at;
            }
        }
    );
    return true;
}

static bool k_global_pool(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    const IrTensor *X = in[0];
    if (X->dtype != F32 || X->rank < 2) return ir_fail(r, "an input of %s or fewer than two axes", ir_dtype_name(X->dtype));
    int64_t dims[IR_MAX_RANK];
    for (uint32_t d = 0; d < X->rank; d++) dims[d] = d < 2 ? X->dims[d] : 1;
    if (!ir_output(r, out[0], F32, X->rank, dims)) return false;
    const uint64_t rows = (uint64_t)X->dims[0] * (uint64_t)X->dims[1], len = rows ? X->count / rows : 0;
    const bool max = OP(r) == IR_GlobalMaxPool;
    for (uint64_t q = 0; q < rows; q++) {
        const float *xc = fdata(X) + q * len;
        double acc = max ? -INFINITY : 0.0;
        for (uint64_t i = 0; i < len; i++) acc = max ? (xc[i] > acc ? xc[i] : acc) : acc + xc[i];
        fdata(out[0])[q] = (float)(max ? acc : acc / (double)(len ? len : 1));
    }
    return true;
}

/* ------------------------------------------------------------------------- normalizations */

static bool k_batch_norm(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 5 || m < 1) return ir_fail(r, "five inputs and an output are needed");
    for (uint32_t k = 0; k < 5; k++) if (!in[k] || in[k]->dtype != F32) return ir_fail(r, "an input missing or not float32");
    const IrTensor *X = in[0];
    if (X->rank < 2) return ir_fail(r, "an input of fewer than two axes");
    if (ir_attr_int(r->node, "training_mode", 0)) return ir_fail(r, "training mode");
    const int64_t C = X->dims[1];
    for (uint32_t k = 1; k < 5; k++) if (in[k]->count != (uint64_t)C) return ir_fail(r, "parameters of another size than the channels");
    if (!ir_output(r, out[0], F32, X->rank, X->dims)) return false;
    const float eps = ir_attr_float(r->node, "epsilon", 1e-5f);
    const uint64_t inner = X->count / (uint64_t)(X->dims[0] && C ? X->dims[0] * C : 1);
    for (int64_t q = 0; q < X->dims[0] * C; q++) {
        const int64_t c = q % C;
        const float scale = fdata(in[1])[c] / sqrtf(fdata(in[4])[c] + eps), shift = fdata(in[2])[c] - fdata(in[3])[c] * scale;
        const float *xc = fdata(X) + (size_t)q * inner;
        float *yc = fdata(out[0]) + (size_t)q * inner;
        for (uint64_t i = 0; i < inner; i++) yc[i] = xc[i] * scale + shift;
    }
    return true;
}

/* LayerNormalization and RMSNormalization over the axes from `axis` on, their scale (and bias) broadcast over those
   axes; LayerNormalization's optional Mean and InvStdDev outputs. Sums in double, in order. */
static bool k_layer_norm(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "an input, a scale and an output are needed");
    const IrTensor *X = in[0], *S = in[1], *Bias = OP(r) == IR_LayerNormalization && n > 2 ? in[2] : NULL;
    if (X->dtype != F32 || S->dtype != F32 || (Bias && Bias->dtype != F32)) return ir_fail(r, "inputs not float32");
    uint32_t axis;
    if (!axis_of(r, ir_attr_int(r->node, "axis", -1), X->rank, &axis)) return false;
    const float eps = ir_attr_float(r->node, "epsilon", 1e-5f);
    if (!ir_output(r, out[0], F32, X->rank, X->dims)) return false;
    uint64_t outer = 1, inner = 1;
    for (uint32_t d = 0; d < axis; d++) outer *= (uint64_t)X->dims[d];
    for (uint32_t d = axis; d < X->rank; d++) inner *= (uint64_t)X->dims[d];
    /* the parameters over the normalized axes */
    IrTensor norm;
    memset(&norm, 0, sizeof norm);
    norm.rank = (uint8_t)(X->rank - axis);
    for (uint32_t d = 0; d < norm.rank; d++) norm.dims[d] = X->dims[axis + d];
    if (S->rank > norm.rank || (Bias && Bias->rank > norm.rank)) return ir_fail(r, "a scale of more axes than are normalized");
    int64_t ss[IR_MAX_RANK], bs[IR_MAX_RANK];
    broadcast_strides(S, norm.rank, norm.dims, ss);
    if (Bias) broadcast_strides(Bias, norm.rank, norm.dims, bs);
    const bool plain_s = S->count == inner, plain_b = !Bias || Bias->count == inner;
    float *mean_out = NULL, *inv_out = NULL;
    if (OP(r) == IR_LayerNormalization && m > 1) {
        int64_t md[IR_MAX_RANK];
        for (uint32_t d = 0; d < X->rank; d++) md[d] = d < axis ? X->dims[d] : 1;
        if (out[1]) { if (!ir_output(r, out[1], F32, X->rank, md)) return false; mean_out = fdata(out[1]); }
        if (m > 2 && out[2]) { if (!ir_output(r, out[2], F32, X->rank, md)) return false; inv_out = fdata(out[2]); }
    }
    const bool rms = OP(r) == IR_RMSNormalization;
    SPINGALETT_PARALLEL_FOR(r->parallel && X->count >= PARALLEL_WORK,
        for (int64_t q = 0; q < (int64_t)outer; q++) {
            const float *x = fdata(X) + (size_t)q * inner;
            float *y = fdata(out[0]) + (size_t)q * inner;
            double mean = 0.0, var = 0.0;
            if (!rms) {
                for (uint64_t i = 0; i < inner; i++) mean += x[i];
                mean /= (double)inner;
            }
            for (uint64_t i = 0; i < inner; i++) var += ((double)x[i] - mean) * ((double)x[i] - mean);
            var /= (double)inner;
            const float inv = (float)(1.0 / sqrt(var + (double)eps)), mf = (float)mean;
            if (mean_out) mean_out[q] = mf;
            if (inv_out) inv_out[q] = inv;
            for (uint64_t i = 0; i < inner; i++) {
                uint64_t so = i, bo = i;
                if (!plain_s || !plain_b) {
                    so = bo = 0;
                    uint64_t t = i;
                    for (uint32_t d = norm.rank; d-- > 0;) {
                        const uint64_t c = t % (uint64_t)norm.dims[d];
                        t /= (uint64_t)norm.dims[d];
                        so += c * (uint64_t)ss[d];
                        if (Bias) bo += c * (uint64_t)bs[d];
                    }
                }
                y[i] = (x[i] - mf) * inv * fdata(S)[so] + (Bias ? fdata(Bias)[bo] : 0.0f);
            }
        }
    );
    return true;
}

/* GroupNormalization (the scale and bias by channel, or by group as opset 18 has them) and InstanceNormalization (a
   group a channel). */
static bool k_group_norm(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 3 || m < 1 || !in[0] || !in[1] || !in[2]) return ir_fail(r, "an input, scale, bias and output are needed");
    const IrTensor *X = in[0], *S = in[1], *Bias = in[2];
    if (X->dtype != F32 || S->dtype != F32 || Bias->dtype != F32 || X->rank < 2) return ir_fail(r, "inputs not float32");
    const int64_t N = X->dims[0], C = X->dims[1];
    const int64_t G = OP(r) == IR_InstanceNormalization ? C : ir_attr_int(r->node, "num_groups", 0);
    if (G < 1 || C % G) return ir_fail(r, "%lld groups of %lld channels", (long long)G, (long long)C);
    if ((S->count != (uint64_t)C && S->count != (uint64_t)G) || Bias->count != S->count) return ir_fail(r, "a scale of another size");
    const bool by_group = S->count == (uint64_t)G && G != C;
    const float eps = ir_attr_float(r->node, "epsilon", 1e-5f);
    if (!ir_output(r, out[0], F32, X->rank, X->dims)) return false;
    const uint64_t pix = X->count / (uint64_t)(N && C ? N * C : 1), per = (uint64_t)(C / G) * pix;
    SPINGALETT_PARALLEL_FOR(r->parallel && X->count >= PARALLEL_WORK,
        for (int64_t q = 0; q < N * G; q++) {
            const float *x = fdata(X) + (size_t)q * per;
            float *y = fdata(out[0]) + (size_t)q * per;
            double mean = 0.0, var = 0.0;
            for (uint64_t i = 0; i < per; i++) mean += x[i];
            mean /= (double)(per ? per : 1);
            for (uint64_t i = 0; i < per; i++) var += ((double)x[i] - mean) * ((double)x[i] - mean);
            var /= (double)(per ? per : 1);
            const float inv = (float)(1.0 / sqrt(var + (double)eps)), mf = (float)mean;
            const int64_t grp = q % G;
            for (int64_t c = 0; c < C / G; c++) {
                const int64_t ch = grp * (C / G) + c, pi = by_group ? grp : ch;
                const float sc = fdata(S)[pi], sh = fdata(Bias)[pi];
                for (uint64_t i = 0; i < pix; i++) {
                    const uint64_t at = (uint64_t)c * pix + i;
                    y[at] = (x[at] - mf) * inv * sc + sh;
                }
            }
        }
    );
    return true;
}

/* ------------------------------------------------------------------------- softmax and reductions */

/* The rows along one axis of a tensor: outer x len x inner, rows `inner` apart. */
typedef struct {
    uint64_t outer, len, inner;
} Rows;

static Rows rows_of(const IrTensor *x, uint32_t first, uint32_t end) {
    Rows w = {1, 1, 1};
    for (uint32_t d = 0; d < x->rank; d++) {
        if (d < first) w.outer *= (uint64_t)x->dims[d];
        else if (d < end) w.len *= (uint64_t)x->dims[d];
        else w.inner *= (uint64_t)x->dims[d];
    }
    return w;
}

/* Softmax and LogSoftmax over an axis (before opset 13: over the axes from it on, the input seen as a matrix). */
static bool k_softmax(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    const IrTensor *X = in[0];
    if (X->dtype != F32) return ir_fail(r, "an input of %s", ir_dtype_name(X->dtype));
    const bool old = r->g->opset < 13;
    uint32_t axis;
    if (!axis_of(r, ir_attr_int(r->node, "axis", old ? 1 : -1), X->rank ? X->rank : 1, &axis)) return false;
    if (!ir_output(r, out[0], F32, X->rank, X->dims)) return false;
    const Rows w = rows_of(X, axis, old ? X->rank : axis + 1);
    const bool log_form = OP(r) == IR_LogSoftmax, hard = OP(r) == IR_Hardmax;
    const float *x = fdata(X);
    float *y = fdata(out[0]);
    FOR_ROWS(r, w.outer * w.inner, X->count, begin, end,
        for (uint64_t q = begin; q < end; q++) {
            const uint64_t o = q / w.inner, i = q % w.inner, step = w.inner;
            const float *xr = x + o * w.len * w.inner + i;
            float *yr = y + o * w.len * w.inner + i;
            float mx = -INFINITY;
            if (hard) {
                /* 1 at the first largest value */
                uint64_t at = 0;
                for (uint64_t l = 0; l < w.len; l++)
                    if (xr[l * step] > xr[at * step]) at = l;
                if (w.len) yr[at * step] = 1.0f;
                continue;
            }
            for (uint64_t l = 0; l < w.len; l++) mx = xr[l * step] > mx || isnan(xr[l * step]) ? xr[l * step] : mx;
            double sum = 0.0;
            for (uint64_t l = 0; l < w.len; l++) {
                const float e = expf(xr[l * step] - mx);
                if (!log_form) yr[l * step] = e;
                sum += e;
            }
            if (log_form) {
                const float ls = (float)log(sum);
                for (uint64_t l = 0; l < w.len; l++) yr[l * step] = xr[l * step] - mx - ls;
            } else {
                const float inv = (float)(1.0 / sum);
                for (uint64_t l = 0; l < w.len; l++) yr[l * step] *= inv;
            }
        }
    );
    return true;
}

/* The Reduce operators over axes (an attribute, or an input from opset 13 for ReduceSum and 18 for the others). */
static bool k_reduce(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    const IrTensor *X = in[0];
    const IrOp op = OP(r);
    const bool floats = X->dtype == F32;
    if (!floats && !is_int(X->dtype) && !(X->dtype == BOOL && (op == IR_ReduceMax || op == IR_ReduceMin)))
        return ir_fail(r, "an input of %s", ir_dtype_name(X->dtype));
    int64_t axes[IR_MAX_RANK];
    const int64_t na = int_list(r->node, "axes", n > 1 ? in[1] : NULL, axes, IR_MAX_RANK);
    const bool keep = ir_attr_int(r->node, "keepdims", 1) != 0;
    if (na <= 0 && ir_attr_int(r->node, "noop_with_empty_axes", 0)) return copy_as(r, X, out[0], X->rank, X->dims);
    bool red[IR_MAX_RANK] = {false};
    for (uint32_t d = 0; d < X->rank; d++) red[d] = na <= 0;
    for (int64_t k = 0; k < na; k++) {
        uint32_t a;
        if (!axis_of(r, axes[k], X->rank, &a)) return false;
        red[a] = true;
    }
    int64_t dims[IR_MAX_RANK];
    uint32_t rank = 0;
    for (uint32_t d = 0; d < X->rank; d++)
        if (!red[d]) dims[rank++] = X->dims[d];
        else if (keep) dims[rank++] = 1;
    if (!ir_output(r, out[0], X->dtype, rank, dims)) return false;
    /* reduced axes that are not one block are moved behind the others */
    uint32_t first = X->rank, end = 0;
    for (uint32_t d = 0; d < X->rank; d++) if (red[d]) { if (d < first) first = d; end = d + 1; }
    bool block = true;
    for (uint32_t d = first; d < end; d++) block &= red[d];
    IrTensor moved;
    memset(&moved, 0, sizeof moved);
    const IrTensor *src = X;
    if (first == X->rank) first = end = 0;
    else if (!block) {
        int64_t perm[IR_MAX_RANK];
        uint32_t k = 0;
        for (uint32_t d = 0; d < X->rank; d++) if (!red[d]) perm[k++] = d;
        first = k;
        for (uint32_t d = 0; d < X->rank; d++) if (red[d]) perm[k++] = d;
        end = X->rank;
        if (!permute(r, X, perm, &moved)) return false;
        src = &moved;
    }
    const Rows w = rows_of(src, first, end);
    FOR_ROWS(r, w.outer * w.inner, src->count, begin, last,
        for (uint64_t q = begin; q < last; q++) {
            const uint64_t o = q / w.inner, i = q % w.inner, base = o * w.len * w.inner + i;
            if (floats) {
                const float *x = fdata(src) + base;
                double acc = op == IR_ReduceProd ? 1.0 : op == IR_ReduceMax ? -INFINITY : op == IR_ReduceMin ? INFINITY : 0.0;
                float mx = -INFINITY;
                if (op == IR_ReduceLogSumExp)
                    for (uint64_t l = 0; l < w.len; l++) mx = x[l * w.inner] > mx ? x[l * w.inner] : mx;
                for (uint64_t l = 0; l < w.len; l++) {
                    const double v = x[l * w.inner];
                    switch (op) {
                        case IR_ReduceMax: acc = isnan(v) || v > acc ? (isnan(acc) ? acc : v) : acc; break;
                        case IR_ReduceMin: acc = isnan(v) || v < acc ? (isnan(acc) ? acc : v) : acc; break;
                        case IR_ReduceProd: acc *= v; break;
                        case IR_ReduceSumSquare: case IR_ReduceL2: acc += v * v; break;
                        case IR_ReduceL1: acc += fabs(v); break;
                        case IR_ReduceLogSumExp: acc += exp(v - mx); break;
                        default: acc += v; break;
                    }
                }
                if (op == IR_ReduceMean) acc /= (double)w.len;
                else if (op == IR_ReduceL2) acc = sqrt(acc);
                else if (op == IR_ReduceLogSumExp) acc = isinf(mx) && mx < 0 ? -INFINITY : log(acc) + mx;
                else if (op == IR_ReduceLogSum) acc = log(acc);
                fdata(out[0])[q] = (float)acc;
            } else {
                int64_t acc = op == IR_ReduceProd ? 1 : op == IR_ReduceMax ? INT64_MIN : op == IR_ReduceMin ? INT64_MAX : 0;
                for (uint64_t l = 0; l < w.len; l++) {
                    const int64_t v = ir_get_int(src, base + l * w.inner);
                    switch (op) {
                        case IR_ReduceMax: acc = v > acc ? v : acc; break;
                        case IR_ReduceMin: acc = v < acc ? v : acc; break;
                        case IR_ReduceProd: acc = (int64_t)((uint64_t)acc * (uint64_t)v); break;
                        case IR_ReduceSumSquare: case IR_ReduceL2: acc += v * v; break;
                        case IR_ReduceL1: acc += v < 0 ? -v : v; break;
                        default: acc += v; break;
                    }
                }
                if (op == IR_ReduceMean) acc = w.len ? acc / (int64_t)w.len : 0;
                else if (op == IR_ReduceL2) acc = (int64_t)sqrt((double)acc);
                put_int(out[0], q, acc);
            }
        }
    );
    ir_release(r->g, &moved);
    return true;
}

static bool k_arg(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    const IrTensor *X = in[0];
    if (X->dtype != F32 && !is_int(X->dtype)) return ir_fail(r, "an input of %s", ir_dtype_name(X->dtype));
    uint32_t axis;
    if (!axis_of(r, ir_attr_int(r->node, "axis", 0), X->rank, &axis)) return false;
    const bool keep = ir_attr_int(r->node, "keepdims", 1) != 0, last = ir_attr_int(r->node, "select_last_index", 0) != 0;
    const bool max = OP(r) == IR_ArgMax;
    if (!X->dims[axis]) return ir_fail(r, "an empty axis");
    int64_t dims[IR_MAX_RANK];
    uint32_t rank = 0;
    for (uint32_t d = 0; d < X->rank; d++) if (d != axis || keep) dims[rank++] = d == axis ? 1 : X->dims[d];
    if (!ir_output(r, out[0], I64, rank, dims)) return false;
    const Rows w = rows_of(X, axis, axis + 1);
    int64_t *y = (int64_t *)out[0]->data;
    for (uint64_t q = 0; q < w.outer * w.inner; q++) {
        const uint64_t base = q / w.inner * w.len * w.inner + q % w.inner;
        double best = ir_get_float(X, base);
        int64_t at = 0;
        for (uint64_t l = 1; l < w.len && !isnan(best); l++) {
            const double v = ir_get_float(X, base + l * w.inner);
            if (isnan(v) || (max ? v > best : v < best) || (last && v == best)) { best = v; at = (int64_t)l; }
        }
        y[q] = at;
    }
    return true;
}

typedef struct {
    double v;
    int64_t i;
} Ranked;

/* larger values first (NaN the largest), equal ones by their index */
static int ranked_down(const void *a, const void *b) {
    const Ranked *x = (const Ranked *)a, *y = (const Ranked *)b;
    if (isnan(x->v) != isnan(y->v)) return isnan(x->v) ? -1 : 1;
    if (x->v != y->v && !isnan(x->v)) return x->v > y->v ? -1 : 1;
    return x->i < y->i ? -1 : x->i > y->i;
}

/* smaller values first (NaN the largest, last) */
static int ranked_up(const void *a, const void *b) {
    const Ranked *x = (const Ranked *)a, *y = (const Ranked *)b;
    if (isnan(x->v) != isnan(y->v)) return isnan(x->v) ? 1 : -1;
    if (x->v != y->v && !isnan(x->v)) return x->v < y->v ? -1 : 1;
    return x->i < y->i ? -1 : x->i > y->i;
}

static bool k_topk(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 2 || !in[0] || !in[1]) return ir_fail(r, "an input, K and two outputs are needed");
    const IrTensor *X = in[0];
    if (X->dtype != F32 && !is_int(X->dtype)) return ir_fail(r, "an input of %s", ir_dtype_name(X->dtype));
    uint32_t axis;
    if (!axis_of(r, ir_attr_int(r->node, "axis", -1), X->rank, &axis)) return false;
    const int64_t K = in[1]->count ? ir_get_int(in[1], 0) : -1;
    if (K < 0 || K > X->dims[axis]) return ir_fail(r, "K of %lld for an axis of %lld", (long long)K, (long long)X->dims[axis]);
    int64_t dims[IR_MAX_RANK];
    for (uint32_t d = 0; d < X->rank; d++) dims[d] = d == axis ? K : X->dims[d];
    if (!ir_output(r, out[0], X->dtype, X->rank, dims) || !ir_output(r, out[1], I64, X->rank, dims)) return false;
    const Rows w = rows_of(X, axis, axis + 1);
    Ranked *row = (Ranked *)malloc((size_t)(w.len ? w.len : 1) * sizeof(Ranked));
    if (!row) return ir_fail(r, "out of memory");
    const bool largest = ir_attr_int(r->node, "largest", 1) != 0;
    for (uint64_t q = 0; q < w.outer * w.inner; q++) {
        const uint64_t o = q / w.inner, i = q % w.inner, base = o * w.len * w.inner + i;
        for (uint64_t l = 0; l < w.len; l++) row[l] = (Ranked){ir_get_float(X, base + l * w.inner), (int64_t)l};
        qsort(row, (size_t)w.len, sizeof(Ranked), largest ? ranked_down : ranked_up);
        const uint64_t ob = o * (uint64_t)K * w.inner + i;
        for (int64_t k = 0; k < K; k++) {
            if (X->dtype == F32) fdata(out[0])[ob + (uint64_t)k * w.inner] = (float)row[k].v;
            else put_int(out[0], ob + (uint64_t)k * w.inner, ir_get_int(X, base + (uint64_t)row[k].i * w.inner));
            ((int64_t *)out[1]->data)[ob + (uint64_t)k * w.inner] = row[k].i;
        }
    }
    free(row);
    return true;
}

static bool k_cumsum(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1] || !in[1]->count) return ir_fail(r, "an input, an axis and an output are needed");
    const IrTensor *X = in[0];
    if (X->dtype != F32 && !is_int(X->dtype)) return ir_fail(r, "an input of %s", ir_dtype_name(X->dtype));
    uint32_t axis;
    if (!axis_of(r, ir_get_int(in[1], 0), X->rank, &axis)) return false;
    if (!ir_output(r, out[0], X->dtype, X->rank, X->dims)) return false;
    const bool exclusive = ir_attr_int(r->node, "exclusive", 0) != 0, reverse = ir_attr_int(r->node, "reverse", 0) != 0;
    const bool prod = OP(r) == IR_CumProd;
    const Rows w = rows_of(X, axis, axis + 1);
    for (uint64_t q = 0; q < w.outer * w.inner; q++) {
        const uint64_t base = q / w.inner * w.len * w.inner + q % w.inner;
        float fs = prod ? 1.0f : 0.0f;
        int64_t is = prod ? 1 : 0;
        for (uint64_t k = 0; k < w.len; k++) {
            const uint64_t at = base + (reverse ? w.len - 1 - k : k) * w.inner;
            if (X->dtype == F32) {
                const float v = fdata(X)[at];
                if (!exclusive) fs = prod ? fs * v : fs + v;
                fdata(out[0])[at] = fs;
                if (exclusive) fs = prod ? fs * v : fs + v;
            } else {
                const int64_t v = ir_get_int(X, at);
                if (!exclusive) is = prod ? is * v : is + v;
                put_int(out[0], at, is);
                if (exclusive) is = prod ? is * v : is + v;
            }
        }
    }
    return true;
}

/* ------------------------------------------------------------------------- resizing */

/* Resize: each output axis's taps (input indices and weights) by the mode and the coordinate transformation, as
   onnx.reference computes them; the axes interpolated one inside the other, the last innermost. */
static bool k_resize(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0]) return ir_fail(r, "an input and an output are needed");
    const IrTensor *X = in[0];
    if (X->dtype != F32 || !X->rank) return ir_fail(r, "an input of %s or no axes", ir_dtype_name(X->dtype));
    const uint32_t R = X->rank;
    /* (Upsample, before opset 10, and Resize of opset 10: scales alone, asymmetric coordinates, nearest by floor) */
    const bool v10 = r->g->opset < 11 || OP(r) == IR_Upsample;
    const IrTensor *roi = v10 || n < 2 ? NULL : in[1];
    const IrTensor *scales = v10 ? (n > 1 ? in[1] : NULL) : n > 2 ? in[2] : NULL, *sizes = !v10 && n > 3 ? in[3] : NULL;
    IrTensor attr_scales;
    const IrAttr *sa = ir_attr(r->node, "scales");
    if (!scales && sa && sa->type == ATTR_FLOATS) {
        memset(&attr_scales, 0, sizeof attr_scales);
        attr_scales.dtype = F32;
        attr_scales.rank = 1;
        attr_scales.dims[0] = sa->count;
        attr_scales.count = sa->count;
        attr_scales.data = sa->floats;
        scales = &attr_scales;
    }
    if (roi && !roi->count) roi = NULL;
    if (scales && !scales->count) scales = NULL;
    if (sizes && !sizes->count) sizes = NULL;
    if (!scales == !sizes) return ir_fail(r, "scales or sizes (one of them) are needed");
    const IrAttr *a;
    const char *mode = (a = ir_attr(r->node, "mode")) && a->type == ATTR_STRING ? a->s : "nearest";
    const char *ctm = (a = ir_attr(r->node, "coordinate_transformation_mode")) && a->type == ATTR_STRING ? a->s
                    : v10 ? "asymmetric" : "half_pixel";
    const char *nearest = (a = ir_attr(r->node, "nearest_mode")) && a->type == ATTR_STRING ? a->s
                        : v10 ? "floor" : "round_prefer_floor";
    const char *policy = (a = ir_attr(r->node, "keep_aspect_ratio_policy")) && a->type == ATTR_STRING ? a->s : "stretch";
    const double A = ir_attr_float(r->node, "cubic_coeff_a", -0.75f);
    const bool exclude = ir_attr_int(r->node, "exclude_outside", 0) != 0;
    const float extrapolation = ir_attr_float(r->node, "extrapolation_value", 0.0f);
    const int taps = !strcmp(mode, "nearest") ? 1 : !strcmp(mode, "linear") || !strcmp(mode, "bilinear") ? 2
                   : !strcmp(mode, "cubic") ? 4 : 0;
    if (!taps) return ir_fail(r, "the mode %s", mode);
    int64_t axes[IR_MAX_RANK];
    int64_t na = int_list(r->node, "axes", NULL, axes, IR_MAX_RANK);
    if (na < 0) { na = R; for (uint32_t d = 0; d < R; d++) axes[d] = d; }
    const IrTensor *given = scales ? scales : sizes;
    if (given->count != (uint64_t)na) return ir_fail(r, "%llu scales or sizes for %lld axes",
                                                     (unsigned long long)given->count, (long long)na);
    double scale[IR_MAX_RANK], width[IR_MAX_RANK], roi_lo[IR_MAX_RANK], roi_hi[IR_MAX_RANK];
    int64_t dims[IR_MAX_RANK];
    bool on[IR_MAX_RANK] = {false};
    for (uint32_t d = 0; d < R; d++) { scale[d] = 1.0; dims[d] = X->dims[d]; roi_lo[d] = 0.0; roi_hi[d] = 1.0; }
    for (int64_t k = 0; k < na; k++) {
        uint32_t d;
        if (!axis_of(r, axes[k], R, &d)) return false;
        on[d] = true;
        if (scales) {
            scale[d] = (double)fdata(scales)[k];
            if (!(scale[d] > 0.0)) return ir_fail(r, "a scale of %g", scale[d]);
            dims[d] = (int64_t)(scale[d] * (double)X->dims[d]);
        } else {
            dims[d] = ir_get_int(sizes, (uint64_t)k);
            if (dims[d] < 0) return ir_fail(r, "a size of %lld", (long long)dims[d]);
            scale[d] = X->dims[d] ? (double)dims[d] / (double)X->dims[d] : 1.0;
        }
        if (roi) {
            if (roi->count != 2 * (uint64_t)na && roi->count != 2 * (uint64_t)R) return ir_fail(r, "a roi of another size");
            const uint64_t ra = roi->count == 2 * (uint64_t)na ? (uint64_t)k : d, rn = roi->count / 2;
            roi_lo[d] = ir_get_float(roi, ra);
            roi_hi[d] = ir_get_float(roi, rn + ra);
        }
    }
    if (sizes && strcmp(policy, "stretch")) {
        const bool smaller = !strcmp(policy, "not_larger");
        if (!smaller && strcmp(policy, "not_smaller")) return ir_fail(r, "the policy %s", policy);
        double sc = smaller ? INFINITY : 0.0;
        for (uint32_t d = 0; d < R; d++) if (on[d]) sc = smaller ? (scale[d] < sc ? scale[d] : sc) : (scale[d] > sc ? scale[d] : sc);
        for (uint32_t d = 0; d < R; d++) if (on[d]) { scale[d] = sc; dims[d] = (int64_t)(sc * (double)X->dims[d] + 0.5); }
    }
    for (uint32_t d = 0; d < R; d++) width[d] = scale[d] * (double)X->dims[d];
    if (ir_attr_int(r->node, "antialias", 0) && taps > 1)
        for (uint32_t d = 0; d < R; d++) if (on[d] && scale[d] < 1.0) return ir_fail(r, "antialiased downsampling");
    if (!ir_output(r, out[0], F32, R, dims) || !out[0]->count) return r->error[0] ? false : true;
    for (uint32_t d = 0; d < R; d++) if (!X->dims[d]) return ir_fail(r, "an empty input resized");
    /* the taps: T[d] per output index of axis d (index -1: the extrapolation value) */
    int T[IR_MAX_RANK];
    size_t total = 0;
    for (uint32_t d = 0; d < R; d++) {
        const bool identity = fabs(scale[d] - 1.0) <= 1e-9 && dims[d] == X->dims[d] && fabs(roi_lo[d]) <= 1e-9
                           && fabs(roi_hi[d] - 1.0) <= 1e-9;
        T[d] = identity ? 1 : taps;
        total += (size_t)dims[d] * (size_t)T[d];
    }
    int64_t *tidx = (int64_t *)malloc(total * sizeof(int64_t) + 1);
    float *tw = (float *)malloc(total * sizeof(float) + 1);
    if (!tidx || !tw) { free(tidx); free(tw); return ir_fail(r, "out of memory"); }
    int64_t *ti[IR_MAX_RANK];
    float *twd[IR_MAX_RANK];
    size_t at = 0;
    for (uint32_t d = 0; d < R; d++) {
        ti[d] = tidx + at;
        twd[d] = tw + at;
        at += (size_t)dims[d] * (size_t)T[d];
        const int64_t size = X->dims[d];
        for (int64_t o = 0; o < dims[d]; o++) {
            int64_t *ix = ti[d] + o * T[d];
            float *wx = twd[d] + o * T[d];
            if (T[d] == 1 && taps > 1) { ix[0] = o; wx[0] = 1.0f; continue; }
            const double s = scale[d], wd = width[d], xo = (double)o;
            double x;
            if (!strcmp(ctm, "align_corners")) x = wd == 1.0 ? 0.0 : xo * (double)(size - 1) / (wd - 1.0);
            else if (!strcmp(ctm, "asymmetric")) x = xo / s;
            else if (!strcmp(ctm, "pytorch_half_pixel")) x = wd == 1.0 ? -0.5 : (xo + 0.5) / s - 0.5;
            else if (!strcmp(ctm, "half_pixel")) x = (xo + 0.5) / s - 0.5;
            else if (!strcmp(ctm, "half_pixel_symmetric")) {
                const double adjustment = (double)dims[d] / wd, center = (double)size / 2.0;
                x = center * (1.0 - adjustment) + (xo + 0.5) / s - 0.5;
            } else if (!strcmp(ctm, "tf_crop_and_resize")) {
                x = wd == 1.0 ? (roi_hi[d] - roi_lo[d]) * (double)(size - 1) / 2.0
                              : xo * (roi_hi[d] - roi_lo[d]) * (double)(size - 1) / (wd - 1.0);
                x += roi_lo[d] * (double)(size - 1);
                if (x < 0.0 || x > (double)(size - 1)) {
                    for (int t = 0; t < T[d]; t++) { ix[t] = -1; wx[t] = 0.0f; }
                    continue;
                }
            } else {
                free(tidx);
                free(tw);
                return ir_fail(r, "the coordinate transformation %s", ctm);
            }
            const double fl = floor(x), ratio = x == fl ? 1.0 : x - fl;
            /* the taps' first index: the T nearest indices, the lower preferred */
            const int64_t first = (int64_t)(x == fl ? fl - 1.0 : fl) - (T[d] == 4 ? 1 : 0);
            double c[4];
            if (T[d] == 1) {
                const bool up = !strcmp(nearest, "round_prefer_floor") ? ratio > 0.5 : !strcmp(nearest, "round_prefer_ceil")
                              ? ratio >= 0.5 : !strcmp(nearest, "ceil") ? true : !strcmp(nearest, "floor") ? false : ratio > 0.5;
                const int64_t lo = (int64_t)(x == fl ? fl - 1.0 : fl);
                const int64_t pick = x == fl ? (int64_t)fl : up ? lo + 1 : lo;
                ix[0] = pick < 0 ? 0 : pick >= size ? size - 1 : pick;
                wx[0] = 1.0f;
                continue;
            }
            if (T[d] == 2) { c[0] = 1.0 - ratio; c[1] = ratio; }
            else {
                const double p = ratio, q = 1.0 - ratio;
                c[0] = ((A * (p + 1) - 5 * A) * (p + 1) + 8 * A) * (p + 1) - 4 * A;
                c[1] = ((A + 2) * p - (A + 3)) * p * p + 1;
                c[2] = ((A + 2) * q - (A + 3)) * q * q + 1;
                c[3] = ((A * (q + 1) - 5 * A) * (q + 1) + 8 * A) * (q + 1) - 4 * A;
            }
            if (exclude) {
                double sum = 0.0;
                for (int t = 0; t < T[d]; t++) {
                    if (first + t < 0 || first + t >= size) c[t] = 0.0;
                    sum += c[t];
                }
                for (int t = 0; t < T[d]; t++) c[t] /= sum;
            }
            for (int t = 0; t < T[d]; t++) {
                const int64_t i = first + t;
                ix[t] = i < 0 ? 0 : i >= size ? size - 1 : i;
                wx[t] = (float)c[t];
            }
        }
    }
    int64_t xs[IR_MAX_RANK];
    int64_t st = 1;
    for (uint32_t d = R; d-- > 0;) { xs[d] = st; st *= X->dims[d]; }
    const uint32_t L = R - 1;
    const int64_t row_len = dims[L];
    const uint64_t rows = out[0]->count / (uint64_t)row_len;
    const float *x = fdata(X);
    float *y = fdata(out[0]);
    FOR_ROWS(r, rows, out[0]->count, begin, end,
        for (uint64_t q = begin; q < end; q++) {
            int64_t o[IR_MAX_RANK], c[IR_MAX_RANK] = {0};
            uint64_t t = q;
            for (uint32_t d = L; d-- > 0;) { o[d] = (int64_t)(t % (uint64_t)dims[d]); t /= (uint64_t)dims[d]; }
            float *yr = y + q * (uint64_t)row_len;
            bool outside = false;
            for (;;) {
                int64_t off = 0;
                float wgt = 1.0f;
                for (uint32_t d = 0; d < L; d++) {
                    const int64_t i = ti[d][o[d] * T[d] + c[d]];
                    if (i < 0) outside = true;
                    off += i * xs[d];
                    wgt *= twd[d][o[d] * T[d] + c[d]];
                }
                if (outside) break;
                const float *src = x + off;
                for (int64_t ol = 0; ol < row_len; ol++) {
                    const int64_t *ix = ti[L] + ol * T[L];
                    const float *wx = twd[L] + ol * T[L];
                    float v = 0.0f;
                    if (ix[0] >= 0) for (int k = 0; k < T[L]; k++) v += wx[k] * src[ix[k]];
                    yr[ol] += wgt * v;
                }
                uint32_t d = L;
                while (d-- > 0) {
                    if (++c[d] < T[d]) break;
                    c[d] = 0;
                }
                if (d == UINT32_MAX) break;
            }
            for (int64_t ol = 0; ol < row_len; ol++)
                if (outside || ti[L][ol * T[L]] < 0) yr[ol] = extrapolation;
        }
    );
    free(tidx);
    free(tw);
    return true;
}

/* ------------------------------------------------------------------------- Einsum */

/* Labels: a-z 0..25, A-Z 26..51, the axes an ellipsis stands for 52 on (aligned to the right). */
enum { EINSUM_LABELS = 52 + IR_MAX_RANK };

typedef struct {
    int label[IR_MAX_RANK];
    uint32_t count;
} Term;

/* One term of an equation into labels; the operand's rank (or -1 for the output) sizes its ellipsis. */
static bool einsum_term(IrRun *r, const char *s, size_t len, int64_t rank, uint32_t ellipsis, Term *t) {
    uint32_t letters = 0;
    bool dots = false;
    for (size_t i = 0; i < len; i++) {
        if (s[i] == '.') {
            if (i + 2 >= len || s[i + 1] != '.' || s[i + 2] != '.' || dots) return ir_fail(r, "an ellipsis malformed");
            dots = true;
            i += 2;
        } else if ((s[i] >= 'a' && s[i] <= 'z') || (s[i] >= 'A' && s[i] <= 'Z')) letters++;
        else return ir_fail(r, "the character '%c' in the equation", s[i]);
    }
    const int64_t e = rank < 0 ? (int64_t)ellipsis : rank - (int64_t)letters;
    if (rank >= 0 && (e < 0 || (!dots && e) || e > (int64_t)ellipsis)) return ir_fail(r, "a term that does not fit its operand");
    if (letters + (dots ? (uint32_t)e : 0) > IR_MAX_RANK) return ir_fail(r, "more than eight axes");
    t->count = 0;
    for (size_t i = 0; i < len; i++) {
        if (s[i] == '.') {
            for (int64_t k = 0; k < e; k++) t->label[t->count++] = 52 + (int)(ellipsis - (uint32_t)e + (uint32_t)k);
            i += 2;
        } else t->label[t->count++] = s[i] >= 'a' ? s[i] - 'a' : 26 + s[i] - 'A';
    }
    return true;
}

static bool k_einsum(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    const IrAttr *eq = ir_attr(r->node, "equation");
    if (!eq || eq->type != ATTR_STRING || !n || n > 8 || m < 1) return ir_fail(r, "an equation, inputs and an output are needed");
    char text[256];
    size_t len = 0;
    for (const char *c = eq->s; *c && len < sizeof text - 1; c++) if (*c != ' ') text[len++] = *c;
    text[len] = 0;
    const char *arrow = strstr(text, "->");
    const size_t lhs = arrow ? (size_t)(arrow - text) : len;
    for (uint32_t k = 0; k < n; k++)
        if (!in[k] || in[k]->dtype != F32) return ir_fail(r, "operands not float32");
    /* the ellipsis's rank: the most axes any operand's stands for */
    Term terms[8], ot;
    uint32_t ellipsis = 0;
    {
        size_t start = 0;
        for (uint32_t k = 0; k < n; k++) {
            size_t end = start;
            while (end < lhs && text[end] != ',') end++;
            uint32_t letters = 0;
            for (size_t i = start; i < end; i++) letters += text[i] != '.';
            if (memchr(text + start, '.', end - start) && in[k]->rank >= letters && in[k]->rank - letters > ellipsis)
                ellipsis = in[k]->rank - letters;
            start = end + 1;
        }
    }
    size_t start = 0;
    for (uint32_t k = 0; k < n; k++) {
        size_t end = start;
        while (end < lhs && text[end] != ',') end++;
        if (start > lhs) return ir_fail(r, "fewer terms than operands");
        if (!einsum_term(r, text + start, end - start, in[k]->rank, ellipsis, &terms[k])) return false;
        start = end + 1;
    }
    int64_t size[EINSUM_LABELS];
    uint32_t uses[EINSUM_LABELS] = {0};
    for (int l = 0; l < EINSUM_LABELS; l++) size[l] = -1;
    for (uint32_t k = 0; k < n; k++)
        for (uint32_t a = 0; a < terms[k].count; a++) {
            const int l = terms[k].label[a];
            const int64_t d = in[k]->dims[a];
            uses[l]++;
            if (size[l] < 0 || size[l] == 1) size[l] = d;
            else if (d != size[l] && d != 1) return ir_fail(r, "sizes of a label that differ");
        }
    if (arrow) {
        if (!einsum_term(r, arrow + 2, len - lhs - 2, -1, ellipsis, &ot)) return false;
    } else {
        ot.count = 0;
        for (uint32_t e = 0; e < ellipsis; e++) ot.label[ot.count++] = 52 + (int)e;
        for (int l = 26; l < 52; l++) if (uses[l] == 1 && ot.count < IR_MAX_RANK) ot.label[ot.count++] = l;
        for (int l = 0; l < 26; l++) if (uses[l] == 1 && ot.count < IR_MAX_RANK) ot.label[ot.count++] = l;
    }
    bool in_out[EINSUM_LABELS] = {false};
    int64_t odims[IR_MAX_RANK];
    for (uint32_t a = 0; a < ot.count; a++) {
        const int l = ot.label[a];
        if (size[l] < 0 || in_out[l]) return ir_fail(r, "an output label not in the inputs, or twice");
        in_out[l] = true;
        odims[a] = size[l];
    }
    /* two operands whose labels are each once in them, all in the other or the output, of the same sizes: a product */
    bool product = n == 2;
    for (uint32_t k = 0; k < n && product; k++) {
        bool seen[EINSUM_LABELS] = {false};
        for (uint32_t a = 0; a < terms[k].count; a++) {
            const int l = terms[k].label[a];
            product &= !seen[l] && in[k]->dims[a] == size[l] && (uses[l] == 2 || in_out[l]);
            seen[l] = true;
        }
    }
    if (product) {
        /* A as [batch, a's own, summed], B as [batch, summed, b's own], their product permuted into the output */
        const Term *ta = &terms[0], *tb = &terms[1];
        int pos_a[EINSUM_LABELS], pos_b[EINSUM_LABELS];
        for (int l = 0; l < EINSUM_LABELS; l++) pos_a[l] = pos_b[l] = -1;
        for (uint32_t a = 0; a < ta->count; a++) pos_a[ta->label[a]] = (int)a;
        for (uint32_t a = 0; a < tb->count; a++) pos_b[tb->label[a]] = (int)a;
        int64_t pa[IR_MAX_RANK], pb[IR_MAX_RANK];
        int cl[IR_MAX_RANK];
        uint32_t na = 0, nb = 0, nc = 0;
        int64_t batch = 1, M = 1, N = 1, K = 1;
        int order[IR_MAX_RANK];         /* the product's labels: batch, a's own, b's own */
        uint32_t no = 0;
        for (uint32_t a = 0; a < ot.count; a++) {
            const int l = ot.label[a];
            if (pos_a[l] >= 0 && pos_b[l] >= 0) {
                pa[na++] = pos_a[l]; pb[nb++] = pos_b[l]; batch *= size[l]; order[no++] = l;
            }
        }
        for (uint32_t a = 0; a < ot.count; a++) {
            const int l = ot.label[a];
            if (pos_a[l] >= 0 && pos_b[l] < 0) { pa[na++] = pos_a[l]; M *= size[l]; order[no++] = l; }
        }
        for (uint32_t a = 0; a < ta->count; a++) {
            const int l = ta->label[a];
            if (!in_out[l]) { pa[na++] = (int64_t)a; cl[nc++] = l; K *= size[l]; }
        }
        for (uint32_t c = 0; c < nc; c++) pb[nb++] = pos_b[cl[c]];
        for (uint32_t a = 0; a < ot.count; a++) {
            const int l = ot.label[a];
            if (pos_b[l] >= 0 && pos_a[l] < 0) { pb[nb++] = pos_b[l]; N *= size[l]; order[no++] = l; }
        }
        IrTensor A, B, C;
        memset(&A, 0, sizeof A);
        memset(&B, 0, sizeof B);
        memset(&C, 0, sizeof C);
        bool ok = permute(r, in[0], pa, &A) && permute(r, in[1], pb, &B);
        if (ok) {
            A.rank = 3; A.dims[0] = batch; A.dims[1] = M; A.dims[2] = K;
            B.rank = 3; B.dims[0] = batch; B.dims[1] = K; B.dims[2] = N;
            ok = matmul_into(r, &A, &B, &C);
        }
        if (ok) {
            C.rank = (uint8_t)no;
            for (uint32_t a = 0; a < no; a++) C.dims[a] = size[order[a]];
            int64_t perm[IR_MAX_RANK];
            for (uint32_t a = 0; a < ot.count; a++)
                for (uint32_t b = 0; b < no; b++) if (order[b] == ot.label[a]) perm[a] = b;
            ok = permute(r, &C, perm, out[0]);
            if (ok && !ot.count) out[0]->rank = 0;
        }
        ir_release(r->g, &A);
        ir_release(r->g, &B);
        ir_release(r->g, &C);
        return ok;
    }
    /* otherwise every output value a sum over the other labels, in their order */
    if (!ir_output(r, out[0], F32, ot.count, odims)) return false;
    int summed[EINSUM_LABELS];
    uint32_t ns = 0;
    for (int l = 0; l < EINSUM_LABELS; l++) if (uses[l] && !in_out[l]) summed[ns++] = l;
    int64_t stride[8][EINSUM_LABELS];
    for (uint32_t k = 0; k < n; k++) {
        for (int l = 0; l < EINSUM_LABELS; l++) stride[k][l] = 0;
        int64_t s = 1;
        for (uint32_t a = terms[k].count; a-- > 0;) {
            if (in[k]->dims[a] != 1 || size[terms[k].label[a]] == 1) stride[k][terms[k].label[a]] += s;
            s *= in[k]->dims[a];
        }
    }
    uint64_t sum_count = 1;
    for (uint32_t i = 0; i < ns; i++) sum_count *= (uint64_t)size[summed[i]];
    FOR_PIECES(r, out[0]->count, begin, end,
        for (uint64_t q = begin; q < end; q++) {
            int64_t base[8] = {0};
            uint64_t t = q;
            for (uint32_t a = ot.count; a-- > 0;) {
                const int64_t i = (int64_t)(t % (uint64_t)odims[a]);
                t /= (uint64_t)odims[a];
                for (uint32_t k = 0; k < n; k++) base[k] += i * stride[k][ot.label[a]];
            }
            double acc = 0.0;
            for (uint64_t c = 0; c < sum_count; c++) {
                uint64_t u = c;
                double p = 1.0;
                int64_t off[8];
                for (uint32_t k = 0; k < n; k++) off[k] = base[k];
                for (uint32_t i = ns; i-- > 0;) {
                    const int64_t v = (int64_t)(u % (uint64_t)size[summed[i]]);
                    u /= (uint64_t)size[summed[i]];
                    for (uint32_t k = 0; k < n; k++) off[k] += v * stride[k][summed[i]];
                }
                for (uint32_t k = 0; k < n; k++) p *= fdata(in[k])[off[k]];
                acc += p;
            }
            fdata(out[0])[q] = (float)acc;
        }
    );
    return true;
}

/* ------------------------------------------------------------------------- composite operators */

/* Attention as ONNX defines it (opset 23 and later): queries, keys and values of three axes (heads given by
   attributes) or four, a past to append the keys and values to, a mask (boolean or added) shorter than the keys or
   not, causal (aligned to the end of the past) or sliding-window masks, a soft cap of the scores, grouped heads.
   A block of 64 queries of a head at a time: their scores by a product, masked and normalized row by row, then
   their outputs by a second product, on one thread each. A row no key is allowed to gives zeros. */
static bool k_attention(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 3 || m < 1 || !in[0] || !in[1] || !in[2]) return ir_fail(r, "queries, keys, values and an output are needed");
    const IrTensor *Q = in[0], *K = in[1], *V = in[2], *mask = n > 3 ? in[3] : NULL;
    const IrTensor *pk = n > 4 ? in[4] : NULL, *pv = n > 5 ? in[5] : NULL, *nonpad = n > 6 ? in[6] : NULL;
    if (Q->dtype != F32 || K->dtype != F32 || V->dtype != F32 || (pk && pk->dtype != F32) || (pv && pv->dtype != F32))
        return ir_fail(r, "inputs not float32");
    if (!pk != !pv) return ir_fail(r, "a past of keys without values, or of values without keys");
    if (nonpad && pk) return ir_fail(r, "nonpad_kv_seqlen with a past");
    const uint32_t R = Q->rank;
    if ((R != 3 && R != 4) || K->rank != R || V->rank != R) return ir_fail(r, "inputs of three or four axes, alike");
    int64_t B = Q->dims[0], Hq, Hk, Sq, Skv, D, Dv;
    if (R == 4) {
        Hq = Q->dims[1]; Sq = Q->dims[2]; D = Q->dims[3];
        Hk = K->dims[1]; Skv = K->dims[2]; Dv = V->dims[3];
        if (K->dims[3] != D || V->dims[1] != Hk || V->dims[2] != Skv) return ir_fail(r, "keys or values of other shapes");
    } else {
        Hq = ir_attr_int(r->node, "q_num_heads", 0);
        Hk = ir_attr_int(r->node, "kv_num_heads", 0);
        if (Hq < 1 || Hk < 1 || Q->dims[2] % Hq || K->dims[2] % Hk || V->dims[2] % Hk)
            return ir_fail(r, "inputs of three axes need q_num_heads and kv_num_heads that divide them");
        Sq = Q->dims[1]; D = Q->dims[2] / Hq; Skv = K->dims[1]; Dv = V->dims[2] / Hk;
        if (K->dims[2] / Hk != D || V->dims[1] != Skv) return ir_fail(r, "keys or values of other shapes");
    }
    if (K->dims[0] != B || V->dims[0] != B || Hq % Hk) return ir_fail(r, "batches or heads that do not match");
    int64_t P = 0;
    if (pk) {
        P = pk->dims[2];
        if (pk->rank != 4 || pv->rank != 4 || pk->dims[0] != B || pk->dims[1] != Hk || pk->dims[3] != D || pv->dims[0] != B
            || pv->dims[1] != Hk || pv->dims[2] != P || pv->dims[3] != Dv) return ir_fail(r, "a past of another shape");
    }
    const int64_t T = P + Skv, group = Hq / Hk;
    const IrAttr *sa = ir_attr(r->node, "scale");
    const float scale = sa ? (sa->type == ATTR_FLOAT ? sa->f : (float)sa->i) : 1.0f / sqrtf((float)D);
    const float softcap = ir_attr_float(r->node, "softcap", 0.0f);
    const bool causal = ir_attr_int(r->node, "is_causal", 0) != 0;
    const int64_t mode = ir_attr_int(r->node, "qk_matmul_output_mode", 0);
    const int64_t left = ir_attr_int(r->node, "left_window_size", -1), right = ir_attr_int(r->node, "right_window_size", -1);
    /* the mask over (batch, heads, queries, keys), its last axis as long as the keys or shorter */
    int64_t mdims[4] = {B, Hq, Sq, mask && mask->rank ? mask->dims[mask->rank - 1] : 0}, ms[4] = {0, 0, 0, 0};
    if (mask) {
        if (mask->rank > 4 || !mask->rank || mdims[3] > T || (mask->dtype != F32 && mask->dtype != BOOL))
            return ir_fail(r, "a mask of another shape or type");
        for (uint32_t d = 0; d < mask->rank; d++) {
            const int64_t want = mdims[4 - mask->rank + d];
            if (mask->dims[d] != want && mask->dims[d] != 1) return ir_fail(r, "a mask that does not broadcast");
        }
        broadcast_strides(mask, 4, mdims, ms);
    }
    if (nonpad && (nonpad->count != (uint64_t)B || !is_int(nonpad->dtype))) return ir_fail(r, "nonpad_kv_seqlen of another shape");
    int64_t ydims[4];
    if (R == 4) { ydims[0] = B; ydims[1] = Hq; ydims[2] = Sq; ydims[3] = Dv; }
    else { ydims[0] = B; ydims[1] = Sq; ydims[2] = Hq * Dv; }
    if (!ir_output(r, out[0], F32, R, ydims)) return false;
    /* the keys and values with their past, as (batch, heads, keys, size): the present outputs, or a copy */
    IrTensor kc, vc;
    memset(&kc, 0, sizeof kc);
    memset(&vc, 0, sizeof vc);
    IrTensor *kt = m > 1 && out[1] ? out[1] : &kc, *vt = m > 2 && out[2] ? out[2] : &vc;
    const int64_t kd[4] = {B, Hk, T, D}, vd[4] = {B, Hk, T, Dv};
    bool ok = ir_output(r, kt, F32, 4, kd) && ir_output(r, vt, F32, 4, vd);
    float *qk = NULL;
    if (ok && m > 3 && out[3]) {
        const int64_t qd[4] = {B, Hq, Sq, T};
        ok = ir_output(r, out[3], F32, 4, qd);
        qk = ok ? fdata(out[3]) : NULL;
    }
    if (!ok) {
        if (kt == &kc) ir_release(r->g, &kc);
        if (vt == &vc) ir_release(r->g, &vc);
        return false;
    }
    for (int64_t b = 0; b < B; b++)
        for (int64_t h = 0; h < Hk; h++)
            for (int64_t t = 0; t < T; t++) {
                float *kr = fdata(kt) + ((b * Hk + h) * T + t) * D, *vr = fdata(vt) + ((b * Hk + h) * T + t) * Dv;
                if (t < P) {
                    memcpy(kr, fdata(pk) + ((b * Hk + h) * P + t) * D, (size_t)D * 4);
                    memcpy(vr, fdata(pv) + ((b * Hk + h) * P + t) * Dv, (size_t)Dv * 4);
                } else if (R == 4) {
                    memcpy(kr, fdata(K) + ((b * Hk + h) * Skv + t - P) * D, (size_t)D * 4);
                    memcpy(vr, fdata(V) + ((b * Hk + h) * Skv + t - P) * Dv, (size_t)Dv * 4);
                } else {
                    memcpy(kr, fdata(K) + (b * Skv + t - P) * Hk * D + h * D, (size_t)D * 4);
                    memcpy(vr, fdata(V) + (b * Skv + t - P) * Hk * Dv + h * Dv, (size_t)Dv * 4);
                }
            }
    const int64_t BLOCK = 64, blocks = (Sq + BLOCK - 1) / BLOCK, items = B * Hq * blocks;
    SpingalettGemmScratch *main = ir_scratch(r);
    if (!main) return false;
    const bool across = r->parallel && items > 1 && (uint64_t)B * Hq * Sq * T * (D + Dv) >= PARALLEL_WORK;
    bool failed = false;
    SPINGALETT_PARALLEL_FOR(across,
        for (int64_t it = 0; it < items; it++) {
            const int64_t blk = it % blocks, h = it / blocks % Hq, b = it / blocks / Hq;
            const int64_t q0 = blk * BLOCK, rows = Sq - q0 < BLOCK ? Sq - q0 : BLOCK, hk = h / group;
            float *s = (float *)spingalett_aligned_alloc((size_t)(rows * T) * sizeof(float) + 64);
            if (!s) { failed = true; continue; }
            SpingalettGemmScratch *scr = across ? ir_thread_scratch(r, spingalett_thread_num()) : main;
            const float *q = R == 4 ? fdata(Q) + ((b * Hq + h) * Sq + q0) * D : fdata(Q) + (b * Sq + q0) * Hq * D + h * D;
            const size_t ldq = R == 4 ? (size_t)D : (size_t)(Hq * D);
            spingalett_gemm_native(scr, false, true, (uint32_t)rows, (uint32_t)T, (uint32_t)D, scale, q, ldq,
                                   fdata(kt) + (b * Hk + hk) * T * D, (size_t)D, 0.0f, s, (size_t)T, !across && r->parallel);
            const int64_t offset = nonpad ? ir_get_int(nonpad, (uint64_t)b) - Sq : P;
            const int64_t valid = nonpad ? ir_get_int(nonpad, (uint64_t)b) : T;
            for (int64_t i = 0; i < rows; i++) {
                float *row = s + i * T;
                float *qrow = qk ? qk + ((b * Hq + h) * Sq + q0 + i) * T : NULL;
                const int64_t qi = q0 + i;
                bool any = false;
                float mx = -INFINITY;
                for (int64_t j = 0; j < T; j++) {
                    float v = row[j];
                    if (softcap > 0.0f) v = softcap * tanhf(v / softcap);
                    if (qrow && mode < 2) qrow[j] = v;
                    float bias = 0.0f;
                    if (mask) {
                        if (j >= mdims[3]) bias = -INFINITY;
                        else {
                            const int64_t at = b * ms[0] + h * ms[1] + qi * ms[2] + j * ms[3];
                            bias = mask->dtype == BOOL ? (((const uint8_t *)mask->data)[at] ? 0.0f : -INFINITY) : fdata(mask)[at];
                        }
                    }
                    const int64_t diff = qi + offset - j;
                    if ((causal && diff < 0) || (left >= 0 && diff > left) || (right >= 0 && -diff > right) || j >= valid)
                        bias = -INFINITY;
                    any |= bias != -INFINITY;
                    v += bias;
                    if (qrow && mode == 2) qrow[j] = v;
                    row[j] = v;
                    mx = v > mx || isnan(v) ? v : mx;
                }
                if (!any) {
                    memset(row, 0, (size_t)T * sizeof(float));
                    if (qrow && mode == 3) memset(qrow, 0, (size_t)T * sizeof(float));
                    continue;
                }
                double sum = 0.0;
                for (int64_t j = 0; j < T; j++) { row[j] = expf(row[j] - (isinf(mx) ? 0.0f : mx)); sum += row[j]; }
                const float inv = sum > 0.0 ? (float)(1.0 / sum) : 0.0f;
                for (int64_t j = 0; j < T; j++) row[j] *= inv;
                if (qrow && mode == 3) memcpy(qrow, row, (size_t)T * sizeof(float));
            }
            float *y = R == 4 ? fdata(out[0]) + ((b * Hq + h) * Sq + q0) * Dv : fdata(out[0]) + (b * Sq + q0) * Hq * Dv + h * Dv;
            spingalett_gemm_native(scr, false, false, (uint32_t)rows, (uint32_t)Dv, (uint32_t)T, 1.0f, s, (size_t)T,
                                   fdata(vt) + (b * Hk + hk) * T * Dv, (size_t)Dv, 0.0f, y,
                                   R == 4 ? (size_t)Dv : (size_t)(Hq * Dv), !across && r->parallel);
            spingalett_aligned_free(s);
        }
    );
    if (kt == &kc) ir_release(r->g, &kc);
    if (vt == &vc) ir_release(r->g, &vc);
    return failed ? ir_fail(r, "out of memory for the scores") : true;
}

/* RotaryEmbedding as ONNX defines it (opset 23): inputs of four axes (batch, heads, sequence, head) or three (heads
   by num_heads), the caches indexed by positions or given per position, the rotation of the first
   rotary_embedding_dim values of each head, by halves or interleaved. */
static bool k_rotary(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 3 || m < 1 || !in[0] || !in[1] || !in[2]) return ir_fail(r, "an input, caches and an output are needed");
    const IrTensor *X = in[0], *C = in[1], *S = in[2], *pos = n > 3 ? in[3] : NULL;
    if (X->dtype != F32 || C->dtype != F32 || S->dtype != F32) return ir_fail(r, "inputs not float32");
    const bool interleaved = ir_attr_int(r->node, "interleaved", 0) != 0;
    int64_t B, H, Sl, D;
    if (X->rank == 4) { B = X->dims[0]; H = X->dims[1]; Sl = X->dims[2]; D = X->dims[3]; }
    else if (X->rank == 3) {
        H = ir_attr_int(r->node, "num_heads", 0);
        if (H < 1 || X->dims[2] % H) return ir_fail(r, "an input of three axes needs num_heads that divides it");
        B = X->dims[0]; Sl = X->dims[1]; D = X->dims[2] / H;
    } else return ir_fail(r, "an input of %u axes", X->rank);
    const int64_t rot = ir_attr_int(r->node, "rotary_embedding_dim", 0) ? ir_attr_int(r->node, "rotary_embedding_dim", 0) : D;
    const int64_t half = rot / 2;
    if (rot > D || rot % 2) return ir_fail(r, "a rotary_embedding_dim of %lld for heads of %lld", (long long)rot, (long long)D);
    if (!C->rank || C->dims[C->rank - 1] != half || S->count != C->count) return ir_fail(r, "caches of another size");
    if (pos && (pos->count != (uint64_t)(B * Sl) && pos->count != (uint64_t)Sl)) return ir_fail(r, "positions of another shape");
    if (!pos && C->count != (uint64_t)(Sl * half) && C->count != (uint64_t)(B * Sl * half)) return ir_fail(r, "caches of another shape");
    const uint64_t cache_rows = C->count / (uint64_t)(half ? half : 1);
    if (!ir_output(r, out[0], F32, X->rank, X->dims)) return false;
    bool bad = false;
    for (int64_t b = 0; b < B; b++)
        for (int64_t t = 0; t < Sl; t++) {
            int64_t at = pos ? ir_get_int(pos, (uint64_t)(pos->count == (uint64_t)Sl ? t : b * Sl + t))
                             : C->count == (uint64_t)(Sl * half) ? t : b * Sl + t;
            if (at < 0 || (uint64_t)at >= cache_rows) { bad = true; at = 0; }
            const float *c = fdata(C) + at * half, *s = fdata(S) + at * half;
            for (int64_t h = 0; h < H; h++) {
                const size_t o = X->rank == 4 ? (size_t)(((b * H + h) * Sl + t) * D) : (size_t)((b * Sl + t) * H * D + h * D);
                const float *x = fdata(X) + o;
                float *y = fdata(out[0]) + o;
                for (int64_t i = 0; i < half; i++) {
                    const int64_t i1 = interleaved ? 2 * i : i, i2 = interleaved ? 2 * i + 1 : i + half;
                    const float x1 = x[i1], x2 = x[i2];
                    y[i1] = c[i] * x1 - s[i] * x2;
                    y[i2] = s[i] * x1 + c[i] * x2;
                }
                for (int64_t i = rot; i < D; i++) y[i] = x[i];
            }
        }
    return bad ? ir_fail(r, "a position beyond the caches") : true;
}

/* ------------------------------------------------------------------------- more element-wise operators */

static bool k_bitwise(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "two inputs and an output are needed");
    IrTensor *a = in[0], *b = in[1];
    if (!is_int(a->dtype) || a->dtype != b->dtype) return ir_fail(r, "inputs of %s and %s", ir_dtype_name(a->dtype),
                                                                  ir_dtype_name(b->dtype));
    uint32_t R;
    int64_t dims[IR_MAX_RANK];
    IrTensor *pair[2] = {a, b};
    if (!broadcast_shape(r, pair, 2, &R, dims) || !ir_output(r, out[0], a->dtype, R, dims)) return false;
    const IrOp op = OP(r);
    const IrAttr *dir = ir_attr(r->node, "direction");
    const bool left = dir && dir->type == ATTR_STRING && !strcmp(dir->s, "LEFT");
    const unsigned bits = (unsigned)ir_dtype_size(a->dtype) * 8u;
    Walk w;
    walk_init(&w, pair, 2, R, dims);
    int64_t idx[IR_MAX_RANK], off[3];
    if (out[0]->count) walk_offsets(&w, 0, idx, off);
    for (uint64_t i = 0; i < out[0]->count; i++) {
        const int64_t x = ir_get_int(a, (uint64_t)off[0]), y = ir_get_int(b, (uint64_t)off[1]);
        int64_t v;
        if (op == IR_BitwiseAnd) v = x & y;
        else if (op == IR_BitwiseOr) v = x | y;
        else if (op == IR_BitwiseXor) v = x ^ y;
        else if (y < 0 || (uint64_t)y >= bits) {
            v = left || x >= 0 ? 0 : -1;    /* (a shift by the width or more: what shifting bit by bit would leave) */
        } else {
            v = left ? (int64_t)((uint64_t)x << y) : x >> y;     /* (signed values shift arithmetically) */
        }
        put_int(out[0], i, v);
        walk_step(&w, idx, off);
    }
    return true;
}

/* SwiGLU (opset 28): Swish of the gate times the value. */
static bool k_swiglu(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "a gate, a value and an output are needed");
    const IrTensor *a = in[0], *b = in[1];
    if (a->dtype != F32 || b->dtype != F32 || a->count != b->count || a->rank != b->rank)
        return ir_fail(r, "a gate and a value of other shapes or types");
    if (!ir_output(r, out[0], F32, a->rank, a->dims)) return false;
    const float alpha = ir_attr_float(r->node, "alpha", 1.0f);
    const float *x = fdata(a), *v = fdata(b);
    float *y = fdata(out[0]);
    FOR_PIECES(r, a->count, begin, end,
        for (uint64_t i = begin; i < end; i++) y[i] = x[i] / (1.0f + expf(-alpha * x[i])) * v[i];
    );
    return true;
}

/* ------------------------------------------------------------------------- more normalizations */

/* LpNormalization: each row along the axis divided by its L1 or L2 norm (a row of zeros stays zeros). */
static bool k_lp_norm(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0] || in[0]->dtype != F32) return ir_fail(r, "an input of float32 and an output are needed");
    const IrTensor *X = in[0];
    uint32_t axis;
    if (!axis_of(r, ir_attr_int(r->node, "axis", -1), X->rank, &axis)) return false;
    const int64_t p = ir_attr_int(r->node, "p", 2);
    if (p != 1 && p != 2) return ir_fail(r, "p of %lld", (long long)p);
    if (!ir_output(r, out[0], F32, X->rank, X->dims)) return false;
    const Rows w = rows_of(X, axis, axis + 1);
    for (uint64_t q = 0; q < w.outer * w.inner; q++) {
        const uint64_t base = q / w.inner * w.len * w.inner + q % w.inner;
        double sum = 0.0;
        for (uint64_t l = 0; l < w.len; l++) {
            const double v = fdata(X)[base + l * w.inner];
            sum += p == 1 ? fabs(v) : v * v;
        }
        const float norm = (float)(p == 1 ? sum : sqrt(sum));
        for (uint64_t l = 0; l < w.len; l++)
            fdata(out[0])[base + l * w.inner] = norm == 0.0f ? 0.0f : fdata(X)[base + l * w.inner] / norm;
    }
    return true;
}

/* MeanVarianceNormalization: (x - mean) / sqrt(var) over the axes (by default the batch's and the spatial ones), as
   ONNX's function computes it (the variance as E[x^2] - E[x]^2, 1e-9 added to its root). */
static bool k_mvn(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0] || in[0]->dtype != F32) return ir_fail(r, "an input of float32 and an output are needed");
    const IrTensor *X = in[0];
    int64_t axes[IR_MAX_RANK] = {0, 2, 3};
    int64_t na = int_list(r->node, "axes", NULL, axes, IR_MAX_RANK);
    if (na < 0) na = X->rank >= 4 ? 3 : 0;
    bool red[IR_MAX_RANK] = {false};
    for (int64_t k = 0; k < na; k++) {
        uint32_t a;
        if (!axis_of(r, axes[k], X->rank, &a)) return false;
        red[a] = true;
    }
    if (!ir_output(r, out[0], F32, X->rank, X->dims)) return false;
    /* each kept index's group: the values over the reduced axes, visited by an odometer */
    int64_t xs[IR_MAX_RANK], s = 1;
    for (uint32_t d = X->rank; d-- > 0;) { xs[d] = s; s *= X->dims[d]; }
    uint64_t groups = 1, per = 1;
    for (uint32_t d = 0; d < X->rank; d++) {
        if (red[d]) per *= (uint64_t)X->dims[d];
        else groups *= (uint64_t)X->dims[d];
    }
    for (uint64_t gi = 0; gi < groups; gi++) {
        int64_t base = 0;
        uint64_t t = gi;
        for (uint32_t d = X->rank; d-- > 0;)
            if (!red[d]) { base += (int64_t)(t % (uint64_t)X->dims[d]) * xs[d]; t /= (uint64_t)X->dims[d]; }
        double sum = 0.0, sq = 0.0;
        for (int pass = 0; pass < 2; pass++) {
            const float mean = (float)(sum / (double)(per ? per : 1));
            const float root = sqrtf((float)(sq / (double)(per ? per : 1)) - mean * mean) + 1e-9f;
            for (uint64_t j = 0; j < per; j++) {
                uint64_t u = j;
                int64_t off = base;
                for (uint32_t d = X->rank; d-- > 0;)
                    if (red[d]) { off += (int64_t)(u % (uint64_t)X->dims[d]) * xs[d]; u /= (uint64_t)X->dims[d]; }
                const float v = fdata(X)[off];
                if (!pass) { sum += v; sq += (double)v * v; }
                else fdata(out[0])[off] = (v - mean) / root;
            }
        }
    }
    return true;
}

/* LRN: each value over (bias + alpha / size * the sum of squares of the channels around it)^beta. */
static bool k_lrn(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0] || in[0]->dtype != F32 || in[0]->rank < 3) return ir_fail(r, "an input of float32 and an output are needed");
    const IrTensor *X = in[0];
    const int64_t size = ir_attr_int(r->node, "size", 0);
    if (size < 1) return ir_fail(r, "a size of %lld", (long long)size);
    const float alpha = ir_attr_float(r->node, "alpha", 1e-4f), beta = ir_attr_float(r->node, "beta", 0.75f);
    const float bias = ir_attr_float(r->node, "bias", 1.0f);
    if (!ir_output(r, out[0], F32, X->rank, X->dims)) return false;
    const int64_t N = X->dims[0], C = X->dims[1], pix = (int64_t)(X->count / (uint64_t)(N && C ? N * C : 1));
    const int64_t c1 = (size - 1) / 2, c2 = size / 2 + 1;
    for (int64_t b = 0; b < N; b++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t p = 0; p < pix; p++) {
                double sum = 0.0;
                for (int64_t k = c - c1 < 0 ? 0 : c - c1; k < (c + c2 < C ? c + c2 : C); k++) {
                    const double v = fdata(X)[(b * C + k) * pix + p];
                    sum += v * v;
                }
                const size_t at = (size_t)((b * C + c) * pix + p);
                fdata(out[0])[at] = fdata(X)[at] / powf(bias + alpha / (float)size * (float)sum, beta);
            }
    return true;
}

/* ------------------------------------------------------------------------- more shapes and data */

static bool k_eye_like(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0] || in[0]->rank != 2) return ir_fail(r, "a matrix and an output are needed");
    const uint8_t dtype = (uint8_t)ir_attr_int(r->node, "dtype", in[0]->dtype);
    const uint8_t type = dtype == 11 ? F32 : dtype;      /* (double computes as float) */
    if (!ir_dtype_size(type)) return ir_fail(r, "an output of ONNX's type %u", dtype);
    if (!ir_output(r, out[0], type, 2, in[0]->dims)) return false;
    const int64_t k = ir_attr_int(r->node, "k", 0), rows = in[0]->dims[0], cols = in[0]->dims[1];
    for (int64_t i = 0; i < rows; i++)
        if (i + k >= 0 && i + k < cols) put_int(out[0], (uint64_t)(i * cols + i + k), 1);
    return true;
}

/* Compress: the slices along the axis (or the flattened values) whose condition is true. */
static bool k_compress(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "an input, a condition and an output are needed");
    const IrTensor *X = in[0], *c = in[1];
    const IrAttr *aa = ir_attr(r->node, "axis");
    const size_t size = ir_dtype_size(X->dtype);
    if (!aa) {
        uint64_t kept = 0;
        for (uint64_t i = 0; i < c->count && i < X->count; i++) kept += ir_get_int(c, i) != 0;
        const int64_t d = (int64_t)kept;
        if (!ir_output(r, out[0], X->dtype, 1, &d)) return false;
        for (uint64_t i = 0, j = 0; i < c->count && i < X->count; i++)
            if (ir_get_int(c, i)) memcpy((uint8_t *)out[0]->data + size * j++, (const uint8_t *)X->data + size * i, size);
        return true;
    }
    uint32_t axis;
    if (!axis_of(r, aa->i, X->rank, &axis)) return false;
    int64_t dims[IR_MAX_RANK];
    for (uint32_t d = 0; d < X->rank; d++) dims[d] = X->dims[d];
    dims[axis] = 0;
    for (uint64_t i = 0; i < c->count && i < (uint64_t)X->dims[axis]; i++) dims[axis] += ir_get_int(c, i) != 0;
    if (!ir_output(r, out[0], X->dtype, X->rank, dims)) return false;
    const Rows w = rows_of(X, axis, axis + 1);
    uint8_t *o = (uint8_t *)out[0]->data;
    for (uint64_t q = 0; q < w.outer; q++)
        for (uint64_t l = 0; l < w.len; l++)
            if (l < c->count && ir_get_int(c, l)) {
                memcpy(o, (const uint8_t *)X->data + (q * w.len + l) * w.inner * size, (size_t)(w.inner * size));
                o += w.inner * size;
            }
    return true;
}

static bool k_reverse_sequence(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1] || in[0]->rank < 2) return ir_fail(r, "an input, lengths and an output are needed");
    const IrTensor *X = in[0], *lens = in[1];
    uint32_t ba, ta;
    if (!axis_of(r, ir_attr_int(r->node, "batch_axis", 1), X->rank, &ba) ||
        !axis_of(r, ir_attr_int(r->node, "time_axis", 0), X->rank, &ta)) return false;
    if (ba == ta || lens->count != (uint64_t)X->dims[ba]) return ir_fail(r, "axes or lengths that do not fit");
    if (!ir_output(r, out[0], X->dtype, X->rank, X->dims)) return false;
    const size_t size = ir_dtype_size(X->dtype);
    int64_t xs[IR_MAX_RANK], s = 1;
    for (uint32_t d = X->rank; d-- > 0;) { xs[d] = s; s *= X->dims[d]; }
    for (uint64_t i = 0; i < X->count; i++) {
        uint64_t t = i;
        int64_t c[IR_MAX_RANK];
        for (uint32_t d = X->rank; d-- > 0;) { c[d] = (int64_t)(t % (uint64_t)X->dims[d]); t /= (uint64_t)X->dims[d]; }
        const int64_t len = ir_get_int(lens, (uint64_t)c[ba]);
        int64_t src = (int64_t)i;
        if (c[ta] < len) src += (len - 1 - 2 * c[ta]) * xs[ta];
        memcpy((uint8_t *)out[0]->data + i * size, (const uint8_t *)X->data + (uint64_t)src * size, size);
    }
    return true;
}

/* The windows of signal processing: Hann, Hamming and Blackman, periodic or symmetric. */
static bool k_window(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 1 || !in[0] || !in[0]->count) return ir_fail(r, "a size and an output are needed");
    const int64_t size = ir_get_int(in[0], 0);
    if (size < 0) return ir_fail(r, "a size of %lld", (long long)size);
    const uint8_t dtype = (uint8_t)ir_attr_int(r->node, "output_datatype", F32);
    const uint8_t type = dtype == 11 || dtype == SPINGALETT_DTYPE_FLOAT16 || dtype == SPINGALETT_DTYPE_BFLOAT16 ? F32 : dtype;
    if (!ir_dtype_size(type)) return ir_fail(r, "an output of ONNX's type %u", dtype);
    if (!ir_output(r, out[0], type, 1, &size)) return false;
    const double N = ir_attr_int(r->node, "periodic", 1) ? (double)size : (double)size - 1.0, pi = 3.14159265358979323846;
    for (int64_t i = 0; i < size; i++) {
        const double x = (double)i;
        double v;
        if (OP(r) == IR_HannWindow) { const double sn = sin(x * pi / N); v = sn * sn; }
        else if (OP(r) == IR_HammingWindow) v = 25.0 / 46.0 - cos(x * pi * 2.0 / N) * (1.0 - 25.0 / 46.0);
        else v = cos(x * pi * 2.0 / N) * -0.5 + cos(x * pi * 4.0 / N) * 0.08 + 0.42;
        if (type == F32) fdata(out[0])[i] = (float)v;
        else put_int(out[0], (uint64_t)i, (int64_t)v);
    }
    return true;
}

/* ------------------------------------------------------------------------- quantization */

/* The scale's (or zero point's) index for value i of x: one for all, one a slice along the axis, or one a block of
   the axis. */
static uint64_t quant_index(const IrTensor *x, const IrTensor *p, uint32_t axis, int64_t block, uint64_t i) {
    if (p->count <= 1) return 0;
    uint64_t t = i, at = 0, stride = 1;
    for (uint32_t d = x->rank; d-- > 0;) {
        const uint64_t c = t % (uint64_t)x->dims[d];
        t /= (uint64_t)x->dims[d];
        if (p->rank == x->rank && block > 0) {
            /* blocked: the scale has every axis, the axis's in blocks */
            at += (d == axis ? c / (uint64_t)block : c) * stride;
            stride *= (uint64_t)p->dims[d];
        } else if (d == axis) {
            return c;
        }
    }
    return at;
}

static bool quant_axis(IrRun *r, const IrTensor *x, const IrTensor *scale, uint32_t *axis, int64_t *block) {
    *axis = 0;
    *block = ir_attr_int(r->node, "block_size", 0);
    if (scale->count > 1) {
        if (!axis_of(r, ir_attr_int(r->node, "axis", 1), x->rank ? x->rank : 1, axis)) return false;
        if (*block > 0) {
            if (scale->rank != x->rank) return ir_fail(r, "a blocked scale of another rank");
            for (uint32_t d = 0; d < x->rank; d++) {
                const int64_t want = d == *axis ? (x->dims[d] + *block - 1) / *block : x->dims[d];
                if (scale->dims[d] != want) return ir_fail(r, "a blocked scale of another shape");
            }
        } else if (scale->rank != 1 || scale->dims[0] != x->dims[*axis]) {
            return ir_fail(r, "a scale of %llu values for an axis of %lld", (unsigned long long)scale->count,
                           (long long)x->dims[*axis]);
        }
    }
    return true;
}

static bool k_quantize(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "an input, a scale and an output are needed");
    const IrTensor *x = in[0], *scale = in[1], *zp = n > 2 ? in[2] : NULL;
    if (x->dtype != F32 || scale->dtype != F32) return ir_fail(r, "an input or scale not float32");
    const uint8_t type = zp ? zp->dtype : (uint8_t)ir_attr_int(r->node, "output_dtype", SPINGALETT_DTYPE_UINT8);
    if (type != SPINGALETT_DTYPE_UINT8 && type != SPINGALETT_DTYPE_INT8 && type != I32)
        return ir_fail(r, "quantization to ONNX's type %u", type);
    uint32_t axis;
    int64_t block;
    if (!quant_axis(r, x, scale, &axis, &block)) return false;
    if (zp && zp->count != scale->count) return ir_fail(r, "a zero point of another shape than the scale");
    if (!ir_output(r, out[0], type, x->rank, x->dims)) return false;
    const int64_t lo = type == SPINGALETT_DTYPE_UINT8 ? 0 : type == SPINGALETT_DTYPE_INT8 ? -128 : INT32_MIN;
    const int64_t hi = type == SPINGALETT_DTYPE_UINT8 ? 255 : type == SPINGALETT_DTYPE_INT8 ? 127 : INT32_MAX;
    for (uint64_t i = 0; i < x->count; i++) {
        const uint64_t q = quant_index(x, scale, axis, block, i);
        const float v = nearbyintf(fdata(x)[i] / fdata(scale)[q]);
        int64_t y = (isnan(v) ? 0 : v > 4e18f ? INT64_MAX / 2 : v < -4e18f ? INT64_MIN / 2 : (int64_t)v) + (zp ? ir_get_int(zp, q) : 0);
        put_int(out[0], i, y < lo ? lo : y > hi ? hi : y);
    }
    return true;
}

static bool k_dequantize(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "an input, a scale and an output are needed");
    const IrTensor *x = in[0], *scale = in[1], *zp = n > 2 ? in[2] : NULL;
    if (!is_int(x->dtype) || scale->dtype != F32) return ir_fail(r, "an input of %s or a scale not float32", ir_dtype_name(x->dtype));
    uint32_t axis;
    int64_t block;
    if (!quant_axis(r, x, scale, &axis, &block)) return false;
    if (zp && zp->count != scale->count) return ir_fail(r, "a zero point of another shape than the scale");
    if (!ir_output(r, out[0], F32, x->rank, x->dims)) return false;
    FOR_PIECES(r, x->count, begin, end,
        for (uint64_t i = begin; i < end; i++) {
            const uint64_t q = quant_index(x, scale, axis, block, i);
            fdata(out[0])[i] = (float)(ir_get_int(x, i) - (zp ? ir_get_int(zp, q) : 0)) * fdata(scale)[q];
        }
    );
    return true;
}

/* DynamicQuantizeLinear: uint8 values, their scale and zero point from the input's range (which holds 0). */
static bool k_dynamic_quantize(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 1 || m < 3 || !in[0] || in[0]->dtype != F32 || !out[1] || !out[2]) return ir_fail(r, "an input of float32 and three outputs are needed");
    const IrTensor *x = in[0];
    float lo = 0.0f, hi = 0.0f;
    for (uint64_t i = 0; i < x->count; i++) {
        lo = fdata(x)[i] < lo ? fdata(x)[i] : lo;
        hi = fdata(x)[i] > hi ? fdata(x)[i] : hi;
    }
    const float scale = (hi - lo) / 255.0f;
    float zf = scale > 0.0f ? nearbyintf(0.0f - lo / scale) : 0.0f;
    zf = zf < 0.0f ? 0.0f : zf > 255.0f ? 255.0f : zf;
    if (!ir_output(r, out[0], SPINGALETT_DTYPE_UINT8, x->rank, x->dims) || !ir_output(r, out[1], F32, 0, NULL) ||
        !ir_output(r, out[2], SPINGALETT_DTYPE_UINT8, 0, NULL)) return false;
    fdata(out[1])[0] = scale;
    ((uint8_t *)out[2]->data)[0] = (uint8_t)zf;
    for (uint64_t i = 0; i < x->count; i++) {
        float v = scale > 0.0f ? nearbyintf(fdata(x)[i] / scale) + zf : zf;
        v = v < 0.0f ? 0.0f : v > 255.0f ? 255.0f : v;
        ((uint8_t *)out[0]->data)[i] = (uint8_t)v;
    }
    return true;
}

/* MatMulInteger (and QLinearMatMul, its result requantized): products of 8-bit integers less their zero points (one
   for all, or one a row of A and a column of B), summed exactly in 32 bits, over broadcast batches. */
static bool k_matmul_integer(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    const bool qlinear = OP(r) == IR_QLinearMatMul;
    const IrTensor *A = in[0], *B = qlinear ? (n > 3 ? in[3] : NULL) : (n > 1 ? in[1] : NULL);
    const IrTensor *azp = qlinear ? (n > 2 ? in[2] : NULL) : n > 2 ? in[2] : NULL;
    const IrTensor *bzp = qlinear ? (n > 5 ? in[5] : NULL) : n > 3 ? in[3] : NULL;
    if (!A || !B || m < 1) return ir_fail(r, "two operands and an output are needed");
    if (qlinear && (n < 7 || !in[1] || !in[4] || !in[6])) return ir_fail(r, "scales are needed");
    if (!is_int(A->dtype) || !is_int(B->dtype) || A->rank < 1 || B->rank < 1) return ir_fail(r, "operands not integers");
    const int64_t M = A->rank >= 2 ? A->dims[A->rank - 2] : 1, K = A->dims[A->rank - 1];
    const int64_t Kb = B->rank >= 2 ? B->dims[B->rank - 2] : B->dims[0], N = B->rank >= 2 ? B->dims[B->rank - 1] : 1;
    if (K != Kb) return ir_fail(r, "operands of %lld and %lld columns and rows", (long long)K, (long long)Kb);
    if (azp && azp->count != 1 && azp->count != (uint64_t)M) return ir_fail(r, "A's zero point of another size");
    if (bzp && bzp->count != 1 && bzp->count != (uint64_t)N) return ir_fail(r, "B's zero point of another size");
    IrTensor ba, bb;
    memset(&ba, 0, sizeof ba);
    memset(&bb, 0, sizeof bb);
    ba.rank = A->rank >= 2 ? A->rank - 2 : 0;
    bb.rank = B->rank >= 2 ? B->rank - 2 : 0;
    for (uint32_t d = 0; d < ba.rank; d++) ba.dims[d] = A->dims[d];
    for (uint32_t d = 0; d < bb.rank; d++) bb.dims[d] = B->dims[d];
    IrTensor *pair[2] = {&ba, &bb};
    uint32_t R;
    int64_t dims[IR_MAX_RANK];
    if (!broadcast_shape(r, pair, 2, &R, dims)) return false;
    uint64_t batches = 1;
    for (uint32_t d = 0; d < R; d++) batches *= (uint64_t)dims[d];
    int64_t odims[IR_MAX_RANK];
    uint32_t orank = R;
    for (uint32_t d = 0; d < R; d++) odims[d] = dims[d];
    if (A->rank >= 2) odims[orank++] = M;
    if (B->rank >= 2) odims[orank++] = N;
    const uint8_t type = qlinear ? (n > 7 && in[7] ? in[7]->dtype : A->dtype) : I32;
    if (orank > IR_MAX_RANK || !ir_output(r, out[0], type, orank, odims)) return r->error[0] ? false : ir_fail(r, "more than eight axes");
    int64_t sa[IR_MAX_RANK], sb[IR_MAX_RANK];
    broadcast_strides(&ba, R, dims, sa);
    broadcast_strides(&bb, R, dims, sb);
    float mult = 1.0f;
    int64_t yzp = 0, lo = INT32_MIN, hi = INT32_MAX;
    if (qlinear) {
        mult = fdata(in[1])[0] * fdata(in[4])[0] / fdata(in[6])[0];
        yzp = n > 7 && in[7] ? ir_get_int(in[7], 0) : 0;
        lo = type == SPINGALETT_DTYPE_UINT8 ? 0 : -128;
        hi = type == SPINGALETT_DTYPE_UINT8 ? 255 : 127;
    }
    SPINGALETT_PARALLEL_FOR(r->parallel && batches * (uint64_t)(M * N * K) >= PARALLEL_WORK,
        for (int64_t q = 0; q < (int64_t)(batches * (uint64_t)M); q++) {
            const uint64_t bt = (uint64_t)q / (uint64_t)M, i = (uint64_t)q % (uint64_t)M;
            int64_t oa = 0, ob = 0;
            uint64_t t = bt;
            for (uint32_t d = R; d-- > 0;) {
                const int64_t c = (int64_t)(t % (uint64_t)dims[d]);
                t /= (uint64_t)dims[d];
                oa += c * sa[d];
                ob += c * sb[d];
            }
            const uint64_t abase = (uint64_t)oa * (uint64_t)(M * K) + i * (uint64_t)K, bbase = (uint64_t)ob * (uint64_t)(K * N);
            const int64_t za = azp ? ir_get_int(azp, azp->count == 1 ? 0 : i) : 0;
            for (int64_t j = 0; j < N; j++) {
                const int64_t zb = bzp ? ir_get_int(bzp, bzp->count == 1 ? 0 : (uint64_t)j) : 0;
                int64_t acc = 0;
                for (int64_t k = 0; k < K; k++)
                    acc += (ir_get_int(A, abase + (uint64_t)k) - za) * (ir_get_int(B, bbase + (uint64_t)(k * N + j)) - zb);
                const uint64_t at = (bt * (uint64_t)M + i) * (uint64_t)N + (uint64_t)j;
                if (qlinear) {
                    double v = nearbyint((double)(int32_t)acc * (double)mult) + (double)yzp;
                    put_int(out[0], at, v < (double)lo ? lo : v > (double)hi ? hi : (int64_t)v);
                } else {
                    put_int(out[0], at, (int32_t)acc);
                }
            }
        }
    );
    return true;
}

/* ------------------------------------------------------------------------- recurrent operators of hybrid models */

/* LinearAttention (opset 27): a state of keys times values per head, updated token by token (linearly, gated by a
   decay, by the delta rule, or both) and read by the queries, as ONNX's reference runs it. */
static bool k_linear_attention(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 3 || m < 1 || !in[0] || !in[1] || !in[2]) return ir_fail(r, "queries, keys, values and an output are needed");
    const IrTensor *Q = in[0], *K = in[1], *V = in[2], *past = n > 3 ? in[3] : NULL;
    const IrTensor *decay = n > 4 ? in[4] : NULL, *beta = n > 5 ? in[5] : NULL;
    const IrAttr *ua = ir_attr(r->node, "update_rule");
    const char *rule = ua && ua->type == ATTR_STRING ? ua->s : "gated_delta";
    const bool gated = !strcmp(rule, "gated") || !strcmp(rule, "gated_delta");
    const bool delta = !strcmp(rule, "delta") || !strcmp(rule, "gated_delta");
    if (!gated && !delta && strcmp(rule, "linear")) return ir_fail(r, "the update rule '%s'", rule);
    if (gated != (decay != NULL) || delta != (beta != NULL)) return ir_fail(r, "decay or beta that the rule does not take");
    const int64_t Hq = ir_attr_int(r->node, "q_num_heads", 0), Hk = ir_attr_int(r->node, "kv_num_heads", 0);
    if (Q->rank != 3 || K->rank != 3 || V->rank != 3 || Hq < 1 || Hk < 1 || Hq % Hk || Q->dims[2] % Hq || V->dims[2] % Hk)
        return ir_fail(r, "inputs of three axes and heads that divide them");
    const int64_t B = Q->dims[0], T = Q->dims[1], Dk = Q->dims[2] / Hq, Dv = V->dims[2] / Hk, group = Hq / Hk;
    if (K->dims[0] != B || K->dims[1] != T || K->dims[2] != Hk * Dk || V->dims[0] != B || V->dims[1] != T)
        return ir_fail(r, "keys or values of other shapes");
    if (decay && (decay->rank != 3 || (decay->dims[2] != Hk && decay->dims[2] != Hk * Dk))) return ir_fail(r, "a decay of another shape");
    if (beta && (beta->rank != 3 || (beta->dims[2] != Hk && beta->dims[2] != 1))) return ir_fail(r, "a beta of another shape");
    if (past && (past->rank != 4 || past->dims[0] != B || past->dims[1] != Hk || past->dims[2] != Dk || past->dims[3] != Dv))
        return ir_fail(r, "a past state of another shape");
    const float sa = ir_attr_float(r->node, "scale", 0.0f), scale = sa != 0.0f ? sa : 1.0f / sqrtf((float)Dk);
    const int64_t ydims[3] = {B, T, Hq * Dv}, sdims[4] = {B, Hk, Dk, Dv};
    if (!ir_output(r, out[0], F32, 3, ydims)) return false;
    IrTensor st;
    memset(&st, 0, sizeof st);
    IrTensor *state = m > 1 && out[1] ? out[1] : &st;
    if (!ir_output(r, state, F32, 4, sdims)) return false;
    if (past) memcpy(state->data, past->data, (size_t)past->count * 4u);
    float *retrieved = (float *)malloc((size_t)(Dv ? Dv : 1) * sizeof(float));
    if (!retrieved) { if (state == &st) ir_release(r->g, &st); return ir_fail(r, "out of memory"); }
    for (int64_t b = 0; b < B; b++)
        for (int64_t t = 0; t < T; t++) {
            for (int64_t h = 0; h < Hk; h++) {
                float *S = fdata(state) + ((b * Hk + h) * Dk) * Dv;
                const float *k = fdata(K) + (b * T + t) * Hk * Dk + h * Dk, *v = fdata(V) + (b * T + t) * Hk * Dv + h * Dv;
                if (gated) {
                    const int64_t per = decay->dims[2] == Hk ? 1 : Dk;
                    const float *g = fdata(decay) + (b * T + t) * decay->dims[2] + h * per;
                    for (int64_t i = 0; i < Dk; i++) {
                        const float e = expf(g[per == 1 ? 0 : i]);
                        for (int64_t j = 0; j < Dv; j++) S[i * Dv + j] *= e;
                    }
                }
                const float *vt = v;
                if (delta) {
                    const float bt = fdata(beta)[(b * T + t) * beta->dims[2] + (beta->dims[2] == 1 ? 0 : h)];
                    for (int64_t j = 0; j < Dv; j++) {
                        float sum = 0.0f;
                        for (int64_t i = 0; i < Dk; i++) sum += S[i * Dv + j] * k[i];
                        retrieved[j] = bt * (v[j] - sum);
                    }
                    vt = retrieved;
                }
                for (int64_t i = 0; i < Dk; i++)
                    for (int64_t j = 0; j < Dv; j++) S[i * Dv + j] += k[i] * vt[j];
            }
            for (int64_t h = 0; h < Hq; h++) {
                const float *S = fdata(state) + ((b * Hk + h / group) * Dk) * Dv;
                const float *q = fdata(Q) + (b * T + t) * Hq * Dk + h * Dk;
                float *y = fdata(out[0]) + (b * T + t) * Hq * Dv + h * Dv;
                for (int64_t j = 0; j < Dv; j++) {
                    float sum = 0.0f;
                    for (int64_t i = 0; i < Dk; i++) sum += q[i] * S[i * Dv + j];
                    y[j] = scale * sum;
                }
            }
        }
    free(retrieved);
    if (state == &st) ir_release(r->g, &st);
    return true;
}

/* CausalConvWithState (opset 27): a causal depthwise convolution over the length, the past's last k - 1 positions
   before the input (zeros without one), SiLU after it if asked; the new state the last k - 1 positions. */
static bool k_causal_conv(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "an input, weights and an output are needed");
    const IrTensor *X = in[0], *W = in[1], *bias = n > 2 ? in[2] : NULL, *past = n > 3 ? in[3] : NULL;
    if (X->rank != 3 || W->rank != 3 || W->dims[0] != X->dims[1] || W->dims[1] != 1 || X->dtype != F32 || W->dtype != F32)
        return ir_fail(r, "an input or weights of other shapes");
    const int64_t B = X->dims[0], C = X->dims[1], L = X->dims[2], k = W->dims[2];
    if (k < 1 || (bias && bias->count != (uint64_t)C)) return ir_fail(r, "a kernel or bias of another size");
    if (past && (past->rank != 3 || past->dims[0] != B || past->dims[1] != C || past->dims[2] != k - 1))
        return ir_fail(r, "a past state of another shape");
    const IrAttr *aa = ir_attr(r->node, "activation");
    const bool silu = aa && aa->type == ATTR_STRING && (!strcmp(aa->s, "silu") || !strcmp(aa->s, "swish"));
    if (!ir_output(r, out[0], F32, 3, X->dims)) return false;
    const int64_t sdims[3] = {B, C, k - 1};
    if (m > 1 && out[1] && !ir_output(r, out[1], F32, 3, sdims)) return false;
    for (int64_t b = 0; b < B; b++)
        for (int64_t c = 0; c < C; c++) {
            const float *x = fdata(X) + (b * C + c) * L, *w = fdata(W) + c * k;
            const float *ps = past ? fdata(past) + (b * C + c) * (k - 1) : NULL;
            float *y = fdata(out[0]) + (b * C + c) * L;
            for (int64_t t = 0; t < L; t++) {
                float sum = 0.0f;
                for (int64_t j = 0; j < k; j++) {
                    const int64_t at = t + j - (k - 1);        /* in the padded sequence: past, then the input */
                    const float v = at >= 0 ? x[at] : ps ? ps[k - 1 + at] : 0.0f;
                    sum += v * w[j];
                }
                if (bias) sum += fdata(bias)[c];
                y[t] = silu ? sum / (1.0f + expf(-sum)) : sum;
            }
            if (m > 1 && out[1]) {
                float *s = fdata(out[1]) + (b * C + c) * (k - 1);
                for (int64_t j = 0; j < k - 1; j++) {
                    const int64_t at = L - (k - 1) + j;
                    s[j] = at >= 0 ? x[at] : ps ? ps[k - 1 + at] : 0.0f;
                }
            }
        }
    return true;
}

/* ------------------------------------------------------------------------- the library's own operators */

/* Linear (fusions make it): X (..., K) times a constant W (K x N), plus a bias of N when given, through an
   activation (the attribute: an ActivationFunction), into (..., N); W's columns packed once. */
static bool k_linear(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 2 || m < 1 || !in[0] || !in[1]) return ir_fail(r, "an input, weights and an output are needed");
    const IrTensor *X = in[0], *W = in[1], *bias = n > 2 ? in[2] : NULL;
    if (X->dtype != F32 || W->dtype != F32 || W->rank != 2 || !X->rank || !W->constant)
        return ir_fail(r, "an input, or weights that are no constant matrix, not float32");
    const int64_t K = W->dims[0], N = W->dims[1];
    if (X->dims[X->rank - 1] != K) return ir_fail(r, "an input of %lld columns for %lld rows of weights",
                                                  (long long)X->dims[X->rank - 1], (long long)K);
    if (bias && (bias->dtype != F32 || bias->count != (uint64_t)N)) return ir_fail(r, "a bias of another size");
    const int64_t act = ir_attr_int(r->node, "activation", ACT_NONE);
    if (act < 0 || act >= ACT_COUNT || act == ACT_SOFTMAX) return ir_fail(r, "the activation %lld", (long long)act);
    int64_t dims[IR_MAX_RANK];
    for (uint32_t d = 0; d < X->rank; d++) dims[d] = d + 1 < X->rank ? X->dims[d] : N;
    if (!K || !N) {
        if (!ir_output(r, out[0], F32, X->rank, dims)) return false;
        for (uint64_t i = 0; bias && i < out[0]->count; i++) fdata(out[0])[i] = fdata(bias)[i % (uint64_t)N];
        if (act != ACT_NONE) apply_activation_bulk(fdata(out[0]), out[0]->count, (ActivationFunction)act);
        return true;
    }
    const Packed *p = packed_of(r, W, false);
    if (!p || !ir_output_raw(r, out[0], F32, X->rank, dims)) return false;
    product_rows(r, p, fdata(X), X->count / (uint64_t)K, fdata(out[0]), bias ? fdata(bias) : NULL, (ActivationFunction)act);
    return true;
}

/* FusedAttention (fusions make it): softmax(Q K^T * scale + mask) V, the softmax over the keys (with "safe", rows
   of no keys 0, as PyTorch's safe softmax). With layout 0, over the broadcast batches of Q (..., S, D), K^T
   (..., D, T), V (..., T, Dv) and a mask broadcast to (..., S, T), into (..., S, Dv); with layout 1 (the transposes
   around attention taken in), Q (B, S, H, D), K (B, T, H or 1, D), V (B, T, H or 1, Dv) into (B, S, H, Dv), the mask
   broadcast to (B, H, S, T). Blocks of 64 queries of a batch and head at a time, each on one thread: their scores by
   a product, then the mask, the softmax (the vector exp) and their outputs by a second product. */
static bool k_fused_attention(IrRun *r, IrTensor *const *in, uint32_t n, IrTensor **out, uint32_t m) {
    if (n < 3 || m < 1 || !in[0] || !in[1] || !in[2]) return ir_fail(r, "queries, keys, values and an output are needed");
    const IrTensor *Q = in[0], *K = in[1], *V = in[2], *mask = n > 3 ? in[3] : NULL;
    if (Q->dtype != F32 || K->dtype != F32 || V->dtype != F32 || (mask && mask->dtype != F32 && mask->dtype != BOOL))
        return ir_fail(r, "inputs not float32");
    const float scale = ir_attr_float(r->node, "scale", 1.0f);
    const bool safe = ir_attr_int(r->node, "safe", 0) != 0, heads_inside = ir_attr_int(r->node, "layout", 0) == 1;
    uint32_t R;                     /* the batch's axes (layout 1: B and H) */
    int64_t dims[IR_MAX_RANK], S, D, T, Dv;
    int64_t sq[IR_MAX_RANK], sk[IR_MAX_RANK], sv[IR_MAX_RANK], sm[IR_MAX_RANK] = {0};
    size_t ldq, ldk, ldv, ldy;
    if (heads_inside) {
        if (Q->rank != 4 || K->rank != 4 || V->rank != 4) return ir_fail(r, "inputs of four axes");
        const int64_t B = Q->dims[0], H = Q->dims[2], Hk = K->dims[2];
        S = Q->dims[1]; D = Q->dims[3]; T = K->dims[1]; Dv = V->dims[3];
        if (K->dims[0] != B || V->dims[0] != B || K->dims[3] != D || V->dims[1] != T || V->dims[2] != Hk || (Hk != H && Hk != 1))
            return ir_fail(r, "keys or values of other shapes");
        R = 2;
        dims[0] = B;
        dims[1] = H;
        /* (offsets of a batch and head's first row, in elements) */
        sq[0] = S * H * D; sq[1] = D;
        sk[0] = T * Hk * D; sk[1] = Hk == H ? D : 0;
        sv[0] = T * Hk * Dv; sv[1] = Hk == H ? Dv : 0;
        ldq = (size_t)(H * D);
        ldk = (size_t)(Hk * D);
        ldv = (size_t)(Hk * Dv);
        ldy = (size_t)(H * Dv);
    } else {
        if (Q->rank < 2 || K->rank < 2 || V->rank < 2) return ir_fail(r, "inputs of fewer than two axes");
        S = Q->dims[Q->rank - 2]; D = Q->dims[Q->rank - 1]; T = K->dims[K->rank - 1]; Dv = V->dims[V->rank - 1];
        if (K->dims[K->rank - 2] != D || V->dims[V->rank - 2] != T) return ir_fail(r, "keys or values of other shapes");
        IrTensor bq, bk, bv;
        memset(&bq, 0, sizeof bq);
        memset(&bk, 0, sizeof bk);
        memset(&bv, 0, sizeof bv);
        bq.rank = Q->rank - 2;
        bk.rank = K->rank - 2;
        bv.rank = V->rank - 2;
        for (uint32_t d = 0; d < bq.rank; d++) bq.dims[d] = Q->dims[d];
        for (uint32_t d = 0; d < bk.rank; d++) bk.dims[d] = K->dims[d];
        for (uint32_t d = 0; d < bv.rank; d++) bv.dims[d] = V->dims[d];
        IrTensor *three[3] = {&bq, &bk, &bv};
        if (!broadcast_shape(r, three, 3, &R, dims) || R + 2 > IR_MAX_RANK) return r->error[0] ? false : ir_fail(r, "too many axes");
        broadcast_strides(&bq, R, dims, sq);
        broadcast_strides(&bk, R, dims, sk);
        broadcast_strides(&bv, R, dims, sv);
        for (uint32_t d = 0; d < R; d++) { sq[d] *= S * D; sk[d] *= D * T; sv[d] *= T * Dv; }
        ldq = (size_t)D;
        ldk = (size_t)T;
        ldv = (size_t)Dv;
        ldy = (size_t)Dv;
    }
    uint64_t batches = 1;
    for (uint32_t d = 0; d < R; d++) batches *= (uint64_t)dims[d];
    int64_t full[IR_MAX_RANK];
    for (uint32_t d = 0; d < R; d++) full[d] = dims[d];
    full[R] = S;
    full[R + 1] = T;
    if (mask) {
        if (mask->rank > R + 2) return ir_fail(r, "a mask of more axes than the scores");
        for (uint32_t d = 0; d < mask->rank; d++) {
            const int64_t want = full[R + 2 - mask->rank + d];
            if (mask->dims[d] != want && mask->dims[d] != 1) return ir_fail(r, "a mask that does not broadcast to the scores");
        }
        broadcast_strides(mask, R + 2, full, sm);
    }
    int64_t odims[IR_MAX_RANK];
    if (heads_inside) {
        odims[0] = dims[0]; odims[1] = S; odims[2] = dims[1]; odims[3] = Dv;
    } else {
        for (uint32_t d = 0; d < R; d++) odims[d] = dims[d];
        odims[R] = S;
        odims[R + 1] = Dv;
    }
    if (!ir_output_raw(r, out[0], F32, R + 2, odims)) return false;
    if (!S || !Dv) return true;
    SpingalettGemmScratch *main = ir_scratch(r);
    if (!main) return false;
    const int64_t BLOCK = 64, blocks = (S + BLOCK - 1) / BLOCK, items = (int64_t)batches * blocks;
    const bool across = r->parallel && items > 1 && (uint64_t)items * BLOCK * T * (D + Dv) >= PARALLEL_WORK;
    bool failed = false;
    SPINGALETT_PARALLEL_FOR(across,
        for (int64_t it = 0; it < items; it++) {
            const int64_t blk = it % blocks, bt = it / blocks, q0 = blk * BLOCK, rows = S - q0 < BLOCK ? S - q0 : BLOCK;
            int64_t oq = 0, ok_ = 0, ov = 0, om = 0;
            uint64_t t = (uint64_t)bt;
            for (uint32_t d = R; d-- > 0;) {
                const int64_t c = (int64_t)(t % (uint64_t)dims[d]);
                t /= (uint64_t)dims[d];
                oq += c * sq[d];
                ok_ += c * sk[d];
                ov += c * sv[d];
                om += c * sm[d];
            }
            float *s = (float *)spingalett_aligned_alloc((size_t)(rows * T) * sizeof(float) + 64);
            if (!s) { failed = true; continue; }
            SpingalettGemmScratch *scr = across ? ir_thread_scratch(r, spingalett_thread_num()) : main;
            /* scores: Q's rows times K^T (layout 1: K's rows, transposed) */
            spingalett_gemm_native(scr, false, heads_inside, (uint32_t)rows, (uint32_t)T, (uint32_t)D, scale,
                                   fdata(Q) + oq + q0 * (int64_t)ldq, ldq, fdata(K) + ok_, ldk, 0.0f, s, (size_t)T,
                                   !across && r->parallel);
            for (int64_t i = 0; i < rows; i++) {
                float *row = s + i * T;
                if (mask) {
                    const int64_t base = om + (q0 + i) * sm[R];
                    if (mask->dtype == BOOL) {
                        const uint8_t *mb = (const uint8_t *)mask->data;
                        for (int64_t j = 0; j < T; j++) if (!mb[base + j * sm[R + 1]]) row[j] = -INFINITY;
                    } else {
                        const float *mf = fdata(mask);
                        for (int64_t j = 0; j < T; j++) row[j] += mf[base + j * sm[R + 1]];
                    }
                }
                float mx = -INFINITY;
                for (int64_t j = 0; j < T; j++) mx = row[j] > mx || isnan(row[j]) ? row[j] : mx;
                const float sum = spingalett_vec_exp_shifted(row, (uint32_t)T, mx);
                const float inv = 1.0f / sum;
                for (int64_t j = 0; j < T; j++) row[j] *= inv;
                if (safe)
                    for (int64_t j = 0; j < T; j++) row[j] = isnan(row[j]) ? 0.0f : row[j];
            }
            float *y = heads_inside ? fdata(out[0]) + (bt / dims[1]) * S * (int64_t)ldy + (bt % dims[1]) * Dv + q0 * (int64_t)ldy
                                    : fdata(out[0]) + ((int64_t)bt * S + q0) * Dv;
            spingalett_gemm_native(scr, false, false, (uint32_t)rows, (uint32_t)Dv, (uint32_t)T, 1.0f, s, (size_t)T,
                                   fdata(V) + ov, ldv, 0.0f, y, ldy, !across && r->parallel);
            spingalett_aligned_free(s);
        }
    );
    return failed ? ir_fail(r, "out of memory for the scores") : true;
}

const IrKernel ir_kernels[IR_OP_COUNT] = {
#define IR_OP(name, kernel) kernel,
#include "Spingalett.IROps.def"
#undef IR_OP
};
