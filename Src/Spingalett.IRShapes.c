/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Graphs of operators: the types and shapes of the values nodes compute, inferred when a graph opens from its inputs'
 * declared shapes and its constants, for the passes whose decisions rest on them (fusions that hold only for some
 * shapes, the layout of convolutions' maps). A size is a number, a symbol (an input's, carried through operators that
 * keep it), or unknown; a value whose type and number of axes are known is declared, with -1 for a symbol's size or
 * an unknown one. Operators without a rule here, and nodes of inputs not declared, leave their outputs undeclared:
 * kernels compute every shape when they run whatever this says, so only the passes' decisions rest on it. Values the
 * file declares keep its declaration (a float16 output keeps its type).
 */

#include "Spingalett.IR.h"
#include <string.h>

#define F32 SPINGALETT_DTYPE_FLOAT32
#define I64 SPINGALETT_DTYPE_INT64
#define BOOL SPINGALETT_DTYPE_BOOL

/* A shape: sizes (-1 for a symbol, sym[d] its index + 1, or unknown, sym[d] 0). */
typedef struct {
    bool known;
    uint8_t dtype, rank;
    int64_t dims[IR_MAX_RANK];
    uint16_t sym[IR_MAX_RANK];
} Shape;

static Shape shape_of(const SpingalettGraph *g, uint32_t v) {
    Shape s;
    memset(&s, 0, sizeof s);
    if (v == IR_NONE) return s;
    const IrValue *x = &g->values[v];
    if (!x->declared || x->rank > IR_MAX_RANK) return s;
    s.known = true;
    s.dtype = x->dtype == SPINGALETT_DTYPE_FLOAT16 || x->dtype == SPINGALETT_DTYPE_BFLOAT16 ? F32 : x->dtype;
    s.rank = x->rank;
    for (uint32_t d = 0; d < x->rank; d++) {
        s.dims[d] = x->dims[d];
        s.sym[d] = x->dims[d] >= 0 ? 0 : x->sym[d];
    }
    return s;
}

/* Output o of node n declared with s (values the file declares left as they are). */
static void declare(SpingalettGraph *g, const IrNode *n, uint32_t o, const Shape *s) {
    if (o >= n->output_count || n->outputs[o] == IR_NONE || !s->known) return;
    IrValue *v = &g->values[n->outputs[o]];
    if (v->declared || v->kind != VALUE_NODE) return;
    v->declared = true;
    v->dtype = s->dtype;
    v->rank = s->rank;
    for (uint32_t d = 0; d < IR_MAX_RANK; d++) {
        v->dims[d] = d < s->rank ? s->dims[d] : -1;
        v->sym[d] = d < s->rank && s->dims[d] < 0 ? s->sym[d] : 0;
    }
}

static Shape with_dtype(Shape s, uint8_t dtype) {
    s.dtype = dtype;
    return s;
}

/* Whether every size is a number; their product. */
static bool numbers(const Shape *s, int64_t *count) {
    int64_t c = 1;
    for (uint32_t d = 0; d < s->rank; d++) {
        if (s->dims[d] < 0) return false;
        c *= s->dims[d];
    }
    if (count) *count = c;
    return true;
}

/* The integers of a constant of int64 or int32: their count, -1 when v is no such constant. */
static int64_t const_ints(const SpingalettGraph *g, uint32_t v, int64_t *out, int64_t most) {
    if (v == IR_NONE) return -1;
    const IrValue *c = &g->values[v];
    if (c->kind != VALUE_CONST || (c->dtype != I64 && c->dtype != SPINGALETT_DTYPE_INT32) || c->rank > 1) return -1;
    const int64_t count = c->rank ? c->dims[0] : 1;
    if (count > most) return -1;
    for (int64_t k = 0; k < count; k++)
        out[k] = c->dtype == I64 ? ((const int64_t *)c->data)[k] : ((const int32_t *)c->data)[k];
    return count;
}

/* A list from input i (a constant) or attribute attr: its count, -1 for none, -2 for an input not a constant. */
static int64_t list_of(const SpingalettGraph *g, const IrNode *n, uint32_t i, const char *attr, int64_t *out, int64_t most) {
    if (i < n->input_count && n->inputs[i] != IR_NONE) {
        const int64_t k = const_ints(g, n->inputs[i], out, most);
        return k < 0 ? -2 : k;
    }
    const IrAttr *a = attr ? ir_attr(n, attr) : NULL;
    if (!a || a->type != ATTR_INTS || (int64_t)a->count > most) return -1;
    for (uint32_t k = 0; k < a->count; k++) out[k] = a->ints[k];
    return a->count;
}

static bool axis_in(int64_t axis, uint32_t rank, uint32_t *out) {
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= (int64_t)rank) return false;
    *out = (uint32_t)axis;
    return true;
}

/* One size of a broadcast from two (right-aligned): false where they cannot broadcast. */
static bool broadcast_dim(int64_t a, uint16_t sa, int64_t b, uint16_t sb, int64_t *d, uint16_t *s) {
    if (a == 1) { *d = b; *s = sb; return true; }
    if (b == 1) { *d = a; *s = sa; return true; }
    if (a >= 0 && b >= 0) {
        if (a != b) return false;
        *d = a; *s = 0;
        return true;
    }
    /* (a number against a symbol: the number, as a graph that runs has them equal) */
    if (a >= 0) { *d = a; *s = 0; return true; }
    if (b >= 0) { *d = b; *s = 0; return true; }
    *d = -1;
    *s = sa && sa == sb ? sa : 0;
    return true;
}

static Shape broadcast(const Shape *in, uint32_t n) {
    Shape r;
    memset(&r, 0, sizeof r);
    uint32_t R = 0;
    for (uint32_t k = 0; k < n; k++) {
        if (!in[k].known) return r;
        if (in[k].rank > R) R = in[k].rank;
    }
    r.rank = (uint8_t)R;
    for (uint32_t d = 0; d < R; d++) r.dims[d] = 1;
    for (uint32_t k = 0; k < n; k++) {
        const uint32_t off = R - in[k].rank;
        for (uint32_t d = 0; d < in[k].rank; d++)
            if (!broadcast_dim(r.dims[off + d], r.sym[off + d], in[k].dims[d], in[k].sym[d], &r.dims[off + d], &r.sym[off + d]))
                return (Shape){0};
    }
    r.known = true;
    return r;
}

/* The output's spatial sizes of a convolution or pooling over x (sizes all numbers) by the CPU's rules. */
static bool windows(SpingalettGraph *g, const IrNode *n, uint32_t index, const Shape *x, const int64_t *kernel,
                    bool ceil_mode, Shape *out) {
    for (uint32_t d = 2; d < x->rank; d++)
        if (x->dims[d] < 0) {
            for (uint32_t e = 2; e < x->rank; e++) out->dims[e] = -1, out->sym[e] = 0;
            return true;
        }
    IrTensor t;
    memset(&t, 0, sizeof t);
    t.rank = x->rank;
    for (uint32_t d = 0; d < x->rank; d++) t.dims[d] = x->dims[d] < 0 ? 1 : x->dims[d];
    IrRun r;
    memset(&r, 0, sizeof r);
    r.g = g;
    r.node = n;
    r.index = index;
    IrGeometry geo;
    if (!ir_geometry(&r, &t, kernel, &geo, ceil_mode)) return false;
    for (uint32_t a = 0; a < geo.spatial; a++) {
        out->dims[2 + a] = geo.out[3 - geo.spatial + a];
        out->sym[2 + a] = 0;
    }
    return true;
}

/* The outputs of node k from its inputs' shapes, where its operator has a rule. */
static void infer(SpingalettGraph *g, uint32_t k) {
    const IrNode *n = &g->nodes[k];
    /* (nodes the layout pass made channels last declare their outputs themselves) */
    if (!n->input_count || n->input_count > 64 || ir_attr_int(n, "channels_last", 0)) return;
    Shape in[64];
    for (uint32_t i = 0; i < n->input_count; i++) in[i] = shape_of(g, n->inputs[i]);
    const Shape *x = &in[0];
    Shape y = *x;
    int64_t list[2 * IR_MAX_RANK];
    switch ((IrOp)n->op) {
        /* the input's shape and type */
        case IR_Abs: case IR_Neg: case IR_Exp: case IR_Log: case IR_Sqrt: case IR_Reciprocal: case IR_Sin: case IR_Cos:
        case IR_Tanh: case IR_Sigmoid: case IR_Erf: case IR_Relu: case IR_LeakyRelu: case IR_Elu: case IR_Selu:
        case IR_Gelu: case IR_HardSigmoid: case IR_HardSwish: case IR_Softplus: case IR_Softsign: case IR_Mish:
        case IR_Floor: case IR_Ceil: case IR_Round: case IR_Sign: case IR_Not: case IR_Tan: case IR_Asin: case IR_Acos:
        case IR_Atan: case IR_Sinh: case IR_Cosh: case IR_Asinh: case IR_Acosh: case IR_Atanh: case IR_Celu:
        case IR_ThresholdedRelu: case IR_Shrink: case IR_Swish: case IR_BitwiseNot: case IR_Clip: case IR_Identity:
        case IR_BatchNormalization: case IR_InstanceNormalization: case IR_GroupNormalization: case IR_RMSNormalization:
        case IR_LpNormalization: case IR_MeanVarianceNormalization: case IR_LRN: case IR_Softmax: case IR_LogSoftmax:
        case IR_Hardmax: case IR_CumSum: case IR_CumProd: case IR_Trilu: case IR_RotaryEmbedding:
            declare(g, n, 0, &y);
            return;
        case IR_IsNaN: case IR_IsInf:
            y = with_dtype(y, BOOL);
            declare(g, n, 0, &y);
            return;
        case IR_Dropout: {
            declare(g, n, 0, &y);
            const Shape mask = with_dtype(y, BOOL);
            declare(g, n, 1, &mask);
            return;
        }
        case IR_LayerNormalization: {
            declare(g, n, 0, &y);
            uint32_t axis;
            if (!x->known || !axis_in(ir_attr_int(n, "axis", -1), x->rank, &axis)) return;
            for (uint32_t d = axis; d < x->rank; d++) y.dims[d] = 1, y.sym[d] = 0;
            declare(g, n, 1, &y);
            declare(g, n, 2, &y);
            return;
        }
        case IR_Cast: case IR_CastLike: {
            int64_t to = n->op == IR_CastLike ? (n->input_count > 1 && in[1].known ? in[1].dtype : -1)
                                              : ir_attr_int(n, "to", -1);
            if (to == 11 || to == SPINGALETT_DTYPE_FLOAT16 || to == SPINGALETT_DTYPE_BFLOAT16) to = F32;
            else if (to == 4 || to == 5) to = SPINGALETT_DTYPE_INT32;
            else if (to == 12 || to == 13) to = I64;
            if (to < 0 || !ir_dtype_size((uint8_t)to)) return;
            y = with_dtype(y, (uint8_t)to);
            declare(g, n, 0, &y);
            return;
        }
        /* broadcasts */
        case IR_Add: case IR_Sub: case IR_Mul: case IR_Div: case IR_Pow: case IR_Mod: case IR_Max: case IR_Min:
        case IR_Sum: case IR_Mean: case IR_PRelu: case IR_And: case IR_Or: case IR_Xor: case IR_BitwiseAnd:
        case IR_BitwiseOr: case IR_BitwiseXor: case IR_BitShift: {
            y = broadcast(in, n->input_count);
            y.dtype = x->dtype;
            declare(g, n, 0, &y);
            return;
        }
        case IR_Equal: case IR_Less: case IR_LessOrEqual: case IR_Greater: case IR_GreaterOrEqual:
            if (n->input_count < 2) return;
            y = with_dtype(broadcast(in, 2), BOOL);
            declare(g, n, 0, &y);
            return;
        case IR_Where:
            if (n->input_count < 3) return;
            y = with_dtype(broadcast(in, 3), in[1].dtype);
            declare(g, n, 0, &y);
            return;
        /* products */
        case IR_MatMul: {
            if (n->input_count < 2 || !x->known || !in[1].known || !x->rank || !in[1].rank) return;
            const Shape *b = &in[1];
            Shape ba = *x, bb = *b;
            ba.rank = x->rank > 2 ? (uint8_t)(x->rank - 2) : 0;
            bb.rank = b->rank > 2 ? (uint8_t)(b->rank - 2) : 0;
            Shape pair[2] = {ba, bb};
            y = broadcast(pair, 2);
            if (!y.known) return;
            if (x->rank > 1) y.dims[y.rank] = x->dims[x->rank - 2], y.sym[y.rank++] = x->sym[x->rank - 2];
            if (b->rank > 1) y.dims[y.rank] = b->dims[b->rank - 1], y.sym[y.rank++] = b->sym[b->rank - 1];
            y.dtype = x->dtype;
            declare(g, n, 0, &y);
            return;
        }
        case IR_Gemm: {
            if (n->input_count < 2 || !x->known || !in[1].known || x->rank != 2 || in[1].rank != 2) return;
            const bool ta = ir_attr_int(n, "transA", 0) != 0, tb = ir_attr_int(n, "transB", 0) != 0;
            y.dims[0] = x->dims[ta ? 1 : 0];
            y.sym[0] = x->sym[ta ? 1 : 0];
            y.dims[1] = in[1].dims[tb ? 0 : 1];
            y.sym[1] = in[1].sym[tb ? 0 : 1];
            declare(g, n, 0, &y);
            return;
        }
        case IR_Linear: {
            if (n->input_count < 2 || !x->known || !in[1].known || !x->rank || in[1].rank != 2) return;
            const bool trans = ir_attr_int(n, "trans_b", 0) != 0;
            y.dims[x->rank - 1] = in[1].dims[trans ? 0 : 1];
            y.sym[x->rank - 1] = 0;
            declare(g, n, 0, &y);
            return;
        }
        case IR_FusedAttention: {
            if (n->input_count < 3 || !x->known || !in[2].known) return;
            const Shape *v = &in[2];
            if (ir_attr_int(n, "layout", 0) == 1) {
                if (x->rank != 4 || v->rank != 4) return;
                y.dims[3] = v->dims[3];
                y.sym[3] = v->sym[3];
            } else {
                if (x->rank < 2 || v->rank < 2 || !in[1].known) return;
                Shape bq = *x, bk = in[1], bv = *v;
                bq.rank = (uint8_t)(x->rank - 2);
                bk.rank = (uint8_t)(in[1].rank - 2);
                bv.rank = (uint8_t)(v->rank - 2);
                Shape three[3] = {bq, bk, bv};
                y = broadcast(three, 3);
                if (!y.known) return;
                y.dims[y.rank] = x->dims[x->rank - 2], y.sym[y.rank++] = x->sym[x->rank - 2];
                y.dims[y.rank] = v->dims[v->rank - 1], y.sym[y.rank++] = v->sym[v->rank - 1];
                y.dtype = F32;
            }
            declare(g, n, 0, &y);
            return;
        }
        case IR_Conv: {
            if (n->input_count < 2 || !x->known || !in[1].known || x->rank < 3 || in[1].rank != x->rank ||
                g->values[n->inputs[1]].kind != VALUE_CONST || !numbers(&in[1], NULL))
                return;
            y.dims[1] = in[1].dims[0];
            y.sym[1] = 0;
            if (!windows(g, n, k, x, in[1].dims + 2, false, &y)) return;
            declare(g, n, 0, &y);
            return;
        }
        case IR_MaxPool: case IR_AveragePool: case IR_LpPool: {
            const IrAttr *ks = ir_attr(n, "kernel_shape");
            if (!x->known || x->rank < 3 || !ks || ks->type != ATTR_INTS || ks->count != x->rank - 2u) return;
            if (!windows(g, n, k, x, ks->ints, ir_attr_int(n, "ceil_mode", 0) != 0, &y)) return;
            declare(g, n, 0, &y);
            const Shape indices = with_dtype(y, I64);
            declare(g, n, 1, &indices);
            return;
        }
        case IR_GlobalAveragePool: case IR_GlobalMaxPool:
            if (!x->known || x->rank < 2) return;
            for (uint32_t d = 2; d < x->rank; d++) y.dims[d] = 1, y.sym[d] = 0;
            declare(g, n, 0, &y);
            return;
        /* shapes */
        case IR_Reshape: {
            const int64_t ns = n->input_count > 1 ? const_ints(g, n->inputs[1], list, IR_MAX_RANK) : -1;
            if (!x->known || ns < 0) return;
            const bool allowzero = ir_attr_int(n, "allowzero", 0) != 0;
            int64_t total = 0, known = 1;
            const bool all = numbers(x, &total);
            int infer_at = -1;
            y.rank = (uint8_t)ns;
            for (uint32_t d = 0; d < y.rank; d++) {
                y.sym[d] = 0;
                if (list[d] == 0 && !allowzero) {
                    if (d >= x->rank) return;
                    y.dims[d] = x->dims[d];
                    y.sym[d] = x->sym[d];
                } else if (list[d] == -1) {
                    if (infer_at >= 0) return;
                    infer_at = (int)d;
                    y.dims[d] = -1;
                    continue;
                } else if (list[d] < 0) {
                    return;
                } else {
                    y.dims[d] = list[d];
                }
                if (y.dims[d] >= 0) known *= y.dims[d];
                else known = -1;
            }
            if (infer_at >= 0) y.dims[infer_at] = all && known > 0 && total % known == 0 ? total / known : -1;
            declare(g, n, 0, &y);
            return;
        }
        case IR_Flatten: {
            uint32_t axis;
            if (!x->known) return;
            int64_t a = ir_attr_int(n, "axis", 1);
            if (a < 0) a += x->rank;
            if (a < 0 || a > x->rank) return;
            axis = (uint32_t)a;
            int64_t p0 = 1, p1 = 1;
            for (uint32_t d = 0; d < x->rank; d++) {
                int64_t *p = d < axis ? &p0 : &p1;
                *p = *p < 0 || x->dims[d] < 0 ? -1 : *p * x->dims[d];
            }
            y.rank = 2;
            y.dims[0] = p0;
            y.dims[1] = p1;
            y.sym[0] = axis == 1 && x->dims[0] < 0 ? x->sym[0] : 0;
            y.sym[1] = axis + 1 == x->rank && x->dims[axis] < 0 ? x->sym[axis] : 0;
            declare(g, n, 0, &y);
            return;
        }
        case IR_Squeeze: case IR_Unsqueeze: {
            if (!x->known) return;
            const int64_t na = list_of(g, n, 1, "axes", list, IR_MAX_RANK);
            if (na == -2) return;
            if (n->op == IR_Squeeze) {
                bool drop[IR_MAX_RANK] = {false};
                for (int64_t a = 0; a < na; a++) {
                    uint32_t d;
                    if (!axis_in(list[a], x->rank, &d)) return;
                    drop[d] = true;
                }
                y.rank = 0;
                for (uint32_t d = 0; d < x->rank; d++) {
                    if (na < 0 ? x->dims[d] == 1 : drop[d]) continue;
                    if (na < 0 && x->dims[d] < 0) return;       /* (a symbol might be 1) */
                    y.dims[y.rank] = x->dims[d];
                    y.sym[y.rank++] = x->sym[d];
                }
            } else {
                if (na < 0) return;
                const uint32_t R = x->rank + (uint32_t)na;
                if (R > IR_MAX_RANK) return;
                bool add[IR_MAX_RANK] = {false};
                for (int64_t a = 0; a < na; a++) {
                    uint32_t d;
                    if (!axis_in(list[a], R, &d)) return;
                    add[d] = true;
                }
                y.rank = (uint8_t)R;
                for (uint32_t d = 0, j = 0; d < R; d++) {
                    y.dims[d] = add[d] ? 1 : x->dims[j];
                    y.sym[d] = add[d] ? 0 : x->sym[j];
                    if (!add[d]) j++;
                }
            }
            declare(g, n, 0, &y);
            return;
        }
        case IR_Transpose: {
            if (!x->known) return;
            const IrAttr *a = ir_attr(n, "perm");
            for (uint32_t d = 0; d < x->rank; d++) {
                const int64_t p = a && a->type == ATTR_INTS && a->count == x->rank ? a->ints[d] : (int64_t)(x->rank - 1 - d);
                if (p < 0 || p >= x->rank) return;
                y.dims[d] = x->dims[p];
                y.sym[d] = x->sym[p];
            }
            declare(g, n, 0, &y);
            return;
        }
        case IR_Concat: {
            uint32_t axis;
            if (!x->known || !axis_in(ir_attr_int(n, "axis", 0), x->rank, &axis)) return;
            int64_t sum = 0;
            for (uint32_t i = 0; i < n->input_count; i++) {
                if (n->inputs[i] == IR_NONE) continue;
                if (!in[i].known || in[i].rank != x->rank) return;
                sum = sum < 0 || in[i].dims[axis] < 0 ? -1 : sum + in[i].dims[axis];
                for (uint32_t d = 0; d < x->rank; d++)
                    if (d != axis && y.dims[d] < 0 && in[i].dims[d] >= 0) y.dims[d] = in[i].dims[d], y.sym[d] = 0;
            }
            y.dims[axis] = sum;
            y.sym[axis] = 0;
            declare(g, n, 0, &y);
            return;
        }
        case IR_Split: {
            uint32_t axis;
            if (!x->known || !axis_in(ir_attr_int(n, "axis", 0), x->rank, &axis)) return;
            int64_t sizes[64];
            int64_t ns = list_of(g, n, 1, "split", sizes, 64);
            if (ns == -2) return;
            const uint32_t m = n->output_count;
            if (ns < 0) {
                if (x->dims[axis] < 0 || m > 64) return;
                const int64_t each = (x->dims[axis] + m - 1) / m;
                for (uint32_t o = 0; o < m; o++) sizes[o] = o + 1 < m ? each : x->dims[axis] - each * (int64_t)(m - 1);
                ns = m;
            }
            if (ns != (int64_t)m) return;
            for (uint32_t o = 0; o < m; o++) {
                y.dims[axis] = sizes[o];
                y.sym[axis] = 0;
                declare(g, n, o, &y);
            }
            return;
        }
        case IR_Slice: {
            if (!x->known) return;
            int64_t starts[IR_MAX_RANK], ends[IR_MAX_RANK], axes[IR_MAX_RANK], steps[IR_MAX_RANK];
            const int64_t ns = list_of(g, n, 1, "starts", starts, IR_MAX_RANK), ne = list_of(g, n, 2, "ends", ends, IR_MAX_RANK);
            int64_t na = list_of(g, n, 3, "axes", axes, IR_MAX_RANK);
            const int64_t nt = list_of(g, n, 4, "steps", steps, IR_MAX_RANK);
            if (ns < 0 || ne != ns || na == -2 || nt == -2) return;
            if (na < 0) {
                for (int64_t a = 0; a < ns; a++) axes[a] = a;
                na = ns;
            }
            if (na != ns || (nt >= 0 && nt != ns)) return;
            for (int64_t a = 0; a < ns; a++) {
                uint32_t d;
                if (!axis_in(axes[a], x->rank, &d)) return;
                const int64_t size = x->dims[d], st = nt >= 0 ? steps[a] : 1;
                y.sym[d] = 0;
                if (size < 0 || !st) {
                    y.dims[d] = -1;
                    continue;
                }
                int64_t s0 = starts[a], e = ends[a];
                if (s0 < 0) s0 += size;
                if (e < 0) e += size;
                if (st > 0) {
                    s0 = s0 < 0 ? 0 : s0 > size ? size : s0;
                    e = e < 0 ? 0 : e > size ? size : e;
                    y.dims[d] = e > s0 ? (e - s0 + st - 1) / st : 0;
                } else {
                    s0 = s0 < 0 ? -1 : s0 > size - 1 ? size - 1 : s0;
                    e = e < -1 ? -1 : e > size - 1 ? size - 1 : e;
                    y.dims[d] = s0 > e ? (s0 - e - st - 1) / -st : 0;
                }
            }
            declare(g, n, 0, &y);
            return;
        }
        case IR_Gather: {
            uint32_t axis;
            if (n->input_count < 2 || !x->known || !in[1].known || !axis_in(ir_attr_int(n, "axis", 0), x->rank, &axis))
                return;
            const uint32_t R = x->rank - 1u + in[1].rank;
            if (R > IR_MAX_RANK) return;
            uint32_t j = 0;
            for (uint32_t d = 0; d < axis; d++) y.dims[j] = x->dims[d], y.sym[j++] = x->sym[d];
            for (uint32_t d = 0; d < in[1].rank; d++) y.dims[j] = in[1].dims[d], y.sym[j++] = in[1].sym[d];
            for (uint32_t d = axis + 1; d < x->rank; d++) y.dims[j] = x->dims[d], y.sym[j++] = x->sym[d];
            y.rank = (uint8_t)R;
            declare(g, n, 0, &y);
            return;
        }
        case IR_Expand: {
            const int64_t ns = n->input_count > 1 ? const_ints(g, n->inputs[1], list, IR_MAX_RANK) : -1;
            if (!x->known || ns < 0) return;
            Shape want;
            memset(&want, 0, sizeof want);
            want.known = true;
            want.rank = (uint8_t)ns;
            for (int64_t d = 0; d < ns; d++) want.dims[d] = list[d];
            Shape pair[2] = {*x, want};
            y = with_dtype(broadcast(pair, 2), x->dtype);
            declare(g, n, 0, &y);
            return;
        }
        case IR_Tile: {
            if (!x->known || n->input_count < 2 || const_ints(g, n->inputs[1], list, IR_MAX_RANK) != x->rank) return;
            for (uint32_t d = 0; d < x->rank; d++) {
                y.dims[d] = x->dims[d] < 0 ? -1 : x->dims[d] * list[d];
                y.sym[d] = list[d] == 1 ? x->sym[d] : 0;
            }
            declare(g, n, 0, &y);
            return;
        }
        case IR_Pad: {
            if (!x->known) return;
            int64_t axes[IR_MAX_RANK];
            const int64_t np = list_of(g, n, 1, "pads", list, 2 * IR_MAX_RANK);
            int64_t na = list_of(g, n, 3, "axes", axes, IR_MAX_RANK);
            if (np < 0 || na == -2) return;
            if (na < 0) {
                for (uint32_t d = 0; d < x->rank; d++) axes[d] = d;
                na = x->rank;
            }
            if (np != 2 * na) return;
            for (int64_t a = 0; a < na; a++) {
                uint32_t d;
                if (!axis_in(axes[a], x->rank, &d)) return;
                const int64_t add = list[a] + list[na + a];
                if (add) {
                    y.dims[d] = x->dims[d] < 0 ? -1 : x->dims[d] + add;
                    y.sym[d] = 0;
                }
            }
            declare(g, n, 0, &y);
            return;
        }
        case IR_Shape: {
            if (!x->known) return;
            const int64_t R = x->rank;
            int64_t start = ir_attr_int(n, "start", 0), end = ir_attr_int(n, "end", R);
            if (start < 0) start += R;
            if (end < 0) end += R;
            start = start < 0 ? 0 : start > R ? R : start;
            end = end < 0 ? 0 : end > R ? R : end;
            Shape s;
            memset(&s, 0, sizeof s);
            s.known = true;
            s.dtype = I64;
            s.rank = 1;
            s.dims[0] = end > start ? end - start : 0;
            declare(g, n, 0, &s);
            return;
        }
        case IR_Size: {
            Shape s;
            memset(&s, 0, sizeof s);
            s.known = true;
            s.dtype = I64;
            declare(g, n, 0, &s);
            return;
        }
        case IR_ConstantOfShape: {
            const int64_t ns = const_ints(g, n->inputs[0], list, IR_MAX_RANK);
            if (ns < 0) return;
            const IrAttr *v = ir_attr(n, "value");
            Shape s;
            memset(&s, 0, sizeof s);
            s.known = true;
            s.dtype = v && v->type == ATTR_TENSOR ? g->values[v->tensor].dtype : F32;
            s.rank = (uint8_t)ns;
            for (int64_t d = 0; d < ns; d++) s.dims[d] = list[d];
            declare(g, n, 0, &s);
            return;
        }
        case IR_ReduceSum: case IR_ReduceMean: case IR_ReduceMax: case IR_ReduceMin: case IR_ReduceProd:
        case IR_ReduceSumSquare: case IR_ReduceL1: case IR_ReduceL2: case IR_ReduceLogSumExp: case IR_ReduceLogSum: {
            if (!x->known) return;
            const int64_t na = list_of(g, n, 1, "axes", list, IR_MAX_RANK);
            if (na == -2) return;
            const bool keep = ir_attr_int(n, "keepdims", 1) != 0;
            if (na <= 0 && ir_attr_int(n, "noop_with_empty_axes", 0)) {
                declare(g, n, 0, &y);
                return;
            }
            bool red[IR_MAX_RANK] = {false};
            for (uint32_t d = 0; d < x->rank; d++) red[d] = na <= 0;
            for (int64_t a = 0; a < na; a++) {
                uint32_t d;
                if (!axis_in(list[a], x->rank, &d)) return;
                red[d] = true;
            }
            y.rank = 0;
            for (uint32_t d = 0; d < x->rank; d++) {
                if (red[d] && !keep) continue;
                y.dims[y.rank] = red[d] ? 1 : x->dims[d];
                y.sym[y.rank++] = red[d] ? 0 : x->sym[d];
            }
            declare(g, n, 0, &y);
            return;
        }
        case IR_ArgMax: case IR_ArgMin: {
            uint32_t axis;
            if (!x->known || !axis_in(ir_attr_int(n, "axis", 0), x->rank, &axis)) return;
            const bool keep = ir_attr_int(n, "keepdims", 1) != 0;
            y.rank = 0;
            for (uint32_t d = 0; d < x->rank; d++) {
                if (d == axis && !keep) continue;
                y.dims[y.rank] = d == axis ? 1 : x->dims[d];
                y.sym[y.rank++] = d == axis ? 0 : x->sym[d];
            }
            y.dtype = I64;
            declare(g, n, 0, &y);
            return;
        }
        case IR_Resize: case IR_Upsample: {
            /* scales or sizes of constants, the default policy (stretch) */
            if (!x->known || ir_attr(n, "axes")) return;
            const IrAttr *pol = ir_attr(n, "keep_aspect_ratio_policy");
            if (pol && pol->type == ATTR_STRING && strcmp(pol->s, "stretch")) return;
            const bool v10 = g->opset < 11 || n->op == IR_Upsample;
            const uint32_t si = v10 ? 1u : 2u, zi = 3u;
            const uint32_t sv = si < n->input_count ? n->inputs[si] : IR_NONE, zv = zi < n->input_count ? n->inputs[zi] : IR_NONE;
            const IrValue *S = sv == IR_NONE ? NULL : &g->values[sv];
            if (S && S->kind == VALUE_CONST && S->dtype == F32 && S->rank == 1 && S->dims[0] == x->rank) {
                for (uint32_t d = 0; d < x->rank; d++) {
                    const double sc = ((const float *)S->data)[d];
                    if (sc == 1.0) continue;
                    y.dims[d] = x->dims[d] < 0 ? -1 : (int64_t)(sc * (double)x->dims[d]);
                    y.sym[d] = 0;
                }
            } else if (zv != IR_NONE && const_ints(g, zv, list, IR_MAX_RANK) == x->rank &&
                       (!S || (S->kind == VALUE_CONST && S->rank == 1 && S->dims[0] == 0))) {
                for (uint32_t d = 0; d < x->rank; d++) {
                    if (list[d] == x->dims[d]) continue;
                    y.dims[d] = list[d];
                    y.sym[d] = 0;
                }
            } else {
                return;
            }
            declare(g, n, 0, &y);
            return;
        }
        case IR_DepthToSpace: case IR_SpaceToDepth: {
            const int64_t b = ir_attr_int(n, "blocksize", 0);
            if (!x->known || x->rank != 4 || b < 1) return;
            const bool to_space = n->op == IR_DepthToSpace;
            y.dims[1] = x->dims[1] < 0 ? -1 : to_space ? x->dims[1] / (b * b) : x->dims[1] * b * b;
            for (uint32_t d = 2; d < 4; d++) y.dims[d] = x->dims[d] < 0 ? -1 : to_space ? x->dims[d] * b : x->dims[d] / b;
            for (uint32_t d = 1; d < 4; d++) y.sym[d] = 0;
            declare(g, n, 0, &y);
            return;
        }
        case IR_SwiGLU:
            if (!x->known || !x->rank) return;
            y.dims[x->rank - 1] = x->dims[x->rank - 1] < 0 ? -1 : x->dims[x->rank - 1] / 2;
            y.sym[x->rank - 1] = 0;
            declare(g, n, 0, &y);
            return;
        case IR_TopK: {
            uint32_t axis;
            if (!x->known || !axis_in(ir_attr_int(n, "axis", -1), x->rank, &axis)) return;
            int64_t kk = -1;
            if (n->input_count > 1 && n->inputs[1] != IR_NONE) {
                if (const_ints(g, n->inputs[1], list, 1) != 1) return;
                kk = list[0];
            } else {
                kk = ir_attr_int(n, "k", -1);
            }
            if (kk < 0) return;
            y.dims[axis] = kk;
            y.sym[axis] = 0;
            declare(g, n, 0, &y);
            const Shape indices = with_dtype(y, I64);
            declare(g, n, 1, &indices);
            return;
        }
        default:
            return;
    }
}

void ir_infer_shapes(SpingalettGraph *g) {
    for (uint32_t k = 0; k < g->node_count; k++) infer(g, k);
}
