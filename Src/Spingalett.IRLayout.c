/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Graphs of operators: the maps of convolutions kept channels last between them. ONNX's convolutions read and write
 * maps channels first ([N, C, spatial...]); the library's kernels, on the CPU and the GPU, compute them channels last
 * ([N, spatial..., C]), which until this pass meant a transpose before and after every convolution. Here a value of
 * three axes or more may have a channels-last twin; convolutions (an attribute "channels_last" of the library's) read
 * and write twins, and so do, where an input has a twin, the operators that compute the same on either layout with
 * their attributes and constants permuted: element-wise operators and Where (other inputs made twins: constants
 * permuted once, values transposed), GroupNormalization and InstanceNormalization, MaxPool and AveragePool (the
 * attribute), GlobalAveragePool and GlobalMaxPool (as a reduction over the spatial axes), the Reduce operators keeping
 * their axes, Softmax, Concat, Split, Slice and Pad along permuted axes, Resize of constant scales or sizes, Reshape
 * that keeps the batch and the channels (merging or splitting spatial axes), Identity. A Transpose of a twin becomes
 * one of the twin's value (none where the two cancel: a view); a Transpose to channels first ([0, r - 1, 1, ...])
 * makes its input the twin of its output, computing nothing. A node of other operators reads values channels first:
 * a value only its twin has is transposed back once, before its first such reader (and the graph's outputs at the
 * end). The pass rests on the inferred shapes (Spingalett.IRShapes.c): values whose rank is unknown keep their
 * layout.
 */

#include "Spingalett.IR.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    SpingalettGraph *g;
    uint32_t *twin;                 /* [cap]: a value's channels-last twin, IR_NONE for none */
    uint8_t *computed;              /* [cap]: whether the value itself is computed (its twin may be all there is) */
    uint32_t cap;
    bool failed;
} Layout;

/* Room in the arrays for every value. */
static bool track(Layout *L) {
    if (L->g->value_count <= L->cap) return true;
    uint32_t cap = L->cap ? L->cap : 256u;
    while (cap < L->g->value_count) cap *= 2u;
    uint32_t *twin = (uint32_t *)realloc(L->twin, cap * sizeof(uint32_t));
    if (twin) L->twin = twin;
    uint8_t *computed = (uint8_t *)realloc(L->computed, cap);
    if (computed) L->computed = computed;
    if (!twin || !computed) return !(L->failed = true);
    for (uint32_t v = L->cap; v < cap; v++) {
        L->twin[v] = IR_NONE;
        L->computed[v] = 1;
    }
    L->cap = cap;
    return true;
}

/* The permutation from channels first to last of rank r ([0, 2, ..., r - 1, 1]: the twin's axis i is axis cl[i]),
   and back ([0, r - 1, 1, ..., r - 2]). */
static void to_last(uint32_t r, int64_t *p) {
    p[0] = 0;
    for (uint32_t i = 1; i + 1 < r; i++) p[i] = (int64_t)i + 1;
    p[r - 1] = 1;
}
static void to_first(uint32_t r, int64_t *p) {
    p[0] = 0;
    p[1] = (int64_t)r - 1;
    for (uint32_t i = 2; i < r; i++) p[i] = (int64_t)i - 1;
}

/* An axis of a channels-first value of rank r as the twin's (the inverse of to_last()). */
static int64_t twin_axis(int64_t axis, uint32_t r) {
    if (axis < 0) axis += r;
    return axis == 0 ? 0 : axis == 1 ? (int64_t)r - 1 : axis - 1;
}

/* A value of rank 3 or more whose shape the graph knows. */
static uint32_t rank_of(const SpingalettGraph *g, uint32_t v) {
    if (v == IR_NONE || !g->values[v].declared) return 0;
    return g->values[v].rank;
}

/* A new value: v's name with a suffix, its type, its sizes permuted by perm (declared where v is). */
static uint32_t value_from(Layout *L, uint32_t v, const char *suffix, const int64_t *perm) {
    SpingalettGraph *g = L->g;
    char name[512];
    snprintf(name, sizeof name, "%s%s", g->values[v].name, suffix);
    const uint32_t w = ir_value(g, name, strlen(name));
    if (w == IR_NONE || !track(L)) {
        L->failed = true;
        return IR_NONE;
    }
    const IrValue *src = &g->values[v];
    IrValue *dst = &g->values[w];
    dst->dtype = src->dtype;
    dst->declared = src->declared;
    dst->rank = src->rank;
    for (uint32_t d = 0; d < src->rank && d < IR_MAX_RANK; d++) {
        dst->dims[d] = src->dims[perm ? perm[d] : d];
        dst->sym[d] = src->sym[perm ? perm[d] : d];
    }
    return w;
}

/* A node appended, of in inputs and out outputs (their arrays of the graph's memory), attributes copied from n with
   up to two added or replaced (names of ints or lists of ints). */
typedef struct {
    const char *name;
    int64_t i;
    const int64_t *ints;
    uint32_t count;                 /* (ints: their count; 0 for an int) */
} NewAttr;

static IrNode *emit(Layout *L, IrOp op, const IrNode *like, const uint32_t *inputs, uint32_t in_count,
                    const uint32_t *outputs, uint32_t out_count, const NewAttr *add, uint32_t add_count) {
    SpingalettGraph *g = L->g;
    const uint32_t k = ir_node(g, op);
    if (k == IR_NONE) {
        L->failed = true;
        return NULL;
    }
    IrNode *n = &g->nodes[k];
    n->inputs = (uint32_t *)ir_alloc(g, (in_count ? in_count : 1u) * sizeof(uint32_t));
    n->outputs = (uint32_t *)ir_alloc(g, (out_count ? out_count : 1u) * sizeof(uint32_t));
    const uint32_t kept = like ? like->attr_count : 0u;
    n->attrs = (IrAttr *)ir_alloc(g, (kept + add_count + 1u) * sizeof(IrAttr));
    if (!n->inputs || !n->outputs || !n->attrs) {
        L->failed = true;
        return NULL;
    }
    memcpy(n->inputs, inputs, in_count * sizeof(uint32_t));
    memcpy(n->outputs, outputs, out_count * sizeof(uint32_t));
    n->input_count = in_count;
    n->output_count = out_count;
    n->name = like ? like->name : g->values[outputs[0]].name;
    for (uint32_t a = 0; a < kept; a++) {
        bool replaced = false;
        for (uint32_t b = 0; b < add_count && !replaced; b++) replaced = !strcmp(like->attrs[a].name, add[b].name);
        if (!replaced) n->attrs[n->attr_count++] = like->attrs[a];
    }
    for (uint32_t b = 0; b < add_count; b++) {
        IrAttr *a = &n->attrs[n->attr_count++];
        memset(a, 0, sizeof *a);
        a->name = ir_strdup(g, add[b].name, strlen(add[b].name));
        if (!a->name) L->failed = true;
        if (add[b].count || add[b].ints) {
            a->type = ATTR_INTS;
            a->count = add[b].count;
            a->ints = (int64_t *)ir_alloc(g, (add[b].count ? add[b].count : 1u) * sizeof(int64_t));
            if (a->ints) memcpy(a->ints, add[b].ints, add[b].count * sizeof(int64_t));
            else L->failed = true;
        } else {
            a->type = ATTR_INT;
            a->i = add[b].i;
        }
    }
    return n;
}

static void emit_transpose(Layout *L, uint32_t src, uint32_t dst, const int64_t *perm, uint32_t r) {
    const NewAttr p = {"perm", 0, perm, r};
    emit(L, IR_Transpose, NULL, &src, 1, &dst, 1, &p, 1);
}

/* A constant's values permuted (any type: elements moved as bytes), into a new constant of the permuted shape;
   IR_NONE when memory runs out. The constant first right-aligned to rank r (axes of size 1 before). */
static uint32_t permuted_constant(Layout *L, uint32_t v, uint32_t r) {
    SpingalettGraph *g = L->g;
    const IrValue *c = &g->values[v];
    const uint32_t lead = r - c->rank;
    int64_t dims[IR_MAX_RANK], perm[IR_MAX_RANK], src_stride[IR_MAX_RANK], st = 1;
    for (uint32_t d = 0; d < r; d++) dims[d] = d < lead ? 1 : c->dims[d - lead];
    for (uint32_t d = r; d-- > 0;) {
        src_stride[d] = st;
        st *= dims[d];
    }
    to_last(r, perm);
    char name[512];
    snprintf(name, sizeof name, "%s.cl", c->name);
    const size_t size = ir_dtype_size(c->dtype);
    const uint8_t dtype = c->dtype;
    const void *data = c->data;
    const uint32_t w = ir_value(g, name, strlen(name));
    if (w == IR_NONE || !track(L)) {
        L->failed = true;
        return IR_NONE;
    }
    IrValue *out = &g->values[w];
    out->kind = VALUE_CONST;
    out->dtype = dtype;
    out->declared = true;
    out->rank = (uint8_t)r;
    int64_t odims[IR_MAX_RANK];
    for (uint32_t d = 0; d < r; d++) out->dims[d] = odims[d] = dims[perm[d]];
    const uint64_t count = (uint64_t)st;
    out->data = ir_alloc(g, (size_t)(count ? count : 1u) * size);
    if (!out->data) {
        L->failed = true;
        return IR_NONE;
    }
    for (uint64_t i = 0; i < count; i++) {
        uint64_t q = i, at = 0;
        for (uint32_t d = r; d-- > 0;) {
            const uint64_t k = q % (uint64_t)odims[d];
            q /= (uint64_t)odims[d];
            at += k * (uint64_t)src_stride[perm[d]];
        }
        memcpy((uint8_t *)out->data + i * size, (const uint8_t *)data + at * size, size);
    }
    return w;
}

static void computed(Layout *L, uint32_t v);

/* The twin of v (rank r): its own, a constant's permuted, or a value's transposed now. */
static uint32_t twin_of(Layout *L, uint32_t v, uint32_t r) {
    /* (a constant's twin is of the rank it is read at: kept for that rank) */
    if (L->twin[v] != IR_NONE && L->g->values[L->twin[v]].rank == r) return L->twin[v];
    uint32_t w;
    if (L->g->values[v].kind == VALUE_CONST) {
        w = permuted_constant(L, v, r);
    } else {
        computed(L, v);
        int64_t perm[IR_MAX_RANK];
        to_last(r, perm);
        w = value_from(L, v, ".cl", perm);
        if (w != IR_NONE) emit_transpose(L, v, w, perm, r);
    }
    if (w != IR_NONE) L->twin[v] = w;
    return w;
}

/* v itself, transposed back from its twin before its first reader that needs it. */
static void computed(Layout *L, uint32_t v) {
    if (v == IR_NONE || L->computed[v]) return;
    int64_t perm[IR_MAX_RANK];
    const uint32_t r = L->g->values[v].rank;
    to_first(r, perm);
    emit_transpose(L, L->twin[v], v, perm, r);
    L->computed[v] = 1;
}

/* Output v of a node computed as its twin: the twin made, v left uncomputed. */
static uint32_t output_twin(Layout *L, uint32_t v) {
    int64_t perm[IR_MAX_RANK];
    to_last(L->g->values[v].rank, perm);
    const uint32_t w = value_from(L, v, ".cl", perm);
    if (w == IR_NONE) return IR_NONE;
    L->twin[v] = w;
    L->computed[v] = 0;
    return w;
}

static bool unary(IrOp op) {
    switch (op) {
        case IR_Abs: case IR_Neg: case IR_Exp: case IR_Log: case IR_Sqrt: case IR_Reciprocal: case IR_Sin: case IR_Cos:
        case IR_Tanh: case IR_Sigmoid: case IR_Erf: case IR_Relu: case IR_LeakyRelu: case IR_Elu: case IR_Selu:
        case IR_Gelu: case IR_HardSigmoid: case IR_HardSwish: case IR_Softplus: case IR_Softsign: case IR_Mish:
        case IR_Floor: case IR_Ceil: case IR_Round: case IR_Sign: case IR_Not: case IR_IsNaN: case IR_IsInf:
        case IR_Celu: case IR_ThresholdedRelu: case IR_Swish: case IR_Clip: case IR_Cast: case IR_Identity:
        case IR_Dropout:
            return true;
        default:
            return false;
    }
}

static bool broadcasting(IrOp op) {
    switch (op) {
        case IR_Add: case IR_Sub: case IR_Mul: case IR_Div: case IR_Pow: case IR_Mod: case IR_Max: case IR_Min:
        case IR_Sum: case IR_Mean: case IR_PRelu: case IR_Equal: case IR_Less: case IR_LessOrEqual: case IR_Greater:
        case IR_GreaterOrEqual: case IR_And: case IR_Or: case IR_Xor: case IR_Where:
            return true;
        default:
            return false;
    }
}

/* The integers of a constant of int64: their count, -1 when v is none. */
static int64_t const_ints(const SpingalettGraph *g, uint32_t v, int64_t *out, int64_t most) {
    if (v == IR_NONE) return -1;
    const IrValue *c = &g->values[v];
    if (c->kind != VALUE_CONST || c->dtype != SPINGALETT_DTYPE_INT64 || c->rank > 1) return -1;
    const int64_t count = c->rank ? c->dims[0] : 1;
    if (count > most) return -1;
    memcpy(out, c->data, (size_t)count * sizeof(int64_t));
    return count;
}

/* A new constant of int64 values (rank 1). */
static uint32_t int_constant(Layout *L, const char *base, const int64_t *values, uint32_t count) {
    SpingalettGraph *g = L->g;
    char name[512];
    snprintf(name, sizeof name, "%s.cl", base);
    const uint32_t w = ir_value(g, name, strlen(name));
    if (w == IR_NONE || !track(L)) {
        L->failed = true;
        return IR_NONE;
    }
    IrValue *c = &g->values[w];
    c->kind = VALUE_CONST;
    c->dtype = SPINGALETT_DTYPE_INT64;
    c->declared = true;
    c->rank = 1;
    c->dims[0] = count;
    c->data = ir_alloc(g, (count ? count : 1u) * sizeof(int64_t));
    if (!c->data) {
        L->failed = true;
        return IR_NONE;
    }
    memcpy(c->data, values, count * sizeof(int64_t));
    return w;
}

/* A constant of floats permuted as a value of rank r's axes (Resize's scales). */
static uint32_t float_list_permuted(Layout *L, uint32_t v, uint32_t r) {
    SpingalettGraph *g = L->g;
    const IrValue *c = &g->values[v];
    if (c->kind != VALUE_CONST || c->dtype != SPINGALETT_DTYPE_FLOAT32 || c->rank != 1 || c->dims[0] != r) return IR_NONE;
    float values[IR_MAX_RANK];
    int64_t perm[IR_MAX_RANK];
    to_last(r, perm);
    for (uint32_t d = 0; d < r; d++) values[d] = ((const float *)c->data)[perm[d]];
    char name[512];
    snprintf(name, sizeof name, "%s.cl", c->name);
    const uint32_t w = ir_value(g, name, strlen(name));
    if (w == IR_NONE || !track(L)) {
        L->failed = true;
        return IR_NONE;
    }
    IrValue *out = &g->values[w];
    out->kind = VALUE_CONST;
    out->dtype = SPINGALETT_DTYPE_FLOAT32;
    out->declared = true;
    out->rank = 1;
    out->dims[0] = r;
    out->data = ir_alloc(g, r * sizeof(float));
    if (!out->data) {
        L->failed = true;
        return IR_NONE;
    }
    memcpy(out->data, values, r * sizeof(float));
    return w;
}

/* Node n (a copy of the original's struct) channels last where its operator allows and an input has a twin (a
   convolution always): emitted so, true; false to leave it as it is. */
static bool as_twin(Layout *L, const IrNode *n) {
    SpingalettGraph *g = L->g;
    const IrOp op = (IrOp)n->op;
    if (!n->input_count || n->inputs[0] == IR_NONE || !n->output_count || n->outputs[0] == IR_NONE) return false;
    const uint32_t x = n->inputs[0], y = n->outputs[0], r = rank_of(g, x);
    const bool has_twin = L->twin[x] != IR_NONE;
    uint32_t ins[64];
    if (n->input_count > 64) return false;
    memcpy(ins, n->inputs, n->input_count * sizeof(uint32_t));
    int64_t perm[IR_MAX_RANK];

    if (op == IR_Conv) {
        /* one or two spatial axes, filters of constants, not a twin's yet */
        if ((r != 3 && r != 4) || rank_of(g, y) != r || n->input_count < 2 || g->values[n->inputs[1]].kind != VALUE_CONST ||
            ir_attr_int(n, "channels_last", 0))
            return false;
        ins[0] = twin_of(L, x, r);
        uint32_t out = output_twin(L, y);
        const NewAttr a = {"channels_last", 1, NULL, 0};
        return ins[0] != IR_NONE && out != IR_NONE && emit(L, op, n, ins, n->input_count, &out, 1, &a, 1);
    }
    if (op == IR_Transpose && r >= 3) {
        const IrAttr *pa = ir_attr(n, "perm");
        int64_t p[IR_MAX_RANK];
        for (uint32_t d = 0; d < r; d++) p[d] = pa && pa->type == ATTR_INTS && pa->count == r ? pa->ints[d] : (int64_t)(r - 1 - d);
        int64_t back[IR_MAX_RANK];
        to_first(r, back);
        if (has_twin) {
            /* of the twin: perm q, q[i] = (the twin's axis of p[i]); none (a view) where they cancel */
            int64_t q[IR_MAX_RANK];
            bool identity = true;
            for (uint32_t d = 0; d < r; d++) {
                q[d] = twin_axis(p[d], r);
                identity = identity && q[d] == (int64_t)d;
            }
            const uint32_t t = L->twin[x];
            if (identity) return emit(L, IR_Identity, n, &t, 1, &y, 1, NULL, 0) != NULL;
            const NewAttr a = {"perm", 0, q, r};
            return emit(L, IR_Transpose, n, &t, 1, &y, 1, &a, 1) != NULL;
        }
        /* to channels first: its input is the output's twin */
        bool first = true;
        for (uint32_t d = 0; d < r; d++) first = first && p[d] == back[d];
        if (first && rank_of(g, y) == r) {
            L->twin[y] = x;
            L->computed[y] = 0;
            return true;
        }
        return false;
    }
    if (!has_twin || r < 3) {
        /* (an element-wise node of a twin's value as its other input) */
        if (!broadcasting(op)) return false;
        bool any = false;
        for (uint32_t i = 0; i < n->input_count; i++) any = any || (n->inputs[i] != IR_NONE && L->twin[n->inputs[i]] != IR_NONE);
        if (!any) return false;
    }
    if (unary(op)) {
        if (n->output_count > 1 && n->outputs[1] != IR_NONE) return false;
        ins[0] = L->twin[x];
        uint32_t out = output_twin(L, y);
        return out != IR_NONE && emit(L, op, n, ins, n->input_count, &out, 1, NULL, 0);
    }
    if (broadcasting(op)) {
        const uint32_t R = rank_of(g, y);
        if (R < 3) return false;
        for (uint32_t i = 0; i < n->input_count; i++) {
            const uint32_t v = n->inputs[i];
            if (v == IR_NONE) continue;
            const IrValue *val = &g->values[v];
            if (L->twin[v] != IR_NONE && val->rank == R) continue;
            /* constants of rank R or fewer are permuted; values must be of rank R */
            if (!val->declared || (val->kind != VALUE_CONST && val->rank != R) || val->rank > R) return false;
        }
        for (uint32_t i = 0; i < n->input_count; i++) {
            if (n->inputs[i] == IR_NONE) continue;
            ins[i] = twin_of(L, n->inputs[i], R);
            if (ins[i] == IR_NONE) return false;
        }
        uint32_t out = output_twin(L, y);
        return out != IR_NONE && emit(L, op, n, ins, n->input_count, &out, 1, NULL, 0);
    }
    switch (op) {
        case IR_GroupNormalization: case IR_InstanceNormalization: case IR_MaxPool: case IR_AveragePool: {
            if ((op == IR_MaxPool || op == IR_AveragePool) && (r != 4 || (n->output_count > 1 && n->outputs[1] != IR_NONE)))
                return false;
            ins[0] = L->twin[x];
            uint32_t out = output_twin(L, y);
            const NewAttr a = {"channels_last", 1, NULL, 0};
            return out != IR_NONE && emit(L, op, n, ins, n->input_count, &out, 1, &a, 1);
        }
        case IR_GlobalAveragePool: case IR_GlobalMaxPool: {
            int64_t axes[IR_MAX_RANK];
            for (uint32_t d = 1; d + 1 < r; d++) axes[d - 1] = d;
            ins[0] = L->twin[x];
            uint32_t out = output_twin(L, y);
            const NewAttr a[2] = {{"axes", 0, axes, r - 2u}, {"keepdims", 1, NULL, 0}};
            return out != IR_NONE && emit(L, op == IR_GlobalAveragePool ? IR_ReduceMean : IR_ReduceMax, NULL, ins, 1, &out,
                                          1, a, 2);
        }
        case IR_ReduceSum: case IR_ReduceMean: case IR_ReduceMax: case IR_ReduceMin: case IR_ReduceProd:
        case IR_ReduceSumSquare: case IR_ReduceL1: case IR_ReduceL2: case IR_ReduceLogSumExp: case IR_ReduceLogSum: {
            if (!ir_attr_int(n, "keepdims", 1)) return false;
            int64_t axes[IR_MAX_RANK];
            int64_t na;
            if (n->input_count > 1 && n->inputs[1] != IR_NONE) na = const_ints(g, n->inputs[1], axes, IR_MAX_RANK);
            else {
                const IrAttr *aa = ir_attr(n, "axes");
                na = aa && aa->type == ATTR_INTS && aa->count <= IR_MAX_RANK ? (int64_t)aa->count : -1;
                for (int64_t k = 0; k < na; k++) axes[k] = aa->ints[k];
            }
            if (na <= 0) return false;
            for (int64_t k = 0; k < na; k++) axes[k] = twin_axis(axes[k], r);
            ins[0] = L->twin[x];
            uint32_t out = output_twin(L, y);
            const NewAttr a = {"axes", 0, axes, (uint32_t)na};
            return out != IR_NONE && emit(L, op, n, ins, 1, &out, 1, &a, 1);
        }
        case IR_Softmax: case IR_LogSoftmax: {
            if (g->opset < 13) return false;
            ins[0] = L->twin[x];
            uint32_t out = output_twin(L, y);
            const NewAttr a = {"axis", twin_axis(ir_attr_int(n, "axis", -1), r), NULL, 0};
            return out != IR_NONE && emit(L, op, n, ins, 1, &out, 1, &a, 1);
        }
        case IR_Concat: {
            for (uint32_t i = 0; i < n->input_count; i++) {
                if (n->inputs[i] == IR_NONE || rank_of(g, n->inputs[i]) != r) return false;
            }
            for (uint32_t i = 0; i < n->input_count; i++) {
                ins[i] = twin_of(L, n->inputs[i], r);
                if (ins[i] == IR_NONE) return false;
            }
            uint32_t out = output_twin(L, y);
            const NewAttr a = {"axis", twin_axis(ir_attr_int(n, "axis", 0), r), NULL, 0};
            return out != IR_NONE && emit(L, op, n, ins, n->input_count, &out, 1, &a, 1);
        }
        case IR_Split: {
            if (n->output_count > 64) return false;
            uint32_t outs[64];
            for (uint32_t o = 0; o < n->output_count; o++) {
                if (n->outputs[o] == IR_NONE || rank_of(g, n->outputs[o]) != r) return false;
            }
            for (uint32_t o = 0; o < n->output_count; o++)
                if ((outs[o] = output_twin(L, n->outputs[o])) == IR_NONE) return false;
            ins[0] = L->twin[x];
            const NewAttr a = {"axis", twin_axis(ir_attr_int(n, "axis", 0), r), NULL, 0};
            return emit(L, op, n, ins, n->input_count, outs, n->output_count, &a, 1) != NULL;
        }
        case IR_Resize: {
            /* constant scales or sizes (and no roi, no axes) permuted */
            if (ir_attr(n, "axes") || g->opset < 11) return false;
            if (n->input_count > 1 && n->inputs[1] != IR_NONE && g->values[n->inputs[1]].rank == 1 &&
                g->values[n->inputs[1]].dims[0] != 0)
                return false;
            for (uint32_t i = 2; i < n->input_count && i < 4; i++) {
                if (n->inputs[i] == IR_NONE) continue;
                const IrValue *c = &g->values[n->inputs[i]];
                if (c->kind != VALUE_CONST) return false;
                if (c->rank == 1 && c->dims[0] == 0) continue;
                if (i == 2) ins[i] = float_list_permuted(L, n->inputs[i], r);
                else {
                    int64_t sizes[IR_MAX_RANK];
                    if (const_ints(g, n->inputs[i], sizes, IR_MAX_RANK) != r) return false;
                    to_last(r, perm);
                    int64_t moved[IR_MAX_RANK];
                    for (uint32_t d = 0; d < r; d++) moved[d] = sizes[perm[d]];
                    ins[i] = int_constant(L, c->name, moved, r);
                }
                if (ins[i] == IR_NONE) return false;
            }
            ins[0] = L->twin[x];
            uint32_t out = output_twin(L, y);
            return out != IR_NONE && emit(L, op, n, ins, n->input_count, &out, 1, NULL, 0);
        }
        case IR_Reshape: {
            /* keeping the batch and the channels: [N, C, S...] to [N, C, S'...] is [N, S..., C] to [N, S'..., C] */
            const uint32_t s = rank_of(g, y);
            if (s < 3 || n->input_count < 2) return false;
            const IrValue *X = &g->values[x], *Y = &g->values[y];
            if (Y->dims[1] < 0 || Y->dims[1] != X->dims[1]) return false;
            if (!(Y->dims[0] >= 0 && Y->dims[0] == X->dims[0]) && !(Y->dims[0] < 0 && Y->sym[0] && Y->sym[0] == X->sym[0]))
                return false;
            int64_t shape[IR_MAX_RANK];
            shape[0] = 0;
            for (uint32_t d = 2; d < s; d++) {
                if (Y->dims[d] < 0) return false;
                shape[d - 1] = Y->dims[d];
            }
            shape[s - 1] = Y->dims[1];
            const uint32_t sv = int_constant(L, g->values[y].name, shape, s);
            if (sv == IR_NONE) return false;
            const uint32_t pair[2] = {L->twin[x], sv};
            uint32_t out = output_twin(L, y);
            const NewAttr a = {"allowzero", 0, NULL, 0};
            return out != IR_NONE && emit(L, op, n, pair, 2, &out, 1, &a, 1);
        }
        case IR_Slice: case IR_Pad: {
            /* constant axes (Slice's, by default its starts' first ones) or pads permuted */
            if (op == IR_Slice) {
                int64_t starts[IR_MAX_RANK], axes[IR_MAX_RANK];
                const int64_t ns = n->input_count > 1 ? const_ints(g, n->inputs[1], starts, IR_MAX_RANK) : -1;
                int64_t na = n->input_count > 3 && n->inputs[3] != IR_NONE ? const_ints(g, n->inputs[3], axes, IR_MAX_RANK) : -2;
                if (ns < 0 || na == -1) return false;
                if (na == -2) {
                    for (int64_t k = 0; k < ns; k++) axes[k] = k;
                    na = ns;
                }
                for (int64_t k = 0; k < na; k++) axes[k] = twin_axis(axes[k], r);
                uint32_t sl[5] = {L->twin[x], n->inputs[1], n->input_count > 2 ? n->inputs[2] : IR_NONE, IR_NONE,
                                  n->input_count > 4 ? n->inputs[4] : IR_NONE};
                if ((sl[3] = int_constant(L, g->values[y].name, axes, (uint32_t)na)) == IR_NONE) return false;
                uint32_t out = output_twin(L, y);
                return out != IR_NONE && emit(L, op, n, sl, n->input_count > 4 ? 5u : 4u, &out, 1, NULL, 0);
            }
            int64_t pads[2 * IR_MAX_RANK];
            if (n->input_count < 2 || (n->input_count > 3 && n->inputs[3] != IR_NONE) ||
                const_ints(g, n->inputs[1], pads, 2 * IR_MAX_RANK) != 2 * (int64_t)r)
                return false;
            int64_t moved[2 * IR_MAX_RANK];
            to_last(r, perm);
            for (uint32_t d = 0; d < r; d++) {
                moved[d] = pads[perm[d]];
                moved[r + d] = pads[r + perm[d]];
            }
            ins[0] = L->twin[x];
            if ((ins[1] = int_constant(L, g->values[n->inputs[1]].name, moved, 2 * r)) == IR_NONE) return false;
            uint32_t out = output_twin(L, y);
            return out != IR_NONE && emit(L, op, n, ins, n->input_count, &out, 1, NULL, 0);
        }
        default:
            return false;
    }
}

bool ir_layout(SpingalettGraph *g) {
    bool convs = false;
    for (uint32_t k = 0; k < g->node_count && !convs; k++)
        convs = g->nodes[k].op == IR_Conv && g->nodes[k].input_count > 0 && g->nodes[k].inputs[0] != IR_NONE &&
                g->values[g->nodes[k].inputs[0]].declared && (g->values[g->nodes[k].inputs[0]].rank == 3 ||
                                                              g->values[g->nodes[k].inputs[0]].rank == 4);
    if (!convs) return true;
    Layout L;
    memset(&L, 0, sizeof L);
    L.g = g;
    if (!track(&L)) return false;
    /* the nodes anew, in their order, with what the twins need */
    const uint32_t count = g->node_count;
    IrNode *old = (IrNode *)malloc((count ? count : 1u) * sizeof(IrNode));
    if (!old) {
        free(L.twin);
        free(L.computed);
        return false;
    }
    memcpy(old, g->nodes, count * sizeof(IrNode));
    g->node_count = 0;
    for (uint32_t k = 0; k < count && !L.failed; k++) {
        const IrNode n = old[k];
        if (as_twin(&L, &n)) continue;
        for (uint32_t i = 0; i < n.input_count; i++) computed(&L, n.inputs[i]);
        const uint32_t at = ir_node(g, (IrOp)n.op);
        if (at == IR_NONE) {
            L.failed = true;
            break;
        }
        g->nodes[at] = n;
    }
    for (uint32_t o = 0; o < g->output_count && !L.failed; o++) computed(&L, g->outputs[o]);
    free(old);
    free(L.twin);
    free(L.computed);
    if (L.failed) set_error(SPINGALETT_ERR_ALLOC, "graph: out of memory");
    return !L.failed;
}
