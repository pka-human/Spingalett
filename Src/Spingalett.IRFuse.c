/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Graphs of operators: what a graph's nodes become when it opens, after its constants are folded. Nodes of one
 * operator on the same inputs with the same attributes become one (common subexpressions). Then fusions, each of
 * which replaces a chain of nodes whose intermediate values nothing else reads by one node, of the library's own
 * operators (Spingalett.IROps.def's last) or of ONNX's, which computes the same in fewer passes over memory
 * (SPINGALETT_GRAPH_FUSE=0 in the environment leaves them out, to compare with):
 *
 *   GELU, SiLU   x * 0.5 * (1 + Erf(x / sqrt 2)), its tanh form, x * Sigmoid(x) and x * Sigmoid(a x), as exporters
 *                write them out, become Gelu and Swish
 *   Linear       MatMul by a constant matrix, then an Add of a constant row, then Relu, Sigmoid, Tanh, Gelu or SiLU
 *                (each optional), or a Gemm by a constant matrix and row then such an activation, becomes Linear:
 *                the matrix packed once, the bias and activation in the product's epilogue (an activation after
 *                a Reshape of the product too, the Reshape then making the activation's output)
 *   attention    MatMul(Q, K^T) (Q and K^T scaled by constants or not), a scaling by a constant, a Where keeping
 *                scores (the rest a large negative), an Add of a mask, Softmax over the last axis (PyTorch's safe
 *                softmax too, which zeroes rows of no keys) and MatMul by V, Casts to float between them, becomes
 *                FusedAttention: blocks of 64 queries at a time, their scores never all in memory; the
 *                transposes of heads around it are taken in
 *   Conv         a channel's Mul and Add of constants and an inference BatchNormalization after a convolution folded
 *                into new filters and a bias; then Relu, Sigmoid, Tanh, Gelu or SiLU: the activation in the
 *                convolution's epilogue (an attribute "activation" of the library's)
 *   norms        layer and RMS normalization over the last axis written out (ReduceMean, Sub, Pow, Sqrt, Div, a
 *                scale and a bias) become LayerNormalization and RMSNormalization; group normalization as exporters
 *                write it (a Reshape into groups, InstanceNormalization, a Reshape back, a channel's Mul and Add of
 *                constants), then Relu, Sigmoid, Tanh, Gelu or SiLU, becomes GroupNormalization of a scale and bias
 *                a channel, the activation in its pass (an attribute "activation" of the library's)
 *
 * A fused node takes the place of the chain's last node, which runs after everything the chain reads.
 */

#include "Spingalett.IR.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    SpingalettGraph *g;
    int32_t *producer;              /* [value_count]: the node computing it, -1 for none */
    uint32_t *uses;                 /* [value_count]: the nodes' inputs reading it, and the graph's outputs */
    uint8_t *dead;                  /* [node_count] */
} Fuse;

/* A value only one node reads (and no output of the graph is). */
static bool single(const Fuse *f, uint32_t v) {
    return v != IR_NONE && f->uses[v] == 1;
}

static IrNode *producer_of(const Fuse *f, uint32_t v, IrOp op) {
    if (v == IR_NONE || f->producer[v] < 0 || f->dead[f->producer[v]]) return NULL;
    IrNode *n = &f->g->nodes[f->producer[v]];
    return n->op == op ? n : NULL;
}

/* The only node reading v, when it is of op and reads v as its input `slot` (-1: any). */
static IrNode *consumer_of(const Fuse *f, uint32_t v, IrOp op, uint32_t *slot) {
    if (!single(f, v)) return NULL;
    for (uint32_t k = 0; k < f->g->node_count; k++) {
        IrNode *n = &f->g->nodes[k];
        if (f->dead[k]) continue;
        for (uint32_t i = 0; i < n->input_count; i++)
            if (n->inputs[i] == v) {
                if (n->op != op) return NULL;
                if (slot) *slot = i;
                return n;
            }
    }
    return NULL;
}

/* A constant of one float32 value. */
static bool scalar_of(const SpingalettGraph *g, uint32_t v, float *x) {
    if (v == IR_NONE) return false;
    const IrValue *c = &g->values[v];
    if (c->kind != VALUE_CONST || c->dtype != SPINGALETT_DTYPE_FLOAT32) return false;
    for (uint32_t d = 0; d < c->rank; d++)
        if (c->dims[d] != 1) return false;
    *x = *(const float *)c->data;
    return true;
}

static bool near(float a, double b) {
    return fabs((double)a - b) <= 1e-6 * fabs(b);
}

/* The other input of a binary node than v (IR_NONE when it reads v twice or is not binary). */
static uint32_t other(const IrNode *n, uint32_t v) {
    if (n->input_count != 2) return IR_NONE;
    if (n->inputs[0] == v && n->inputs[1] != v) return n->inputs[1];
    if (n->inputs[1] == v && n->inputs[0] != v) return n->inputs[0];
    return IR_NONE;
}

/* A node left out: what it read is read once less. */
static void kill(Fuse *f, const IrNode *n) {
    if (f->dead[n - f->g->nodes]) return;
    f->dead[n - f->g->nodes] = 1;
    for (uint32_t i = 0; i < n->input_count; i++)
        if (n->inputs[i] != IR_NONE && f->uses[n->inputs[i]]) f->uses[n->inputs[i]]--;
}

/* Node `at` rewritten as op of the inputs and one output, its attributes replaced. */
static bool rewrite(Fuse *f, IrNode *at, IrOp op, const uint32_t *inputs, uint32_t count, uint32_t output,
                    const IrAttr *attrs, uint32_t attr_count) {
    SpingalettGraph *g = f->g;
    uint32_t *in = (uint32_t *)ir_alloc(g, (count ? count : 1u) * sizeof(uint32_t));
    uint32_t *out = (uint32_t *)ir_alloc(g, sizeof(uint32_t));
    IrAttr *a = (IrAttr *)ir_alloc(g, (attr_count ? attr_count : 1u) * sizeof(IrAttr));
    if (!in || !out || !a) return false;
    memcpy(in, inputs, count * sizeof(uint32_t));
    out[0] = output;
    for (uint32_t k = 0; k < attr_count; k++) a[k] = attrs[k];
    at->op = (uint16_t)op;
    at->inputs = in;
    at->input_count = count;
    at->outputs = out;
    at->output_count = 1;
    at->attrs = a;
    at->attr_count = attr_count;
    f->producer[output] = (int32_t)(at - g->nodes);
    f->dead[at - g->nodes] = 0;
    for (uint32_t k = 0; k < count; k++)
        if (inputs[k] != IR_NONE) f->uses[inputs[k]]++;
    return true;
}

/* ------------------------------------------------------------------------- GELU and SiLU */

/* The node computing v from x with a constant: a single reader's op(x, c) or op(c, x) with c the constant (one
   value near want); NULL otherwise. */
static IrNode *with_constant(const Fuse *f, uint32_t v, IrOp op, double want, uint32_t *x) {
    IrNode *n = producer_of(f, v, op);
    float c;
    if (!n || !single(f, v) || n->input_count != 2) return NULL;
    for (int k = 0; k < 2; k++)
        if (scalar_of(f->g, n->inputs[k], &c) && near(c, want) && (op != IR_Div || k == 1)) {
            *x = n->inputs[1 - k];
            return n;
        }
    return NULL;
}

/* After 1 + erf(...) or 1 + tanh(...) at v: the products by x and by 0.5, in any order and grouping; the value they
   make, IR_NONE when they are not there. The nodes are marked in eaten. */
static uint32_t times_half_x(Fuse *f, uint32_t v, uint32_t x, IrNode **eaten, int *count) {
    bool half = false, by_x = false;
    for (int step = 0; step < 2 && !(half && by_x); step++) {
        IrNode *m = consumer_of(f, v, IR_Mul, NULL);
        if (!m) return IR_NONE;
        const uint32_t o = other(m, v);
        float c;
        uint32_t base;
        IrNode *h;
        if (!half && scalar_of(f->g, o, &c) && c == 0.5f) half = true;
        else if (!by_x && o == x) by_x = true;
        else if (!half && !by_x && (h = with_constant(f, o, IR_Mul, 0.5, &base)) && base == x) {
            half = by_x = true;
            eaten[(*count)++] = h;
        } else return IR_NONE;
        eaten[(*count)++] = m;
        v = m->outputs[0];
    }
    return half && by_x ? v : IR_NONE;
}

static bool replace_by(Fuse *f, IrNode **eaten, int count, IrOp op, uint32_t x, uint32_t y, const IrAttr *attrs,
                       uint32_t attr_count) {
    IrNode *last = eaten[0];
    for (int k = 0; k < count; k++) {
        kill(f, eaten[k]);
        if (eaten[k] > last) last = eaten[k];
    }
    return rewrite(f, last, op, &x, 1, y, attrs, attr_count);
}

/* x * 0.5 * (1 + erf(x / sqrt 2)), or with x * (1 / sqrt 2): Gelu. */
static bool fuse_gelu(Fuse *f, IrNode *erf) {
    IrNode *eaten[8];
    int count = 0;
    uint32_t x = IR_NONE;
    IrNode *pre = with_constant(f, erf->inputs[0], IR_Div, 1.4142135623730951, &x);
    if (!pre) pre = with_constant(f, erf->inputs[0], IR_Mul, 0.7071067811865476, &x);
    if (!pre) return true;
    uint32_t one;
    IrNode *add = consumer_of(f, erf->outputs[0], IR_Add, NULL);
    float c;
    if (!add || !scalar_of(f->g, other(add, erf->outputs[0]), &c) || c != 1.0f) return true;
    (void)one;
    eaten[count++] = pre;
    eaten[count++] = erf;
    eaten[count++] = add;
    const uint32_t y = times_half_x(f, add->outputs[0], x, eaten, &count);
    return y == IR_NONE ? true : replace_by(f, eaten, count, IR_Gelu, x, y, NULL, 0);
}

/* x * 0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 x^3))), x^3 as Pow(x, 3): Gelu, approximate "tanh". */
static bool fuse_gelu_tanh(Fuse *f, IrNode *tanh_node) {
    IrNode *eaten[12];
    int count = 0;
    uint32_t inner, x = IR_NONE, cube;
    IrNode *scale = with_constant(f, tanh_node->inputs[0], IR_Mul, 0.7978845608028654, &inner);
    IrNode *sum = scale ? producer_of(f, inner, IR_Add) : NULL;
    if (!sum || !single(f, inner)) return true;
    IrNode *cubic = NULL;
    for (int k = 0; k < 2 && !cubic; k++) {
        cubic = with_constant(f, sum->inputs[k], IR_Mul, 0.044715, &cube);
        if (cubic) x = sum->inputs[1 - k];
    }
    uint32_t base;
    IrNode *pow = cubic ? with_constant(f, cube, IR_Pow, 3.0, &base) : NULL;
    if (!pow || base != x) return true;
    float c;
    IrNode *add = consumer_of(f, tanh_node->outputs[0], IR_Add, NULL);
    if (!add || !scalar_of(f->g, other(add, tanh_node->outputs[0]), &c) || c != 1.0f) return true;
    eaten[count++] = pow;
    eaten[count++] = cubic;
    eaten[count++] = sum;
    eaten[count++] = scale;
    eaten[count++] = tanh_node;
    eaten[count++] = add;
    const uint32_t y = times_half_x(f, add->outputs[0], x, eaten, &count);
    if (y == IR_NONE) return true;
    IrAttr attr;
    memset(&attr, 0, sizeof attr);
    attr.name = ir_strdup(f->g, "approximate", 11);
    attr.s = ir_strdup(f->g, "tanh", 4);
    attr.type = ATTR_STRING;
    if (!attr.name || !attr.s) return false;
    return replace_by(f, eaten, count, IR_Gelu, x, y, &attr, 1);
}

/* x * Sigmoid(x): Swish (SiLU); x * Sigmoid(alpha x) (CLIP's quick GELU, alpha 1.702): Swish of that alpha. */
static bool fuse_silu(Fuse *f, IrNode *sigmoid) {
    uint32_t x = sigmoid->inputs[0];
    IrNode *scaled = producer_of(f, x, IR_Mul);
    float alpha = 1.0f;
    if (scaled && single(f, x) && scaled->input_count == 2) {
        for (int k = 0; k < 2; k++)
            if (scalar_of(f->g, scaled->inputs[k], &alpha)) {
                x = scaled->inputs[1 - k];
                break;
            }
        if (x == sigmoid->inputs[0]) scaled = NULL;
    } else scaled = NULL;
    IrNode *m = consumer_of(f, sigmoid->outputs[0], IR_Mul, NULL);
    if (!m || other(m, sigmoid->outputs[0]) != x) return true;
    IrNode *eaten[3] = {sigmoid, m, scaled};
    IrAttr attr;
    memset(&attr, 0, sizeof attr);
    attr.name = ir_strdup(f->g, "alpha", 5);
    attr.type = ATTR_FLOAT;
    attr.f = alpha;
    if (!attr.name) return false;
    return replace_by(f, eaten, scaled ? 3 : 2, IR_Swish, x, m->outputs[0], &attr, scaled ? 1u : 0u);
}

/* ------------------------------------------------------------------------- Linear */

/* The only reader of v when it is an activation the products' epilogue applies (Relu, Sigmoid, Tanh, SiLU, Gelu),
   and its ActivationFunction; NULL otherwise. */
static IrNode *activation_reader(const Fuse *f, uint32_t v, int64_t *act) {
    if (!single(f, v)) return NULL;
    for (uint32_t k = 0; k < f->g->node_count; k++) {
        IrNode *n = &f->g->nodes[k];
        if (f->dead[k] || n->input_count < 1 || n->inputs[0] != v) continue;
        if (n->op == IR_Relu) *act = ACT_RELU;
        else if (n->op == IR_Sigmoid) *act = ACT_SIGMOID;
        else if (n->op == IR_Tanh) *act = ACT_TANH;
        else if (n->op == IR_Swish && (!ir_attr(n, "alpha") || ir_attr_float(n, "alpha", 1.0f) == 1.0f)) *act = ACT_SILU;
        else if (n->op == IR_Gelu) {
            const IrAttr *ap = ir_attr(n, "approximate");
            *act = ap && ap->type == ATTR_STRING && !strcmp(ap->s, "tanh") ? ACT_GELU_TANH : ACT_GELU;
        } else return NULL;
        return n;
    }
    return NULL;
}

static bool fuse_linear(Fuse *f, IrNode *mm) {
    SpingalettGraph *g = f->g;
    if (mm->input_count < 2) return true;
    const uint32_t a = mm->inputs[0], w = mm->inputs[1];
    if (a == IR_NONE || w == IR_NONE) return true;
    const IrValue *W = &g->values[w];
    if (W->kind != VALUE_CONST || W->dtype != SPINGALETT_DTYPE_FLOAT32 || W->rank != 2 || g->values[a].kind == VALUE_CONST)
        return true;
    /* Gemm: of A as it is (no transA), alpha and beta 1, C a constant row or none; transB's matrix taken as it is */
    const bool gemm = mm->op == IR_Gemm;
    const bool trans = gemm && ir_attr_int(mm, "transB", 0) != 0;
    if (gemm && (ir_attr_int(mm, "transA", 0) || ir_attr_float(mm, "alpha", 1.0f) != 1.0f ||
                 ir_attr_float(mm, "beta", 1.0f) != 1.0f)) return true;
    const int64_t N = trans ? W->dims[0] : W->dims[1];
    uint32_t y = mm->outputs[0], bias = IR_NONE;
    if (gemm && mm->input_count > 2 && mm->inputs[2] != IR_NONE) {
        const IrValue *C = &g->values[mm->inputs[2]];
        bool row = C->kind == VALUE_CONST && C->dtype == SPINGALETT_DTYPE_FLOAT32 && C->rank >= 1 && C->dims[C->rank - 1] == N;
        for (uint32_t d = 0; row && d + 1 < C->rank; d++) row = C->dims[d] == 1;
        if (!row) return true;
        bias = mm->inputs[2];
    }
    IrNode *last = mm, *add = bias == IR_NONE ? consumer_of(f, y, IR_Add, NULL) : NULL, *act_node = NULL;
    if (add) {
        const uint32_t b = other(add, y);
        const IrValue *B = b == IR_NONE ? NULL : &g->values[b];
        bool row = B && B->kind == VALUE_CONST && B->dtype == SPINGALETT_DTYPE_FLOAT32 && B->rank >= 1 &&
                   B->dims[B->rank - 1] == N;
        for (uint32_t d = 0; row && d + 1 < B->rank; d++) row = B->dims[d] == 1;
        /* (a bias of more axes than the product would make the sum's shape larger) */
        if (row && B->rank <= 1) {
            bias = b;
            last = add;
            y = add->outputs[0];
        }
    }
    int64_t act = ACT_NONE;
    act_node = activation_reader(f, y, &act);
    /* an activation after a Reshape of the product (they commute): into the epilogue, the Reshape then making the
       activation's output */
    IrNode *hop = NULL;
    uint32_t slot;
    if (!act_node && (hop = consumer_of(f, y, IR_Reshape, &slot)) && slot == 0) {
        act_node = activation_reader(f, hop->outputs[0], &act);
        if (!act_node) hop = NULL;
    }
    if (gemm && !act_node) return true;        /* (a Gemm by a constant packs it once as it is) */
    if (act_node && hop) {
        hop->outputs[0] = act_node->outputs[0];
        f->producer[hop->outputs[0]] = (int32_t)(hop - g->nodes);
    } else if (act_node) {
        last = act_node;
        y = act_node->outputs[0];
    }
    IrAttr attrs[2];
    memset(attrs, 0, sizeof attrs);
    attrs[0].name = ir_strdup(g, "activation", 10);
    attrs[0].type = ATTR_INT;
    attrs[0].i = act;
    attrs[1].name = ir_strdup(g, "trans_b", 7);
    attrs[1].type = ATTR_INT;
    attrs[1].i = 1;
    if (!attrs[0].name || !attrs[1].name) return false;
    kill(f, mm);
    if (add && last != mm && bias != IR_NONE && !gemm) kill(f, add);
    if (act_node) kill(f, act_node);
    const uint32_t ins[3] = {a, w, bias};
    return rewrite(f, last, IR_Linear, ins, bias != IR_NONE ? 3u : 2u, y, attrs, trans ? 2u : 1u);
}

/* ------------------------------------------------------------------------- attention */

/* A Transpose of four axes by a permutation (its digits, 0x0213 for (0, 2, 1, 3)). */
static bool perm_is(const IrNode *n, unsigned digits) {
    const IrAttr *p = ir_attr(n, "perm");
    if (!p || p->type != ATTR_INTS || p->count != 4) return false;
    for (int k = 0; k < 4; k++)
        if (p->ints[k] != (int64_t)((digits >> (12 - 4 * k)) & 15u)) return false;
    return true;
}

/* Whether v is the value src of four axes with its axes permuted by `want` (digits as perm_is() reads them), through a
   chain of Reshapes and Transposes each read by nothing else (PyTorch's exports write K^T of heads as a Transpose, a
   Reshape merging the batch and the heads, a Transpose of the last two axes and a Reshape splitting them again); the
   chain's nodes, from v back, in chain (at most 8). The sizes are the graph's declared ones, numbers all. Axes are
   tracked as runs of the source's axes of sizes other than 1: a Transpose permutes them, a Reshape merges neighbours
   or splits a run where its source axes meet; axes of size 1, which may go anywhere, are matched last. */
static bool permuted(const Fuse *f, uint32_t v, unsigned want, uint32_t *src, IrNode **chain, uint32_t *length) {
    const SpingalettGraph *g = f->g;
    uint32_t values[9] = {v}, count = 0;
    while (count < 8) {
        const uint32_t x = values[count];
        if (!single(f, x) || f->producer[x] < 0 || f->dead[f->producer[x]]) break;
        IrNode *n = &g->nodes[f->producer[x]];
        if (n->op != IR_Transpose && n->op != IR_Reshape) break;
        chain[count++] = n;
        values[count] = n->inputs[0];
    }
    for (uint32_t i = count; i > 0; i--) {
        /* the farthest value of four axes the chain from it makes v of: Q, K and V before their heads' transposes */
        const IrValue *s0 = &g->values[values[i]];
        if (!s0->declared || s0->rank != 4) continue;
        bool ok = true;
        for (uint32_t d = 0; d < 4; d++) ok = ok && s0->dims[d] > 0;
        if (!ok) continue;
        /* axes as runs of source axes: run[a] (runs[a] of them) */
        uint8_t run[IR_MAX_RANK][4], runs[IR_MAX_RANK];
        uint32_t rank = 4;
        for (uint32_t a = 0; a < 4; a++) {
            run[a][0] = (uint8_t)a;
            runs[a] = s0->dims[a] != 1;
        }
        for (uint32_t j = i; ok && j > 0; j--) {
            const IrNode *n = chain[j - 1];
            const IrValue *y = &g->values[n->outputs[0]];
            if (n->op == IR_Transpose) {
                const IrAttr *pa = ir_attr(n, "perm");
                ok = pa && pa->type == ATTR_INTS && pa->count == rank;
                uint8_t r2[IR_MAX_RANK][4], n2[IR_MAX_RANK];
                for (uint32_t a = 0; ok && a < rank; a++) {
                    ok = pa->ints[a] >= 0 && pa->ints[a] < (int64_t)rank;
                    if (ok) memcpy(r2[a], run[pa->ints[a]], 4), n2[a] = runs[pa->ints[a]];
                }
                if (ok) memcpy(run, r2, sizeof r2), memcpy(runs, n2, sizeof n2);
            } else {
                /* the source axes in order, grouped again by the new sizes */
                uint8_t flat[4];
                uint32_t total = 0;
                for (uint32_t a = 0; a < rank; a++)
                    for (uint32_t b = 0; b < runs[a]; b++) flat[total++] = run[a][b];
                ok = y->declared && y->rank <= IR_MAX_RANK && y->rank > 0;
                uint32_t at = 0;
                for (uint32_t a = 0; ok && a < y->rank; a++) {
                    int64_t size = 1;
                    runs[a] = 0;
                    while (size < y->dims[a] && at < total) {
                        size *= s0->dims[flat[at]];
                        run[a][runs[a]++] = flat[at++];
                    }
                    ok = size == y->dims[a];
                }
                ok = ok && at == total;
                rank = y->rank;
            }
        }
        if (!ok || rank != 4) continue;
        /* each axis one source axis as wanted, or none where the wanted one is of size 1 */
        for (uint32_t a = 0; a < 4 && ok; a++) {
            const uint8_t w = (uint8_t)((want >> (12 - 4 * a)) & 15u);
            ok = runs[a] == 1 ? run[a][0] == w : runs[a] == 0 && s0->dims[w] == 1;
        }
        if (!ok) continue;
        *src = values[i];
        *length = i;
        return true;
    }
    return false;
}

/* The rank of a value where the graph says it: declared, or made by a Transpose's permutation or a Reshape to a
   constant shape; -1 otherwise. */
static int64_t rank_of(const Fuse *f, uint32_t v) {
    const SpingalettGraph *g = f->g;
    if (v == IR_NONE) return -1;
    if (g->values[v].declared) return g->values[v].rank;
    if (f->producer[v] < 0) return -1;
    const IrNode *n = &g->nodes[f->producer[v]];
    if (n->op == IR_Transpose) {
        const IrAttr *p = ir_attr(n, "perm");
        return p && p->type == ATTR_INTS ? (int64_t)p->count : -1;
    }
    if (n->op == IR_Reshape && n->input_count > 1 && n->inputs[1] != IR_NONE && g->values[n->inputs[1]].kind == VALUE_CONST &&
        g->values[n->inputs[1]].rank == 1) return g->values[n->inputs[1]].dims[0];
    return -1;
}

static bool fuse_attention(Fuse *f, IrNode *softmax) {
    SpingalettGraph *g = f->g;
    /* the softmax over the last axis only (-1, or the last of a rank the graph tells) */
    const IrAttr *ax = ir_attr(softmax, "axis");
    if (ax && ax->type == ATTR_INT && ax->i >= 0) {
        IrNode *mm = NULL;
        uint32_t v = softmax->inputs[0];
        /* (the scores' rank: their product's operands') */
        for (int hops = 0; hops < 3 && !mm && v != IR_NONE && f->producer[v] >= 0; hops++) {
            IrNode *n = &g->nodes[f->producer[v]];
            if (n->op == IR_MatMul) mm = n;
            else v = n->inputs[0];
        }
        const int64_t rank = mm ? (rank_of(f, mm->inputs[0]) > rank_of(f, mm->inputs[1]) ? rank_of(f, mm->inputs[0])
                                                                                          : rank_of(f, mm->inputs[1])) : -1;
        if (rank < 2 || ax->i != rank - 1) return true;
    } else if (!(ax ? ax->type == ATTR_INT && ax->i == -1 : g->opset >= 13)) {
        return true;
    }
    /* back from the softmax's input: the mask added, a Where keeping some scores (the rest a large negative fill),
       Casts to float (which change nothing of floats), a scaling, the product: in that order */
    uint32_t v = softmax->inputs[0], mask = IR_NONE, keep = IR_NONE;
    float scale = 1.0f, fill = 0.0f, c;
    IrNode *eaten[16], *qk = NULL;
    int count = 0, stage = 4;               /* 3 mask, 2 Where, 1 scale */
    while (count < 12) {
        if (!single(f, v) || f->producer[v] < 0 || f->dead[f->producer[v]]) return true;
        IrNode *n = &g->nodes[f->producer[v]];
        if (n->op == IR_MatMul) {
            qk = n;
            break;
        }
        if (n->op == IR_Cast && ir_attr_int(n, "to", 0) == SPINGALETT_DTYPE_FLOAT32) {
            v = n->inputs[0];
        } else if (n->op == IR_Add && stage > 3) {
            /* scores + mask: the scores the operand the chain goes on from */
            uint32_t s0 = IR_NONE;
            for (int k = 0; k < 2 && s0 == IR_NONE; k++) {
                const uint32_t o = n->inputs[k];
                if (o != IR_NONE && single(f, o) && f->producer[o] >= 0) {
                    const IrOp op = (IrOp)g->nodes[f->producer[o]].op;
                    if (op == IR_MatMul || op == IR_Mul || op == IR_Div || op == IR_Where || op == IR_Cast) {
                        s0 = o;
                        mask = n->inputs[1 - k];
                    }
                }
            }
            if (s0 == IR_NONE) return true;
            v = s0;
            stage = 3;
        } else if (n->op == IR_Where && stage > 2 && n->input_count == 3 && scalar_of(g, n->inputs[2], &fill) &&
                   fill <= -1e30f) {
            keep = n->inputs[0];
            v = n->inputs[1];
            stage = 2;
        } else if ((n->op == IR_Mul || n->op == IR_Div) && stage > 1 && n->input_count == 2) {
            if (n->op == IR_Div && scalar_of(g, n->inputs[1], &c) && c != 0.0f) {
                scale = 1.0f / c;
                v = n->inputs[0];
            } else if (n->op == IR_Mul && scalar_of(g, n->inputs[1], &c)) {
                scale = c;
                v = n->inputs[0];
            } else if (n->op == IR_Mul && scalar_of(g, n->inputs[0], &c)) {
                scale = c;
                v = n->inputs[1];
            } else return true;
            stage = 1;
        } else {
            return true;
        }
        eaten[count++] = n;
    }
    if (!qk) return true;
    /* the probabilities (a Cast to float after them changes nothing), or PyTorch's safe softmax of them:
       Where(IsNaN(p), 0, p), which makes rows of no keys 0 */
    uint32_t probs = softmax->outputs[0], slot;
    IrNode *isnan_node = NULL, *where = NULL, *cast = consumer_of(f, probs, IR_Cast, NULL);
    if (cast && ir_attr_int(cast, "to", 0) == SPINGALETT_DTYPE_FLOAT32) probs = cast->outputs[0];
    else cast = NULL;
    if (f->uses[probs] == 2) {
        for (uint32_t k = 0; k < g->node_count && (!isnan_node || !where); k++) {
            IrNode *n = &g->nodes[k];
            if (f->dead[k]) continue;
            if (n->op == IR_IsNaN && n->inputs[0] == probs) isnan_node = n;
            if (n->op == IR_Where && n->input_count == 3 && n->inputs[2] == probs) where = n;
        }
        if (!isnan_node || !where || where->inputs[0] != isnan_node->outputs[0] || !single(f, isnan_node->outputs[0]) ||
            !scalar_of(g, where->inputs[1], &c) || c != 0.0f) return true;
        probs = where->outputs[0];
    }
    IrNode *pv = consumer_of(f, probs, IR_MatMul, &slot);
    if (!pv || slot != 0) return true;
    /* Q and K^T each scaled by a constant (PyTorch's scaled_dot_product_attention scales both by a root) */
    uint32_t q = qk->inputs[0], kt = qk->inputs[1], base;
    IrNode *qs = producer_of(f, q, IR_Mul), *ks = producer_of(f, kt, IR_Mul);
    if (qs && single(f, q) && qs->input_count == 2 && (scalar_of(g, qs->inputs[1], &c) || scalar_of(g, qs->inputs[0], &c))) {
        base = scalar_of(g, qs->inputs[1], &c) ? qs->inputs[0] : qs->inputs[1];
        scale *= c;
        q = base;
    } else qs = NULL;
    if (ks && single(f, kt) && ks->input_count == 2 && (scalar_of(g, ks->inputs[1], &c) || scalar_of(g, ks->inputs[0], &c))) {
        base = scalar_of(g, ks->inputs[1], &c) ? ks->inputs[0] : ks->inputs[1];
        scale *= c;
        kt = base;
    } else ks = NULL;
    /* the transposes of heads around it (Q, K and V as (B, S, H, D), the output's back to that): taken in */
    uint32_t vv = pv->inputs[1], y = pv->outputs[0], xq, xk, xv, lq = 0, lk = 0, lv = 0;
    IrNode *cq[8], *ck[8], *cv[8];
    const bool heads = permuted(f, q, 0x0213, &xq, cq, &lq) && permuted(f, kt, 0x0231, &xk, ck, &lk) &&
                       permuted(f, vv, 0x0213, &xv, cv, &lv);
    IrNode *ty = consumer_of(f, y, IR_Transpose, NULL);
    const bool layout = heads && ty && perm_is(ty, 0x0213) && xv != xq && xk != xq;
    if (layout) {
        q = xq;
        kt = xk;
        vv = xv;
        y = ty->outputs[0];
    }
    const uint32_t ins[5] = {q, kt, vv, mask, keep};
    IrAttr attrs[4];
    memset(attrs, 0, sizeof attrs);
    attrs[0].name = ir_strdup(g, "scale", 5);
    attrs[0].type = ATTR_FLOAT;
    attrs[0].f = scale;
    attrs[1].name = ir_strdup(g, "safe", 4);
    attrs[1].type = ATTR_INT;
    attrs[1].i = where != NULL;
    attrs[2].name = ir_strdup(g, "layout", 6);
    attrs[2].type = ATTR_INT;
    attrs[2].i = layout;
    attrs[3].name = ir_strdup(g, "fill", 4);
    attrs[3].type = ATTR_FLOAT;
    attrs[3].f = fill;
    if (!attrs[0].name || !attrs[1].name || !attrs[2].name || !attrs[3].name) return false;
    if (layout) {
        for (uint32_t j = 0; j < lq; j++) kill(f, cq[j]);
        for (uint32_t j = 0; j < lk; j++) kill(f, ck[j]);
        for (uint32_t j = 0; j < lv; j++) kill(f, cv[j]);
        kill(f, ty);
    }
    kill(f, qk);
    kill(f, pv);
    if (qs) kill(f, qs);
    if (ks) kill(f, ks);
    for (int k = 0; k < count; k++) kill(f, eaten[k]);
    kill(f, softmax);
    if (cast) kill(f, cast);
    if (isnan_node) kill(f, isnan_node);
    if (where) kill(f, where);
    const uint32_t inputs = keep != IR_NONE ? 5u : mask != IR_NONE ? 4u : 3u;
    return rewrite(f, layout ? ty : pv, IR_FusedAttention, ins, inputs, y, attrs, keep != IR_NONE ? 4u : 3u);
}

/* ------------------------------------------------------------------------- normalizations written out */

/* A ReduceMean over the last axis alone (its axes an attribute or a constant input), keeping it. */
static bool mean_of_last(const Fuse *f, const IrNode *n) {
    if (n->op != IR_ReduceMean || ir_attr_int(n, "keepdims", 1) == 0) return false;
    const IrAttr *a = ir_attr(n, "axes");
    if (a) return a->type == ATTR_INTS && a->count == 1 && a->ints[0] == -1;
    if (n->input_count < 2 || n->inputs[1] == IR_NONE) return false;
    const IrValue *axes = &f->g->values[n->inputs[1]];
    return axes->kind == VALUE_CONST && axes->dtype == SPINGALETT_DTYPE_INT64 && axes->rank == 1 && axes->dims[0] == 1 &&
           ((const int64_t *)axes->data)[0] == -1;
}

/* A constant of float32 of one axis (a scale or a bias over the last axis), or of one value. */
static bool row_constant(const SpingalettGraph *g, uint32_t v) {
    if (v == IR_NONE) return false;
    const IrValue *c = &g->values[v];
    return c->kind == VALUE_CONST && c->dtype == SPINGALETT_DTYPE_FLOAT32 && c->rank == 1;
}

/* LayerNormalization as exporters write it out: mean = ReduceMean(x), d = x - mean, var = ReduceMean(d^2),
   (d / Sqrt(var + eps)) * gamma + beta (gamma and beta optional); and RMSNormalization: x / Sqrt(ReduceMean(x^2) + eps)
   (or x * 1 / that) * gamma. */
static bool fuse_layer_norm(Fuse *f, IrNode *first) {
    SpingalettGraph *g = f->g;
    if (!mean_of_last(f, first)) return true;
    const uint32_t x = first->inputs[0];
    float c, eps;
    /* d = x - mean: read by the square and the division */
    IrNode *sub = NULL, *pow = NULL, *mean2 = NULL;
    for (uint32_t k = 0; k < g->node_count; k++) {
        IrNode *n = &g->nodes[k];
        if (!f->dead[k] && n->op == IR_Sub && n->input_count == 2 && n->inputs[0] == x && n->inputs[1] == first->outputs[0]) sub = n;
    }
    bool rms = false;
    uint32_t centered = IR_NONE;
    if (sub && single(f, first->outputs[0]) && f->uses[sub->outputs[0]] == 2) {
        centered = sub->outputs[0];
    } else if (!sub) {
        /* RMS: the first mean is of x^2 */
        IrNode *sq = producer_of(f, x, IR_Pow);
        if (!sq || !single(f, x) || !scalar_of(g, sq->inputs[1], &c) || c != 2.0f) return true;
        rms = true;
        pow = sq;
        mean2 = first;
        centered = sq->inputs[0];
    } else {
        return true;
    }
    if (!rms) {
        for (uint32_t k = 0; k < g->node_count && !pow; k++) {
            IrNode *n = &g->nodes[k];
            if (!f->dead[k] && n->op == IR_Pow && n->inputs[0] == centered && scalar_of(g, n->inputs[1], &c) && c == 2.0f) pow = n;
        }
        if (!pow || !single(f, pow->outputs[0])) return true;
        mean2 = consumer_of(f, pow->outputs[0], IR_ReduceMean, NULL);
        if (!mean2 || !mean_of_last(f, mean2)) return true;
    }
    IrNode *add = consumer_of(f, mean2->outputs[0], IR_Add, NULL);
    if (!add || !scalar_of(g, other(add, mean2->outputs[0]), &eps)) return true;
    IrNode *root = consumer_of(f, add->outputs[0], IR_Sqrt, NULL);
    if (!root) return true;
    /* d / root, or d * Reciprocal(root) */
    IrNode *div = consumer_of(f, root->outputs[0], IR_Div, NULL), *inv = NULL, *scale = NULL;
    uint32_t normed;
    if (div && div->inputs[0] == centered && div->inputs[1] == root->outputs[0]) {
        normed = div->outputs[0];
    } else if ((inv = consumer_of(f, root->outputs[0], IR_Reciprocal, NULL)) &&
               (scale = consumer_of(f, inv->outputs[0], IR_Mul, NULL)) && other(scale, inv->outputs[0]) == centered) {
        div = NULL;
        normed = scale->outputs[0];
    } else return true;
    if (!rms && !(single(f, normed) || !consumer_of(f, normed, IR_Mul, NULL))) return true;
    if (rms && f->uses[centered] != 2) return true;
    /* * gamma, + beta */
    IrNode *mg = consumer_of(f, normed, IR_Mul, NULL), *ab = NULL;
    uint32_t gamma = IR_NONE, beta = IR_NONE, y = normed;
    if (mg && row_constant(g, other(mg, normed))) {
        gamma = other(mg, normed);
        y = mg->outputs[0];
        ab = rms ? NULL : consumer_of(f, y, IR_Add, NULL);
        if (ab && row_constant(g, other(ab, y))) {
            beta = other(ab, y);
            y = ab->outputs[0];
        } else ab = NULL;
    } else mg = NULL;
    if (gamma != IR_NONE && beta != IR_NONE && g->values[gamma].dims[0] != g->values[beta].dims[0]) return true;
    /* the scale: gamma, or ones (a constant made here) */
    if (gamma == IR_NONE) return true;      /* (normalizations without a scale stay written out) */
    IrNode *eaten[10] = {first, sub, pow, mean2, add, root, div, inv, scale, mg};
    IrNode *last = mg;
    if (ab) last = ab;
    for (int k = 0; k < 10; k++) if (eaten[k]) kill(f, eaten[k]);
    if (ab) kill(f, ab);
    IrAttr attrs[2];
    memset(attrs, 0, sizeof attrs);
    attrs[0].name = ir_strdup(g, "axis", 4);
    attrs[0].type = ATTR_INT;
    attrs[0].i = -1;
    attrs[1].name = ir_strdup(g, "epsilon", 7);
    attrs[1].type = ATTR_FLOAT;
    attrs[1].f = eps;
    if (!attrs[0].name || !attrs[1].name) return false;
    const uint32_t ins[3] = {rms ? centered : x, gamma, beta};
    return rewrite(f, last, rms ? IR_RMSNormalization : IR_LayerNormalization, ins, beta != IR_NONE ? 3u : 2u, y, attrs, 2);
}

static uint64_t value_count(const IrValue *v) {
    uint64_t count = 1;
    for (uint32_t d = 0; d < v->rank; d++) count *= (uint64_t)(v->dims[d] < 0 ? 0 : v->dims[d]);
    return count;
}

/* The integers of a constant of int64 (at most most of them): their count, -1 for no such constant. */
static int64_t const_ints(const SpingalettGraph *g, uint32_t v, int64_t *out, int64_t most) {
    if (v == IR_NONE) return -1;
    const IrValue *c = &g->values[v];
    if (c->kind != VALUE_CONST || c->dtype != SPINGALETT_DTYPE_INT64 || c->rank != 1 || c->dims[0] > most) return -1;
    memcpy(out, c->data, (size_t)c->dims[0] * sizeof(int64_t));
    return c->dims[0];
}

static uint32_t new_constant(SpingalettGraph *g, const char *name, uint32_t rank, const int64_t *dims, float **data);
static const float *per_channel(const SpingalettGraph *g, uint32_t v, uint32_t rank, int64_t channels);

/* Reshape(x into [N, G, ...]), InstanceNormalization of constants a group, Reshape back to x's shape, Mul and Add of
   constants a channel, then an activation (optional): GroupNormalization of x with the scale and bias a channel
   folded together. */
static bool fuse_group_norm(Fuse *f, IrNode *inorm) {
    SpingalettGraph *g = f->g;
    if (inorm->input_count < 3 || inorm->output_count < 1) return true;
    const uint32_t grouped = inorm->inputs[0];
    IrNode *r1 = producer_of(f, grouped, IR_Reshape);
    if (!r1 || !single(f, grouped) || r1->input_count < 2) return true;
    const uint32_t x = r1->inputs[0];
    const IrValue *X = &g->values[x];
    if (!X->declared || X->rank < 3 || X->dtype != SPINGALETT_DTYPE_FLOAT32 || X->dims[1] <= 0) return true;
    const int64_t C = X->dims[1];
    const IrValue *S = &g->values[inorm->inputs[1]], *Bv = &g->values[inorm->inputs[2]];
    if (S->kind != VALUE_CONST || Bv->kind != VALUE_CONST || S->dtype != SPINGALETT_DTYPE_FLOAT32 ||
        Bv->dtype != SPINGALETT_DTYPE_FLOAT32 || S->rank != 1 || Bv->rank != 1 || S->dims[0] != Bv->dims[0])
        return true;
    const int64_t G = S->dims[0];
    int64_t shape[IR_MAX_RANK];
    /* into groups: [N or 0, G, the rest] */
    const int64_t n1 = const_ints(g, r1->inputs[1], shape, IR_MAX_RANK);
    if (n1 != 3 || G < 1 || C % G || shape[1] != G || (shape[0] != 0 && shape[0] != X->dims[0]) ||
        (shape[2] != -1 && (X->dims[2] < 0 || shape[2] * G != C * X->dims[2] * (X->rank > 3 ? X->dims[3] : 1))))
        return true;
    IrNode *r2 = consumer_of(f, inorm->outputs[0], IR_Reshape, NULL);
    if (!r2 || r2->input_count < 2 || r2->inputs[0] != inorm->outputs[0]) return true;
    /* back: x's shape, each size its own or 0 (copied) */
    const int64_t n2 = const_ints(g, r2->inputs[1], shape, IR_MAX_RANK);
    if (n2 != X->rank) return true;
    for (uint32_t d = 0; d < X->rank; d++)
        if (shape[d] != 0 && (X->dims[d] < 0 || shape[d] != X->dims[d])) return true;
    IrNode *mul = consumer_of(f, r2->outputs[0], IR_Mul, NULL);
    if (!mul) return true;
    const float *gamma = per_channel(g, other(mul, r2->outputs[0]), X->rank, C);
    IrNode *add = gamma ? consumer_of(f, mul->outputs[0], IR_Add, NULL) : NULL;
    const float *beta = add ? per_channel(g, other(add, mul->outputs[0]), X->rank, C) : NULL;
    if (!gamma || !beta) return true;
    /* (per_channel()'s constants: a value a channel, or one for all) */
    const bool g_one = value_count(&g->values[other(mul, r2->outputs[0])]) == 1;
    const bool b_one = value_count(&g->values[other(add, mul->outputs[0])]) == 1;
    int64_t act = ACT_NONE;
    IrNode *last = add, *act_node = activation_reader(f, add->outputs[0], &act);
    uint32_t y = add->outputs[0];
    if (act_node) {
        last = act_node;
        y = act_node->outputs[0];
    }
    /* the scale and bias a channel: gamma s[g], gamma b[g] + beta (the values may move as constants are added) */
    const float *gs = (const float *)S->data, *gb = (const float *)Bv->data;
    const uint32_t old_count = g->value_count;
    char name[64];
    snprintf(name, sizeof name, "groupnorm_scale_%u", (unsigned)(inorm - g->nodes));
    const int64_t cdims[1] = {C};
    float *sc, *bi;
    const uint32_t scale = new_constant(g, name, 1, cdims, &sc);
    snprintf(name, sizeof name, "groupnorm_bias_%u", (unsigned)(inorm - g->nodes));
    const uint32_t bias = scale == IR_NONE ? IR_NONE : new_constant(g, name, 1, cdims, &bi);
    if (bias == IR_NONE) return false;
    for (int64_t c = 0; c < C; c++) {
        const int64_t grp = c / (C / G);
        const float gm = gamma[g_one ? 0 : c], bt = beta[b_one ? 0 : c];
        sc[c] = gm * gs[grp];
        bi[c] = gm * gb[grp] + bt;
    }
    int32_t *producer = (int32_t *)realloc(f->producer, (size_t)g->value_count * sizeof(int32_t));
    if (producer) f->producer = producer;
    uint32_t *uses = (uint32_t *)realloc(f->uses, (size_t)g->value_count * sizeof(uint32_t));
    if (uses) f->uses = uses;
    if (!producer || !uses) return false;
    for (uint32_t v = old_count; v < g->value_count; v++) {
        f->producer[v] = -1;
        f->uses[v] = 0;
    }
    IrAttr attrs[3];
    memset(attrs, 0, sizeof attrs);
    attrs[0].name = ir_strdup(g, "num_groups", 10);
    attrs[0].type = ATTR_INT;
    attrs[0].i = G;
    attrs[1].name = ir_strdup(g, "epsilon", 7);
    attrs[1].type = ATTR_FLOAT;
    attrs[1].f = ir_attr_float(inorm, "epsilon", 1e-5f);
    attrs[2].name = ir_strdup(g, "activation", 10);
    attrs[2].type = ATTR_INT;
    attrs[2].i = act;
    if (!attrs[0].name || !attrs[1].name || !attrs[2].name) return false;
    kill(f, r1);
    kill(f, inorm);
    kill(f, r2);
    kill(f, mul);
    if (act_node) kill(f, add);
    kill(f, last);
    const uint32_t ins[3] = {x, scale, bias};
    return rewrite(f, last, IR_GroupNormalization, ins, 3, y, attrs, act != ACT_NONE ? 3u : 2u);
}

/* ------------------------------------------------------------------------- convolutions */

/* A constant of float32 of one value a channel of a convolution's output of `rank` axes (channels second): its
   values, NULL otherwise. */
static const float *per_channel(const SpingalettGraph *g, uint32_t v, uint32_t rank, int64_t channels) {
    if (v == IR_NONE) return NULL;
    const IrValue *c = &g->values[v];
    if (c->kind != VALUE_CONST || c->dtype != SPINGALETT_DTYPE_FLOAT32 || c->rank > rank) return NULL;
    /* (right-aligned to the output's axes: the channels' axis of their count or 1, every other axis 1) */
    for (uint32_t d = 0; d < c->rank; d++)
        if (c->dims[d] != 1 && !(d + rank - c->rank == 1 && c->dims[d] == channels)) return NULL;
    return (const float *)c->data;
}

/* A new constant of float32 (the graph's memory), its value's index; IR_NONE when memory runs out. */
static uint32_t new_constant(SpingalettGraph *g, const char *name, uint32_t rank, const int64_t *dims, float **data) {
    uint64_t count = 1;
    for (uint32_t d = 0; d < rank; d++) count *= (uint64_t)dims[d];
    const uint32_t v = ir_value(g, name, strlen(name));
    if (v == IR_NONE) return IR_NONE;
    IrValue *value = &g->values[v];
    value->kind = VALUE_CONST;
    value->dtype = SPINGALETT_DTYPE_FLOAT32;
    value->rank = (uint8_t)rank;
    value->declared = true;
    for (uint32_t d = 0; d < IR_MAX_RANK; d++) value->dims[d] = d < rank ? dims[d] : -1;
    value->data = ir_alloc(g, (size_t)(count ? count : 1u) * sizeof(float));
    *data = (float *)value->data;
    return value->data ? v : IR_NONE;
}

/* Conv with constant filters then, a channel each, Mul by a constant, Add of a constant or an inference
   BatchNormalization of constants: the filters and the bias made anew with them folded in, once. */
static bool fuse_conv_affine(Fuse *f, IrNode *conv) {
    SpingalettGraph *g = f->g;
    if (conv->input_count < 2 || conv->inputs[1] == IR_NONE || ir_attr(conv, "activation")) return true;
    const IrValue *W = &g->values[conv->inputs[1]];
    const IrValue *B = conv->input_count > 2 && conv->inputs[2] != IR_NONE ? &g->values[conv->inputs[2]] : NULL;
    if (W->kind != VALUE_CONST || W->dtype != SPINGALETT_DTYPE_FLOAT32 || W->rank < 3 ||
        (B && (B->kind != VALUE_CONST || B->dtype != SPINGALETT_DTYPE_FLOAT32 || B->rank != 1))) return true;
    const uint32_t rank = W->rank;
    const int64_t OC = W->dims[0];
    uint64_t per = 1;
    for (uint32_t d = 1; d < rank; d++) per *= (uint64_t)W->dims[d];
    double *scale = (double *)malloc((size_t)OC * sizeof(double)), *shift = (double *)malloc((size_t)OC * sizeof(double));
    if (!scale || !shift) { free(scale); free(shift); return false; }
    for (int64_t c = 0; c < OC; c++) {
        scale[c] = 1.0;
        shift[c] = B ? ((const float *)B->data)[c] : 0.0;
    }
    uint32_t y = conv->outputs[0];
    IrNode *eaten[8];
    int count = 0;
    while (count < 8 && single(f, y)) {
        IrNode *n = NULL;
        for (uint32_t k = 0; k < g->node_count && !n; k++)
            if (!f->dead[k] && g->nodes[k].input_count > 0 && (g->nodes[k].inputs[0] == y ||
                (g->nodes[k].input_count > 1 && g->nodes[k].inputs[1] == y))) n = &g->nodes[k];
        if (!n) break;
        const float *k1;
        if ((n->op == IR_Mul || n->op == IR_Add) && n->input_count == 2 && (k1 = per_channel(g, other(n, y), rank, OC))) {
            const IrValue *kv = &g->values[other(n, y)];
            uint64_t kc = 1;
            for (uint32_t d = 0; d < kv->rank; d++) kc *= (uint64_t)kv->dims[d];
            for (int64_t c = 0; c < OC; c++) {
                const double a = k1[kc == 1 ? 0 : c];
                if (n->op == IR_Mul) { scale[c] *= a; shift[c] *= a; }
                else shift[c] += a;
            }
        } else if (n->op == IR_BatchNormalization && n->inputs[0] == y && n->input_count >= 5 && n->output_count >= 1 &&
                   !ir_attr_int(n, "training_mode", 0)) {
            const float *gm = per_channel(g, n->inputs[1], 1, OC), *bt = per_channel(g, n->inputs[2], 1, OC);
            const float *mu = per_channel(g, n->inputs[3], 1, OC), *var = per_channel(g, n->inputs[4], 1, OC);
            bool used = false;
            for (uint32_t o = 1; o < n->output_count; o++) used |= n->outputs[o] != IR_NONE;
            if (!gm || !bt || !mu || !var || used) break;
            const double eps = ir_attr_float(n, "epsilon", 1e-5f);
            for (int64_t c = 0; c < OC; c++) {
                const double k = (double)gm[c] / sqrt((double)var[c] + eps);
                scale[c] *= k;
                shift[c] = (shift[c] - mu[c]) * k + bt[c];
            }
        } else break;
        eaten[count++] = n;
        y = n->outputs[0];
    }
    if (!count) { free(scale); free(shift); return true; }
    char name[96];
    snprintf(name, sizeof name, "%.80s/filters", conv->name ? conv->name : "conv");
    float *w, *b;
    const int64_t bias_dims[1] = {OC};
    const uint32_t wv = new_constant(g, name, rank, W->dims, &w);
    snprintf(name, sizeof name, "%.80s/bias", conv->name ? conv->name : "conv");
    const uint32_t bv = wv == IR_NONE ? IR_NONE : new_constant(g, name, 1, bias_dims, &b);
    if (bv == IR_NONE) { free(scale); free(shift); return false; }
    W = &g->values[conv->inputs[1]];        /* (ir_value may have moved the values) */
    for (int64_t c = 0; c < OC; c++) {
        for (uint64_t i = 0; i < per; i++) w[(uint64_t)c * per + i] = (float)(((const float *)W->data)[(uint64_t)c * per + i] * scale[c]);
        b[c] = (float)shift[c];
    }
    free(scale);
    free(shift);
    /* the values' tables grew: the fusion's own copies follow */
    int32_t *producer = (int32_t *)realloc(f->producer, (size_t)g->value_count * sizeof(int32_t));
    uint32_t *uses = (uint32_t *)realloc(f->uses, (size_t)g->value_count * sizeof(uint32_t));
    if (!producer || !uses) {
        if (producer) f->producer = producer;
        if (uses) f->uses = uses;
        return false;
    }
    f->producer = producer;
    f->uses = uses;
    for (uint32_t v = g->value_count - 2; v < g->value_count; v++) { f->producer[v] = -1; f->uses[v] = 0; }
    const uint32_t ins[3] = {conv->inputs[0], wv, bv};
    IrNode *last = eaten[count - 1];
    kill(f, conv);
    for (int k = 0; k < count; k++) kill(f, eaten[k]);
    return rewrite(f, last, IR_Conv, ins, 3, y, conv->attrs, conv->attr_count);
}

/* Conv then Relu, Sigmoid, Tanh, Gelu or Swish: the activation an attribute of the Conv, applied in its epilogue. */
static bool fuse_conv_activation(Fuse *f, IrNode *conv) {
    const uint32_t y = conv->outputs[0];
    if (ir_attr(conv, "activation")) return true;
    int64_t act = ACT_NONE;
    IrNode *act_node = activation_reader(f, y, &act);
    if (!act_node) return true;
    IrAttr *attrs = (IrAttr *)ir_alloc(f->g, (conv->attr_count + 1u) * sizeof(IrAttr));
    if (!attrs) return false;
    memcpy(attrs, conv->attrs, conv->attr_count * sizeof(IrAttr));
    IrAttr *a = &attrs[conv->attr_count];
    memset(a, 0, sizeof *a);
    a->name = ir_strdup(f->g, "activation", 10);
    a->type = ATTR_INT;
    a->i = act;
    if (!a->name) return false;
    uint32_t ins[3];
    const uint32_t count = conv->input_count < 3 ? conv->input_count : 3;
    memcpy(ins, conv->inputs, count * sizeof(uint32_t));
    const uint32_t out = act_node->outputs[0];
    kill(f, conv);
    kill(f, act_node);
    return rewrite(f, act_node, IR_Conv, ins, count, out, attrs, conv->attr_count + 1u);
}

/* ------------------------------------------------------------------------- common subexpressions */

static bool same_attrs(const IrNode *a, const IrNode *b) {
    if (a->attr_count != b->attr_count) return false;
    for (uint32_t k = 0; k < a->attr_count; k++) {
        const IrAttr *x = &a->attrs[k], *y = ir_attr(b, x->name);
        if (!y || x->type != y->type) return false;
        switch (x->type) {
            case ATTR_INT: if (x->i != y->i) return false; break;
            case ATTR_FLOAT: if (memcmp(&x->f, &y->f, sizeof x->f)) return false; break;
            case ATTR_INTS: if (x->count != y->count || (x->count && memcmp(x->ints, y->ints, x->count * sizeof(int64_t)))) return false; break;
            case ATTR_FLOATS: if (x->count != y->count || (x->count && memcmp(x->floats, y->floats, x->count * sizeof(float)))) return false; break;
            case ATTR_STRING: if (strcmp(x->s, y->s)) return false; break;
            default: if (x->tensor != y->tensor) return false; break;
        }
    }
    return true;
}

/* Nodes of one operator on the same inputs with the same attributes compute the same: each later one is left out
   and what read its outputs reads the first's (not where an output of the graph would lose its name). */
static bool common_subexpressions(SpingalettGraph *g) {
    const uint32_t N = g->node_count;
    uint32_t *rename = (uint32_t *)malloc((size_t)(g->value_count ? g->value_count : 1u) * sizeof(uint32_t));
    uint8_t *output = (uint8_t *)calloc(g->value_count ? g->value_count : 1u, 1);
    uint32_t *first = (uint32_t *)malloc((size_t)(N ? N : 1u) * sizeof(uint32_t));
    uint8_t *gone = (uint8_t *)calloc(N ? N : 1u, 1);
    /* a table of the nodes by a hash of their operator and inputs */
    uint32_t cap = 64;
    while (cap < 2u * N + 2u) cap *= 2;
    uint32_t *table = (uint32_t *)calloc(cap, sizeof(uint32_t));
    if (!rename || !output || !first || !gone || !table) {
        free(rename); free(output); free(first); free(gone); free(table);
        set_error(SPINGALETT_ERR_ALLOC, "graph: out of memory");
        return false;
    }
    for (uint32_t v = 0; v < g->value_count; v++) rename[v] = v;
    for (uint32_t o = 0; o < g->output_count; o++) output[g->outputs[o]] = 1;
    for (uint32_t k = 0; k < N; k++) {
        IrNode *n = &g->nodes[k];
        for (uint32_t i = 0; i < n->input_count; i++)
            if (n->inputs[i] != IR_NONE) n->inputs[i] = rename[n->inputs[i]];
        bool candidate = n->output_count > 0;
        for (uint32_t o = 0; o < n->output_count && candidate; o++)
            candidate = n->outputs[o] != IR_NONE && !output[n->outputs[o]];
        if (!candidate) continue;
        uint32_t h = 2166136261u ^ n->op;
        for (uint32_t i = 0; i < n->input_count; i++) h = (h ^ n->inputs[i]) * 16777619u;
        uint32_t at = h & (cap - 1u);
        for (; table[at]; at = (at + 1u) & (cap - 1u)) {
            const IrNode *m = &g->nodes[table[at] - 1u];
            if (m->op != n->op || m->input_count != n->input_count || m->output_count != n->output_count ||
                memcmp(m->inputs, n->inputs, n->input_count * sizeof(uint32_t)) || !same_attrs(m, n)) continue;
            bool outputs_ok = true;
            for (uint32_t o = 0; o < m->output_count && outputs_ok; o++) outputs_ok = m->outputs[o] != IR_NONE;
            if (!outputs_ok) continue;
            for (uint32_t o = 0; o < n->output_count; o++) rename[n->outputs[o]] = m->outputs[o];
            gone[k] = 1;
            break;
        }
        if (!gone[k]) table[at] = k + 1u;
    }
    uint32_t kept = 0;
    for (uint32_t k = 0; k < N; k++)
        if (!gone[k]) g->nodes[kept++] = g->nodes[k];
    g->node_count = kept;
    free(rename); free(output); free(first); free(gone); free(table);
    return true;
}

bool ir_fuse(SpingalettGraph *g) {
    if (!common_subexpressions(g)) return false;
    /* (SPINGALETT_GRAPH_FUSE=0 leaves the graph as its file has it, to compare the fusions with it) */
    const char *fuse = getenv("SPINGALETT_GRAPH_FUSE");
    if (fuse && !strcmp(fuse, "0")) return true;
    Fuse f;
    f.g = g;
    f.producer = (int32_t *)malloc((size_t)(g->value_count ? g->value_count : 1u) * sizeof(int32_t));
    f.uses = (uint32_t *)calloc(g->value_count ? g->value_count : 1u, sizeof(uint32_t));
    f.dead = (uint8_t *)calloc(g->node_count ? g->node_count : 1u, 1);
    bool ok = f.producer && f.uses && f.dead;
    for (uint32_t v = 0; ok && v < g->value_count; v++) f.producer[v] = -1;
    for (uint32_t k = 0; ok && k < g->node_count; k++) {
        const IrNode *n = &g->nodes[k];
        for (uint32_t o = 0; o < n->output_count; o++)
            if (n->outputs[o] != IR_NONE) f.producer[n->outputs[o]] = (int32_t)k;
        for (uint32_t i = 0; i < n->input_count; i++)
            if (n->inputs[i] != IR_NONE) f.uses[n->inputs[i]]++;
    }
    for (uint32_t o = 0; ok && o < g->output_count; o++) f.uses[g->outputs[o]] += 2;     /* (never single) */
    /* GELU first (Linear takes it as its activation), attention before Linear (whose products it would take) */
    for (uint32_t k = 0; ok && k < g->node_count; k++)
        if (!f.dead[k] && g->nodes[k].op == IR_ReduceMean) ok = fuse_layer_norm(&f, &g->nodes[k]);
    for (uint32_t k = 0; ok && k < g->node_count; k++) {
        if (f.dead[k]) continue;
        if (g->nodes[k].op == IR_Erf) ok = fuse_gelu(&f, &g->nodes[k]);
        else if (g->nodes[k].op == IR_Tanh) ok = fuse_gelu_tanh(&f, &g->nodes[k]);
        else if (g->nodes[k].op == IR_Sigmoid) ok = fuse_silu(&f, &g->nodes[k]);
    }
    for (uint32_t k = 0; ok && k < g->node_count; k++)
        if (!f.dead[k] && g->nodes[k].op == IR_InstanceNormalization) ok = fuse_group_norm(&f, &g->nodes[k]);
    for (uint32_t k = 0; ok && k < g->node_count; k++)
        if (!f.dead[k] && g->nodes[k].op == IR_Softmax) ok = fuse_attention(&f, &g->nodes[k]);
    for (uint32_t k = 0; ok && k < g->node_count; k++)
        if (!f.dead[k] && (g->nodes[k].op == IR_MatMul || g->nodes[k].op == IR_Gemm)) ok = fuse_linear(&f, &g->nodes[k]);
    for (uint32_t k = 0; ok && k < g->node_count; k++)
        if (!f.dead[k] && g->nodes[k].op == IR_Conv) ok = fuse_conv_affine(&f, &g->nodes[k]);
    for (uint32_t k = 0; ok && k < g->node_count; k++)
        if (!f.dead[k] && g->nodes[k].op == IR_Conv) ok = fuse_conv_activation(&f, &g->nodes[k]);
    if (ok) {
        uint32_t kept = 0;
        for (uint32_t k = 0; k < g->node_count; k++)
            if (!f.dead[k]) g->nodes[kept++] = g->nodes[k];
        g->node_count = kept;
    } else {
        set_error(SPINGALETT_ERR_ALLOC, "graph: out of memory");
    }
    free(f.producer);
    free(f.uses);
    free(f.dead);
    /* the fused nodes' outputs again (inference before the fusions stops where an input is not declared, as at a mask
       of attention; the fused node may not read it for its shape), then the maps of convolutions channels last
       (Spingalett.IRLayout.c) */
    if (ok) ir_infer_shapes(g);
    return ok && ir_layout(g);
}
