/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Graphs of operators: the fusions a graph gets when it opens, after its constants are folded. Each replaces a
 * chain of nodes whose intermediate values nothing else reads by one node of the library's own operators
 * (Spingalett.IROps.def's last), which computes the same in fewer passes over memory:
 *
 *   GELU, SiLU   x * 0.5 * (1 + Erf(x / sqrt 2)), its tanh form, and x * Sigmoid(x), as exporters write them out,
 *                become Gelu and Swish
 *   Linear       MatMul by a constant matrix, then an Add of a constant row, then Relu, Sigmoid, Tanh or Gelu
 *                (each optional), becomes Linear: the matrix packed once, the bias and activation in the
 *                product's epilogue
 *   attention    MatMul(Q, K^T) (Q and K^T scaled by constants or not), a scaling by a constant, an Add of a
 *                mask, Softmax over the last axis (PyTorch's safe softmax too, which zeroes rows of no keys) and
 *                MatMul by V becomes FusedAttention: blocks of 64 queries at a time, their scores never all in
 *                memory
 *
 *   Conv         Conv then Relu, Sigmoid, Tanh, Gelu or SiLU: the activation in the convolution's epilogue (an
 *                attribute "activation" of the library's)
 *
 * A fused node takes the place of the chain's last node, which runs after everything the chain reads.
 */

#include "Spingalett.IR.h"
#include <math.h>
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

static bool fuse_linear(Fuse *f, IrNode *mm) {
    SpingalettGraph *g = f->g;
    const uint32_t a = mm->inputs[0], w = mm->inputs[1];
    if (a == IR_NONE || w == IR_NONE) return true;
    const IrValue *W = &g->values[w];
    if (W->kind != VALUE_CONST || W->dtype != SPINGALETT_DTYPE_FLOAT32 || W->rank != 2 || g->values[a].kind == VALUE_CONST)
        return true;
    const int64_t N = W->dims[1];
    uint32_t y = mm->outputs[0], bias = IR_NONE;
    IrNode *last = mm, *add = consumer_of(f, y, IR_Add, NULL), *act_node = NULL;
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
    if (single(f, y)) {
        for (uint32_t k = 0; k < g->node_count && !act_node; k++) {
            IrNode *n = &g->nodes[k];
            if (f->dead[k] || n->input_count < 1 || n->inputs[0] != y) continue;
            if (n->op == IR_Relu) act = ACT_RELU;
            else if (n->op == IR_Sigmoid) act = ACT_SIGMOID;
            else if (n->op == IR_Tanh) act = ACT_TANH;
            else if (n->op == IR_Swish && !ir_attr(n, "alpha")) act = ACT_SILU;
            else if (n->op == IR_Gelu) {
                const IrAttr *ap = ir_attr(n, "approximate");
                act = ap && ap->type == ATTR_STRING && !strcmp(ap->s, "tanh") ? ACT_GELU_TANH : ACT_GELU;
            }
            if (act != ACT_NONE) act_node = n;
        }
    }
    if (last == mm && !act_node) {
        /* a bare product by a constant: Linear all the same, for its packed matrix */
        IrAttr none[1];
        const uint32_t ins[2] = {a, w};
        kill(f, mm);
        return rewrite(f, mm, IR_Linear, ins, 2, mm->outputs[0], none, 0);
    }
    if (act_node) {
        last = act_node;
        y = act_node->outputs[0];
    }
    IrAttr attr;
    memset(&attr, 0, sizeof attr);
    attr.name = ir_strdup(g, "activation", 10);
    attr.type = ATTR_INT;
    attr.i = act;
    if (!attr.name) return false;
    kill(f, mm);
    if (add && last != mm && bias != IR_NONE) kill(f, add);
    if (act_node) kill(f, act_node);
    const uint32_t ins[3] = {a, w, bias};
    return rewrite(f, last, IR_Linear, ins, bias != IR_NONE ? 3u : 2u, y, &attr, 1);
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

/* The Transpose by perm computing v, read by nothing else, and its input. */
static IrNode *transposed(const Fuse *f, uint32_t v, unsigned digits, uint32_t *x) {
    IrNode *t = producer_of(f, v, IR_Transpose);
    if (!t || !single(f, v) || !perm_is(t, digits)) return NULL;
    *x = t->inputs[0];
    return t;
}

static bool fuse_attention(Fuse *f, IrNode *softmax) {
    SpingalettGraph *g = f->g;
    /* the softmax over the last axis only */
    const IrAttr *ax = ir_attr(softmax, "axis");
    if (!(ax ? ax->type == ATTR_INT && ax->i == -1 : g->opset >= 13)) return true;
    uint32_t scores = softmax->inputs[0], mask = IR_NONE;
    if (!single(f, scores)) return true;
    IrNode *add = producer_of(f, scores, IR_Add), *scale_node = NULL;
    uint32_t before = scores;
    if (add) {
        /* scores + mask: the operand a product (scaled or not) makes is the scores */
        for (int k = 0; k < 2 && mask == IR_NONE; k++) {
            const uint32_t s = add->inputs[k];
            if (single(f, s) && (producer_of(f, s, IR_MatMul) || producer_of(f, s, IR_Mul) || producer_of(f, s, IR_Div))) {
                mask = add->inputs[1 - k];
                before = s;
            }
        }
        if (mask == IR_NONE) return true;
    }
    float scale = 1.0f, c;
    IrNode *mul = producer_of(f, before, IR_Mul), *div = producer_of(f, before, IR_Div);
    uint32_t product = before;
    if (mul && single(f, before)) {
        for (int k = 0; k < 2 && product == before; k++)
            if (scalar_of(g, mul->inputs[k], &c)) { scale = c; product = mul->inputs[1 - k]; scale_node = mul; }
    } else if (div && single(f, before) && scalar_of(g, div->inputs[1], &c) && c != 0.0f) {
        scale = 1.0f / c;
        product = div->inputs[0];
        scale_node = div;
    }
    IrNode *qk = producer_of(f, product, IR_MatMul);
    if (!qk || !single(f, product)) return true;
    /* the probabilities, or PyTorch's safe softmax of them: Where(IsNaN(p), 0, p), which makes rows of no keys 0 */
    uint32_t probs = softmax->outputs[0], slot;
    IrNode *isnan_node = NULL, *where = NULL;
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
    uint32_t v = pv->inputs[1], y = pv->outputs[0], xq, xk, xv;
    IrNode *tq = transposed(f, q, 0x0213, &xq), *tk = transposed(f, kt, 0x0231, &xk), *tv = transposed(f, v, 0x0213, &xv);
    IrNode *ty = consumer_of(f, y, IR_Transpose, NULL);
    const bool layout = tq && tk && tv && ty && perm_is(ty, 0x0213) && xv != xq && xk != xq;
    if (layout) {
        q = xq;
        kt = xk;
        v = xv;
        y = ty->outputs[0];
    }
    const uint32_t ins[4] = {q, kt, v, mask};
    IrAttr attrs[3];
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
    if (!attrs[0].name || !attrs[1].name || !attrs[2].name) return false;
    if (layout) {
        kill(f, tq);
        kill(f, tk);
        kill(f, tv);
        kill(f, ty);
    }
    kill(f, qk);
    kill(f, pv);
    if (qs) kill(f, qs);
    if (ks) kill(f, ks);
    if (scale_node) kill(f, scale_node);
    if (add) kill(f, add);
    kill(f, softmax);
    if (isnan_node) kill(f, isnan_node);
    if (where) kill(f, where);
    return rewrite(f, layout ? ty : pv, IR_FusedAttention, ins, mask != IR_NONE ? 4u : 3u, y, attrs, 3);
}

/* ------------------------------------------------------------------------- convolutions */

/* Conv then Relu, Sigmoid, Tanh, Gelu or Swish: the activation an attribute of the Conv, applied in its epilogue. */
static bool fuse_conv_activation(Fuse *f, IrNode *conv) {
    const uint32_t y = conv->outputs[0];
    if (ir_attr(conv, "activation") || !single(f, y)) return true;
    IrNode *act_node = NULL;
    int64_t act = ACT_NONE;
    for (uint32_t k = 0; k < f->g->node_count && !act_node; k++) {
        IrNode *n = &f->g->nodes[k];
        if (f->dead[k] || n->input_count < 1 || n->inputs[0] != y) continue;
        if (n->op == IR_Relu) act = ACT_RELU;
        else if (n->op == IR_Sigmoid) act = ACT_SIGMOID;
        else if (n->op == IR_Tanh) act = ACT_TANH;
        else if (n->op == IR_Swish && !ir_attr(n, "alpha")) act = ACT_SILU;
        else if (n->op == IR_Gelu) {
            const IrAttr *ap = ir_attr(n, "approximate");
            act = ap && ap->type == ATTR_STRING && !strcmp(ap->s, "tanh") ? ACT_GELU_TANH : ACT_GELU;
        }
        if (act != ACT_NONE) act_node = n;
    }
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

bool ir_fuse(SpingalettGraph *g) {
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
    for (uint32_t k = 0; ok && k < g->node_count; k++) {
        if (f.dead[k]) continue;
        if (g->nodes[k].op == IR_Erf) ok = fuse_gelu(&f, &g->nodes[k]);
        else if (g->nodes[k].op == IR_Tanh) ok = fuse_gelu_tanh(&f, &g->nodes[k]);
        else if (g->nodes[k].op == IR_Sigmoid) ok = fuse_silu(&f, &g->nodes[k]);
    }
    for (uint32_t k = 0; ok && k < g->node_count; k++)
        if (!f.dead[k] && g->nodes[k].op == IR_Softmax) ok = fuse_attention(&f, &g->nodes[k]);
    for (uint32_t k = 0; ok && k < g->node_count; k++)
        if (!f.dead[k] && g->nodes[k].op == IR_MatMul) ok = fuse_linear(&f, &g->nodes[k]);
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
    return ok;
}
