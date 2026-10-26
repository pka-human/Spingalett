/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Graphs of operators on the GPU (the compute modes COMPUTE_CUDA and COMPUTE_VULKAN): a node runs on the device where
 * its operator has kernels for the node's types and attributes, on the host otherwise.
 *
 * A run whose inputs have shapes no plan has makes a plan as it runs. Each node in order goes to the device when it
 * reads a value only the device has, or when its outputs (or inputs) are large and its kernels take it; else to the
 * host, whose kernels are the CPU's (Spingalett.IROps.c): shapes, indices, small masks, and whatever has no kernels
 * here. The device's nodes gather into pieces, each a command buffer recorded once (a CUDA graph on CUDA) over memory
 * of its own, in which values take ranges planned from their lifetimes (a range is free again after its value's last
 * reader). A piece ends where a host node needs a value the device computed, which it downloads, and at the graph's
 * end, which downloads the outputs. The host's values that the device's nodes read are uploaded by the piece that
 * reads them, constants once a graph. Values that only change their shape (Reshape, Flatten, Squeeze, Unsqueeze,
 * Identity) are views of their input's memory on the device as on the host.
 *
 * Later runs with inputs of the same types and shapes replay the plan: the host's nodes run again, each piece is
 * submitted as recorded (its uploads taken from the host's values of this run). A host node whose outputs differ in
 * type or shape from the plan's, or differ in value where the device's nodes read them as parameters (a Reshape's
 * shape, a Slice's starts: their dispatches hold them), ends the replay: the run makes a new plan. A graph keeps the
 * plans of its last PLANS shapes of inputs.
 *
 * The kernels sum in orders fixed by the shapes, as the CPU's do: a graph gives the same bits on every run on one
 * device, and the CPU's results up to rounding.
 */

#include "Spingalett.Gpu.h"
#include "Spingalett.IR.h"
#include "Spingalett.GpuKernels.h"
#include "Spingalett.Recorder.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PLANS 4u
#define HEAVY 4096u                 /* values from which a node of the host's values goes to the device (or
                                       SPINGALETT_GRAPH_GPU_MIN's: 0 sends every node the kernels take there) */
#define ALIGN 256u                  /* bytes values' memory is aligned to */
#define NONE UINT32_MAX
#define MAX_GROUPS 65535u

enum { ON_HOST = 1, ON_DEVICE = 2 };
enum { STEP_HOST, STEP_PIECE };

/* A range of a piece's memory that a value and its views share. */
typedef struct {
    uint32_t piece;
    uint64_t at, bytes;
    uint32_t refs;                  /* values holding it, while planning */
} Block;

typedef struct { uint64_t lo, hi; } Span;

/* A value copied between the host and a piece's memory, through its staging memory (an upload before the piece's
   node `before` in its order: its memory is the value's from there on only). */
typedef struct {
    uint32_t value, before;
    uint64_t stage, bytes;
} Transfer;

typedef struct {
    SpgGpuCommands *commands;
    SpgGpuBuffer memory;            /* its values' */
    SpgGpuBuffer staging;           /* visible to the host: the uploads, then the downloads */
    uint64_t size, staged;          /* bytes of each */
    Transfer *ups, *downs;
    uint32_t up_count, up_cap, down_count, down_cap;
    uint32_t *nodes;                /* the device's nodes, in order */
    uint32_t node_count, node_cap;
    Span *free;                     /* its free memory, while planning */
    uint32_t free_count, free_cap;
} Piece;

typedef struct {
    uint8_t kind;                   /* STEP_HOST, STEP_PIECE */
    uint32_t index;                 /* the node, or the piece */
} Step;

/* A device node's scratch memory, in its piece. */
typedef struct {
    uint64_t at, bytes;
} Scratch;

typedef struct {
    uint64_t *key;                  /* the inputs' types and shapes */
    uint32_t key_len;
    uint64_t used;                  /* the run that last took it */
    bool host_only;                 /* no node on the device: the CPU's run serves better */
    uint64_t heavy;                 /* HEAVY, or SPINGALETT_GRAPH_GPU_MIN's */
    uint32_t device_nodes;
    Step *steps;
    uint32_t step_count, step_cap;
    Piece *pieces;
    uint32_t piece_count, piece_cap;
    Block *blocks;
    uint32_t block_count, block_cap;
    IrTensor *meta;                 /* [value_count]: types and shapes (no data) */
    uint8_t *on;                    /* [value_count]: ON_HOST, ON_DEVICE */
    uint32_t *block_of;             /* [value_count]: a value's block on the device (NONE: none) */
    uint8_t **param;                /* [value_count]: the values of host values the device's nodes read as parameters */
    Scratch *scratch;               /* [node_count] */
} Plan;

typedef struct {
    SpgBackend backend;
    Plan *plans[PLANS];
    SpgGpuBuffer *consts;           /* [value_count]: constants on the device (buffer NULL: not there) */
    SpgGpuArena **arenas;           /* their memory */
    uint32_t arena_count, arena_cap;
    uint64_t runs;
} GpuGraph;

/* The planning or recording of one node on the device. */
typedef struct {
    SpingalettGraph *g;
    GpuGraph *gg;
    Plan *p;
    uint32_t k;
    const IrNode *n;
    bool record;
    Recorder *r;
    uint32_t params;                /* planning: inputs read on the host (bits) */
    uint32_t views;                 /* planning: outputs that are views of input 0 (bits) */
    uint64_t scratch;               /* planning: bytes asked for; recording: their address */
} GCtx;

/* A device node: planning (c->record false) checks that the kernels take it (false: the host runs it) and sets its
   outputs' types and shapes in out[], with ask_param() for inputs read on the host and as_view() for outputs that
   are views; recording dispatches its kernels. in[] and out[] are the run's tensors (a host value's data, else
   none). */
typedef bool (*GpuOp)(GCtx *c, IrTensor *const *in, IrTensor **out);

static bool fail(const char *message) {
    set_error(SPINGALETT_ERR_INVALID, message);
    return false;
}

static bool grow(void **list, uint32_t *cap, uint32_t count, size_t size) {
    if (count < *cap) return true;
    const uint32_t c = *cap ? 2u * *cap : 16u;
    void *grown = realloc(*list, (size_t)c * size);
    if (!grown) return false;
    *list = grown;
    *cap = c;
    return true;
}

static uint64_t bytes_of(const IrTensor *t) {
    return t->count * ir_dtype_size(t->dtype);
}

static uint64_t align_up(uint64_t x) {
    return (x + ALIGN - 1u) & ~(uint64_t)(ALIGN - 1u);
}

static uint32_t group_count(uint64_t items, uint32_t per) {
    const uint64_t g = (items + per - 1u) / per;
    return g > MAX_GROUPS ? MAX_GROUPS : (uint32_t)g;
}

/* ------------------------------------------------------------------------- memory of pieces */

/* bytes from the piece's free memory (first fit, the piece growing past its end), at ALIGN */
static uint64_t piece_take(Piece *pc, uint64_t bytes) {
    bytes = align_up(bytes ? bytes : 1u);
    for (uint32_t k = 0; k < pc->free_count; k++) {
        Span *s = &pc->free[k];
        if (s->hi - s->lo >= bytes) {
            const uint64_t at = s->lo;
            s->lo += bytes;
            if (s->lo == s->hi) {
                memmove(s, s + 1, (pc->free_count - k - 1u) * sizeof *s);
                pc->free_count--;
            }
            return at;
        }
    }
    /* (the last free range, if it ends the memory, grows) */
    uint64_t at = pc->size;
    if (pc->free_count && pc->free[pc->free_count - 1u].hi == pc->size) {
        at = pc->free[--pc->free_count].lo;
    }
    pc->size = at + bytes;
    return at;
}

static bool piece_give(Piece *pc, uint64_t at, uint64_t bytes) {
    bytes = align_up(bytes ? bytes : 1u);
    uint32_t k = 0;
    while (k < pc->free_count && pc->free[k].lo < at) k++;
    if (!grow((void **)&pc->free, &pc->free_cap, pc->free_count, sizeof(Span))) return false;
    memmove(pc->free + k + 1, pc->free + k, (pc->free_count - k) * sizeof(Span));
    pc->free[k] = (Span){at, at + bytes};
    pc->free_count++;
    /* merged with its neighbours */
    if (k + 1u < pc->free_count && pc->free[k].hi == pc->free[k + 1u].lo) {
        pc->free[k].hi = pc->free[k + 1u].hi;
        memmove(pc->free + k + 1, pc->free + k + 2, (pc->free_count - k - 2u) * sizeof(Span));
        pc->free_count--;
    }
    if (k > 0 && pc->free[k - 1u].hi == pc->free[k].lo) {
        pc->free[k - 1u].hi = pc->free[k].hi;
        memmove(pc->free + k, pc->free + k + 1, (pc->free_count - k - 1u) * sizeof(Span));
        pc->free_count--;
    }
    return true;
}

/* A new block of the current piece for bytes; NONE when memory runs out. */
static uint32_t block_new(Plan *p, uint64_t bytes) {
    if (!grow((void **)&p->blocks, &p->block_cap, p->block_count, sizeof(Block))) return NONE;
    Piece *pc = &p->pieces[p->piece_count - 1u];
    p->blocks[p->block_count] = (Block){p->piece_count - 1u, piece_take(pc, bytes), bytes, 1};
    return p->block_count++;
}

/* A value lets go of its block: free memory again once nothing holds it (in the current piece only: the memory of
   pieces before it is theirs). */
static void block_drop(Plan *p, uint32_t b) {
    if (b == NONE || !p->blocks[b].refs) return;
    if (--p->blocks[b].refs) return;
    if (p->blocks[b].piece == p->piece_count - 1u)
        piece_give(&p->pieces[p->blocks[b].piece], p->blocks[b].at, p->blocks[b].bytes);
}

/* ------------------------------------------------------------------------- plans */

static void plan_free(Plan *p) {
    if (!p) return;
    for (uint32_t k = 0; k < p->piece_count; k++) {
        Piece *pc = &p->pieces[k];
        if (pc->commands) spg_gpu_commands_free(pc->commands);
        spg_gpu_buffer_free(&pc->memory);
        spg_gpu_buffer_free(&pc->staging);
        free(pc->ups);
        free(pc->downs);
        free(pc->nodes);
        free(pc->free);
    }
    free(p->pieces);
    free(p->steps);
    free(p->blocks);
    free(p->meta);
    free(p->on);
    free(p->block_of);
    if (p->param) {
        for (uint32_t v = 0; p->param && p->param[v] != (uint8_t *)p; v++) free(p->param[v]);
    }
    free(p->param);
    free(p->scratch);
    free(p->key);
    free(p);
}

/* The key of a run's inputs: each input's type, rank and sizes. */
static uint64_t *run_key(const SpingalettGraph *g, uint32_t *len) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < g->input_count; i++) n += 2u + g->tensors[g->inputs[i]].rank;
    uint64_t *key = (uint64_t *)malloc((n ? n : 1u) * sizeof(uint64_t));
    if (!key) return NULL;
    uint32_t at = 0;
    for (uint32_t i = 0; i < g->input_count; i++) {
        const IrTensor *t = &g->tensors[g->inputs[i]];
        key[at++] = t->dtype;
        key[at++] = t->rank;
        for (uint32_t d = 0; d < t->rank; d++) key[at++] = (uint64_t)t->dims[d];
    }
    *len = n;
    return key;
}

static Plan *plan_new(const SpingalettGraph *g) {
    Plan *p = (Plan *)calloc(1, sizeof *p);
    if (!p) return NULL;
    const uint32_t V = g->value_count ? g->value_count : 1u, N = g->node_count ? g->node_count : 1u;
    p->meta = (IrTensor *)calloc(V, sizeof(IrTensor));
    p->on = (uint8_t *)calloc(V, 1);
    p->block_of = (uint32_t *)malloc(V * sizeof(uint32_t));
    /* (one past the values: the plan itself, which ends the list for plan_free()) */
    p->param = (uint8_t **)calloc(V + 1u, sizeof(uint8_t *));
    p->scratch = (Scratch *)calloc(N, sizeof(Scratch));
    if (!p->meta || !p->on || !p->block_of || !p->param || !p->scratch) {
        if (p->param) p->param[V] = (uint8_t *)p;
        plan_free(p);
        return NULL;
    }
    p->param[g->value_count] = (uint8_t *)p;
    for (uint32_t v = 0; v < g->value_count; v++) p->block_of[v] = NONE;
    const char *min = getenv("SPINGALETT_GRAPH_GPU_MIN");
    p->heavy = min && *min ? strtoull(min, NULL, 10) : HEAVY;
    return p;
}

static bool add_step(Plan *p, uint8_t kind, uint32_t index) {
    if (!grow((void **)&p->steps, &p->step_cap, p->step_count, sizeof(Step))) return false;
    p->steps[p->step_count++] = (Step){kind, index};
    return true;
}

static bool piece_open(Plan *p) {
    if (!grow((void **)&p->pieces, &p->piece_cap, p->piece_count, sizeof(Piece))) return false;
    memset(&p->pieces[p->piece_count++], 0, sizeof(Piece));
    return true;
}

/* ------------------------------------------------------------------------- addresses */

/* The device address of a value (a constant's, or its block's); 0 for none. */
static uint64_t address_of(const GCtx *c, uint32_t v) {
    if (v == NONE) return 0;
    if (c->g->values[v].kind == VALUE_CONST) return c->gg->consts[v].address;
    const uint32_t b = c->p->block_of[v];
    if (b == NONE) return 0;
    const Block *k = &c->p->blocks[b];
    return c->p->pieces[k->piece].memory.address + k->at;
}

static uint64_t in_at(const GCtx *c, uint32_t i) {
    return i < c->n->input_count ? address_of(c, c->n->inputs[i]) : 0;
}
static uint64_t out_at(const GCtx *c, uint32_t o) {
    return o < c->n->output_count ? address_of(c, c->n->outputs[o]) : 0;
}

static Range range_of(const GCtx *c, uint32_t v) {
    const uint64_t a = address_of(c, v);
    return (Range){a, a + bytes_of(&c->g->tensors[v])};
}

/* ------------------------------------------------------------------------- what device nodes ask while planned */

/* Input i read on the host (its values a parameter of the dispatches): its tensor, NULL when the host has none. */
static const IrTensor *ask_param(GCtx *c, IrTensor *const *in, uint32_t i) {
    if (i >= c->n->input_count || !in[i]) return NULL;
    const uint32_t v = c->n->inputs[i];
    if (!(c->p->on[v] & ON_HOST) && c->g->values[v].kind != VALUE_CONST) return NULL;
    c->params |= 1u << i;
    return in[i];
}

static void as_view(GCtx *c, uint32_t o) {
    c->views |= 1u << o;
}

/* Output o of type dtype and shape (planning). */
static bool set_shape(IrTensor *t, uint8_t dtype, uint32_t rank, const int64_t *dims) {
    if (rank > IR_MAX_RANK) return false;
    t->dtype = dtype;
    t->rank = (uint8_t)rank;
    t->count = 1;
    for (uint32_t d = 0; d < IR_MAX_RANK; d++) {
        t->dims[d] = d < rank ? dims[d] : 0;
        if (d < rank) t->count *= (uint64_t)dims[d];
    }
    t->data = NULL;
    t->owned = false;
    t->staged = false;
    t->constant = false;
    return true;
}

/* Records a kernel after the barrier its reads and writes need. */
static void dispatch(GCtx *c, const Access *a, SpgKernel kernel, const uint32_t *spec, uint32_t spec_count,
                     const void *push, uint32_t push_size, uint32_t gx, uint32_t gy, uint32_t gz) {
    need(c->r, a);
    spg_gpu_dispatch(c->r->c, kernel, spec, spec_count, push, push_size, gx, gy, gz);
}

/* ------------------------------------------------------------------------- element-wise operators */

static uint32_t type_code(uint8_t dtype) {
    switch (dtype) {
        case SPINGALETT_DTYPE_FLOAT32: return SPG_T_F32;
        case SPINGALETT_DTYPE_BOOL: return SPG_T_BOOL;
        case SPINGALETT_DTYPE_UINT8: return SPG_T_U8;
        case SPINGALETT_DTYPE_INT8: return SPG_T_I8;
        case SPINGALETT_DTYPE_INT32: return SPG_T_I32;
        case SPINGALETT_DTYPE_INT64: return SPG_T_I64;
        default: return NONE;
    }
}

/* The broadcast shape of up to three tensors; false when they do not broadcast. */
static bool broadcast(IrTensor *const *t, uint32_t n, uint32_t *rank, int64_t *dims) {
    uint32_t R = 0;
    for (uint32_t k = 0; k < n; k++)
        if (t[k] && t[k]->rank > R) R = t[k]->rank;
    for (uint32_t d = 0; d < R; d++) dims[d] = 1;
    for (uint32_t k = 0; k < n; k++) {
        if (!t[k]) continue;
        const uint32_t off = R - t[k]->rank;
        for (uint32_t d = 0; d < t[k]->rank; d++) {
            const int64_t s = t[k]->dims[d];
            if (s == dims[off + d] || s == 1) continue;
            if (dims[off + d] != 1) return false;
            dims[off + d] = s;
        }
    }
    *rank = R;
    return true;
}

/* The element-wise dispatch of y = op(t[0..n)) over the output's shape: the operands' strides over its axes, axes of
   size 1 dropped and neighbours merged where every operand runs through both as one (SPG_GEW_AXES at most: false when
   more remain). */
typedef struct {
    SpgGewPush push;
    uint32_t flags;
} Gew;

static bool gew_layout(Gew *e, const IrTensor *y, const IrTensor *const *t, uint32_t n) {
    memset(e, 0, sizeof *e);
    const uint32_t R = y->rank;
    int64_t dims[IR_MAX_RANK], s[3][IR_MAX_RANK];
    for (uint32_t k = 0; k < n; k++) {
        int64_t st = 1;
        for (uint32_t d = R; d-- > 0;) {
            const int32_t td = (int32_t)d - (int32_t)(R - t[k]->rank);
            const int64_t size = td >= 0 ? t[k]->dims[td] : 1;
            s[k][d] = size == 1 ? 0 : st;
            if (td >= 0) st *= size;
        }
    }
    /* axes of size 1 out, then merged from the innermost */
    uint32_t m = 0;
    int64_t md[IR_MAX_RANK], ms[3][IR_MAX_RANK];
    for (uint32_t d = 0; d < R; d++) {
        if (y->dims[d] == 1) continue;
        md[m] = y->dims[d];
        for (uint32_t k = 0; k < n; k++) ms[k][m] = s[k][d];
        m++;
    }
    uint32_t q = 0;
    for (uint32_t d = 0; d < m; d++) {
        bool merge = q > 0;
        for (uint32_t k = 0; k < n && merge; k++) merge = ms[k][q - 1u] == ms[k][d] * md[d];
        if (merge) {
            dims[q - 1u] *= md[d];
            for (uint32_t k = 0; k < n; k++) s[k][q - 1u] = ms[k][d];
        } else {
            dims[q] = md[d];
            for (uint32_t k = 0; k < n; k++) s[k][q] = ms[k][d];
            q++;
        }
    }
    if (q > SPG_GEW_AXES || y->count > UINT32_MAX) return false;
    const uint32_t lead = SPG_GEW_AXES - q;
    for (uint32_t d = 0; d < SPG_GEW_AXES; d++) {
        e->push.dims[d] = d < lead ? 1u : (uint32_t)dims[d - lead];
        e->push.sa[d] = d < lead || n < 1 ? 0u : (uint32_t)s[0][d - lead];
        e->push.sb[d] = d < lead || n < 2 ? 0u : (uint32_t)s[1][d - lead];
        e->push.sc[d] = d < lead || n < 3 ? 0u : (uint32_t)s[2][d - lead];
    }
    for (uint32_t k = 0; k < n; k++) {
        /* 1: at the output's index (its strides those of the output), 2: one value */
        bool flat = t[k]->count == y->count, one = t[k]->count == 1;
        uint64_t st = 1;
        for (uint32_t d = q; d-- > 0;) {
            flat = flat && (uint64_t)s[k][d] == st;
            st *= (uint64_t)dims[d];
        }
        e->flags |= (one ? 2u : flat ? 1u : 0u) << (2u * k);
    }
    e->push.total = (uint32_t)y->count;
    return true;
}

/* Records one gew dispatch: y = op(a, b, c) with the operands' types. */
static void gew_dispatch(GCtx *c, const Gew *e, uint32_t op, uint32_t types, uint64_t y, uint64_t a, uint64_t b,
                         uint64_t cc, Range ry, Range ra, Range rb, Range rc) {
    SpgGewPush push = e->push;
    push.y = y;
    push.a = a;
    push.b = b ? b : a;                 /* (an address for loads that never happen) */
    push.c = cc ? cc : a;
    const uint32_t spec[3] = {op, types, e->flags};
    Access acc = {0};
    reads(&acc, ra);
    if (b) reads(&acc, rb);
    if (cc) reads(&acc, rc);
    writes(&acc, ry);
    dispatch(c, &acc, SPG_KERNEL_gew, spec, 3, &push, sizeof push, group_count(push.total, 256u), 1, 1);
}

/* The operator code of a node's unary or binary operator in gew, its alpha and beta; NONE for none. */
static uint32_t gew_op(const IrNode *n, float *alpha, float *beta) {
    *alpha = 0.0f;
    *beta = 0.0f;
    switch ((IrOp)n->op) {
        case IR_Identity: case IR_Dropout: return SPG_GEW_COPY;
        case IR_Neg: return SPG_GEW_NEG;
        case IR_Abs: return SPG_GEW_ABS;
        case IR_Exp: return SPG_GEW_EXP;
        case IR_Log: return SPG_GEW_LOG;
        case IR_Sqrt: return SPG_GEW_SQRT;
        case IR_Reciprocal: return SPG_GEW_RECIPROCAL;
        case IR_Sigmoid: return SPG_GEW_SIGMOID;
        case IR_Tanh: return SPG_GEW_TANH;
        case IR_Relu: return SPG_GEW_RELU;
        case IR_LeakyRelu: *alpha = ir_attr_float(n, "alpha", 0.01f); return SPG_GEW_LEAKY;
        case IR_Elu: *alpha = ir_attr_float(n, "alpha", 1.0f); return SPG_GEW_ELU;
        case IR_Selu:
            *alpha = ir_attr_float(n, "alpha", 1.67326319217681884765625f);
            *beta = ir_attr_float(n, "gamma", 1.05070102214813232421875f);
            return SPG_GEW_SELU;
        case IR_Gelu: {
            const IrAttr *a = ir_attr(n, "approximate");
            return a && a->type == ATTR_STRING && !strcmp(a->s, "tanh") ? SPG_GEW_GELU_TANH : SPG_GEW_GELU;
        }
        case IR_Swish: *alpha = ir_attr_float(n, "alpha", 1.0f); return SPG_GEW_SILU;
        case IR_HardSigmoid:
            *alpha = ir_attr_float(n, "alpha", 0.2f);
            *beta = ir_attr_float(n, "beta", 0.5f);
            return SPG_GEW_HARD_SIGMOID;
        case IR_HardSwish: return SPG_GEW_HARD_SWISH;
        case IR_Softplus: return SPG_GEW_SOFTPLUS;
        case IR_Softsign: return SPG_GEW_SOFTSIGN;
        case IR_Sin: return SPG_GEW_SIN;
        case IR_Cos: return SPG_GEW_COS;
        case IR_Erf: return SPG_GEW_ERF;
        case IR_Floor: return SPG_GEW_FLOOR;
        case IR_Ceil: return SPG_GEW_CEIL;
        case IR_Round: return SPG_GEW_ROUND;
        case IR_Sign: return SPG_GEW_SIGN;
        case IR_Not: return SPG_GEW_NOT;
        case IR_IsNaN: return SPG_GEW_ISNAN;
        case IR_IsInf:
            *alpha = (float)ir_attr_int(n, "detect_positive", 1);
            *beta = (float)ir_attr_int(n, "detect_negative", 1);
            return SPG_GEW_ISINF;
        case IR_Mish: return SPG_GEW_MISH;
        case IR_Celu: *alpha = ir_attr_float(n, "alpha", 1.0f); return SPG_GEW_CELU;
        case IR_ThresholdedRelu: *alpha = ir_attr_float(n, "alpha", 1.0f); return SPG_GEW_THRESHOLDED_RELU;
        case IR_Add: case IR_Sum: case IR_Mean: return SPG_GEW_ADD;
        case IR_Sub: return SPG_GEW_SUB;
        case IR_Mul: return SPG_GEW_MUL;
        case IR_Div: return SPG_GEW_DIV;
        case IR_Pow: return SPG_GEW_POW;
        case IR_Max: return SPG_GEW_MAX;
        case IR_Min: return SPG_GEW_MIN;
        case IR_PRelu: return SPG_GEW_PRELU;
        case IR_Equal: return SPG_GEW_EQUAL;
        case IR_Less: return SPG_GEW_LESS;
        case IR_LessOrEqual: return SPG_GEW_LESS_EQUAL;
        case IR_Greater: return SPG_GEW_GREATER;
        case IR_GreaterOrEqual: return SPG_GEW_GREATER_EQUAL;
        case IR_And: return SPG_GEW_AND;
        case IR_Or: return SPG_GEW_OR;
        case IR_Xor: return SPG_GEW_XOR;
        case IR_Mod: return ir_attr_int(n, "fmod", 0) ? SPG_GEW_FMOD : SPG_GEW_MOD;
        case IR_Where: return SPG_GEW_WHERE;
        default: return NONE;
    }
}

static const uint8_t F32 = SPINGALETT_DTYPE_FLOAT32, BOOL = SPINGALETT_DTYPE_BOOL;

/* Unary, binary and variadic element-wise operators and Where, of floats (comparisons giving booleans, logic of
   booleans). */
static bool op_elementwise(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    float alpha, beta;
    uint32_t op = gew_op(n, &alpha, &beta);
    if (op == NONE || !n->input_count || n->output_count < 1 || !out[0]) return false;
    if (op >= SPG_GEW_ADD && op < SPG_GEW_WHERE && n->input_count == 1) op = SPG_GEW_COPY;    /* (Sum of one, ...) */
    const uint32_t count = op == SPG_GEW_WHERE ? 3u : op >= SPG_GEW_ADD ? n->input_count : 1u;
    if (n->input_count < count || count > 32u) return false;
    for (uint32_t i = 0; i < count; i++)
        if (!in[i]) return false;
    const bool logic = op == SPG_GEW_AND || op == SPG_GEW_OR || op == SPG_GEW_XOR || op == SPG_GEW_NOT;
    const bool compare = op >= SPG_GEW_EQUAL && op <= SPG_GEW_GREATER_EQUAL;
    for (uint32_t i = 0; i < count; i++) {
        const uint8_t want = logic || (op == SPG_GEW_WHERE && i == 0) ? BOOL : F32;
        if (in[i]->dtype != want) return false;
    }
    uint8_t ytype = logic || compare || op == SPG_GEW_ISNAN || op == SPG_GEW_ISINF ? BOOL : F32;
    if (n->op == IR_Dropout && n->output_count > 1 && out[1]) return false;
    uint32_t R;
    int64_t dims[IR_MAX_RANK];
    if (!broadcast(in, count, &R, dims)) return false;
    if (!c->record) {
        if (!set_shape(out[0], ytype, R, dims)) return false;
        /* (each step of a chain of more than two operands through gew's merged axes) */
        for (uint32_t i = 0; i < count; i++) {
            Gew e;
            IrTensor *const pair[3] = {i == 0 ? in[0] : out[0], in[i], in[i]};
            if (!gew_layout(&e, out[0], (const IrTensor *const *)pair, op == SPG_GEW_WHERE ? 3u : count == 1 ? 1u : 2u))
                return false;
            if (op == SPG_GEW_WHERE) {
                if (!gew_layout(&e, out[0], (const IrTensor *const *)in, 3u)) return false;
                break;
            }
        }
        return true;
    }
    const uint32_t ty = type_code(ytype) << 12;
    const uint64_t y = out_at(c, 0);
    const Range ry = range_of(c, n->outputs[0]);
    Gew e;
    if (count == 1 || op == SPG_GEW_WHERE) {
        gew_layout(&e, out[0], (const IrTensor *const *)in, count);
        uint32_t types = ty;
        for (uint32_t i = 0; i < count; i++) types |= type_code(in[i]->dtype) << (4u * i);
        e.push.alpha = alpha;
        e.push.beta = beta;
        gew_dispatch(c, &e, op, types, y, in_at(c, 0), count > 1 ? in_at(c, 1) : 0, count > 2 ? in_at(c, 2) : 0, ry,
                     range_of(c, n->inputs[0]), count > 1 ? range_of(c, n->inputs[1]) : ry,
                     count > 2 ? range_of(c, n->inputs[2]) : ry);
        return true;
    }
    /* y = a op b, then y = y op each further operand */
    for (uint32_t i = 1; i < count; i++) {
        IrTensor *const pair[2] = {i == 1 ? in[0] : out[0], in[i]};
        gew_layout(&e, out[0], (const IrTensor *const *)pair, 2u);
        const uint64_t a = i == 1 ? in_at(c, 0) : y;
        const Range ra = i == 1 ? range_of(c, n->inputs[0]) : ry;
        gew_dispatch(c, &e, op, ty | type_code(pair[0]->dtype) | type_code(pair[1]->dtype) << 4, y, a, in_at(c, i), 0,
                     ry, ra, range_of(c, n->inputs[i]), ry);
    }
    if (n->op == IR_Mean) {
        IrTensor *const one[1] = {out[0]};
        gew_layout(&e, out[0], (const IrTensor *const *)one, 1u);
        e.push.alpha = 1.0f / (float)count;
        gew_dispatch(c, &e, SPG_GEW_AFFINE, ty | type_code(F32), y, y, 0, 0, ry, ry, ry, ry);
    }
    return true;
}

/* Cast and CastLike between floats, booleans and integers of 32 and 64 bits (to float16 and bfloat16: rounded, kept
   as floats). */
static bool op_cast(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    if (!in[0] || !out[0]) return false;
    int64_t to;
    if (n->op == IR_CastLike) {
        if (n->input_count < 2 || !in[1]) return false;
        to = in[1]->dtype;
    } else {
        to = ir_attr_int(n, "to", 0);
        if (to == 11) to = F32;
        else if (to == 4 || to == 5) to = SPINGALETT_DTYPE_INT32;
        else if (to == 12 || to == 13) to = SPINGALETT_DTYPE_INT64;
    }
    const uint32_t op = to == SPINGALETT_DTYPE_FLOAT16 ? SPG_GEW_TO_HALF : to == SPINGALETT_DTYPE_BFLOAT16
                      ? SPG_GEW_TO_BF16 : SPG_GEW_COPY;
    const uint8_t ytype = op != SPG_GEW_COPY ? F32 : (uint8_t)to;
    if (type_code(in[0]->dtype) == NONE || type_code(ytype) == NONE || (op != SPG_GEW_COPY && in[0]->dtype != F32))
        return false;
    if (!c->record) return set_shape(out[0], ytype, in[0]->rank, in[0]->dims);
    Gew e;
    IrTensor *const one[1] = {in[0]};
    gew_layout(&e, out[0], (const IrTensor *const *)one, 1u);
    const Range ry = range_of(c, n->outputs[0]);
    gew_dispatch(c, &e, op, type_code(ytype) << 12 | type_code(in[0]->dtype), out_at(c, 0), in_at(c, 0), 0, 0, ry,
                 range_of(c, n->inputs[0]), ry, ry);
    return true;
}

/* Clip of floats, its bounds read on the host. */
static bool op_clip(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    if (!in[0] || !out[0] || in[0]->dtype != F32) return false;
    float lo = ir_attr_float(n, "min", -INFINITY), hi = ir_attr_float(n, "max", INFINITY);
    for (uint32_t i = 1; i < 3 && i < n->input_count; i++) {
        if (!in[i]) continue;
        const IrTensor *t = ask_param(c, in, i);
        if (!t || t->dtype != F32 || t->count != 1) return false;
        if (i == 1) lo = ((const float *)t->data)[0];
        else hi = ((const float *)t->data)[0];
    }
    if (!c->record) return set_shape(out[0], F32, in[0]->rank, in[0]->dims);
    Gew e;
    IrTensor *const one[1] = {in[0]};
    gew_layout(&e, out[0], (const IrTensor *const *)one, 1u);
    e.push.alpha = lo;
    e.push.beta = hi;
    const Range ry = range_of(c, n->outputs[0]);
    gew_dispatch(c, &e, SPG_GEW_CLIP, type_code(F32) << 12, out_at(c, 0), in_at(c, 0), 0, 0, ry, range_of(c, n->inputs[0]),
                 ry, ry);
    return true;
}

/* ------------------------------------------------------------------------- views */

/* Reshape, Flatten, Squeeze, Unsqueeze and Identity of a value on the device: the CPU's kernel gives the shape (its
   parameters read on the host), the output a view of the input's memory. */
static bool op_view(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    if (!in[0] || !out[0] || n->output_count != 1) return false;
    if (c->record) return true;
    for (uint32_t i = 1; i < n->input_count; i++)
        if (in[i] && !ask_param(c, in, i)) return false;
    /* (the input's data on the host, if any, is not read: the kernel makes a view) */
    if (!ir_run_node(c->g, c->k, c->g->tensors, false)) {
        spingalett_clear_error();
        return false;
    }
    out[0]->data = NULL;
    out[0]->owned = false;
    out[0]->staged = false;
    as_view(c, 0);
    return true;
}

static const GpuOp ops[IR_OP_COUNT] = {
    [IR_Identity] = op_view, [IR_Reshape] = op_view, [IR_Flatten] = op_view, [IR_Squeeze] = op_view,
    [IR_Unsqueeze] = op_view, [IR_Dropout] = op_view,
    [IR_Neg] = op_elementwise, [IR_Abs] = op_elementwise, [IR_Exp] = op_elementwise, [IR_Log] = op_elementwise,
    [IR_Sqrt] = op_elementwise, [IR_Reciprocal] = op_elementwise, [IR_Sigmoid] = op_elementwise,
    [IR_Tanh] = op_elementwise, [IR_Relu] = op_elementwise, [IR_LeakyRelu] = op_elementwise, [IR_Elu] = op_elementwise,
    [IR_Selu] = op_elementwise, [IR_Gelu] = op_elementwise, [IR_Swish] = op_elementwise,
    [IR_HardSigmoid] = op_elementwise, [IR_HardSwish] = op_elementwise, [IR_Softplus] = op_elementwise,
    [IR_Softsign] = op_elementwise, [IR_Sin] = op_elementwise, [IR_Cos] = op_elementwise, [IR_Erf] = op_elementwise,
    [IR_Floor] = op_elementwise, [IR_Ceil] = op_elementwise, [IR_Round] = op_elementwise, [IR_Sign] = op_elementwise,
    [IR_Not] = op_elementwise, [IR_IsNaN] = op_elementwise, [IR_IsInf] = op_elementwise, [IR_Mish] = op_elementwise,
    [IR_Celu] = op_elementwise, [IR_ThresholdedRelu] = op_elementwise, [IR_Add] = op_elementwise,
    [IR_Sub] = op_elementwise, [IR_Mul] = op_elementwise, [IR_Div] = op_elementwise, [IR_Pow] = op_elementwise,
    [IR_Max] = op_elementwise, [IR_Min] = op_elementwise, [IR_Sum] = op_elementwise, [IR_Mean] = op_elementwise,
    [IR_PRelu] = op_elementwise, [IR_Equal] = op_elementwise, [IR_Less] = op_elementwise,
    [IR_LessOrEqual] = op_elementwise, [IR_Greater] = op_elementwise, [IR_GreaterOrEqual] = op_elementwise,
    [IR_And] = op_elementwise, [IR_Or] = op_elementwise, [IR_Xor] = op_elementwise, [IR_Mod] = op_elementwise,
    [IR_Where] = op_elementwise, [IR_Cast] = op_cast, [IR_CastLike] = op_cast, [IR_Clip] = op_clip,
};

/* The operators whose kernels read only their first input's shape: on the host, without its values. */
static bool shape_only(IrOp op) {
    return op == IR_Shape || op == IR_Size;
}

static bool is_view(IrOp op) {
    return op == IR_Identity || op == IR_Reshape || op == IR_Flatten || op == IR_Squeeze || op == IR_Unsqueeze ||
           op == IR_Dropout;
}

/* ------------------------------------------------------------------------- constants on the device */

/* The constants the nodes of piece pc read on the device, uploaded once a graph (in an arena for those not there
   yet, through memory the host fills). */
static bool upload_constants(SpingalettGraph *g, GpuGraph *gg, Plan *p, Piece *pc) {
    uint32_t *list = NULL, count = 0, cap = 0;
    uint64_t total = 0;
    for (uint32_t j = 0; j < pc->node_count; j++) {
        const IrNode *n = &g->nodes[pc->nodes[j]];
        for (uint32_t i = 0; i < n->input_count; i++) {
            const uint32_t v = n->inputs[i];
            if (v == NONE || g->values[v].kind != VALUE_CONST || gg->consts[v].buffer || !(p->on[v] & ON_DEVICE)) continue;
            bool listed = false;
            for (uint32_t m = 0; m < count && !listed; m++) listed = list[m] == v;
            if (listed) continue;
            if (!grow((void **)&list, &cap, count, sizeof(uint32_t))) {
                free(list);
                return fail("graph on the GPU: out of memory");
            }
            list[count++] = v;
            total += align_up(bytes_of(&g->tensors[v]) ? bytes_of(&g->tensors[v]) : 4u);
        }
    }
    if (!count) return true;
    SpgGpuArena *arena = spg_gpu_arena_create();
    if (!arena || !grow((void **)&gg->arenas, &gg->arena_cap, gg->arena_count, sizeof(SpgGpuArena *))) {
        spg_gpu_arena_free(arena);
        free(list);
        return fail("graph on the GPU: out of memory");
    }
    for (uint32_t m = 0; m < count; m++) {
        const uint64_t bytes = bytes_of(&g->tensors[list[m]]);
        spg_gpu_arena_add(arena, &gg->consts[list[m]], bytes ? bytes : 4u, SPG_MEMORY_DEVICE);
    }
    if (!spg_gpu_arena_commit(arena)) {
        spg_gpu_arena_free(arena);
        for (uint32_t m = 0; m < count; m++) memset(&gg->consts[list[m]], 0, sizeof(SpgGpuBuffer));
        free(list);
        return fail("graph on the GPU: out of device memory for the constants");
    }
    gg->arenas[gg->arena_count++] = arena;
    SpgGpuBuffer stage;
    SpgGpuCommands *cmd = NULL;
    bool ok = spg_gpu_buffer_create(&stage, total, true);
    if (ok) {
        cmd = spg_gpu_commands_create();
        ok = cmd && spg_gpu_record_begin(cmd);
    }
    uint64_t at = 0;
    for (uint32_t m = 0; ok && m < count; m++) {
        const uint32_t v = list[m];
        const uint64_t bytes = bytes_of(&g->tensors[v]);
        if (bytes) memcpy((uint8_t *)stage.mapped + at, g->tensors[v].data, bytes);
        spg_gpu_copy(cmd, &stage, at, &gg->consts[v], 0, bytes);
        at += align_up(bytes ? bytes : 4u);
    }
    ok = ok && spg_gpu_record_end(cmd) && spg_gpu_submit(cmd) && spg_gpu_wait(cmd);
    if (cmd) spg_gpu_commands_free(cmd);
    spg_gpu_buffer_free(&stage);
    free(list);
    return ok ? true : fail("graph on the GPU: the constants could not be uploaded");
}

/* ------------------------------------------------------------------------- pieces */

static void downloaded(SpingalettGraph *g, Plan *p, const Piece *pc) {
    for (uint32_t k = 0; k < pc->down_count; k++) {
        const Transfer *t = &pc->downs[k];
        IrTensor *x = &g->tensors[t->value];
        x->data = (uint8_t *)pc->staging.mapped + t->stage;
        x->staged = true;
        x->owned = false;
        p->on[t->value] |= ON_HOST;
    }
}

/* Submits a piece (its uploads taken from the host's tensors first), waits, and sets its downloads' tensors. */
static bool piece_run(SpingalettGraph *g, Plan *p, Piece *pc) {
    for (uint32_t k = 0; k < pc->up_count; k++) {
        const Transfer *t = &pc->ups[k];
        if (t->bytes) memcpy((uint8_t *)pc->staging.mapped + t->stage, g->tensors[t->value].data, t->bytes);
    }
    if (pc->commands && (!spg_gpu_submit(pc->commands) || !spg_gpu_wait(pc->commands)))
        return fail("graph on the GPU: the device failed to run a piece");
    downloaded(g, p, pc);
    return true;
}

/* Ends the current piece: downloads of the values `want` (count of them) at its end, its memory, its constants, its
   commands recorded; then runs it. */
static bool piece_close(SpingalettGraph *g, GpuGraph *gg, Plan *p, const uint32_t *want, uint32_t count) {
    const uint32_t index = p->piece_count - 1u;
    Piece *pc = &p->pieces[index];
    for (uint32_t k = 0; k < count; k++) {
        if (!grow((void **)&pc->downs, &pc->down_cap, pc->down_count, sizeof(Transfer)))
            return fail("graph on the GPU: out of memory");
        const uint64_t bytes = bytes_of(&g->tensors[want[k]]);
        pc->downs[pc->down_count++] = (Transfer){want[k], pc->node_count, pc->staged, bytes};
        pc->staged += align_up(bytes ? bytes : 1u);
    }
    if (!pc->node_count && !pc->up_count && !pc->down_count) {
        /* (nothing to run: an empty piece is dropped, unless pieces before it hold memory) */
        return add_step(p, STEP_PIECE, index) ? true : fail("graph on the GPU: out of memory");
    }
    if (!upload_constants(g, gg, p, pc)) return false;
    if (pc->size && !spg_gpu_buffer_create(&pc->memory, pc->size, false))
        return fail("graph on the GPU: out of device memory");
    if (pc->staged && !spg_gpu_buffer_create(&pc->staging, pc->staged, true))
        return fail("graph on the GPU: out of memory");
    pc->commands = spg_gpu_commands_create();
    if (!pc->commands || !spg_gpu_record_begin(pc->commands)) return fail("graph on the GPU: no commands");
    Recorder r;
    memset(&r, 0, sizeof r);
    r.c = pc->commands;
    GCtx c = {g, gg, p, 0, NULL, true, &r, 0, 0, 0};
    uint32_t up = 0;
    for (uint32_t j = 0; j < pc->node_count; j++) {
        /* the uploads this node is the first to read */
        for (; up < pc->up_count && pc->ups[up].before == j; up++) {
            const Transfer *t = &pc->ups[up];
            Access a = {0};
            writes(&a, range_of(&c, t->value));
            need(&r, &a);
            const Block *b = &p->blocks[p->block_of[t->value]];
            spg_gpu_copy(pc->commands, &pc->staging, t->stage, &p->pieces[b->piece].memory, b->at, t->bytes);
        }
        const uint32_t k = pc->nodes[j];
        const IrNode *n = &g->nodes[k];
        IrTensor *in[64], *out[64];
        for (uint32_t i = 0; i < n->input_count; i++) in[i] = n->inputs[i] == NONE ? NULL : &g->tensors[n->inputs[i]];
        for (uint32_t o = 0; o < n->output_count; o++) out[o] = n->outputs[o] == NONE ? NULL : &g->tensors[n->outputs[o]];
        c.k = k;
        c.n = n;
        c.scratch = p->scratch[k].bytes ? pc->memory.address + p->scratch[k].at : 0;
        ops[n->op](&c, in, out);
    }
    for (uint32_t k = 0; k < pc->down_count; k++) {
        const Transfer *t = &pc->downs[k];
        Access a = {0};
        reads(&a, range_of(&c, t->value));
        need(&r, &a);
        const uint32_t b = p->block_of[t->value];
        const SpgGpuBuffer *src = g->values[t->value].kind == VALUE_CONST ? &gg->consts[t->value]
                                                                          : &p->pieces[p->blocks[b].piece].memory;
        spg_gpu_copy(pc->commands, src, g->values[t->value].kind == VALUE_CONST ? 0 : p->blocks[b].at, &pc->staging,
                     t->stage, t->bytes);
    }
    spg_gpu_barrier_host(pc->commands);
    if (!spg_gpu_record_end(pc->commands)) return fail("graph on the GPU: its commands could not be recorded");
    if (!add_step(p, STEP_PIECE, index)) return fail("graph on the GPU: out of memory");
    return piece_run(g, p, pc);
}

/* ------------------------------------------------------------------------- making a plan */

static bool has_device(const Plan *p, uint32_t v) {
    return p->on[v] & ON_DEVICE;
}

/* Whether node k goes to the device, planned there (its outputs' shapes set): its operator's kernels take it, and it
   reads a value only the device has, or its outputs or the host's inputs it reads are large. */
static bool plan_device(SpingalettGraph *g, GpuGraph *gg, Plan *p, uint32_t k, GCtx *c) {
    const IrNode *n = &g->nodes[k];
    const GpuOp op = ops[n->op];
    if (!op || shape_only((IrOp)n->op) || n->input_count > 32 || n->output_count > 32) return false;
    bool device_input = false, large = false;
    for (uint32_t i = 0; i < n->input_count; i++) {
        const uint32_t v = n->inputs[i];
        if (v == NONE) continue;
        if (has_device(p, v) && !(p->on[v] & ON_HOST)) device_input = true;
        if (g->values[v].kind != VALUE_CONST && g->tensors[v].count >= p->heavy) large = true;
    }
    /* (a view goes where its input is) */
    if (is_view((IrOp)n->op) && (n->inputs[0] == NONE || !has_device(p, n->inputs[0]))) return false;
    if (!device_input && !large && !is_view((IrOp)n->op)) {
        /* the outputs' sizes decide: planned to see them */
        GCtx probe = {g, gg, p, k, n, false, NULL, 0, 0, 0};
        IrTensor saved[32];
        IrTensor *in[32], *out[32];
        for (uint32_t i = 0; i < n->input_count; i++) in[i] = n->inputs[i] == NONE ? NULL : &g->tensors[n->inputs[i]];
        for (uint32_t o = 0; o < n->output_count; o++) {
            out[o] = n->outputs[o] == NONE ? NULL : &g->tensors[n->outputs[o]];
            if (out[o]) saved[o] = *out[o];
        }
        bool big = false;
        if (op(&probe, in, out))
            for (uint32_t o = 0; o < n->output_count; o++) big = big || (out[o] && out[o]->count >= p->heavy);
        for (uint32_t o = 0; o < n->output_count; o++)
            if (out[o]) *out[o] = saved[o];
        if (!big) return false;
    }
    IrTensor *in[32], *out[32];
    for (uint32_t i = 0; i < n->input_count; i++) in[i] = n->inputs[i] == NONE ? NULL : &g->tensors[n->inputs[i]];
    for (uint32_t o = 0; o < n->output_count; o++) out[o] = n->outputs[o] == NONE ? NULL : &g->tensors[n->outputs[o]];
    *c = (GCtx){g, gg, p, k, n, false, NULL, 0, 0, 0};
    IrTensor saved[32];
    for (uint32_t o = 0; o < n->output_count; o++)
        if (out[o]) saved[o] = *out[o];
    if (!op(c, in, out)) {
        for (uint32_t o = 0; o < n->output_count; o++)
            if (out[o]) *out[o] = saved[o];
        return false;
    }
    return true;
}

/* Makes the plan of node k on the device after plan_device(): its inputs uploaded where the host has them only, its
   outputs' memory, its scratch. */
static bool place_device(SpingalettGraph *g, Plan *p, uint32_t k, const GCtx *c) {
    const IrNode *n = &g->nodes[k];
    Piece *pc = &p->pieces[p->piece_count - 1u];
    for (uint32_t i = 0; i < n->input_count; i++) {
        const uint32_t v = n->inputs[i];
        if (v == NONE || (c->params >> i & 1u)) continue;
        if (g->values[v].kind == VALUE_CONST) {
            p->on[v] |= ON_DEVICE;
            continue;
        }
        if (has_device(p, v)) continue;
        /* uploaded by this piece */
        const uint64_t bytes = bytes_of(&g->tensors[v]);
        const uint32_t b = block_new(p, bytes);
        if (b == NONE || !grow((void **)&pc->ups, &pc->up_cap, pc->up_count, sizeof(Transfer))) return false;
        pc->ups[pc->up_count++] = (Transfer){v, pc->node_count, pc->staged, bytes};
        pc->staged += align_up(bytes ? bytes : 1u);
        p->block_of[v] = b;
        p->on[v] |= ON_DEVICE;
    }
    for (uint32_t i = 0; i < n->input_count; i++) {
        const uint32_t v = n->inputs[i];
        if (v == NONE || !(c->params >> i & 1u) || g->values[v].kind == VALUE_CONST || p->param[v]) continue;
        /* a parameter of the dispatches: its values kept, which replays check */
        const uint64_t bytes = bytes_of(&g->tensors[v]);
        p->param[v] = (uint8_t *)malloc(bytes ? bytes : 1u);
        if (!p->param[v]) return false;
        if (bytes) memcpy(p->param[v], g->tensors[v].data, bytes);
    }
    for (uint32_t o = 0; o < n->output_count; o++) {
        const uint32_t v = n->outputs[o];
        if (v == NONE) continue;
        if (c->views >> o & 1u) {
            const uint32_t src = n->inputs[0];
            if (g->values[src].kind == VALUE_CONST) return false;
            p->block_of[v] = p->block_of[src];
            p->blocks[p->block_of[v]].refs++;
        } else {
            const uint32_t b = block_new(p, bytes_of(&g->tensors[v]));
            if (b == NONE) return false;
            p->block_of[v] = b;
        }
        p->on[v] = ON_DEVICE;
    }
    if (c->scratch) {
        Piece *cur = &p->pieces[p->piece_count - 1u];
        p->scratch[k].bytes = c->scratch;
        p->scratch[k].at = piece_take(cur, c->scratch);
    }
    if (!grow((void **)&pc->nodes, &pc->node_cap, pc->node_count, sizeof(uint32_t))) return false;
    pc->nodes[pc->node_count++] = k;
    if (c->scratch) piece_give(&p->pieces[p->piece_count - 1u], p->scratch[k].at, p->scratch[k].bytes);
    return true;
}

/* Lets go of node k's inputs read for the last time, and of outputs nothing reads. */
static void release_after(SpingalettGraph *g, Plan *p, uint32_t k) {
    const IrNode *n = &g->nodes[k];
    for (uint32_t i = 0; i < n->input_count; i++) {
        const uint32_t v = n->inputs[i];
        if (v == NONE || g->values[v].kind == VALUE_CONST || g->values[v].last != (int32_t)k) continue;
        bool again = false;             /* (an input read twice by the node: once) */
        for (uint32_t j = 0; j < i && !again; j++) again = n->inputs[j] == v;
        if (!again && p->block_of[v] != NONE) block_drop(p, p->block_of[v]);
    }
    for (uint32_t o = 0; o < n->output_count; o++) {
        const uint32_t v = n->outputs[o];
        if (v != NONE && g->values[v].last < 0 && p->block_of[v] != NONE) block_drop(p, p->block_of[v]);
    }
}

/* Runs the graph while making plan p of it (the run's inputs and constants bound in g->tensors). */
static bool plan_run(SpingalettGraph *g, GpuGraph *gg, Plan *p) {
    for (uint32_t v = 0; v < g->value_count; v++)
        p->on[v] = g->values[v].kind == VALUE_CONST || g->values[v].kind == VALUE_INPUT ? ON_HOST : 0;
    for (uint32_t v = 0; v < g->value_count; v++)
        if (g->values[v].kind == VALUE_CONST && gg->consts[v].buffer) p->on[v] |= ON_DEVICE;
    if (!piece_open(p)) return fail("graph on the GPU: out of memory");
    const bool parallel = spingalett_get_num_threads() > 1u;
    uint32_t *want = (uint32_t *)malloc(64 * sizeof(uint32_t) + 4u * (g->output_count + 1u));
    if (!want) return fail("graph on the GPU: out of memory");
    bool ok = true, devices = false;
    for (uint32_t k = 0; ok && k < g->node_count; k++) {
        const IrNode *n = &g->nodes[k];
        GCtx c;
        if (plan_device(g, gg, p, k, &c)) {
            ok = place_device(g, p, k, &c) || fail("graph on the GPU: out of memory");
            devices = true;
            p->device_nodes++;
        } else {
            /* on the host: the values it reads that only the device has, downloaded (the piece ends) */
            uint32_t count = 0;
            for (uint32_t i = 0; i < n->input_count && i < 64; i++) {
                const uint32_t v = n->inputs[i];
                if (v == NONE || (p->on[v] & ON_HOST) || (i == 0 && shape_only((IrOp)n->op))) continue;
                bool listed = false;
                for (uint32_t j = 0; j < count && !listed; j++) listed = want[j] == v;
                if (!listed) want[count++] = v;
            }
            if (count) {
                ok = piece_close(g, gg, p, want, count) && piece_open(p);
                if (!ok) break;
            }
            ok = ir_run_node(g, k, g->tensors, parallel) && add_step(p, STEP_HOST, k);
            for (uint32_t o = 0; ok && o < n->output_count; o++)
                if (n->outputs[o] != NONE) p->on[n->outputs[o]] = ON_HOST;
        }
        if (ok) release_after(g, p, k);
    }
    /* the outputs only the device has, downloaded by the last piece */
    uint32_t count = 0;
    for (uint32_t o = 0; ok && o < g->output_count; o++) {
        const uint32_t v = g->outputs[o];
        if (p->on[v] & ON_HOST) continue;
        bool listed = false;
        for (uint32_t j = 0; j < count && !listed; j++) listed = want[j] == v;
        if (!listed) want[count++] = v;
    }
    ok = ok && piece_close(g, gg, p, want, count);
    free(want);
    p->host_only = !devices;
    if (ok) spingalett_log(LOG_DEBUG, "graph on the GPU: %u of %u nodes on the device, %u pieces", p->device_nodes,
                           g->node_count, p->piece_count);
    /* the plan's types and shapes, for the checks of replays */
    for (uint32_t v = 0; ok && v < g->value_count; v++) {
        p->meta[v] = g->tensors[v];
        p->meta[v].data = NULL;
        p->meta[v].owned = false;
        p->meta[v].staged = false;
    }
    return ok;
}

/* ------------------------------------------------------------------------- replaying a plan */

static bool same_shape(const IrTensor *a, const IrTensor *b) {
    if (a->dtype != b->dtype || a->rank != b->rank) return false;
    for (uint32_t d = 0; d < a->rank; d++)
        if (a->dims[d] != b->dims[d]) return false;
    return true;
}

/* 1: ran, 0: failed (the error set), -1: a host node's outputs left the plan (a new one is needed). */
static int replay(SpingalettGraph *g, Plan *p) {
    for (uint32_t v = 0; v < g->value_count; v++) {
        if (g->values[v].kind != VALUE_NODE || !(p->on[v] & ON_DEVICE)) continue;
        g->tensors[v] = p->meta[v];
    }
    const bool parallel = spingalett_get_num_threads() > 1u;
    for (uint32_t s = 0; s < p->step_count; s++) {
        const Step *st = &p->steps[s];
        if (st->kind == STEP_PIECE) {
            if (!piece_run(g, p, &p->pieces[st->index])) return 0;
            continue;
        }
        if (!ir_run_node(g, st->index, g->tensors, parallel)) return 0;
        const IrNode *n = &g->nodes[st->index];
        for (uint32_t o = 0; o < n->output_count; o++) {
            const uint32_t v = n->outputs[o];
            if (v == NONE) continue;
            const IrTensor *t = &g->tensors[v];
            if (!same_shape(t, &p->meta[v])) return -1;
            if (p->param[v] && bytes_of(t) && memcmp(p->param[v], t->data, (size_t)bytes_of(t))) return -1;
        }
    }
    return 1;
}

/* ------------------------------------------------------------------------- the graph's entry points */

static GpuGraph *gpu_of(SpingalettGraph *g) {
    GpuGraph *gg = (GpuGraph *)g->gpu;
    if (gg) return gg;
    gg = (GpuGraph *)calloc(1, sizeof *gg);
    if (!gg) return NULL;
    gg->consts = (SpgGpuBuffer *)calloc(g->value_count ? g->value_count : 1u, sizeof(SpgGpuBuffer));
    if (!gg->consts) {
        free(gg);
        return NULL;
    }
    gg->backend = spg_gpu_using();
    g->gpu = gg;
    return gg;
}

int spingalett_gpu_graph_run(SpingalettGraph *g) {
    GpuGraph *gg = gpu_of(g);
    if (!gg) return -1;
    if (gg->backend != spg_gpu_using()) {
        /* (another backend than its constants'): made again */
        spingalett_gpu_graph_free(g);
        if (!(gg = gpu_of(g))) return -1;
    }
    gg->runs++;
    uint32_t key_len;
    uint64_t *key = run_key(g, &key_len);
    if (!key) return -1;
    Plan *p = NULL;
    uint32_t slot = 0;
    for (uint32_t k = 0; k < PLANS; k++) {
        Plan *q = gg->plans[k];
        if (q && q->key_len == key_len && !memcmp(q->key, key, key_len * sizeof(uint64_t))) {
            p = q;
            slot = k;
            break;
        }
    }
    if (p && p->host_only) {
        free(key);
        return -1;
    }
    if (p) {
        free(key);
        p->used = gg->runs;
        const int r = replay(g, p);
        if (r >= 0) return r;
        /* (the run goes on from where the replay stopped as a new plan: its host nodes run again) */
        plan_free(p);
        gg->plans[slot] = NULL;
        key = run_key(g, &key_len);
        if (!key) return -1;
        for (uint32_t v = 0; v < g->value_count; v++)
            if (g->values[v].kind == VALUE_NODE) g->tensors[v].staged = false;
    }
    /* a new plan, in the free slot or the least recently used one's */
    slot = 0;
    for (uint32_t k = 0; k < PLANS; k++) {
        if (!gg->plans[k]) { slot = k; break; }
        if (gg->plans[k]->used < gg->plans[slot]->used) slot = k;
    }
    plan_free(gg->plans[slot]);
    gg->plans[slot] = NULL;
    p = plan_new(g);
    if (!p) {
        free(key);
        return -1;
    }
    p->key = key;
    p->key_len = key_len;
    p->used = gg->runs;
    if (!plan_run(g, gg, p)) {
        spingalett_log(LOG_WARNING, "graph on the GPU: %s. Running on the CPU.", spingalett_last_error_message());
        spingalett_clear_error();
        plan_free(p);
        for (uint32_t v = 0; v < g->value_count; v++) g->tensors[v].staged = false;
        return -1;
    }
    gg->plans[slot] = p;
    return 1;
}

void spingalett_gpu_graph_free(SpingalettGraph *g) {
    GpuGraph *gg = (GpuGraph *)g->gpu;
    if (!gg) return;
    const SpgBackend saved = spg_gpu_using();
    spg_gpu_use(gg->backend);
    for (uint32_t k = 0; k < PLANS; k++) plan_free(gg->plans[k]);
    for (uint32_t k = 0; k < gg->arena_count; k++) spg_gpu_arena_free(gg->arenas[k]);
    free(gg->arenas);
    free(gg->consts);
    free(gg);
    g->gpu = NULL;
    spg_gpu_use(saved);
}
