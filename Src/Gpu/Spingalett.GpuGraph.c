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
#include <stdio.h>
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
   node `before` in its order: its memory is the value's from there on only; a download of one kept as bfloat16 through
   its floats at `wide` - 1 in the piece's memory). */
typedef struct {
    uint32_t value, before;
    uint64_t stage, bytes, wide;
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
    SpgGpuBuffer params;            /* its nodes' parameter blocks (strides and the like, written when recorded) */
    uint8_t *param_bytes;           /* (while recorded) */
    uint64_t params_size, params_used;
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
    uint64_t *offset;               /* [value_count]: a value's bytes from its block's start (views of part of one) */
    uint8_t *half;                  /* [value_count]: a value kept as bfloat16 (its readers products on the matrix
                                       units and attention, through copies and views) */
    uint32_t *half_block;           /* [value_count]: the block of a value's copy rounded to bfloat16 (NONE: none) */
    uint32_t *half_maker;           /* [value_count]: the node that makes it, the first of the products that read it */
    uint8_t **param;                /* [value_count]: the values of host values the device's nodes read as parameters */
    Scratch *scratch;               /* [node_count] */
} Plan;

typedef struct {
    SpgBackend backend;
    Plan *plans[PLANS];
    SpgGpuBuffer *consts;           /* [value_count]: constants on the device (buffer NULL: not there) */
    SpgGpuBuffer *prepared;         /* [2 node_count]: what a node's kernels made of its constants, once (the forms of
                                       single precision, then of bfloat16) */
    uint32_t *reader_at, *readers;  /* [value_count + 1], [...]: the nodes reading each value (its inputs' places), the
                                       graph's outputs as NONE */
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
    uint32_t replaced;              /* planning: constant inputs the node reads as prepared() made them (bits) */
    uint32_t views;                 /* planning: outputs that are views of input 0 (bits) */
    uint64_t view_at[8];            /* planning: their bytes from input 0's start (as_view_at()) */
    uint64_t scratch;               /* planning: bytes asked for; recording: their address */
    uint64_t param;                 /* planning: bytes of parameter blocks asked for (param_put()) */
    uint32_t halves;                /* planning: inputs whose copies rounded to bfloat16 the node reads (bits) */
    uint32_t half_out;              /* planning: outputs the node can write as bfloat16 (bits) */
    uint32_t half_force;            /* planning: outputs it writes so whatever their readers (copies of bfloat16) */
    uint64_t half_count[4];         /* (their values: inputs 0 to 3) */
    Piece *piece;                   /* recording: the piece */
    uint32_t half_ins;              /* planning: inputs it reads as bfloat16 where they are kept so (bits; views'
                                       input 0 always) */
} GCtx;

/* A device node: planning (c->record false) checks that the kernels take it (false: the host runs it) and sets its
   outputs' types and shapes in out[], with ask_param() for inputs read on the host and as_view() for outputs that
   are views; recording dispatches its kernels. in[] and out[] are the run's tensors (a host value's data, else
   none). */
typedef bool (*GpuOp)(GCtx *c, IrTensor *const *in, IrTensor **out);

static bool epilogue_act(int64_t act);
static bool product_bf16(uint32_t M, uint32_t N, uint32_t K);

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
        spg_gpu_buffer_free(&pc->params);
        free(pc->param_bytes);
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
    free(p->offset);
    free(p->half);
    free(p->half_block);
    free(p->half_maker);
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
    p->offset = (uint64_t *)calloc(V, sizeof(uint64_t));
    p->half = (uint8_t *)calloc(V, 1);
    p->half_block = (uint32_t *)malloc(V * sizeof(uint32_t));
    p->half_maker = (uint32_t *)malloc(V * sizeof(uint32_t));
    /* (one past the values: the plan itself, which ends the list for plan_free()) */
    p->param = (uint8_t **)calloc(V + 1u, sizeof(uint8_t *));
    p->scratch = (Scratch *)calloc(N, sizeof(Scratch));
    if (!p->meta || !p->on || !p->block_of || !p->offset || !p->half || !p->half_block || !p->half_maker || !p->param ||
        !p->scratch) {
        if (p->param) p->param[V] = (uint8_t *)p;
        plan_free(p);
        return NULL;
    }
    p->param[g->value_count] = (uint8_t *)p;
    for (uint32_t v = 0; v < g->value_count; v++) p->block_of[v] = p->half_block[v] = p->half_maker[v] = NONE;
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
    return c->p->pieces[k->piece].memory.address + k->at + c->p->offset[v];
}

static uint64_t in_at(const GCtx *c, uint32_t i) {
    return i < c->n->input_count ? address_of(c, c->n->inputs[i]) : 0;
}
static uint64_t out_at(const GCtx *c, uint32_t o) {
    return o < c->n->output_count ? address_of(c, c->n->outputs[o]) : 0;
}

static Range range_of(const GCtx *c, uint32_t v);

/* The buffer holding a value on the device and the value's offset in it (fills and copies take buffers). */
static const SpgGpuBuffer *buffer_of(const GCtx *c, uint32_t v, uint64_t *offset) {
    if (c->g->values[v].kind == VALUE_CONST) {
        *offset = 0;
        return &c->gg->consts[v];
    }
    const Block *k = &c->p->blocks[c->p->block_of[v]];
    *offset = k->at + c->p->offset[v];
    return &c->p->pieces[k->piece].memory;
}

/* Records output o filled with a 32-bit pattern (bytes: a multiple of four, from its start). */
static void fill_output(GCtx *c, uint32_t o, uint64_t bytes, uint32_t value) {
    const uint32_t v = c->n->outputs[o];
    Access a = {0};
    writes(&a, range_of(c, v));
    need(c->r, &a);
    uint64_t at;
    const SpgGpuBuffer *b = buffer_of(c, v, &at);
    spg_gpu_fill(c->r->c, b, (size_t)at, (size_t)(bytes & ~(uint64_t)3u), value);
}

/* Bytes a value takes on the device: its type's, or two a value kept as bfloat16. */
static uint64_t stored_bytes(const Plan *p, const SpingalettGraph *g, uint32_t v) {
    const IrTensor *t = &g->tensors[v];
    return p->half[v] ? 2u * t->count : bytes_of(t);
}

static Range range_of(const GCtx *c, uint32_t v) {
    const uint64_t a = address_of(c, v);
    return (Range){a, a + stored_bytes(c->p, c->g, v)};
}

/* Bytes of an element of input i as the device keeps it. */
static uint32_t in_size(const GCtx *c, uint32_t i) {
    const uint32_t v = c->n->inputs[i];
    return c->p->half[v] ? 2u : (uint32_t)ir_dtype_size(c->g->tensors[v].dtype);
}

static bool half_in(const GCtx *c, uint32_t i) {
    return i < c->n->input_count && c->n->inputs[i] != NONE && c->p->half[c->n->inputs[i]];
}
static bool half_output(const GCtx *c, uint32_t o) {
    return o < c->n->output_count && c->n->outputs[o] != NONE && c->p->half[c->n->outputs[o]];
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

/* Output o a view of part of input 0's memory, from bytes `at` on (outputs below 8). */
static void as_view_at(GCtx *c, uint32_t o, uint64_t at) {
    c->views |= 1u << o;
    c->view_at[o] = at;
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

/* What the node's kernels read in place of constant input i: `bytes` that fill() writes from the constant's values,
   made once a graph and node (while the node is planned, on the device then); its address (0 when it could not be
   made: memory ran out). The constant itself is not uploaded for the node. */
static uint64_t prepared_form(GCtx *c, uint32_t i, uint32_t form, uint64_t bytes,
                              void (*fill)(void *dst, const IrTensor *src, const void *arg), const void *arg) {
    GpuGraph *gg = c->gg;
    SpgGpuBuffer *b = &gg->prepared[2u * c->k + form];
    if (!c->record && !b->buffer) {
        SpgGpuBuffer stage;
        if (!spg_gpu_buffer_create(&stage, bytes, true)) return 0;
        fill(stage.mapped, &c->g->tensors[c->n->inputs[i]], arg);
        SpgGpuCommands *cmd = NULL;
        bool ok = spg_gpu_buffer_create(b, bytes, false);
        if (ok) {
            cmd = spg_gpu_commands_create();
            ok = cmd && spg_gpu_record_begin(cmd);
        }
        if (ok) spg_gpu_copy(cmd, &stage, 0, b, 0, bytes);
        ok = ok && spg_gpu_record_end(cmd) && spg_gpu_submit(cmd) && spg_gpu_wait(cmd);
        if (cmd) spg_gpu_commands_free(cmd);
        spg_gpu_buffer_free(&stage);
        if (!ok) {
            spg_gpu_buffer_free(b);
            return 0;
        }
    }
    if (!c->record) c->replaced |= 1u << i;
    return b->address;
}

static uint64_t prepared(GCtx *c, uint32_t i, uint64_t bytes, void (*fill)(void *dst, const IrTensor *src, const void *arg),
                         const void *arg) {
    return prepared_form(c, i, 0, bytes, fill, arg);
}

static void convert_strided(GCtx *c, uint64_t y, uint64_t x, uint32_t rank, const int64_t *dims, const int64_t *sx,
                            const int64_t *sy, Range ry, Range rx);

/* Input i's values rounded to bfloat16 (count of them, contiguous: the operands of products on the matrix units): one
   copy a value, made by the first node that reads it so, in its piece, kept until the value's last reader. Its
   address while recorded (0 while planned). */
static uint64_t half_input(GCtx *c, uint32_t i, uint64_t count) {
    if (!c->record) {
        if (i < 4) {
            c->halves |= 1u << i;
            c->half_count[i] = count;
        }
        return 0;
    }
    const uint32_t v = c->n->inputs[i];
    const Block *b = &c->p->blocks[c->p->half_block[v]];
    const uint64_t at = c->p->pieces[b->piece].memory.address + b->at;
    if (c->p->half_maker[v] == c->k) {
        const int64_t flat[1] = {(int64_t)count}, one[1] = {1};
        convert_strided(c, at, in_at(c, i), 1, flat, one, one, (Range){at, at + 2u * count}, range_of(c, v));
    }
    return at;
}

/* A block of parameters for a kernel: asked for while planned (bytes), written while recorded: its device address. */
/* Bytes of a parameter block at `at` in the device's memory at once, waited for; false when they could not be. */
static bool param_upload(GCtx *c, uint64_t at, const void *data, uint32_t bytes) {
    const SpgGpuBuffer *params = &c->piece->params;
    SpgGpuBuffer stage;
    if (!spg_gpu_buffer_create(&stage, bytes, true)) return false;
    memcpy(stage.mapped, data, bytes);
    SpgGpuCommands *cmd = spg_gpu_commands_create();
    bool ok = cmd && spg_gpu_record_begin(cmd);
    if (ok) spg_gpu_copy(cmd, &stage, 0, params, (size_t)(at - params->address), bytes);
    ok = ok && spg_gpu_record_end(cmd) && spg_gpu_submit(cmd) && spg_gpu_wait(cmd);
    if (cmd) spg_gpu_commands_free(cmd);
    spg_gpu_buffer_free(&stage);
    return ok;
}

static uint64_t param_put(GCtx *c, const void *data, uint32_t bytes) {
    if (!c->record) {
        c->param += (bytes + 15u) & ~15u;
        return 0;
    }
    Piece *pc = c->piece;
    const uint64_t at = pc->params_used;
    memcpy(pc->param_bytes + at, data, bytes);
    pc->params_used += (bytes + 15u) & ~15u;
    return pc->params.address + at;
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

/* ------------------------------------------------------------------------- copies through strides */

/* A list of integers from input i (read on the host as a parameter) or else from attribute attr: their count, -1 for
   none, -2 for an input the host does not have. */
static int64_t ints_of(GCtx *c, IrTensor *const *in, uint32_t i, const char *attr, int64_t *out, int64_t most) {
    if (i < c->n->input_count && in[i]) {
        const IrTensor *t = ask_param(c, in, i);
        if (!t || (t->dtype != SPINGALETT_DTYPE_INT64 && t->dtype != SPINGALETT_DTYPE_INT32) || (int64_t)t->count > most)
            return -2;
        for (uint64_t k = 0; k < t->count; k++) out[k] = ir_get_int(t, k);
        return (int64_t)t->count;
    }
    const IrAttr *a = attr ? ir_attr(c->n, attr) : NULL;
    if (!a || a->type != ATTR_INTS) return -1;
    if ((int64_t)a->count > most) return -2;
    for (uint32_t k = 0; k < a->count; k++) out[k] = a->ints[k];
    return a->count;
}

static bool axis_in(int64_t axis, uint32_t rank, uint32_t *out) {
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= (int64_t)rank) return false;
    *out = (uint32_t)axis;
    return true;
}

/* Row-major strides of a shape, in elements. */
static void strides_of(uint32_t rank, const int64_t *dims, int64_t *s) {
    int64_t st = 1;
    for (uint32_t d = rank; d-- > 0;) {
        s[d] = st;
        st *= dims[d];
    }
}

/* Records y[sum c sy] = x[sum c sx] over the coordinates c of dims (strides in elements of `size` bytes, x and y the
   addresses of the first values): axes of size 1 dropped, neighbours merged where both run through them as one,
   vectors of 16 bytes where the innermost axis is contiguous in both; dispatches over the outer axes where more than
   SPG_COPY_AXES remain. */
static void copy_strided_as(GCtx *c, uint64_t y, uint64_t x, uint32_t rank, const int64_t *dims, const int64_t *sx,
                            const int64_t *sy, uint32_t size, Range ry, Range rx, bool to_bf16);

static void copy_strided(GCtx *c, uint64_t y, uint64_t x, uint32_t rank, const int64_t *dims, const int64_t *sx,
                         const int64_t *sy, uint32_t size, Range ry, Range rx) {
    copy_strided_as(c, y, x, rank, dims, sx, sy, size, ry, rx, false);
}

/* The same of floats rounded to bfloat16 as they are written (gcvt): y's strides in its elements of two bytes. */
static void convert_strided(GCtx *c, uint64_t y, uint64_t x, uint32_t rank, const int64_t *dims, const int64_t *sx,
                            const int64_t *sy, Range ry, Range rx) {
    copy_strided_as(c, y, x, rank, dims, sx, sy, 4u, ry, rx, true);
}

static void copy_strided_as(GCtx *c, uint64_t y, uint64_t x, uint32_t rank, const int64_t *dims, const int64_t *sx,
                            const int64_t *sy, uint32_t size, Range ry, Range rx, bool to_bf16) {
    int64_t d_[IR_MAX_RANK * 2], x_[IR_MAX_RANK * 2], y_[IR_MAX_RANK * 2];
    uint32_t q = 0;
    uint64_t total = 1;
    for (uint32_t d = 0; d < rank; d++) {
        total *= (uint64_t)dims[d];
        if (dims[d] == 1) continue;
        if (q > 0 && x_[q - 1] == sx[d] * dims[d] && y_[q - 1] == sy[d] * dims[d]) {
            d_[q - 1] *= dims[d];
            x_[q - 1] = sx[d];
            y_[q - 1] = sy[d];
        } else {
            d_[q] = dims[d];
            x_[q] = sx[d];
            y_[q] = sy[d];
            q++;
        }
    }
    if (!total) return;
    if (q == 0) {
        d_[0] = 1;
        x_[0] = y_[0] = 1;
        q = 1;
    }
    /* sixteen bytes at a time */
    if (!to_bf16 && x_[q - 1] == 1 && y_[q - 1] == 1 && ((uint64_t)d_[q - 1] * size) % 16u == 0 && x % 16u == 0 &&
        y % 16u == 0) {
        bool aligned = true;
        for (uint32_t d = 0; d + 1 < q && aligned; d++)
            aligned = ((x_[d] * (int64_t)size) % 16 == 0) && ((y_[d] * (int64_t)size) % 16 == 0);
        if (aligned) {
            const int64_t per = 16 / (int64_t)size;
            d_[q - 1] /= per;
            for (uint32_t d = 0; d + 1 < q; d++) {
                x_[d] /= per;
                y_[d] /= per;
            }
            size = 16;
            total /= (uint64_t)per;
        }
    }
    /* the outer axes beyond SPG_COPY_AXES: a dispatch each */
    const uint32_t outer = q > SPG_COPY_AXES ? q - SPG_COPY_AXES : 0;
    uint64_t runs = 1;
    for (uint32_t d = 0; d < outer; d++) runs *= (uint64_t)d_[d];
    Access a = {0};
    reads(&a, rx);
    writes(&a, ry);
    need(c->r, &a);
    const uint32_t spec[1] = {size};
    for (uint64_t run = 0; run < runs; run++) {
        int64_t ox = 0, oy = 0;
        uint64_t r = run;
        for (uint32_t d = outer; d-- > 0;) {
            const int64_t k = (int64_t)(r % (uint64_t)d_[d]);
            r /= (uint64_t)d_[d];
            ox += k * x_[d];
            oy += k * y_[d];
        }
        SpgCopyPush push;
        memset(&push, 0, sizeof push);
        push.y = y + (uint64_t)(oy * (int64_t)size);
        push.total = (uint32_t)(total / runs);
        const uint32_t lead = SPG_COPY_AXES - (q - outer);
        /* (x's address that of its lowest value read, which negative strides put before the first) */
        int64_t low = 0;
        for (uint32_t d = outer; d < q; d++)
            if (x_[d] < 0) low += (d_[d] - 1) * x_[d];
        push.x = x + (uint64_t)((ox + low) * (int64_t)size);
        push.base = (uint32_t)-low;
        if (to_bf16) push.y = y + (uint64_t)(oy * 2);
        for (uint32_t d = 0; d < SPG_COPY_AXES; d++) {
            push.dims[d] = d < lead ? 1u : (uint32_t)d_[outer + d - lead];
            push.sx[d] = d < lead ? 0 : (int32_t)x_[outer + d - lead];
            push.sy[d] = d < lead ? 0 : (int32_t)y_[outer + d - lead];
        }
        spg_gpu_dispatch(c->r->c, to_bf16 ? SPG_KERNEL_gcvt : SPG_KERNEL_gcopy, spec, to_bf16 ? 0u : 1u, &push,
                         sizeof push, group_count(push.total, 256u), 1, 1);
    }
}

/* Whether a copy of these strides fits gcopy's 32-bit offsets. */
static bool copy_fits(uint64_t xcount, uint64_t ycount) {
    return xcount < ((uint64_t)1 << 31) && ycount < ((uint64_t)1 << 31);
}

static uint32_t esize(const IrTensor *t) {
    return (uint32_t)ir_dtype_size(t->dtype);
}

static bool op_transpose(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrTensor *x = in[0];
    if (!x || !out[0] || !copy_fits(x->count, x->count)) return false;
    const uint32_t R = x->rank;
    int64_t perm[IR_MAX_RANK], dims[IR_MAX_RANK];
    const IrAttr *a = ir_attr(c->n, "perm");
    bool seen[IR_MAX_RANK] = {false};
    for (uint32_t k = 0; k < R; k++) {
        perm[k] = a && a->type == ATTR_INTS && a->count == R ? a->ints[k] : (int64_t)(R - 1 - k);
        if (perm[k] < 0 || perm[k] >= (int64_t)R || seen[perm[k]]) return false;
        seen[perm[k]] = true;
        dims[k] = x->dims[perm[k]];
    }
    /* (moving axes of size 1 only: a view) */
    const bool view = ir_moves_ones(x->dims, perm, R) && c->g->values[c->n->inputs[0]].kind != VALUE_CONST;
    if (!c->record) {
        c->half_ins = 1u;
        if (view) as_view(c, 0);
        /* (bfloat16 copied as such; floats rounded to it where the readers want it) */
        else if (half_in(c, 0)) c->half_force |= 1u;
        else if (x->dtype == F32) c->half_out |= 1u;
        return set_shape(out[0], x->dtype, R, dims);
    }
    if (view) return true;
    int64_t xs[IR_MAX_RANK], sx[IR_MAX_RANK], sy[IR_MAX_RANK];
    strides_of(R, x->dims, xs);
    strides_of(R, dims, sy);
    for (uint32_t k = 0; k < R; k++) sx[k] = xs[perm[k]];
    if (half_output(c, 0) && !half_in(c, 0))
        convert_strided(c, out_at(c, 0), in_at(c, 0), R, dims, sx, sy, range_of(c, c->n->outputs[0]),
                        range_of(c, c->n->inputs[0]));
    else
        copy_strided(c, out_at(c, 0), in_at(c, 0), R, dims, sx, sy, in_size(c, 0), range_of(c, c->n->outputs[0]),
                     range_of(c, c->n->inputs[0]));
    return true;
}

static bool op_slice(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrTensor *x = in[0];
    if (!x || !out[0] || !copy_fits(x->count, x->count)) return false;
    const uint32_t R = x->rank, n = c->n->input_count;
    int64_t starts[IR_MAX_RANK], ends[IR_MAX_RANK], axes[IR_MAX_RANK], steps[IR_MAX_RANK];
    const int64_t ns = ints_of(c, in, n > 1 ? 1u : 99u, "starts", starts, IR_MAX_RANK);
    const int64_t ne = ints_of(c, in, n > 2 ? 2u : 99u, "ends", ends, IR_MAX_RANK);
    int64_t na = ints_of(c, in, n > 3 ? 3u : 99u, "axes", axes, IR_MAX_RANK);
    const int64_t nt = ints_of(c, in, n > 4 ? 4u : 99u, "steps", steps, IR_MAX_RANK);
    if (ns < 0 || ne != ns || na == -2 || nt == -2) return false;
    if (na < 0) {
        for (int64_t k = 0; k < ns; k++) axes[k] = k;
        na = ns;
    }
    if (na != ns || (nt >= 0 && nt != ns)) return false;
    int64_t first[IR_MAX_RANK], step[IR_MAX_RANK], dims[IR_MAX_RANK];
    for (uint32_t d = 0; d < R; d++) { first[d] = 0; step[d] = 1; dims[d] = x->dims[d]; }
    for (int64_t k = 0; k < ns; k++) {
        uint32_t a;
        if (!axis_in(axes[k], R, &a)) return false;
        const int64_t size = x->dims[a], st = nt >= 0 ? steps[k] : 1;
        if (!st) return false;
        int64_t s0 = starts[k], e = ends[k];
        if (s0 < 0) s0 += size;
        if (e < 0) e += size;
        if (st > 0) {
            s0 = s0 < 0 ? 0 : s0 > size ? size : s0;
            e = e < 0 ? 0 : e > size ? size : e;
            dims[a] = e > s0 ? (e - s0 + st - 1) / st : 0;
        } else {
            s0 = s0 < 0 ? -1 : s0 > size - 1 ? size - 1 : s0;
            e = e < -1 ? -1 : e > size - 1 ? size - 1 : e;
            dims[a] = s0 > e ? (s0 - e - st - 1) / -st : 0;
        }
        first[a] = s0;
        step[a] = st;
    }
    int64_t xs[IR_MAX_RANK], sx[IR_MAX_RANK], sy[IR_MAX_RANK], base = 0;
    strides_of(R, x->dims, xs);
    strides_of(R, dims, sy);
    for (uint32_t d = 0; d < R; d++) {
        if (dims[d]) base += first[d] * xs[d];
        sx[d] = step[d] * xs[d];
    }
    /* a run of the input's memory (the axes before one sliced with step 1 all of size 1, those after it whole): a
       view from the run's start */
    bool run = c->g->values[c->n->inputs[0]].kind != VALUE_CONST;
    uint32_t a = 0;
    while (a < R && dims[a] == x->dims[a] && x->dims[a] == 1) a++;
    if (a < R && step[a] != 1) run = false;
    for (uint32_t d = a + 1; d < R && run; d++) run = dims[d] == x->dims[d];
    if (!c->record) {
        c->half_ins = 1u;
        if (run) as_view_at(c, 0, (uint64_t)base * in_size(c, 0));
        else if (half_in(c, 0)) c->half_force |= 1u;
        return set_shape(out[0], x->dtype, R, dims);
    }
    if (run) return true;
    copy_strided(c, out_at(c, 0), in_at(c, 0) + (uint64_t)(base * (int64_t)in_size(c, 0)), R, dims, sx, sy, in_size(c, 0),
                 range_of(c, c->n->outputs[0]), range_of(c, c->n->inputs[0]));
    return true;
}

static bool op_concat(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    const IrTensor *first = NULL;
    for (uint32_t k = 0; k < n->input_count && !first; k++) first = in[k];
    if (!first || !out[0]) return false;
    uint32_t axis;
    if (!axis_in(ir_attr_int(n, "axis", 0), first->rank, &axis)) return false;
    int64_t dims[IR_MAX_RANK];
    for (uint32_t d = 0; d < first->rank; d++) dims[d] = first->dims[d];
    dims[axis] = 0;
    for (uint32_t k = 0; k < n->input_count; k++) {
        if (!in[k]) continue;
        if (in[k]->rank != first->rank || in[k]->dtype != first->dtype) return false;
        for (uint32_t d = 0; d < first->rank; d++)
            if (d != axis && in[k]->dims[d] != first->dims[d]) return false;
        dims[axis] += in[k]->dims[axis];
    }
    if (!c->record) {
        uint64_t total = 1;
        for (uint32_t d = 0; d < first->rank; d++) total *= (uint64_t)dims[d];
        return copy_fits(total, total) && set_shape(out[0], first->dtype, first->rank, dims);
    }
    int64_t sy[IR_MAX_RANK];
    strides_of(first->rank, dims, sy);
    int64_t at = 0;
    for (uint32_t k = 0; k < n->input_count; k++) {
        if (!in[k]) continue;
        int64_t sx[IR_MAX_RANK];
        strides_of(in[k]->rank, in[k]->dims, sx);
        copy_strided(c, out_at(c, 0) + (uint64_t)(at * sy[axis] * (int64_t)esize(first)), in_at(c, k), first->rank,
                     in[k]->dims, sx, sy, esize(first), range_of(c, n->outputs[0]), range_of(c, n->inputs[k]));
        at += in[k]->dims[axis];
    }
    return true;
}

static bool op_split(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    const IrTensor *x = in[0];
    const uint32_t m = n->output_count;
    if (!x || m > 64 || !copy_fits(x->count, x->count)) return false;
    uint32_t axis;
    if (!axis_in(ir_attr_int(n, "axis", 0), x->rank, &axis)) return false;
    int64_t sizes[64];
    int64_t ns = ints_of(c, in, n->input_count > 1 ? 1u : 99u, "split", sizes, 64);
    if (ns == -2) return false;
    const int64_t total = x->dims[axis];
    if (ns < 0) {
        const int64_t parts = (int64_t)m, each = (total + parts - 1) / parts;
        for (int64_t k = 0; k < parts; k++) sizes[k] = k + 1 < parts ? each : total - each * (parts - 1);
        ns = parts;
    }
    if (ns != (int64_t)m) return false;
    int64_t sum = 0;
    for (int64_t k = 0; k < ns; k++) {
        if (sizes[k] < 0) return false;
        sum += sizes[k];
    }
    if (sum != total) return false;
    int64_t xs[IR_MAX_RANK];
    strides_of(x->rank, x->dims, xs);
    /* (the axes before the split's all of size 1: each output a run of the input's memory, a view) */
    bool runs = m <= 8 && c->g->values[n->inputs[0]].kind != VALUE_CONST;
    for (uint32_t d = 0; d < axis && runs; d++) runs = x->dims[d] == 1;
    int64_t at = 0;
    for (uint32_t k = 0; k < m; k++) {
        int64_t dims[IR_MAX_RANK];
        for (uint32_t d = 0; d < x->rank; d++) dims[d] = x->dims[d];
        dims[axis] = sizes[k];
        if (out[k]) {
            if (!c->record) {
                if (!set_shape(out[k], x->dtype, x->rank, dims)) return false;
                c->half_ins = 1u;
                if (runs) as_view_at(c, k, (uint64_t)(at * xs[axis]) * in_size(c, 0));
                else if (half_in(c, 0) && k < 32) c->half_force |= 1u << k;
            } else if (!runs) {
                int64_t sy[IR_MAX_RANK];
                strides_of(x->rank, dims, sy);
                copy_strided(c, out_at(c, k), in_at(c, 0) + (uint64_t)(at * xs[axis] * (int64_t)in_size(c, 0)), x->rank,
                             dims, xs, sy, in_size(c, 0), range_of(c, n->outputs[k]), range_of(c, n->inputs[0]));
            }
        }
        at += sizes[k];
    }
    return true;
}

/* x's strides over the axes of a shape it broadcasts to (0 where it does). */
static void broadcast_strides(const IrTensor *t, uint32_t R, const int64_t *dims, int64_t *strides) {
    int64_t st = 1;
    for (uint32_t d = R; d-- > 0;) {
        const int32_t td = (int32_t)d - (int32_t)(R - t->rank);
        const int64_t size = td >= 0 ? t->dims[td] : 1;
        strides[d] = size == 1 && dims[d] != 1 ? 0 : st;
        if (td >= 0) st *= size;
    }
}

static bool op_expand(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrTensor *x = in[0];
    if (!x || !out[0] || c->n->input_count < 2) return false;
    int64_t shape[IR_MAX_RANK];
    const int64_t ns = ints_of(c, in, 1, NULL, shape, IR_MAX_RANK);
    if (ns < 0) return false;
    IrTensor want;
    memset(&want, 0, sizeof want);
    want.rank = (uint8_t)ns;
    for (int64_t k = 0; k < ns; k++) want.dims[k] = shape[k];
    IrTensor *pair[2] = {(IrTensor *)x, &want};
    uint32_t R;
    int64_t dims[IR_MAX_RANK];
    if (!broadcast(pair, 2, &R, dims)) return false;
    if (!c->record) {
        uint64_t total = 1;
        for (uint32_t d = 0; d < R; d++) total *= (uint64_t)dims[d];
        return copy_fits(x->count, total) && set_shape(out[0], x->dtype, R, dims);
    }
    int64_t sx[IR_MAX_RANK], sy[IR_MAX_RANK];
    broadcast_strides(x, R, dims, sx);
    strides_of(R, dims, sy);
    copy_strided(c, out_at(c, 0), in_at(c, 0), R, dims, sx, sy, esize(x), range_of(c, c->n->outputs[0]),
                 range_of(c, c->n->inputs[0]));
    return true;
}

static bool op_tile(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrTensor *x = in[0];
    if (!x || !out[0] || c->n->input_count < 2) return false;
    int64_t reps[IR_MAX_RANK];
    if (ints_of(c, in, 1, NULL, reps, IR_MAX_RANK) != (int64_t)x->rank) return false;
    const uint32_t R = x->rank;
    int64_t dims[IR_MAX_RANK];
    uint64_t total = 1;
    for (uint32_t d = 0; d < R; d++) {
        if (reps[d] < 0) return false;
        dims[d] = x->dims[d] * reps[d];
        total *= (uint64_t)dims[d];
    }
    if (!c->record) return copy_fits(x->count, total) && set_shape(out[0], x->dtype, R, dims);
    /* (each axis as its repeats, outer, over the input's size) */
    int64_t d2[2 * IR_MAX_RANK], sx[2 * IR_MAX_RANK], sy[2 * IR_MAX_RANK], xs[IR_MAX_RANK], ys[IR_MAX_RANK];
    strides_of(R, x->dims, xs);
    strides_of(R, dims, ys);
    for (uint32_t d = 0; d < R; d++) {
        d2[2 * d] = reps[d];
        sx[2 * d] = 0;
        sy[2 * d] = x->dims[d] * ys[d];
        d2[2 * d + 1] = x->dims[d];
        sx[2 * d + 1] = xs[d];
        sy[2 * d + 1] = ys[d];
    }
    copy_strided(c, out_at(c, 0), in_at(c, 0), 2 * R, d2, sx, sy, esize(x), range_of(c, c->n->outputs[0]),
                 range_of(c, c->n->inputs[0]));
    return true;
}

/* Pad of the constant mode: the output filled with the value, the input copied inside it (negative pads crop). */
static bool op_pad(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    const IrTensor *x = in[0];
    if (!x || !out[0] || esize(x) != 4u || !copy_fits(x->count, x->count)) return false;
    const IrAttr *mode = ir_attr(n, "mode");
    if (mode && mode->type == ATTR_STRING && strcmp(mode->s, "constant")) return false;
    const uint32_t R = x->rank;
    int64_t pads[2 * IR_MAX_RANK], axes[IR_MAX_RANK];
    const int64_t np = ints_of(c, in, n->input_count > 1 ? 1u : 99u, "pads", pads, 2 * IR_MAX_RANK);
    int64_t na = ints_of(c, in, n->input_count > 3 ? 3u : 99u, "axes", axes, IR_MAX_RANK);
    if (np < 0 || na == -2) return false;
    if (na < 0) {
        for (uint32_t k = 0; k < R; k++) axes[k] = k;
        na = R;
    }
    if (np != 2 * na) return false;
    int64_t before[IR_MAX_RANK] = {0}, after[IR_MAX_RANK] = {0}, dims[IR_MAX_RANK];
    for (int64_t k = 0; k < na; k++) {
        uint32_t a;
        if (!axis_in(axes[k], R, &a)) return false;
        before[a] = pads[k];
        after[a] = pads[na + k];
    }
    uint64_t total = 1;
    for (uint32_t d = 0; d < R; d++) {
        dims[d] = x->dims[d] + before[d] + after[d];
        if (dims[d] < 0) return false;
        total *= (uint64_t)dims[d];
    }
    uint32_t fill = 0;
    if (n->input_count > 2 && in[2]) {
        const IrTensor *v = ask_param(c, in, 2);
        if (!v || v->dtype != x->dtype || v->count != 1) return false;
        memcpy(&fill, v->data, 4);
    } else if (ir_attr(n, "value")) {
        const float v = ir_attr_float(n, "value", 0.0f);
        if (x->dtype == F32) memcpy(&fill, &v, 4);
        else fill = (uint32_t)(int32_t)v;
    }
    if (!c->record) return copy_fits(total, total) && set_shape(out[0], x->dtype, R, dims);
    const Range ry = range_of(c, n->outputs[0]);
    fill_output(c, 0, ry.hi - ry.lo, fill);
    /* the input's part inside: from x's first kept value to y's place for it */
    int64_t xs[IR_MAX_RANK], ys[IR_MAX_RANK], part[IR_MAX_RANK], ox = 0, oy = 0;
    strides_of(R, x->dims, xs);
    strides_of(R, dims, ys);
    for (uint32_t d = 0; d < R; d++) {
        const int64_t lo = before[d] < 0 ? -before[d] : 0, hi = x->dims[d] + (after[d] < 0 ? after[d] : 0);
        part[d] = hi > lo ? hi - lo : 0;
        ox += lo * xs[d];
        oy += (before[d] > 0 ? before[d] : 0) * ys[d];
    }
    copy_strided(c, out_at(c, 0) + (uint64_t)oy * 4u, in_at(c, 0) + (uint64_t)ox * 4u, R, part, xs, ys, 4u, ry,
                 range_of(c, n->inputs[0]));
    return true;
}

/* Gather along an axis by indices of 32 or 64 bits (on the device, uploaded where the host has them). */
static bool op_gather(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrTensor *x = in[0], *idx = in[1];
    if (!x || !idx || !out[0] || (idx->dtype != SPINGALETT_DTYPE_INT64 && idx->dtype != SPINGALETT_DTYPE_INT32)) return false;
    const uint32_t size = esize(x);
    if (size != 1u && size != 4u && size != 8u) return false;
    uint32_t axis;
    if (!axis_in(ir_attr_int(c->n, "axis", 0), x->rank, &axis)) return false;
    const uint32_t R = x->rank - 1u + idx->rank;
    if (R > IR_MAX_RANK) return false;
    int64_t dims[IR_MAX_RANK];
    uint32_t k = 0;
    uint64_t inner = 1;
    for (uint32_t d = 0; d < axis; d++) dims[k++] = x->dims[d];
    for (uint32_t d = 0; d < idx->rank; d++) dims[k++] = idx->dims[d];
    for (uint32_t d = axis + 1; d < x->rank; d++) {
        dims[k++] = x->dims[d];
        inner *= (uint64_t)x->dims[d];
    }
    /* a constant index of one value, the axes before it all of size 1: a run of the input's memory, a view */
    bool run = c->g->values[c->n->inputs[1]].kind == VALUE_CONST && idx->count == 1 &&
               c->g->values[c->n->inputs[0]].kind != VALUE_CONST;
    for (uint32_t d = 0; d < axis && run; d++) run = x->dims[d] == 1;
    int64_t at = run ? ir_get_int(idx, 0) : 0;
    if (at < 0) at += x->dims[axis];
    run = run && at >= 0 && at < x->dims[axis];
    if (!c->record) {
        if (run) {
            as_view_at(c, 0, (uint64_t)at * inner * size);
            return set_shape(out[0], x->dtype, R, dims);
        }
        return x->count < UINT32_MAX && x->count * idx->count / (x->dims[axis] ? (uint64_t)x->dims[axis] : 1u) <
               UINT32_MAX && set_shape(out[0], x->dtype, R, dims);
    }
    if (run) return true;
    if (!out[0]->count) return true;
    SpgGatherPush push = {out_at(c, 0), in_at(c, 0), in_at(c, 1), (uint32_t)out[0]->count, (uint32_t)inner,
                          (uint32_t)idx->count, (uint32_t)x->dims[axis]};
    const uint32_t spec[2] = {size, type_code(idx->dtype)};
    Access a = {0};
    reads(&a, range_of(c, c->n->inputs[0]));
    reads(&a, range_of(c, c->n->inputs[1]));
    writes(&a, range_of(c, c->n->outputs[0]));
    dispatch(c, &a, SPG_KERNEL_ggather, spec, 2, &push, sizeof push, group_count(push.total, 256u), 1, 1);
    return true;
}

/* ------------------------------------------------------------------------- rows: reductions, softmax, normalizations */

/* Records a grow dispatch over push's rows (outer x inner of len values): threads a row by their length (one where
   the rows interleave: inner above 1). */
static void row_dispatch(GCtx *c, SpgRowPush *push, uint32_t op, Range rx, Range ry, Range ra, Range rb, Range raux) {
    uint32_t tpr = 1;
    if (push->inner == 1u)
        while (tpr < 256u && 4u * tpr * 2u <= push->len) tpr *= 2u;
    push->rows = push->outer * push->inner;
    /* (addresses lavapipe may load from whatever the branches: x's where there is nothing) */
    if (!push->a) push->a = push->x;
    if (!push->b) push->b = push->a;
    if (!push->aux) push->aux = push->y;
    Access a = {0};
    reads(&a, rx);
    reads(&a, ra);
    reads(&a, rb);
    writes(&a, ry);
    writes(&a, raux);
    const uint32_t spec[2] = {op, tpr};
    dispatch(c, &a, SPG_KERNEL_grow, spec, 2, push, sizeof *push, group_count(push->rows, 256u / tpr), 1, 1);
}

/* The rows of a tensor over axes [first, end): outer x len x inner; false past 32 bits. */
static bool rows_over(const IrTensor *x, uint32_t first, uint32_t end, SpgRowPush *push) {
    uint64_t outer = 1, len = 1, inner = 1;
    for (uint32_t d = 0; d < x->rank; d++) {
        if (d < first) outer *= (uint64_t)x->dims[d];
        else if (d < end) len *= (uint64_t)x->dims[d];
        else inner *= (uint64_t)x->dims[d];
    }
    if (x->count >= UINT32_MAX || outer * inner >= UINT32_MAX) return false;
    memset(push, 0, sizeof *push);
    push->outer = (uint32_t)outer;
    push->len = (uint32_t)len;
    push->inner = (uint32_t)inner;
    return true;
}

static uint32_t reduce_code(IrOp op) {
    switch (op) {
        case IR_ReduceSum: return SPG_ROW_SUM;
        case IR_ReduceMean: return SPG_ROW_MEAN;
        case IR_ReduceMax: return SPG_ROW_MAX;
        case IR_ReduceMin: return SPG_ROW_MIN;
        case IR_ReduceProd: return SPG_ROW_PROD;
        case IR_ReduceL1: return SPG_ROW_L1;
        case IR_ReduceL2: return SPG_ROW_L2;
        case IR_ReduceSumSquare: return SPG_ROW_SUMSQ;
        case IR_ReduceLogSumExp: return SPG_ROW_LOGSUMEXP;
        case IR_ReduceLogSum: return SPG_ROW_LOGSUM;
        default: return NONE;
    }
}

/* The Reduce operators of floats over axes that are one block (the others by the host). */
static bool op_reduce(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    const IrTensor *x = in[0];
    const uint32_t code = reduce_code((IrOp)n->op);
    if (!x || !out[0] || x->dtype != F32 || code == NONE) return false;
    int64_t axes[IR_MAX_RANK];
    const int64_t na = ints_of(c, in, n->input_count > 1 ? 1u : 99u, "axes", axes, IR_MAX_RANK);
    if (na == -2 || (na <= 0 && ir_attr_int(n, "noop_with_empty_axes", 0))) return false;
    const bool keep = ir_attr_int(n, "keepdims", 1) != 0;
    bool red[IR_MAX_RANK] = {false};
    for (uint32_t d = 0; d < x->rank; d++) red[d] = na <= 0;
    for (int64_t k = 0; k < na; k++) {
        uint32_t a;
        if (!axis_in(axes[k], x->rank, &a)) return false;
        red[a] = true;
    }
    uint32_t first = x->rank, end = 0;
    for (uint32_t d = 0; d < x->rank; d++)
        if (red[d]) {
            if (d < first) first = d;
            end = d + 1;
        }
    if (first == x->rank) first = end = 0;
    for (uint32_t d = first; d < end; d++)
        if (!red[d] && x->dims[d] != 1) return false;
    int64_t dims[IR_MAX_RANK];
    uint32_t rank = 0;
    for (uint32_t d = 0; d < x->rank; d++)
        if (!red[d]) dims[rank++] = x->dims[d];
        else if (keep) dims[rank++] = 1;
    SpgRowPush push;
    if (!rows_over(x, first, end, &push)) return false;
    if (!c->record) return set_shape(out[0], F32, rank, dims);
    push.x = in_at(c, 0);
    push.y = out_at(c, 0);
    const Range none = {0, 0};
    row_dispatch(c, &push, code, range_of(c, n->inputs[0]), range_of(c, n->outputs[0]), none, none, none);
    return true;
}

static bool op_arg(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    const IrTensor *x = in[0];
    if (!x || !out[0] || x->dtype != F32) return false;
    uint32_t axis;
    if (!axis_in(ir_attr_int(n, "axis", 0), x->rank, &axis) || !x->dims[axis]) return false;
    const bool keep = ir_attr_int(n, "keepdims", 1) != 0, last = ir_attr_int(n, "select_last_index", 0) != 0;
    int64_t dims[IR_MAX_RANK];
    uint32_t rank = 0;
    for (uint32_t d = 0; d < x->rank; d++)
        if (d != axis || keep) dims[rank++] = d == axis ? 1 : x->dims[d];
    SpgRowPush push;
    if (!rows_over(x, axis, axis + 1, &push) || x->dims[axis] >= (1 << 24)) return false;
    if (!c->record) return set_shape(out[0], SPINGALETT_DTYPE_INT64, rank, dims);
    push.x = in_at(c, 0);
    push.y = out_at(c, 0);
    push.flags = last ? SPG_ROW_LAST : 0u;
    const Range none = {0, 0};
    row_dispatch(c, &push, n->op == IR_ArgMax ? SPG_ROW_ARGMAX : SPG_ROW_ARGMIN, range_of(c, n->inputs[0]),
                 range_of(c, n->outputs[0]), none, none, none);
    return true;
}

/* Softmax, LogSoftmax and Hardmax over an axis (before opset 13: over the axes from it on). */
static bool op_softmax(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    const IrTensor *x = in[0];
    if (!x || !out[0] || x->dtype != F32) return false;
    const bool old = c->g->opset < 13;
    uint32_t axis;
    if (!axis_in(ir_attr_int(n, "axis", old ? 1 : -1), x->rank ? x->rank : 1u, &axis)) return false;
    SpgRowPush push;
    if (!rows_over(x, axis, old ? x->rank : axis + 1, &push)) return false;
    if (!c->record) return set_shape(out[0], F32, x->rank, x->dims);
    if (!x->count) return true;
    push.x = in_at(c, 0);
    push.y = out_at(c, 0);
    const Range none = {0, 0};
    row_dispatch(c, &push, n->op == IR_LogSoftmax ? SPG_ROW_LOGSOFTMAX : n->op == IR_Hardmax ? SPG_ROW_HARDMAX
                 : SPG_ROW_SOFTMAX, range_of(c, n->inputs[0]), range_of(c, n->outputs[0]), none, none, none);
    return true;
}

/* LayerNormalization and RMSNormalization over the axes from `axis` on, a scale (and bias) of each of those values
   or one for all; LayerNormalization's means and inverse deviations not asked for. */
static bool op_layernorm(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    const bool rms = n->op == IR_RMSNormalization;
    const IrTensor *x = in[0], *sc = n->input_count > 1 ? in[1] : NULL, *bias = !rms && n->input_count > 2 ? in[2] : NULL;
    if (!x || !sc || !out[0] || x->dtype != F32 || sc->dtype != F32 || (bias && bias->dtype != F32)) return false;
    for (uint32_t o = 1; o < n->output_count; o++)
        if (n->outputs[o] != NONE && (c->g->values[n->outputs[o]].last >= 0)) return false;
    uint32_t axis;
    if (!axis_in(ir_attr_int(n, "axis", -1), x->rank, &axis)) return false;
    SpgRowPush push;
    if (!rows_over(x, axis, x->rank, &push)) return false;
    if ((sc->count != push.len && sc->count != 1) || (bias && bias->count != push.len && bias->count != 1)) return false;
    if (!c->record) {
        if (spg_gpu_using() == SPG_BACKEND_CUDA) c->half_out |= 1u;
        for (uint32_t o = 1; o < n->output_count; o++)
            if (out[o]) {
                int64_t md[IR_MAX_RANK];
                for (uint32_t d = 0; d < x->rank; d++) md[d] = d < axis ? x->dims[d] : 1;
                set_shape(out[o], F32, x->rank, md);
            }
        return set_shape(out[0], F32, x->rank, x->dims);
    }
    if (!x->count) return true;
    push.x = in_at(c, 0);
    push.y = out_at(c, 0);
    push.a = in_at(c, 1);
    push.b = bias ? in_at(c, 2) : 0;
    push.eps = ir_attr_float(n, "epsilon", 1e-5f);
    push.flags = (sc->count == 1 ? SPG_ROW_A_ONE : 0u) | (bias ? SPG_ROW_B : 0u) |
                 (bias && bias->count == 1 ? SPG_ROW_B_ONE : 0u) | (half_output(c, 0) ? SPG_ROW_HALF_Y : 0u);
    const Range none = {0, 0};
    row_dispatch(c, &push, rms ? SPG_ROW_RMSNORM : SPG_ROW_LAYERNORM, range_of(c, n->inputs[0]),
                 range_of(c, n->outputs[0]), range_of(c, n->inputs[1]), bias ? range_of(c, n->inputs[2]) : none, none);
    return true;
}

/* GroupNormalization (a scale and bias a channel, or a group as opset 18 has them) and InstanceNormalization (a group
   a channel). */
static bool op_groupnorm(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    const IrTensor *x = in[0], *sc = n->input_count > 1 ? in[1] : NULL, *bias = n->input_count > 2 ? in[2] : NULL;
    if (!x || !sc || !bias || !out[0] || x->dtype != F32 || sc->dtype != F32 || bias->dtype != F32 || x->rank < 2)
        return false;
    const bool cl = ir_attr_int(n, "channels_last", 0) != 0;
    const int64_t N = x->dims[0], C = x->dims[cl ? x->rank - 1 : 1];
    const int64_t G = n->op == IR_InstanceNormalization ? C : ir_attr_int(n, "num_groups", 0);
    const int64_t act = ir_attr_int(n, "activation", SPINGALETT_ACT_NONE);
    if (G < 1 || C % G || (sc->count != (uint64_t)C && sc->count != (uint64_t)G) || bias->count != sc->count ||
        !epilogue_act(act))
        return false;
    if (x->count >= UINT32_MAX || !N || !C) return false;
    const uint64_t pix = x->count / (uint64_t)(N * C);
    if (!c->record) {
        if (spg_gpu_using() == SPG_BACKEND_CUDA) c->half_out |= 1u;
        return set_shape(out[0], F32, x->rank, x->dims);
    }
    SpgRowPush push;
    memset(&push, 0, sizeof push);
    push.outer = (uint32_t)(N * G);
    push.len = (uint32_t)((uint64_t)(C / G) * pix);
    push.inner = 1;
    push.groups = (uint32_t)G;
    push.cpg = (uint32_t)(C / G);
    push.pix = (uint32_t)(pix ? pix : 1u);
    push.x = in_at(c, 0);
    push.y = out_at(c, 0);
    push.a = in_at(c, 1);
    push.b = in_at(c, 2);
    push.eps = ir_attr_float(n, "epsilon", 1e-5f);
    push.flags = (sc->count == (uint64_t)G && G != C ? SPG_ROW_BY_GROUP : 0u) | (half_output(c, 0) ? SPG_ROW_HALF_Y : 0u);
    push.act = (uint32_t)act;
    const Range none = {0, 0};
    row_dispatch(c, &push, cl ? SPG_ROW_GROUPNORM_LAST : SPG_ROW_GROUPNORM, range_of(c, n->inputs[0]), range_of(c, n->outputs[0]),
                 range_of(c, n->inputs[1]), range_of(c, n->inputs[2]), none);
    return true;
}

/* ------------------------------------------------------------------------- convolutions and pooling */

/* The windows of the node over x by the CPU's rules (ir_geometry()). */
static bool windows_of(GCtx *c, const IrTensor *x, const int64_t *kernel, IrGeometry *g, bool ceil_mode) {
    IrRun r;
    memset(&r, 0, sizeof r);
    r.g = c->g;
    r.node = c->n;
    r.index = c->k;
    return ir_geometry(&r, x, kernel, g, ceil_mode);
}

/* OIHW filters as OHWI: [out][kh][kw][in a group] (prepared()), of floats or (half) bfloat16. */
typedef struct { uint64_t outs, ins, taps; bool half; } FilterShape;
static void filters_ohwi(void *dst, const IrTensor *src, const void *arg) {
    const FilterShape *f = (const FilterShape *)arg;
    const float *w = (const float *)src->data;
    for (uint64_t o = 0; o < f->outs; o++)
        for (uint64_t i = 0; i < f->ins; i++)
            for (uint64_t k = 0; k < f->taps; k++) {
                const uint64_t at = (o * f->taps + k) * f->ins + i;
                const float v = w[(o * f->ins + i) * f->taps + k];
                if (f->half) ((uint16_t *)dst)[at] = spingalett_float_to_bf16(v);
                else ((float *)dst)[at] = v;
            }
}

/* Conv of one or two spatial axes (dilations 1) on the library's product over channels-last maps, its windows read
   through a geometry (pointwise ones as rows): the input transposed into scratch memory channels last, the product
   (its bias and the fused activation in its epilogue) into scratch, transposed back. */
static bool op_conv(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    const IrTensor *X = in[0], *W = in[1], *Bias = n->input_count > 2 ? in[2] : NULL;
    if (!X || !W || !out[0] || X->dtype != F32 || W->dtype != F32 || W->rank != X->rank || X->rank < 3 || X->rank > 4 ||
        c->g->values[n->inputs[1]].kind != VALUE_CONST || (Bias && (Bias->dtype != F32 || Bias->count != (uint64_t)W->dims[0])))
        return false;
    /* (channels last, the layout pass's: [N, spatial..., C] in and out, seen channels first for the windows) */
    const bool cl = ir_attr_int(n, "channels_last", 0) != 0;
    IrTensor xc = *X;
    if (cl) {
        xc.dims[1] = X->dims[X->rank - 1];
        for (uint32_t d = 2; d < X->rank; d++) xc.dims[d] = X->dims[d - 1];
    }
    const int64_t G = ir_attr_int(n, "group", 1), N = X->dims[0], C = xc.dims[1], OC = W->dims[0];
    const int64_t act = ir_attr_int(n, "activation", SPINGALETT_ACT_NONE);
    if (G < 1 || C % G || OC % G || W->dims[1] != C / G || !epilogue_act(act)) return false;
    IrGeometry geo;
    if (!windows_of(c, &xc, W->dims + 2, &geo, false)) {
        spingalett_clear_error();
        return false;
    }
    if (geo.in[0] != 1 || geo.k[0] != 1 || geo.dil[1] != 1 || geo.dil[2] != 1) return false;
    const int64_t h = geo.in[1], w = geo.in[2], oh = geo.out[1], ow = geo.out[2], kh = geo.k[1], kw = geo.k[2];
    const int64_t CG = C / G, OG = OC / G, taps = kh * kw, K = taps * CG;
    const uint64_t pixels = (uint64_t)N * (uint64_t)(oh * ow), xin = (uint64_t)N * (uint64_t)(h * w * C);
    if (pixels * (uint64_t)OC >= ((uint64_t)1 << 31) || xin >= ((uint64_t)1 << 31)) return false;
    const bool pointwise = taps == 1 && geo.s[1] == 1 && geo.s[2] == 1 && !geo.pb[1] && !geo.pb[2] && G == 1 &&
                           oh == h && ow == w;
    const size_t geo_size = pointwise ? 0 : spg_conv_forward(NULL, (uint32_t)h, (uint32_t)w, (uint32_t)C, (uint32_t)oh,
                                                             (uint32_t)ow, (uint32_t)G, (uint32_t)kh, (uint32_t)kw,
                                                             (uint32_t)geo.s[1], (uint32_t)geo.s[2], (uint32_t)geo.pb[1],
                                                             (uint32_t)geo.pb[2]);
    if (!pointwise && !geo_size) return false;
    /* (in bfloat16: the filters' copy, the input rounded as it is transposed, or kept so) */
    const bool half = (cl && half_in(c, 0)) || product_bf16((uint32_t)pixels, (uint32_t)OG, (uint32_t)K);
    const uint32_t width = half ? 2u : 4u;
    const FilterShape fs = {(uint64_t)OC, (uint64_t)CG, (uint64_t)taps, half};
    const uint64_t wt = prepared_form(c, 1, half ? 1u : 0u, (uint64_t)OC * (uint64_t)K * width, filters_ohwi, &fs);
    if (!wt) return false;
    int64_t dims[4] = {N, OC, oh, ow};
    if (X->rank == 3) dims[2] = ow;
    if (cl) {
        dims[1] = X->rank == 3 ? ow : oh;
        dims[2] = X->rank == 3 ? OC : ow;
        dims[3] = OC;
    }
    /* scratch: the input channels last (rounded to bfloat16 or not), the output before its transpose; channels last
       in bfloat16, the input's copy so rounded */
    const uint64_t in_bytes = cl ? 0 : align_up(xin * width), out_bytes = cl ? 0 : align_up(pixels * (uint64_t)OC * 4u);
    const uint64_t x16 = cl && half ? (half_in(c, 0) ? in_at(c, 0) : half_input(c, 0, xin)) : 0;
    uint32_t *table = NULL;
    if (!pointwise) {
        table = (uint32_t *)calloc(geo_size, sizeof(uint32_t));
        if (!table) return false;
        spg_conv_forward(table, (uint32_t)h, (uint32_t)w, (uint32_t)C, (uint32_t)oh, (uint32_t)ow, (uint32_t)G,
                         (uint32_t)kh, (uint32_t)kw, (uint32_t)geo.s[1], (uint32_t)geo.s[2], (uint32_t)geo.pb[1],
                         (uint32_t)geo.pb[2]);
    }
    const uint64_t geo_at = pointwise ? 0 : param_put(c, table, (uint32_t)(geo_size * sizeof(uint32_t)));
    /* (now, not with the other parameters when the piece is recorded: the product's tiles are timed as it is recorded, on
       this geometry; before it was there they read wherever garbage sent them, which faults on lavapipe, and at times on CUDA) */
    const bool geo_ready = !c->record || pointwise || param_upload(c, geo_at, table, (uint32_t)(geo_size * sizeof(uint32_t)));
    free(table);
    if (!geo_ready) return false;
    if (!c->record) {
        c->scratch = in_bytes + out_bytes;
        if (cl && half) c->half_out |= 1u, c->half_ins = 1u;
        return set_shape(out[0], F32, X->rank, dims);
    }
    const uint64_t xs = cl ? (half ? x16 : in_at(c, 0)) : c->scratch, ys = cl ? out_at(c, 0) : c->scratch + in_bytes;
    const Range rxs = cl && !half ? range_of(c, n->inputs[0]) : (Range){xs, xs + xin * width};
    const Range rys = cl ? range_of(c, n->outputs[0]) : (Range){ys, ys + pixels * (uint64_t)OC * 4u};
    /* channels last: [N][pixels][C] from [N][C][pixels] (or as they are, rounded to bfloat16) */
    const int64_t ip = h * w, op = oh * ow;
    if (!cl) {
        const int64_t d3[3] = {N, ip, C}, sx[3] = {C * ip, 1, ip}, sy[3] = {ip * C, C, 1};
        if (half) convert_strided(c, xs, in_at(c, 0), 3, d3, sx, sy, rxs, range_of(c, n->inputs[0]));
        else copy_strided(c, xs, in_at(c, 0), 3, d3, sx, sy, 4u, rxs, range_of(c, n->inputs[0]));
    }
    const Range none = {0, 0}, rw = {wt, wt + (uint64_t)OC * (uint64_t)K * width};
    SpgGemmPush push = {
        .a = xs, .b = wt, .c = ys, .e0 = Bias ? in_at(c, 2) : 0, .geo = geo_at, .M = (uint32_t)pixels,
        .N = (uint32_t)(pointwise ? OC : OG), .K = (uint32_t)K, .lda = (uint32_t)C, .ldb = (uint32_t)K,
        .ldc = (uint32_t)OC, .a_group = pointwise ? 0u : (uint32_t)CG, .b_group = pointwise ? 0u : (uint32_t)(OG * K),
        .c_group = pointwise ? 0u : (uint32_t)OG, .alpha = 1.0f, .flags = Bias ? SPG_GEMM_BIAS : 0u,
    };
    Access a = {0};
    reads(&a, rxs);
    reads(&a, rw);
    if (Bias) reads(&a, range_of(c, n->inputs[2]));
    if (!pointwise) reads(&a, (Range){geo_at, geo_at + geo_size * 4u});
    writes(&a, rys);
    need(c->r, &a);
    SpgGemmMode m = {pointwise ? SPG_A_ROW : SPG_A_CONV, SPG_B_COL, SPG_EPI_BIAS_ACT, (uint32_t)act,
                     pointwise ? 1u : (uint32_t)G, false, pointwise || (CG % 4 == 0 && C % 4 == 0), true, 0,
                     pixels * (uint64_t)OC, half, half ? (half_output(c, 0) ? 7u : 3u) : 0u, false, false};
    m.wide_a = half && (pointwise || (CG % 8 == 0 && C % 8 == 0));
    m.wide_b = half;
    spg_gemm(c->r->c, &push, &m);
    (void)none;
    /* back to channels first */
    if (!cl) {
        const int64_t d3[3] = {N, OC, op}, sx[3] = {op * OC, 1, OC}, sy[3] = {OC * op, op, 1};
        copy_strided(c, out_at(c, 0), ys, 3, d3, sx, sy, 4u, range_of(c, n->outputs[0]), rys);
    }
    return true;
}

/* MaxPool (without indices), AveragePool and LpPool over channels-first maps. */
static bool op_pool(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    const IrTensor *X = in[0];
    if (!X || !out[0] || X->dtype != F32 || X->rank < 3 || X->rank > 5) return false;
    if (n->output_count > 1 && n->outputs[1] != NONE) return false;
    int64_t kernel[3];
    const IrAttr *ks = ir_attr(n, "kernel_shape");
    if (!ks || ks->type != ATTR_INTS || ks->count != X->rank - 2u) return false;
    for (uint32_t k = 0; k < ks->count; k++) kernel[k] = ks->ints[k];
    /* (channels last, the layout pass's: [N, H, W, C], seen channels first for the windows) */
    const bool cl = ir_attr_int(n, "channels_last", 0) != 0;
    if (cl && (X->rank != 4 || n->op == IR_LpPool)) return false;
    IrTensor xc = *X;
    if (cl) {
        xc.dims[1] = X->dims[3];
        xc.dims[2] = X->dims[1];
        xc.dims[3] = X->dims[2];
    }
    IrGeometry g;
    if (!windows_of(c, &xc, kernel, &g, ir_attr_int(n, "ceil_mode", 0) != 0)) {
        spingalett_clear_error();
        return false;
    }
    int64_t dims[IR_MAX_RANK] = {X->dims[0], X->dims[1]};
    for (uint32_t a = 0; a < g.spatial; a++) dims[2 + a] = g.out[3 - g.spatial + a];
    if (cl) {
        dims[1] = g.out[1];
        dims[2] = g.out[2];
        dims[3] = xc.dims[1];
    }
    uint64_t total = (uint64_t)X->dims[0] * (uint64_t)xc.dims[1];
    for (uint32_t a = 0; a < 3; a++) total *= (uint64_t)g.out[a];
    if (total >= UINT32_MAX || X->count >= UINT32_MAX) return false;
    if (!c->record) return set_shape(out[0], F32, X->rank, dims);
    SpgPoolGPush push;
    memset(&push, 0, sizeof push);
    push.x = in_at(c, 0);
    push.y = out_at(c, 0);
    push.rows = (uint32_t)(X->dims[0] * X->dims[1]);
    push.total = (uint32_t)total;
    for (uint32_t a = 0; a < 3; a++) {
        push.in[a] = (uint32_t)g.in[a];
        push.out[a] = (uint32_t)g.out[a];
        push.k[a] = (uint32_t)g.k[a];
        push.s[a] = (uint32_t)g.s[a];
        push.pb[a] = (uint32_t)g.pb[a];
        push.pe[a] = (uint32_t)g.pe[a];
        push.dil[a] = (uint32_t)g.dil[a];
    }
    push.flags = (n->op == IR_MaxPool ? SPG_GPOOL_MAX : 0u) | (n->op == IR_LpPool ? SPG_GPOOL_LP : 0u) |
                 (ir_attr_int(n, "count_include_pad", 0) ? SPG_GPOOL_INCLUDE : 0u) | (cl ? SPG_GPOOL_LAST : 0u);
    if (cl) push.rows = (uint32_t)xc.dims[1];       /* (channels last: the channels) */
    push.p = (float)ir_attr_int(n, "p", 2);
    Access a = {0};
    reads(&a, range_of(c, n->inputs[0]));
    writes(&a, range_of(c, n->outputs[0]));
    dispatch(c, &a, SPG_KERNEL_gpool, NULL, 0, &push, sizeof push, group_count(total, 256u), 1, 1);
    return true;
}

/* GlobalAveragePool and GlobalMaxPool: rows of each map's values. */
static bool op_global_pool(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    const IrTensor *X = in[0];
    if (!X || !out[0] || X->dtype != F32 || X->rank < 2) return false;
    int64_t dims[IR_MAX_RANK];
    for (uint32_t d = 0; d < X->rank; d++) dims[d] = d < 2 ? X->dims[d] : 1;
    SpgRowPush push;
    if (!rows_over(X, 2, X->rank, &push)) return false;
    if (!c->record) return set_shape(out[0], F32, X->rank, dims);
    push.x = in_at(c, 0);
    push.y = out_at(c, 0);
    const Range none = {0, 0};
    row_dispatch(c, &push, n->op == IR_GlobalMaxPool ? SPG_ROW_MAX : SPG_ROW_MEAN, range_of(c, n->inputs[0]),
                 range_of(c, n->outputs[0]), none, none, none);
    return true;
}

/* Resize and Upsample of up to four axes through the CPU's taps (ir_resize_taps()): its scales, sizes and roi read on
   the host, the taps a parameter block. */
static bool op_resize(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    const IrTensor *X = in[0];
    if (!X || !out[0] || X->dtype != F32 || !X->rank || X->rank > 4 || X->count >= UINT32_MAX) return false;
    for (uint32_t i = 1; i < n->input_count; i++)
        if (in[i] && !ask_param(c, in, i)) return false;
    IrRun r;
    memset(&r, 0, sizeof r);
    r.g = c->g;
    r.node = n;
    r.index = c->k;
    IrResizeTaps plan;
    if (!ir_resize_taps(&r, (IrTensor *const *)in, n->input_count, &plan)) {
        ir_resize_free(&plan);
        return false;
    }
    const uint32_t R = X->rank, lead = 4u - R;
    uint64_t total = 1, pairs = 0;
    for (uint32_t d = 0; d < R; d++) {
        total *= (uint64_t)plan.dims[d];
        pairs += (uint64_t)plan.dims[d] * (uint64_t)(plan.T[d] ? plan.T[d] : 1);
    }
    pairs += lead;
    if (total >= UINT32_MAX || pairs >= ((uint64_t)1 << 26)) {
        ir_resize_free(&plan);
        return false;
    }
    if (!c->record) {
        param_put(c, NULL, (uint32_t)(pairs * 8u));
        ir_resize_free(&plan);
        return set_shape(out[0], F32, R, plan.dims);
    }
    SpgResizePush push;
    memset(&push, 0, sizeof push);
    int32_t *table = (int32_t *)calloc(pairs ? pairs : 1u, 8u);
    if (!table) {
        ir_resize_free(&plan);
        return false;
    }
    int64_t xs[IR_MAX_RANK];
    strides_of(R, X->dims, xs);
    uint32_t at = 0;
    for (uint32_t d = 0; d < 4; d++) {
        push.at[d] = at;
        if (d < lead || !total) {
            push.dims[d] = d < lead ? 1u : (uint32_t)plan.dims[d - lead];
            push.T[d] = 1;
            table[2u * at] = 0;
            const float one = 1.0f;
            memcpy(&table[2u * at + 1u], &one, 4);
            at++;
            continue;
        }
        const uint32_t a = d - lead;
        push.dims[d] = (uint32_t)plan.dims[a];
        push.T[d] = (uint32_t)plan.T[a];
        push.xs[d] = (uint32_t)xs[a];
        for (uint64_t k = 0; k < (uint64_t)plan.dims[a] * (uint64_t)plan.T[a]; k++, at++) {
            table[2u * at] = (int32_t)plan.index[a][k];
            memcpy(&table[2u * at + 1u], &plan.weight[a][k], 4);
        }
    }
    push.x = in_at(c, 0);
    push.y = out_at(c, 0);
    push.taps = param_put(c, table, (uint32_t)(pairs * 8u));
    push.total = (uint32_t)total;
    push.extrapolation = plan.extrapolation;
    free(table);
    ir_resize_free(&plan);
    if (!total) return true;
    Access a = {0};
    reads(&a, range_of(c, n->inputs[0]));
    writes(&a, range_of(c, n->outputs[0]));
    dispatch(c, &a, SPG_KERNEL_gresize, NULL, 0, &push, sizeof push, group_count(total, 256u), 1, 1);
    return true;
}

/* ------------------------------------------------------------------------- attention */

static void gemm(GCtx *c, SpgGemmPush *push, uint32_t amode, uint32_t bmode, uint32_t epi, uint32_t act, uint32_t z,
                 Range ra, Range rb, Range rbias, Range rc);

/* FusedAttention as products: the scores (scale Q K^T) of each batch's heads into scratch memory, the mask added,
   a softmax over each row, then the probabilities times V into the output; st holds the operands' strides over
   (B, H) and rows (op_attention()). */
static bool attention_products(GCtx *c, IrTensor *const *in, IrTensor **out, int64_t B, int64_t H, int64_t S, int64_t T,
                               int64_t D, int64_t Dv, const uint32_t *st, uint32_t orank, const int64_t *odims) {
    const IrNode *n = c->n;
    const uint64_t scores = (uint64_t)B * H * S * T;
    if (!c->record) {
        c->scratch = scores * 4u;
        return set_shape(out[0], F32, orank, odims);
    }
    if (!out[0]->count) return true;
    const uint64_t sc = c->scratch;
    const Range rs = {sc, sc + scores * 4u}, none = {0, 0};
    const Range rq = range_of(c, n->inputs[0]), rk = range_of(c, n->inputs[1]), rv = range_of(c, n->inputs[2]);
    const Range ry = range_of(c, n->outputs[0]);
    /* (a key's values contiguous: K as N x K; else the keys contiguous: K^T as K x N) */
    const bool key_rows = st[20] == 1u;
    for (int64_t b = 0; b < B; b++) {
        SpgGemmPush push = {
            .a = in_at(c, 0) + 4u * (uint64_t)b * st[0], .b = in_at(c, 1) + 4u * (uint64_t)b * st[3],
            .c = sc + 4u * (uint64_t)b * (uint64_t)(H * S * T), .M = (uint32_t)S, .N = (uint32_t)T, .K = (uint32_t)D,
            .lda = st[2], .ldb = key_rows ? st[5] : st[20], .ldc = (uint32_t)T, .a_group = st[1], .b_group = st[4],
            .c_group = (uint32_t)(S * T), .alpha = ir_attr_float(n, "scale", 1.0f),
        };
        gemm(c, &push, SPG_A_ROW, key_rows ? SPG_B_COL : SPG_B_ROW, SPG_EPI_STORE, SPINGALETT_ACT_NONE, (uint32_t)H, rq, rk,
             none, rs);
    }
    const IrTensor *mask = n->input_count > 3 ? in[3] : NULL;
    if (mask) {
        IrTensor s4;
        const int64_t d4[4] = {B, H, S, T};
        set_shape(&s4, F32, 4, d4);
        Gew e;
        IrTensor *const pair[2] = {&s4, (IrTensor *)mask};
        gew_layout(&e, &s4, (const IrTensor *const *)pair, 2u);
        gew_dispatch(c, &e, SPG_GEW_ADD, type_code(F32) << 12, sc, sc, in_at(c, 3), 0, rs, rs, range_of(c, n->inputs[3]),
                     rs);
    }
    SpgRowPush rows;
    memset(&rows, 0, sizeof rows);
    rows.outer = (uint32_t)(B * H * S);
    rows.len = (uint32_t)T;
    rows.inner = 1;
    rows.x = sc;
    rows.y = sc;
    row_dispatch(c, &rows, SPG_ROW_SOFTMAX, rs, rs, none, none, none);
    for (int64_t b = 0; b < B; b++) {
        SpgGemmPush push = {
            .a = sc + 4u * (uint64_t)b * (uint64_t)(H * S * T), .b = in_at(c, 2) + 4u * (uint64_t)b * st[6],
            .c = out_at(c, 0) + 4u * (uint64_t)b * st[9], .M = (uint32_t)S, .N = (uint32_t)Dv, .K = (uint32_t)T,
            .lda = (uint32_t)T, .ldb = st[8], .ldc = st[11], .a_group = (uint32_t)(S * T), .b_group = st[7],
            .c_group = st[10], .alpha = 1.0f,
        };
        gemm(c, &push, SPG_A_ROW, SPG_B_ROW, SPG_EPI_STORE, SPINGALETT_ACT_NONE, (uint32_t)H, rs, rv, none, ry);
    }
    return true;
}

/* FusedAttention (fusions make it) through gattn, heads of up to 128 values (the values' as long as the queries'):
   layout 0, Q (..., S, D) over up to two axes of batches broadcast between Q, K (..., D, T as K^T: the keys' rows its
   columns), V (..., T, Dv) and the output (..., S, Dv); layout 1, Q (B, S, H, D), K and V (B, T, H or 1, D) and the
   output (B, S, H, Dv); layout 2, Q, K and V where the attribute "views" puts them in the inputs (offsets and strides
   over B, H and rows) and the output (B, S, H, Dv) with "heads", else (B, H, S, Dv). The mask and keep broadcast to
   the scores. */
static bool op_attention(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    const IrTensor *Q = in[0], *K = in[1], *V = in[2];
    const IrTensor *mask = n->input_count > 3 ? in[3] : NULL, *keep = n->input_count > 4 ? in[4] : NULL;
    if (!Q || !K || !V || !out[0] || Q->dtype != F32 || K->dtype != F32 || V->dtype != F32 ||
        (mask && mask->dtype != F32 && mask->dtype != BOOL) || (keep && keep->dtype != BOOL))
        return false;
    const int64_t layout = ir_attr_int(n, "layout", 0);
    const bool layout1 = layout == 1;
    int64_t S, D, T, Dv, B = 1, H = 1;
    uint32_t st[21] = {0};
    int64_t odims[IR_MAX_RANK];
    uint32_t orank;
    int64_t bdims[2] = {1, 1};          /* the batches as (B, H) */
    int64_t at[3] = {0, 0, 0};          /* (layout 2: the views' offsets in their inputs) */
    if (layout == 2) {
        const IrAttr *va = ir_attr(n, "views"), *sa = ir_attr(n, "shape");
        if (!va || va->type != ATTR_INTS || va->count != 18 || !sa || sa->type != ATTR_INTS || sa->count > IR_MAX_RANK)
            return false;
        const int64_t *w = va->ints;
        B = w[0]; H = w[1]; S = w[2]; T = w[3]; D = w[4]; Dv = w[5];
        const IrTensor *x3[3] = {Q, K, V};
        const int64_t rows[3] = {S, T, T}, len[3] = {D, D, Dv};
        for (uint32_t k = 0; k < 3; k++) {
            const int64_t *v = w + 6 + 4 * k;
            if (B < 1 || H < 1 || v[0] < 0 || v[1] < 0 || v[2] < 0 || v[3] < 0 || v[1] >= UINT32_MAX ||
                v[2] >= UINT32_MAX || v[3] >= UINT32_MAX ||
                (uint64_t)(v[0] + (B - 1) * v[1] + (H - 1) * v[2] + (rows[k] - 1) * v[3] + len[k] - 1) >= x3[k]->count)
                return false;
            at[k] = v[0];
        }
        const bool heads = ir_attr_int(n, "heads", 0) != 0;
        const int64_t s[12] = {w[7], w[8], w[9], w[11], w[12], w[13], w[15], w[16], w[17],
                               heads ? S * H * Dv : H * S * Dv, heads ? Dv : S * Dv, heads ? H * Dv : Dv};
        for (uint32_t k = 0; k < 12; k++) st[k] = (uint32_t)s[k];
        st[20] = 1;
        orank = sa->count;
        int64_t count = 1;
        for (uint32_t d = 0; d < orank; d++) count *= (odims[d] = sa->ints[d]);
        if (count != B * H * S * Dv || Dv != D || D > 128) return false;
        bdims[0] = B;
        bdims[1] = H;
    } else if (layout1) {
        if (Q->rank != 4 || K->rank != 4 || V->rank != 4) return false;
        B = Q->dims[0];
        H = Q->dims[2];
        S = Q->dims[1];
        D = Q->dims[3];
        T = K->dims[1];
        Dv = V->dims[3];
        const int64_t Hk = K->dims[2];
        if (K->dims[0] != B || V->dims[0] != B || K->dims[3] != D || V->dims[1] != T || V->dims[2] != Hk ||
            (Hk != H && Hk != 1))
            return false;
        const int64_t s[12] = {S * H * D, D, H * D, T * Hk * D, Hk == H ? D : 0, Hk * D,
                               T * Hk * Dv, Hk == H ? Dv : 0, Hk * Dv, S * H * Dv, Dv, H * Dv};
        for (uint32_t k = 0; k < 12; k++) st[k] = (uint32_t)s[k];
        st[20] = 1;
        odims[0] = B; odims[1] = S; odims[2] = H; odims[3] = Dv;
        orank = 4;
        bdims[0] = B;
        bdims[1] = H;
    } else {
        if (Q->rank < 2 || K->rank < 2 || V->rank < 2) return false;
        S = Q->dims[Q->rank - 2];
        D = Q->dims[Q->rank - 1];
        T = K->dims[K->rank - 1];
        Dv = V->dims[V->rank - 1];
        if (K->dims[K->rank - 2] != D || V->dims[V->rank - 2] != T) return false;
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
        uint32_t R;
        int64_t bd[IR_MAX_RANK];
        if (!broadcast(three, 3, &R, bd) || R > 2) return false;
        for (uint32_t d = 0; d < R; d++) bdims[2 - R + d] = bd[d];
        B = bdims[0];
        H = bdims[1];
        /* the operands' strides over (B, H), the keys' as rows: K^T's columns, T apart... the keys' values D apart */
        int64_t sq[2] = {0, 0}, sk[2] = {0, 0}, sv[2] = {0, 0}, tmp[IR_MAX_RANK];
        broadcast_strides(&bq, R, bd, tmp);
        for (uint32_t d = 0; d < R; d++) sq[2 - R + d] = tmp[d] * S * D;
        broadcast_strides(&bk, R, bd, tmp);
        for (uint32_t d = 0; d < R; d++) sk[2 - R + d] = tmp[d] * D * T;
        broadcast_strides(&bv, R, bd, tmp);
        for (uint32_t d = 0; d < R; d++) sv[2 - R + d] = tmp[d] * T * Dv;
        for (uint32_t d = 0; d < R; d++) odims[d] = bd[d];
        odims[R] = S;
        odims[R + 1] = Dv;
        orank = R + 2;
        /* (K is K^T: a key's values T apart, the keys 1) */
        const int64_t s[12] = {sq[0], sq[1], D, sk[0], sk[1], 1, sv[0], sv[1], Dv, H * S * Dv, S * Dv, Dv};
        for (uint32_t k = 0; k < 12; k++) st[k] = (uint32_t)s[k];
        st[20] = (uint32_t)T;
    }
    if (D < 1 || Dv < 1 || B * H > 65535 || S >= (1 << 30) || T >= (1 << 30)) return false;
    /* heads past gattn's: through products and a softmax over the scores in scratch memory (a mask of floats added;
       without keep and safe) */
    if (Dv != D || D > 128) {
        if (keep || (mask && mask->dtype != F32) || ir_attr_int(n, "safe", 0) || (uint64_t)B * H * S * T >= ((uint64_t)1 << 31))
            return false;
        return attention_products(c, in, out, B, H, S, T, D, Dv, st, orank, odims);
    }
    /* the masks' strides over (B, H, S, T) */
    const int64_t full[4] = {B, H, S, T};
    for (uint32_t k = 0; k < 2; k++) {
        const IrTensor *m = k ? keep : mask;
        if (!m) continue;
        if (m->rank > 4) return false;
        for (uint32_t d = 0; d < m->rank; d++) {
            const int64_t want = full[4 - m->rank + d];
            if (m->dims[d] != want && m->dims[d] != 1) return false;
        }
        int64_t ms[4];
        broadcast_strides(m, 4, full, ms);
        for (uint32_t d = 0; d < 4; d++) st[12 + 4 * k + d] = (uint32_t)ms[d];
    }
    const bool flash = spingalett_get_gpu_precision() == PRECISION_BFLOAT16 && spg_gpu_mma_bf16() &&
                       spg_gpu_kernel_built(SPG_KERNEL_gflash);
    if (!c->record) {
        /* (inputs kept as bfloat16 only for gflash, which can write its output so) */
        if ((half_in(c, 0) || half_in(c, 1) || half_in(c, 2)) && !flash) return false;
        if (flash) c->half_out |= 1u, c->half_ins = 7u;
        param_put(c, st, sizeof st);
        return set_shape(out[0], F32, orank, odims);
    }
    if (!out[0]->count) return true;
    const uint32_t unit = D <= 32 ? 32u : D <= 64 ? 64u : 128u;
    SpgGattnPush push = {in_at(c, 0) + (uint64_t)at[0] * in_size(c, 0), in_at(c, 1) + (uint64_t)at[1] * in_size(c, 1),
                         in_at(c, 2) + (uint64_t)at[2] * in_size(c, 2), out_at(c, 0), mask ? in_at(c, 3) : in_at(c, 0),
                         keep ? in_at(c, 4) : in_at(c, 0), param_put(c, st, sizeof st), (uint32_t)S, (uint32_t)T,
                         (uint32_t)D, (uint32_t)H, ir_attr_float(n, "scale", 1.0f), ir_attr_float(n, "fill", -INFINITY), 0, 0};
    push.flags = (mask ? (mask->dtype == BOOL ? SPG_GATTN_BOOL : SPG_GATTN_MASK) : 0u) | (keep ? SPG_GATTN_KEEP : 0u) |
                 (ir_attr_int(n, "safe", 0) ? SPG_GATTN_SAFE : 0u);
    bool vec = (uint32_t)D == unit && push.q % 16u == 0 && push.k % 16u == 0 && push.v % 16u == 0 && push.y % 16u == 0;
    for (uint32_t k = 0; k < 12 && vec; k++) vec = st[k] % 8u == 0;
    vec = vec && st[20] == 1u;
    if (vec) push.flags |= SPG_GATTN_VEC;
    push.flags |= (half_in(c, 0) ? SPG_GATTN_HALF_Q : 0u) | (half_in(c, 1) ? SPG_GATTN_HALF_K : 0u) |
                  (half_in(c, 2) ? SPG_GATTN_HALF_V : 0u) | (half_output(c, 0) ? SPG_GATTN_HALF_Y : 0u);
    Access a = {0};
    for (uint32_t i = 0; i < n->input_count && i < 5; i++)
        if (n->inputs[i] != NONE) reads(&a, range_of(c, n->inputs[i]));
    writes(&a, range_of(c, n->outputs[0]));
    const uint32_t spec[1] = {unit};
    /* in bfloat16 on the matrix units (gflash: blocks of 128 queries, of 64 for heads of 128), else in single
       precision (gattn) */
    if (flash) {
        /* (blocks of 64 queries where those of 128 would leave the device short of blocks) */
        const bool half_rows = unit <= 64u && (uint64_t)((S + 127) / 128) * (uint64_t)(B * H) < 64u;
        const uint32_t rows = unit > 64u || half_rows ? 64u : 128u, spec2[2] = {unit, half_rows ? 1u : 2u};
        dispatch(c, &a, SPG_KERNEL_gflash, spec2, 2, &push, sizeof push, (uint32_t)((S + rows - 1) / rows),
                 (uint32_t)(B * H), 1);
        return true;
    }
    dispatch(c, &a, SPG_KERNEL_gattn, spec, 1, &push, sizeof push, (uint32_t)((S + SPG_GATTN_ROWS - 1) / SPG_GATTN_ROWS),
             (uint32_t)(B * H), 1);
    return true;
}

/* ------------------------------------------------------------------------- products */

/* Whether products multiply in bfloat16 on the matrix units (spingalett_set_gpu_precision()), for a product of this
   size: those smaller than the units' blocks stay in single precision (as the networks' do). */
static bool product_bf16(uint32_t M, uint32_t N, uint32_t K) {
    return spingalett_get_gpu_precision() == PRECISION_BFLOAT16 && K >= 32u && M >= 16u && N >= 16u &&
           spg_gpu_mma_bf16() && spg_gpu_bf16_storage() && spg_gpu_kernel_built(SPG_KERNEL_gcvt);
}

/* Records C = A B through spg_gemm(): z products (A, B and C each z_a, z_b, z_c elements apart: 0 for one shared). */
static void gemm_half(GCtx *c, SpgGemmPush *push, uint32_t amode, uint32_t bmode, uint32_t epi, uint32_t act, uint32_t z,
                      Range ra, Range rb, Range rbias, Range rc, uint32_t half);

static void gemm(GCtx *c, SpgGemmPush *push, uint32_t amode, uint32_t bmode, uint32_t epi, uint32_t act, uint32_t z,
                 Range ra, Range rb, Range rbias, Range rc) {
    gemm_half(c, push, amode, bmode, epi, act, z, ra, rb, rbias, rc, 0);
}

/* The same with operands kept as bfloat16 (half: A 1, B 2), multiplied on the matrix units. */
static void gemm_half(GCtx *c, SpgGemmPush *push, uint32_t amode, uint32_t bmode, uint32_t epi, uint32_t act, uint32_t z,
                      Range ra, Range rb, Range rbias, Range rc, uint32_t half) {
    Access a = {0};
    reads(&a, ra);
    reads(&a, rb);
    reads(&a, rbias);
    writes(&a, rc);
    if (epi == SPG_EPI_STORE && push->beta != 0.0f) reads(&a, rc);
    need(c->r, &a);
    /* (operands of floats stay in single precision: the matrix units' kernels that round floats as they read them read
       a value at a time, slower than single precision) */
    SpgGemmMode m = {amode, bmode, epi, act, z, false, true, true, 0, (uint64_t)z * push->M * push->N, half != 0, half,
                     half != 0, half != 0};
    if (push->alpha == 0.0f && epi != SPG_EPI_STORE) push->alpha = 1.0f;
    spg_gemm(c->r->c, push, &m);
}

/* A K x N matrix of floats as N x K (prepared()). */
static void transposed(void *dst, const IrTensor *src, const void *arg) {
    (void)arg;
    const uint64_t K = (uint64_t)src->dims[0], N = (uint64_t)src->dims[1];
    const float *w = (const float *)src->data;
    float *t = (float *)dst;
    for (uint64_t k = 0; k < K; k++)
        for (uint64_t j = 0; j < N; j++) t[j * K + k] = w[k * N + j];
}

/* A K x N matrix of floats as N x K of bfloat16 (prepared()); an N x K one (arg not NULL) as it is, rounded. */
static void transposed_bf16(void *dst, const IrTensor *src, const void *arg) {
    const uint64_t R = (uint64_t)src->dims[0], C = (uint64_t)src->dims[1];
    const float *w = (const float *)src->data;
    uint16_t *t = (uint16_t *)dst;
    if (arg) {
        for (uint64_t i = 0; i < R * C; i++) t[i] = spingalett_float_to_bf16(w[i]);
        return;
    }
    for (uint64_t k = 0; k < R; k++)
        for (uint64_t j = 0; j < C; j++) t[j * R + k] = spingalett_float_to_bf16(w[k * C + j]);
}

/* The activations of the products' epilogues (ActivationFunction): those the kernels hold. */
static bool epilogue_act(int64_t act) {
    return act == SPINGALETT_ACT_NONE || act == SPINGALETT_ACT_RELU || act == SPINGALETT_ACT_LEAKY_RELU ||
           act == SPINGALETT_ACT_SIGMOID || act == SPINGALETT_ACT_TANH || act == SPINGALETT_ACT_GELU ||
           act == SPINGALETT_ACT_GELU_TANH || act == SPINGALETT_ACT_SILU;
}

/* X (..., K) times W (K x N, or N x K with trans), plus a bias (of N values, or a row of them), through an activation:
   Linear (fusions make it), and Gemm by a constant matrix. */
static bool linear_product(GCtx *c, IrTensor *const *in, IrTensor **out, bool trans, int64_t act) {
    const IrNode *n = c->n;
    const IrTensor *x = in[0], *w = in[1], *bias = n->input_count > 2 ? in[2] : NULL;
    if (!x || !w || !out[0] || x->dtype != F32 || w->dtype != F32 || w->rank != 2 || !x->rank) return false;
    const int64_t K = trans ? w->dims[1] : w->dims[0], N = trans ? w->dims[0] : w->dims[1];
    if (x->dims[x->rank - 1] != K || !epilogue_act(act) || (bias && (bias->dtype != F32 || bias->count != (uint64_t)N)))
        return false;
    const uint64_t M = K ? x->count / (uint64_t)K : 0;
    if (M > UINT32_MAX || N > UINT32_MAX || K > UINT32_MAX || !K) return false;
    int64_t dims[IR_MAX_RANK];
    for (uint32_t d = 0; d < x->rank; d++) dims[d] = d + 1 < x->rank ? x->dims[d] : N;
    const bool constant = c->g->values[n->inputs[1]].kind == VALUE_CONST;
    /* in bfloat16: the weights' copy as N x K, the input as it is kept so, or its copy rounded; the output kept so
       where its readers want it */
    if (constant && (half_in(c, 0) || product_bf16((uint32_t)M, (uint32_t)N, (uint32_t)K))) {
        static const int as_is = 1;
        const uint64_t w16 = prepared_form(c, 1, 1, (uint64_t)K * (uint64_t)N * 2u, transposed_bf16,
                                           trans ? &as_is : NULL);
        if (!w16) return false;
        const uint64_t a16 = half_in(c, 0) ? in_at(c, 0) : half_input(c, 0, M * (uint64_t)K);
        if (!c->record) {
            c->half_out |= 1u;
            c->half_ins = 1u;
            return set_shape(out[0], F32, x->rank, dims);
        }
        const Range r16 = {a16, a16 + M * (uint64_t)K * 2u};
        SpgGemmPush push = {
            .a = a16, .b = w16, .c = out_at(c, 0), .e0 = bias ? in_at(c, 2) : 0, .M = (uint32_t)M, .N = (uint32_t)N,
            .K = (uint32_t)K, .lda = (uint32_t)K, .ldb = (uint32_t)K, .ldc = (uint32_t)N, .alpha = 1.0f,
            .flags = bias ? SPG_GEMM_BIAS : 0u,
        };
        const Range none = {0, 0}, rw = {w16, w16 + (uint64_t)K * (uint64_t)N * 2u};
        gemm_half(c, &push, SPG_A_ROW, SPG_B_COL, SPG_EPI_BIAS_ACT, (uint32_t)act, 1, r16, rw,
                  bias ? range_of(c, n->inputs[2]) : none, range_of(c, n->outputs[0]), half_output(c, 0) ? 7u : 3u);
        return true;
    }
    /* constant weights as N x K, the layout of the products' forward kernels (made once) */
    uint64_t wt = 0;
    if (!trans && constant) {
        wt = prepared(c, 1, (uint64_t)K * (uint64_t)N * 4u, transposed, NULL);
        if (!wt) return false;
    }
    if (!c->record) return set_shape(out[0], F32, x->rank, dims);
    const bool col = trans || wt;
    SpgGemmPush push = {
        .a = in_at(c, 0), .b = wt ? wt : in_at(c, 1), .c = out_at(c, 0), .e0 = bias ? in_at(c, 2) : 0,
        .M = (uint32_t)M, .N = (uint32_t)N, .K = (uint32_t)K, .lda = (uint32_t)K, .ldb = col ? (uint32_t)K : (uint32_t)N,
        .ldc = (uint32_t)N, .alpha = 1.0f, .flags = bias ? SPG_GEMM_BIAS : 0u,
    };
    const Range none = {0, 0}, rw = wt ? (Range){wt, wt + (uint64_t)K * (uint64_t)N * 4u} : range_of(c, n->inputs[1]);
    gemm(c, &push, SPG_A_ROW, col ? SPG_B_COL : SPG_B_ROW, SPG_EPI_BIAS_ACT, (uint32_t)act, 1, range_of(c, n->inputs[0]),
         rw, bias ? range_of(c, n->inputs[2]) : none, range_of(c, n->outputs[0]));
    return true;
}

/* Linear (fusions make it): X (..., K) times W (K x N, or N x K with trans_b), plus a bias, through an activation. */
static bool op_linear(GCtx *c, IrTensor *const *in, IrTensor **out) {
    return linear_product(c, in, out, ir_attr_int(c->n, "trans_b", 0) != 0,
                          ir_attr_int(c->n, "activation", SPINGALETT_ACT_NONE));
}

/* Gemm: alpha A' B' + beta C, A' and B' A and B or their transposes, C broadcast to the output (a bias of the columns
   in the product's epilogue; else copied into the output, which the product adds to). */
static bool op_gemm(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    const IrTensor *A = in[0], *B = in[1], *C = n->input_count > 2 ? in[2] : NULL;
    if (!A || !B || !out[0] || A->dtype != F32 || B->dtype != F32 || A->rank != 2 || B->rank != 2 ||
        (C && (C->dtype != F32 || C->rank > 2)))
        return false;
    const bool ta = ir_attr_int(n, "transA", 0) != 0, tb = ir_attr_int(n, "transB", 0) != 0;
    const float alpha = ir_attr_float(n, "alpha", 1.0f), beta = ir_attr_float(n, "beta", 1.0f);
    /* by a constant matrix, alpha 1, C none or a bias of the columns with beta 1: Linear's product */
    if (!ta && alpha == 1.0f && c->g->values[n->inputs[1]].kind == VALUE_CONST &&
        (!C || (beta == 1.0f && C->count == (uint64_t)(tb ? B->dims[0] : B->dims[1]) && (C->rank < 2 || C->dims[0] == 1))))
        return linear_product(c, in, out, tb, SPINGALETT_ACT_NONE);
    const int64_t M = ta ? A->dims[1] : A->dims[0], K = ta ? A->dims[0] : A->dims[1];
    const int64_t N = tb ? B->dims[0] : B->dims[1];
    if ((tb ? B->dims[1] : B->dims[0]) != K || M > UINT32_MAX || N > UINT32_MAX || K > UINT32_MAX) return false;
    const int64_t dims[2] = {M, N};
    IrTensor y;
    set_shape(&y, F32, 2, dims);
    if (C) {
        IrTensor *pair[2] = {&y, (IrTensor *)C};
        uint32_t R;
        int64_t bd[IR_MAX_RANK];
        if (!broadcast(pair, 2, &R, bd) || bd[0] != M || bd[1] != N) return false;
    }
    /* a bias of the columns (C of N values, or of one) with beta 1 and alpha 1: the epilogue's */
    const bool bias = C && beta == 1.0f && alpha == 1.0f && C->count == (uint64_t)N && (C->rank < 2 || C->dims[0] == 1);
    /* (both transposed: the products have no such mode, A is transposed into scratch memory first) */
    const bool both = ta && tb;
    if (!c->record) {
        if (both) c->scratch = (uint64_t)M * (uint64_t)K * 4u;
        return set_shape(out[0], F32, 2, dims);
    }
    const Range none = {0, 0}, ry = range_of(c, n->outputs[0]);
    const bool add = C && beta != 0.0f && !bias;
    if (add) {
        /* y = C broadcast, which the product adds to */
        Gew e;
        IrTensor *const one[1] = {(IrTensor *)C};
        gew_layout(&e, &y, (const IrTensor *const *)one, 1u);
        gew_dispatch(c, &e, SPG_GEW_COPY, type_code(F32) << 12 | type_code(F32), out_at(c, 0), in_at(c, 2), 0, 0, ry,
                     range_of(c, n->inputs[2]), ry, ry);
    }
    if (!K) {
        if (!add && !bias) fill_output(c, 0, ry.hi - ry.lo, 0);
        if (!bias) return true;
    }
    uint64_t a = in_at(c, 0);
    Range ra = range_of(c, n->inputs[0]);
    if (both) {
        const int64_t ad[2] = {M, K}, sx[2] = {1, M}, sy[2] = {K, 1};
        const Range rs = {c->scratch, c->scratch + (uint64_t)M * (uint64_t)K * 4u};
        copy_strided(c, c->scratch, a, 2, ad, sx, sy, 4u, rs, ra);
        a = c->scratch;
        ra = rs;
    }
    SpgGemmPush push = {
        .a = a, .b = in_at(c, 1), .c = out_at(c, 0), .e0 = bias ? in_at(c, 2) : 0,
        .M = (uint32_t)M, .N = (uint32_t)N, .K = (uint32_t)K, .lda = ta && !both ? (uint32_t)M : (uint32_t)K,
        .ldb = tb ? (uint32_t)K : (uint32_t)N, .ldc = (uint32_t)N, .alpha = alpha, .beta = add ? beta : 0.0f,
        .flags = bias ? SPG_GEMM_BIAS : 0u,
    };
    gemm(c, &push, ta && !both ? SPG_A_COL : SPG_A_ROW, tb ? SPG_B_COL : SPG_B_ROW,
         bias ? SPG_EPI_BIAS_ACT : SPG_EPI_STORE, SPINGALETT_ACT_NONE, 1, ra, range_of(c, n->inputs[1]),
         bias ? range_of(c, n->inputs[2]) : none, ry);
    return true;
}

/* MatMul with numpy's broadcasting of the batches: one product when B has no batch (A's batches its rows), else a
   product a batch (in the z of one dispatch where each operand's batches are evenly apart, or not there). */
static bool op_matmul(GCtx *c, IrTensor *const *in, IrTensor **out) {
    const IrNode *n = c->n;
    const IrTensor *A = in[0], *B = in[1];
    if (!A || !B || !out[0] || A->dtype != F32 || B->dtype != F32 || !A->rank || !B->rank) return false;
    /* vectors as matrices of one row (A) or column (B), the axis dropped from the output */
    const int64_t M = A->rank > 1 ? A->dims[A->rank - 2] : 1, K = A->dims[A->rank - 1];
    const int64_t KB = B->rank > 1 ? B->dims[B->rank - 2] : B->dims[0], N = B->rank > 1 ? B->dims[B->rank - 1] : 1;
    if (K != KB || M > UINT32_MAX || N > UINT32_MAX || K > UINT32_MAX) return false;
    IrTensor ba, bb;
    memset(&ba, 0, sizeof ba);
    memset(&bb, 0, sizeof bb);
    ba.rank = A->rank > 2 ? (uint8_t)(A->rank - 2) : 0;
    bb.rank = B->rank > 2 ? (uint8_t)(B->rank - 2) : 0;
    for (uint32_t d = 0; d < ba.rank; d++) ba.dims[d] = A->dims[d];
    for (uint32_t d = 0; d < bb.rank; d++) bb.dims[d] = B->dims[d];
    IrTensor *pair[2] = {&ba, &bb};
    uint32_t R;
    int64_t bd[IR_MAX_RANK];
    if (!broadcast(pair, 2, &R, bd)) return false;
    int64_t dims[IR_MAX_RANK];
    uint32_t rank = 0;
    uint64_t batches = 1;
    for (uint32_t d = 0; d < R; d++) {
        dims[rank++] = bd[d];
        batches *= (uint64_t)bd[d];
    }
    if (A->rank > 1) dims[rank++] = M;
    if (B->rank > 1) dims[rank++] = N;
    if (rank > IR_MAX_RANK) return false;
    /* each operand's batch strides over the broadcast batches, merged into one where they run evenly */
    int64_t sa[IR_MAX_RANK], sb[IR_MAX_RANK];
    for (uint32_t k = 0; k < 2; k++) {
        const IrTensor *t = k ? &bb : &ba;
        int64_t *s = k ? sb : sa;
        broadcast_strides(t, R, bd, s);
        const int64_t matrix = k ? K * N : M * K;
        for (uint32_t d = 0; d < R; d++) s[d] *= matrix;
    }
    /* the batches as one axis (z) where both operands' strides run evenly through them */
    bool even = true;
    for (uint32_t d = 0; d + 1 < R && even; d++) {
        if (bd[d] == 1) continue;
        even = sa[d] == sa[d + 1] * bd[d + 1] && sb[d] == sb[d + 1] * bd[d + 1];
    }
    if (!c->record) return (even || batches <= 4096u) && set_shape(out[0], F32, rank, dims);
    const Range none = {0, 0};
    const Range ra = range_of(c, n->inputs[0]), rb = range_of(c, n->inputs[1]), ry = range_of(c, n->outputs[0]);
    if (!K) {
        fill_output(c, 0, ry.hi - ry.lo, 0);
        return true;
    }
    /* B shared by every batch, A's batches contiguous: one product of all of A's rows */
    bool b_shared = true;
    for (uint32_t d = 0; d < R; d++) b_shared = b_shared && sb[d] == 0;
    if (b_shared && even && batches * (uint64_t)M <= UINT32_MAX) {
        SpgGemmPush push = {.a = in_at(c, 0), .b = in_at(c, 1), .c = out_at(c, 0), .M = (uint32_t)(batches * (uint64_t)M),
                            .N = (uint32_t)N, .K = (uint32_t)K, .lda = (uint32_t)K, .ldb = (uint32_t)N,
                            .ldc = (uint32_t)N, .alpha = 1.0f};
        /* (A's batches broadcast over B's absent ones would repeat rows: not this case, as B has none) */
        bool a_full = true;
        for (uint32_t d = 0; d < R; d++) a_full = a_full && (bd[d] == 1 || sa[d] != 0);
        if (a_full) {
            gemm(c, &push, SPG_A_ROW, SPG_B_ROW, SPG_EPI_STORE, SPINGALETT_ACT_NONE, 1, ra, rb, none, ry);
            return true;
        }
    }
    /* a product a batch: in z where even (its strides those of the innermost batch axis of size above 1), else a
       dispatch per batch */
    int64_t za = 0, zb = 0;
    for (uint32_t d = R; d-- > 0;)
        if (bd[d] != 1) {
            za = sa[d];
            zb = sb[d];
            break;
        }
    const uint64_t per = even ? batches : 1u;
    for (uint64_t first = 0; first < batches; first += per) {
        int64_t oa = 0, ob = 0;
        uint64_t r = first;
        for (uint32_t d = R; d-- > 0 && !even;) {
            const int64_t k = (int64_t)(r % (uint64_t)bd[d]);
            r /= (uint64_t)bd[d];
            oa += k * sa[d];
            ob += k * sb[d];
        }
        for (uint64_t z0 = 0; z0 < per; z0 += 65535u) {
            const uint32_t z = (uint32_t)(per - z0 < 65535u ? per - z0 : 65535u);
            SpgGemmPush push = {
                .a = in_at(c, 0) + (uint64_t)(oa + (int64_t)z0 * za) * 4u,
                .b = in_at(c, 1) + (uint64_t)(ob + (int64_t)z0 * zb) * 4u,
                .c = out_at(c, 0) + (first + z0) * (uint64_t)(M * N) * 4u,
                .M = (uint32_t)M, .N = (uint32_t)N, .K = (uint32_t)K, .lda = (uint32_t)K, .ldb = (uint32_t)N,
                .ldc = (uint32_t)N, .a_group = (uint32_t)za, .b_group = (uint32_t)zb, .c_group = (uint32_t)(M * N),
                .alpha = 1.0f,
            };
            gemm(c, &push, SPG_A_ROW, SPG_B_ROW, SPG_EPI_STORE, SPINGALETT_ACT_NONE, z, ra, rb, none, ry);
        }
    }
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
    [IR_Transpose] = op_transpose, [IR_Slice] = op_slice, [IR_Concat] = op_concat, [IR_Split] = op_split,
    [IR_Expand] = op_expand, [IR_Tile] = op_tile, [IR_Pad] = op_pad, [IR_Gather] = op_gather,
    [IR_Linear] = op_linear, [IR_Gemm] = op_gemm, [IR_MatMul] = op_matmul,
    [IR_ReduceSum] = op_reduce, [IR_ReduceMean] = op_reduce, [IR_ReduceMax] = op_reduce, [IR_ReduceMin] = op_reduce,
    [IR_ReduceProd] = op_reduce, [IR_ReduceL1] = op_reduce, [IR_ReduceL2] = op_reduce,
    [IR_ReduceSumSquare] = op_reduce, [IR_ReduceLogSumExp] = op_reduce, [IR_ReduceLogSum] = op_reduce,
    [IR_ArgMax] = op_arg, [IR_ArgMin] = op_arg, [IR_Softmax] = op_softmax, [IR_LogSoftmax] = op_softmax,
    [IR_Hardmax] = op_softmax, [IR_LayerNormalization] = op_layernorm, [IR_RMSNormalization] = op_layernorm,
    [IR_GroupNormalization] = op_groupnorm, [IR_InstanceNormalization] = op_groupnorm,
    [IR_FusedAttention] = op_attention, [IR_Conv] = op_conv, [IR_MaxPool] = op_pool, [IR_AveragePool] = op_pool,
    [IR_LpPool] = op_pool, [IR_GlobalAveragePool] = op_global_pool, [IR_GlobalMaxPool] = op_global_pool,
    [IR_Resize] = op_resize, [IR_Upsample] = op_resize,
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
    if (pc->commands && (!spg_gpu_submit(pc->commands) || !spg_gpu_wait(pc->commands))) {
        for (uint32_t j = 0; j < pc->node_count; j++)
            spingalett_log(LOG_DEBUG, "graph on the GPU: the failed piece's node %u: %s '%s'", j,
                           ir_op_name((IrOp)g->nodes[pc->nodes[j]].op), g->nodes[pc->nodes[j]].name);
        return fail("graph on the GPU: the device failed to run a piece");
    }
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
        /* (a value kept as bfloat16, which a node the host runs reads: widened first) */
        const uint64_t wide = p->half[want[k]] ? piece_take(pc, bytes) + 1u : 0;
        pc->downs[pc->down_count++] = (Transfer){want[k], pc->node_count, pc->staged, bytes, wide};
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
    if (pc->params_size) {
        pc->param_bytes = (uint8_t *)calloc(1, pc->params_size);
        if (!pc->param_bytes || !spg_gpu_buffer_create(&pc->params, pc->params_size, false))
            return fail("graph on the GPU: out of memory");
    }
    pc->commands = spg_gpu_commands_create();
    if (!pc->commands || !spg_gpu_record_begin(pc->commands)) return fail("graph on the GPU: no commands");
    Recorder r;
    memset(&r, 0, sizeof r);
    r.c = pc->commands;
    GCtx c = {g, gg, p, 0, NULL, true, &r, 0, 0, 0, {0}, 0, 0, 0, 0, 0, {0}, pc};
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
        if (t->wide) {
            SpgGewPush push;
            memset(&push, 0, sizeof push);
            push.y = pc->memory.address + t->wide - 1u;
            push.a = push.b = push.c = address_of(&c, t->value);
            push.total = (uint32_t)g->tensors[t->value].count;
            const uint32_t spec[3] = {SPG_GEW_COPY, SPG_T_BF16 | SPG_T_F32 << 12, 1u};
            reads(&a, range_of(&c, t->value));
            writes(&a, (Range){push.y, push.y + t->bytes});
            dispatch(&c, &a, SPG_KERNEL_gew, spec, 3, &push, sizeof push, group_count(push.total, 256u), 1, 1);
            memset(&a, 0, sizeof a);
            reads(&a, (Range){push.y, push.y + t->bytes});
            need(&r, &a);
            spg_gpu_copy(pc->commands, &pc->memory, t->wide - 1u, &pc->staging, t->stage, t->bytes);
            continue;
        }
        reads(&a, range_of(&c, t->value));
        need(&r, &a);
        const uint32_t b = p->block_of[t->value];
        const SpgGpuBuffer *src = g->values[t->value].kind == VALUE_CONST ? &gg->consts[t->value]
                                                                          : &p->pieces[p->blocks[b].piece].memory;
        spg_gpu_copy(pc->commands, src, g->values[t->value].kind == VALUE_CONST ? 0 : p->blocks[b].at + p->offset[t->value],
                     &pc->staging,
                     t->stage, t->bytes);
    }
    spg_gpu_barrier_host(pc->commands);
    if (!spg_gpu_record_end(pc->commands)) return fail("graph on the GPU: its commands could not be recorded");
    if (pc->params_size) {
        /* the parameter blocks, once (through memory the host fills) */
        SpgGpuBuffer stage;
        SpgGpuCommands *cmd = NULL;
        bool ok = spg_gpu_buffer_create(&stage, pc->params_size, true);
        if (ok) {
            memcpy(stage.mapped, pc->param_bytes, pc->params_size);
            cmd = spg_gpu_commands_create();
            ok = cmd && spg_gpu_record_begin(cmd);
        }
        if (ok) spg_gpu_copy(cmd, &stage, 0, &pc->params, 0, pc->params_size);
        ok = ok && spg_gpu_record_end(cmd) && spg_gpu_submit(cmd) && spg_gpu_wait(cmd);
        if (cmd) spg_gpu_commands_free(cmd);
        spg_gpu_buffer_free(&stage);
        free(pc->param_bytes);
        pc->param_bytes = NULL;
        if (!ok) return fail("graph on the GPU: the parameters could not be uploaded");
    }
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
        GCtx probe = {g, gg, p, k, n, false, NULL, 0, 0, 0, {0}, 0, 0, 0, 0, 0, {0}, NULL};
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
    *c = (GCtx){g, gg, p, k, n, false, NULL, 0, 0, 0, {0}, 0, 0, 0, 0, 0, {0}, NULL};
    IrTensor saved[32];
    for (uint32_t o = 0; o < n->output_count; o++)
        if (out[o]) saved[o] = *out[o];
    bool ok = op(c, in, out);
    /* (an input kept as bfloat16 that the node's kernels would read as floats: the host's, widened) */
    for (uint32_t i = 0; ok && i < n->input_count; i++) {
        const uint32_t v = n->inputs[i];
        ok = v == NONE || !p->half[v] || (i < 32 && (c->half_ins >> i & 1u)) || (i == 0 && c->views);
    }
    if (!ok) {
        for (uint32_t o = 0; o < n->output_count; o++)
            if (out[o]) *out[o] = saved[o];
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------------- values kept as bfloat16 */

/* Values the graph declares: their number of values (0 when a size is unknown). */
static uint64_t declared_count(const SpingalettGraph *g, uint32_t v) {
    const IrValue *x = &g->values[v];
    if (!x->declared) return 0;
    uint64_t count = 1;
    for (uint32_t d = 0; d < x->rank; d++) {
        if (x->dims[d] < 0) return 0;
        count *= (uint64_t)x->dims[d];
    }
    return count;
}

/* Whether products run in bfloat16 on the matrix units with values kept as such between them (CUDA's kernels). */
static bool half_values(void) {
    return spingalett_get_gpu_precision() == PRECISION_BFLOAT16 && spg_gpu_using() == SPG_BACKEND_CUDA &&
           spg_gpu_mma_bf16() && spg_gpu_kernel_built(SPG_KERNEL_gcvt) && spg_gpu_kernel_built(SPG_KERNEL_gflash);
}

/* Whether node n reads v (only as its input `slot`) as a product's A operand on the matrix units or as attention's
   input in bfloat16, by the shapes the graph infers. */
static bool half_reader(const SpingalettGraph *g, const IrNode *n, uint32_t v) {
    uint32_t slot = NONE;
    for (uint32_t i = 0; i < n->input_count; i++)
        if (n->inputs[i] == v) {
            /* (attention may read one value as two or three of Q, K and V: views of it) */
            if (slot != NONE && !(n->op == IR_FusedAttention && i < 3)) return false;
            slot = i;
        }
    const uint64_t count = declared_count(g, v);
    if (slot == NONE || !count) return false;
    switch ((IrOp)n->op) {
        case IR_Linear: case IR_Gemm: {
            if (slot != 0 || n->input_count < 2 || g->values[n->inputs[1]].kind != VALUE_CONST) return false;
            const IrValue *W = &g->values[n->inputs[1]];
            bool trans;
            if (n->op == IR_Gemm) {
                if (ir_attr_int(n, "transA", 0) || ir_attr_float(n, "alpha", 1.0f) != 1.0f) return false;
                if (n->input_count > 2 && n->inputs[2] != NONE) {
                    const uint64_t cc = declared_count(g, n->inputs[2]);
                    if (ir_attr_float(n, "beta", 1.0f) != 1.0f || !cc ||
                        cc != (uint64_t)W->dims[ir_attr_int(n, "transB", 0) ? 0 : 1])
                        return false;
                }
                trans = ir_attr_int(n, "transB", 0) != 0;
            } else {
                trans = ir_attr_int(n, "trans_b", 0) != 0;
            }
            if (W->rank != 2) return false;
            const int64_t K = trans ? W->dims[1] : W->dims[0], N = trans ? W->dims[0] : W->dims[1];
            return K > 0 && count % (uint64_t)K == 0 && product_bf16((uint32_t)(count / (uint64_t)K), (uint32_t)N, (uint32_t)K);
        }
        case IR_Conv: {
            if (slot != 0 || !ir_attr_int(n, "channels_last", 0) || n->input_count < 2 ||
                g->values[n->inputs[1]].kind != VALUE_CONST || n->outputs[0] == NONE)
                return false;
            const IrValue *W = &g->values[n->inputs[1]], *Y = &g->values[n->outputs[0]];
            const uint64_t out_count = declared_count(g, n->outputs[0]);
            const int64_t G = ir_attr_int(n, "group", 1), OC = W->dims[0];
            if (!out_count || G < 1 || OC % G || !Y->rank) return false;
            uint64_t K = (uint64_t)W->dims[1];
            for (uint32_t d = 2; d < W->rank; d++) K *= (uint64_t)W->dims[d];
            return product_bf16((uint32_t)(out_count / (uint64_t)OC), (uint32_t)(OC / G), (uint32_t)K);
        }
        case IR_FusedAttention: {
            if (slot > 2 || n->input_count < 3) return false;
            if (ir_attr_int(n, "layout", 0) == 2) {
                const IrAttr *va = ir_attr(n, "views");
                return va && va->type == ATTR_INTS && va->count == 18 && va->ints[4] > 0 && va->ints[4] <= 128 &&
                       va->ints[4] == va->ints[5];
            }
            const IrValue *Q = &g->values[n->inputs[0]], *V = &g->values[n->inputs[2]];
            if (!Q->declared || !V->declared || !Q->rank || !V->rank) return false;
            const int64_t D = Q->dims[Q->rank - 1], Dv = V->dims[V->rank - 1];
            return D > 0 && D <= 128 && D == Dv;
        }
        default:
            return false;
    }
}

/* Whether every reader of v reads it so (copies and views passing it on to such readers), none the graph's outputs. */
static bool half_readers(const GpuGraph *gg, const SpingalettGraph *g, uint32_t v, uint32_t depth) {
    if (depth > 8 || gg->reader_at[v] == gg->reader_at[v + 1]) return false;
    for (uint32_t r = gg->reader_at[v]; r < gg->reader_at[v + 1]; r++) {
        const uint32_t k = gg->readers[r];
        if (k == NONE) return false;
        const IrNode *n = &g->nodes[k];
        switch ((IrOp)n->op) {
            case IR_Reshape: case IR_Flatten: case IR_Squeeze: case IR_Unsqueeze: case IR_Identity: case IR_Transpose:
            case IR_Slice: case IR_Split:
                if (n->inputs[0] != v) return false;
                for (uint32_t i = 1; i < n->input_count; i++)
                    if (n->inputs[i] == v) return false;
                for (uint32_t o = 0; o < n->output_count; o++)
                    if (n->outputs[o] != NONE && !half_readers(gg, g, n->outputs[o], depth + 1)) return false;
                break;
            default:
                if (!half_reader(g, n, v)) return false;
        }
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
        if (v == NONE || ((c->params | c->replaced) >> i & 1u)) continue;
        if (g->values[v].kind == VALUE_CONST) {
            p->on[v] |= ON_DEVICE;
            continue;
        }
        if (has_device(p, v)) continue;
        /* uploaded by this piece */
        const uint64_t bytes = bytes_of(&g->tensors[v]);
        const uint32_t b = block_new(p, bytes);
        if (b == NONE || !grow((void **)&pc->ups, &pc->up_cap, pc->up_count, sizeof(Transfer))) return false;
        pc->ups[pc->up_count++] = (Transfer){v, pc->node_count, pc->staged, bytes, 0};
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
            p->offset[v] = p->offset[src] + (o < 8 ? c->view_at[o] : 0);
            p->half[v] = p->half[src];
            p->blocks[p->block_of[v]].refs++;
        } else {
            /* kept as bfloat16: a copy of such, or what its readers all read so */
            p->half[v] = o < 32 && ((c->half_force >> o & 1u) ||
                                    ((c->half_out >> o & 1u) && half_values() && half_readers(c->gg, g, v, 0)));
            const uint32_t b = block_new(p, stored_bytes(p, g, v));
            if (b == NONE) return false;
            p->block_of[v] = b;
        }
        p->on[v] = ON_DEVICE;
    }
    for (uint32_t i = 0; i < 4 && i < n->input_count; i++) {
        const uint32_t v = n->inputs[i];
        if (!(c->halves >> i & 1u) || v == NONE || p->half_block[v] != NONE) continue;
        if ((p->half_block[v] = block_new(p, 2u * c->half_count[i])) == NONE) return false;
        p->half_maker[v] = k;
    }
    if (c->scratch) {
        Piece *cur = &p->pieces[p->piece_count - 1u];
        p->scratch[k].bytes = c->scratch;
        p->scratch[k].at = piece_take(cur, c->scratch);
    }
    if (!grow((void **)&pc->nodes, &pc->node_cap, pc->node_count, sizeof(uint32_t))) return false;
    pc->nodes[pc->node_count++] = k;
    pc->params_size += c->param;
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
        if (!again && p->half_block[v] != NONE) block_drop(p, p->half_block[v]);
    }
    for (uint32_t o = 0; o < n->output_count; o++) {
        const uint32_t v = n->outputs[o];
        if (v != NONE && g->values[v].last < 0 && p->block_of[v] != NONE) block_drop(p, p->block_of[v]);
    }
}

static void report_tensor(const IrTensor *t, bool half) {
    fprintf(stderr, " %s[", half ? "bfloat16" : ir_dtype_name(t->dtype));
    for (uint32_t d = 0; d < t->rank; d++) fprintf(stderr, "%s%lld", d ? "," : "", (long long)t->dims[d]);
    fprintf(stderr, "]");
}

/* SPINGALETT_GRAPH_GPU_REPORT: 1 lists the operators the host ran (but for those of shapes), 2 with their inputs'
   types and shapes, 3 every node in order, where it runs, its inputs and outputs as the device keeps them, the
   inputs it rounds to bfloat16 (^) and those of sizes the graph does not declare (?). */
static void report_plan(const SpingalettGraph *g, const Plan *p, char level) {
    for (uint32_t s = 0; s < p->step_count; s++) {
        const bool host = p->steps[s].kind == STEP_HOST;
        const Piece *pc = host ? NULL : &p->pieces[p->steps[s].index];
        for (uint32_t j = 0; j < (host ? 1u : pc->node_count); j++) {
            const uint32_t k = host ? p->steps[s].index : pc->nodes[j];
            const IrNode *n = &g->nodes[k];
            if ((!host && level < '3') || shape_only((IrOp)n->op)) continue;
            fprintf(stderr, "graph on the GPU: %s %s '%s'", host ? "on the host:" : level < '3' ? "" : "  device:",
                    ir_op_name((IrOp)n->op), n->name);
            for (uint32_t i = 0; i < n->input_count && level >= '2'; i++) {
                const uint32_t v = n->inputs[i];
                if (v == NONE) continue;
                report_tensor(&g->tensors[v], !host && p->half[v]);
                if (!host && p->half_maker[v] == k) fprintf(stderr, "^");
                if (level >= '3' && !declared_count(g, v)) fprintf(stderr, "?");
            }
            if (level >= '3') {
                fprintf(stderr, " ->");
                for (uint32_t o = 0; o < n->output_count; o++)
                    if (n->outputs[o] != NONE) report_tensor(&g->tensors[n->outputs[o]], !host && p->half[n->outputs[o]]);
            }
            fprintf(stderr, "\n");
        }
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
    const char *report = getenv("SPINGALETT_GRAPH_GPU_REPORT");
    if (ok && report && *report && *report != '0') report_plan(g, p, *report);
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
    gg->prepared = (SpgGpuBuffer *)calloc(2u * (g->node_count ? g->node_count : 1u), sizeof(SpgGpuBuffer));
    if (!gg->consts || !gg->prepared) {
        free(gg->consts);
        free(gg->prepared);
        free(gg);
        return NULL;
    }
    /* the readers of each value (the graph's outputs NONE), for the values kept as bfloat16 */
    gg->reader_at = (uint32_t *)calloc(g->value_count + 2u, sizeof(uint32_t));
    uint64_t total = g->output_count;
    for (uint32_t k = 0; k < g->node_count; k++) total += g->nodes[k].input_count;
    gg->readers = (uint32_t *)malloc((total ? total : 1u) * sizeof(uint32_t));
    if (!gg->reader_at || !gg->readers) {
        free(gg->reader_at);
        free(gg->readers);
        free(gg->consts);
        free(gg->prepared);
        free(gg);
        return NULL;
    }
    for (uint32_t k = 0; k < g->node_count; k++)
        for (uint32_t i = 0; i < g->nodes[k].input_count; i++)
            if (g->nodes[k].inputs[i] != NONE) gg->reader_at[g->nodes[k].inputs[i] + 1u]++;
    for (uint32_t o = 0; o < g->output_count; o++) gg->reader_at[g->outputs[o] + 1u]++;
    for (uint32_t v = 0; v < g->value_count; v++) gg->reader_at[v + 1u] += gg->reader_at[v];
    uint32_t *fill = (uint32_t *)calloc(g->value_count + 1u, sizeof(uint32_t));
    if (!fill) {
        free(gg->reader_at);
        free(gg->readers);
        free(gg->consts);
        free(gg->prepared);
        free(gg);
        return NULL;
    }
    for (uint32_t k = 0; k < g->node_count; k++)
        for (uint32_t i = 0; i < g->nodes[k].input_count; i++) {
            const uint32_t v = g->nodes[k].inputs[i];
            if (v != NONE) gg->readers[gg->reader_at[v] + fill[v]++] = k;
        }
    for (uint32_t o = 0; o < g->output_count; o++) gg->readers[gg->reader_at[g->outputs[o]] + fill[g->outputs[o]]++] = NONE;
    free(fill);
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
    for (uint32_t k = 0; k < 2u * g->node_count; k++) spg_gpu_buffer_free(&gg->prepared[k]);
    free(gg->arenas);
    free(gg->consts);
    free(gg->prepared);
    free(gg->reader_at);
    free(gg->readers);
    free(gg);
    g->gpu = NULL;
    spg_gpu_use(saved);
}
