/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Graphs of operators (Spingalett.IR.h): their memory, values and nodes, what happens when a graph opens (floats of
 * 16 bits widened, the nodes' order checked, constants folded, what no output needs left out, every value's last
 * reader found), and runs: the inputs bound to the graph's values (their shapes against the declared ones and the
 * symbols' sizes), each node's kernel in the nodes' order, a value freed after its last reader, the outputs handed
 * out. The public functions of spingalett_graph_*() are here.
 */

#include "Spingalett.IR.h"
#include "Spingalett.Thread.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *const op_names[IR_OP_COUNT] = {
#define IR_OP(name, kernel) #name,
#include "Spingalett.IROps.def"
#undef IR_OP
};

const char *ir_op_name(IrOp op) {
    return (unsigned)op < IR_OP_COUNT ? op_names[op] : "?";
}

IrOp ir_op_of(const char *op_type, size_t len, const char *domain, size_t domain_len) {
    const bool onnx = domain_len == 0 || (domain_len == 7 && !memcmp(domain, "ai.onnx", 7)) ||
                      (domain_len == 10 && !memcmp(domain, "spingalett", 10));
    if (!onnx) return IR_OP_COUNT;
    for (int k = 0; k < IR_OP_COUNT; k++)
        if (strlen(op_names[k]) == len && !memcmp(op_names[k], op_type, len)) return (IrOp)k;
    return IR_OP_COUNT;
}

size_t ir_dtype_size(uint8_t dtype) {
    switch (dtype) {
        case SPINGALETT_DTYPE_FLOAT32: case SPINGALETT_DTYPE_INT32: return 4;
        case SPINGALETT_DTYPE_INT64: return 8;
        case SPINGALETT_DTYPE_FLOAT16: case SPINGALETT_DTYPE_BFLOAT16: return 2;
        case SPINGALETT_DTYPE_UINT8: case SPINGALETT_DTYPE_INT8: case SPINGALETT_DTYPE_BOOL: return 1;
        default: return 0;
    }
}

const char *ir_dtype_name(uint8_t dtype) {
    switch (dtype) {
        case SPINGALETT_DTYPE_FLOAT32: return "float32";
        case SPINGALETT_DTYPE_INT32: return "int32";
        case SPINGALETT_DTYPE_INT64: return "int64";
        case SPINGALETT_DTYPE_FLOAT16: return "float16";
        case SPINGALETT_DTYPE_BFLOAT16: return "bfloat16";
        case SPINGALETT_DTYPE_UINT8: return "uint8";
        case SPINGALETT_DTYPE_INT8: return "int8";
        case SPINGALETT_DTYPE_BOOL: return "bool";
        default: return "an unsupported type";
    }
}

/* ------------------------------------------------------------------------- the graph's memory */

SpingalettGraph *ir_graph_new(void) {
    SpingalettGraph *g = (SpingalettGraph *)calloc(1, sizeof *g);
    if (!g) {
        set_error(SPINGALETT_ERR_ALLOC, "graph: out of memory");
        return NULL;
    }
    g->lock = spg_signal_create();
    if (!g->lock) {
        free(g);
        set_error(SPINGALETT_ERR_ALLOC, "graph: out of memory");
        return NULL;
    }
    return g;
}

static void free_run(SpingalettGraph *g) {
    if (!g->tensors) return;
    for (uint32_t v = 0; v < g->value_count; v++) ir_release(g, &g->tensors[v]);
    free(g->tensors);
    g->tensors = NULL;
}

/* ------------------------------------------------------------------------- the pool */

void *ir_pool_take(SpingalettGraph *g, size_t bytes, size_t *capacity) {
    if (!bytes) bytes = 1;
    /* the smallest free block that holds it without wasting more than its size */
    uint32_t best = UINT32_MAX;
    for (uint32_t b = 0; b < g->pool_count; b++) {
        const IrBlock *k = &g->pool[b];
        if (!k->refs && k->bytes >= bytes && k->bytes / 2 <= bytes && (best == UINT32_MAX || k->bytes < g->pool[best].bytes))
            best = b;
    }
    if (best == UINT32_MAX) {
        if (g->pool_count == g->pool_cap) {
            const uint32_t cap = g->pool_cap ? 2u * g->pool_cap : 64u;
            IrBlock *grown = (IrBlock *)realloc(g->pool, cap * sizeof(IrBlock));
            if (!grown) return NULL;
            g->pool = grown;
            g->pool_cap = cap;
        }
        const size_t rounded = (bytes + 4095u) & ~(size_t)4095u;
        void *data = spingalett_aligned_alloc(rounded);
        if (!data) return NULL;
        best = g->pool_count++;
        g->pool[best].data = data;
        g->pool[best].bytes = rounded;
    }
    g->pool[best].refs = 1;
    g->pool[best].used = g->runs;
    *capacity = g->pool[best].bytes;
    return g->pool[best].data;
}

static IrBlock *block_of(SpingalettGraph *g, const void *data) {
    for (uint32_t b = g->pool_count; b-- > 0;)
        if ((const uint8_t *)data >= (const uint8_t *)g->pool[b].data &&
            (const uint8_t *)data < (const uint8_t *)g->pool[b].data + g->pool[b].bytes) return &g->pool[b];
    return NULL;
}

void ir_pool_give(SpingalettGraph *g, void *data) {
    IrBlock *b = block_of(g, data);
    if (b && b->refs) b->refs--;
}

void ir_view(IrRun *r, IrTensor *t, const IrTensor *x, uint32_t rank, const int64_t *dims) {
    if (t != x) ir_release(r->g, t);
    IrBlock *b = x->owned ? block_of(r->g, x->data) : NULL;
    if (b) b->refs++;
    t->data = x->data;
    t->owned = b != NULL;
    t->constant = x->constant;
    t->capacity = x->capacity;
    t->dtype = x->dtype;
    t->rank = (uint8_t)rank;
    for (uint32_t k = 0; k < IR_MAX_RANK; k++) t->dims[k] = k < rank ? dims[k] : 0;
    t->count = x->count;
}

/* After a run: the free blocks it did not take are freed (shapes that changed leave blocks no run takes). */
static void pool_trim(SpingalettGraph *g) {
    uint32_t kept = 0;
    for (uint32_t b = 0; b < g->pool_count; b++) {
        if (!g->pool[b].refs && g->pool[b].used < g->runs) spingalett_aligned_free(g->pool[b].data);
        else g->pool[kept++] = g->pool[b];
    }
    g->pool_count = kept;
}

void ir_release(SpingalettGraph *g, IrTensor *t) {
    if (t->owned) ir_pool_give(g, t->data);
    t->owned = false;
    t->data = NULL;
}

void *ir_prepared(IrRun *r) {
    return r->g->prepared ? r->g->prepared[r->index] : NULL;
}

void ir_prepare(IrRun *r, void *data, void (*free_fn)(void *)) {
    SpingalettGraph *g = r->g;
    if (!g->prepared) {
        g->prepared = (void **)calloc(g->node_count ? g->node_count : 1u, sizeof(void *));
        g->prepared_free = (void (**)(void *))calloc(g->node_count ? g->node_count : 1u, sizeof(void (*)(void *)));
        if (!g->prepared || !g->prepared_free) {
            free(g->prepared);
            free(g->prepared_free);
            g->prepared = NULL;
            g->prepared_free = NULL;
            free_fn(data);
            return;
        }
    }
    g->prepared[r->index] = data;
    g->prepared_free[r->index] = free_fn;
}

static double seconds(void) {
#if defined(TIME_UTC)
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

/* The time of each operator over the graph's runs, longest first (SPINGALETT_GRAPH_PROFILE). */
static void print_profile(const SpingalettGraph *g) {
    double total = 0.0;
    for (int k = 0; k < IR_OP_COUNT; k++) total += g->profile[k];
    fprintf(stderr, "graph profile: %.2f ms in %llu runs after the first\n", total * 1e3,
            (unsigned long long)(g->runs ? g->runs - 1 : 0));
    bool shown[IR_OP_COUNT] = {false};
    for (;;) {
        int best = -1;
        for (int k = 0; k < IR_OP_COUNT; k++)
            if (!shown[k] && g->profile_nodes[k] && (best < 0 || g->profile[k] > g->profile[best])) best = k;
        if (best < 0) break;
        shown[best] = true;
        fprintf(stderr, "  %-24s %8llu nodes %10.3f ms %5.1f%%\n", op_names[best], (unsigned long long)g->profile_nodes[best],
                g->profile[best] * 1e3, total > 0.0 ? 100.0 * g->profile[best] / total : 0.0);
    }
}

void ir_graph_free(SpingalettGraph *g) {
    if (!g) return;
    if (g->profile) print_profile(g);
    free(g->profile);
    free(g->profile_nodes);
    free_run(g);
    for (uint32_t b = 0; b < g->pool_count; b++) spingalett_aligned_free(g->pool[b].data);
    free(g->pool);
    for (uint32_t k = 0; g->prepared && k < g->node_count; k++)
        if (g->prepared[k]) g->prepared_free[k](g->prepared[k]);
    free(g->prepared);
    free(g->prepared_free);
    for (uint32_t b = 0; b < g->block_count; b++) spingalett_aligned_free(g->blocks[b]);
    free(g->blocks);
    free(g->values);
    free(g->names);
    free(g->nodes);
    free(g->inputs);
    free(g->outputs);
    free(g->symbols);
    for (uint32_t k = 0; k < g->external_count; k++) spingalett_file_close(&g->externals[k]);
    free(g->externals);
    free(g->external_paths);
    if (g->file.data) spingalett_file_close(&g->file);
    spingalett_aligned_free(g->copy);
    if (g->scratch) spingalett_gemm_scratch_free(g->scratch);
    for (int t = 0; g->thread_scratch && t < g->scratch_threads; t++) spingalett_gemm_scratch_free(g->thread_scratch[t]);
    free(g->thread_scratch);
    if (g->lock) spg_signal_free((SpgSignal *)g->lock);
    free(g);
}

void *ir_alloc(SpingalettGraph *g, size_t bytes) {
    if (g->block_count == g->block_cap) {
        uint32_t cap = g->block_cap ? 2u * g->block_cap : 64u;
        void **grown = (void **)realloc(g->blocks, cap * sizeof(void *));
        if (!grown) {
            set_error(SPINGALETT_ERR_ALLOC, "graph: out of memory");
            return NULL;
        }
        g->blocks = grown;
        g->block_cap = cap;
    }
    void *p = spingalett_aligned_alloc(bytes ? bytes : 1u);
    if (!p) {
        set_error(SPINGALETT_ERR_ALLOC, "graph: out of memory");
        return NULL;
    }
    memset(p, 0, bytes ? bytes : 1u);
    g->blocks[g->block_count++] = p;
    return p;
}

char *ir_strdup(SpingalettGraph *g, const char *s, size_t n) {
    char *c = (char *)ir_alloc(g, n + 1);
    if (c) {
        if (n) memcpy(c, s, n);
        c[n] = '\0';
    }
    return c;
}

static uint32_t name_hash(const char *name, size_t len) {
    uint32_t h = 2166136261u;                   /* FNV-1a */
    for (size_t i = 0; i < len; i++) h = (h ^ (uint8_t)name[i]) * 16777619u;
    return h;
}

/* Value v's name into the table (a later value of the same name takes its slot, so that the last one is found). */
static void name_insert(SpingalettGraph *g, uint32_t v) {
    const char *name = g->values[v].name;
    const size_t len = strlen(name);
    for (uint32_t at = name_hash(name, len) & (g->name_cap - 1u);; at = (at + 1u) & (g->name_cap - 1u)) {
        const uint32_t e = g->names[at];
        if (!e || !strcmp(g->values[e - 1u].name, name)) {
            g->names[at] = v + 1u;
            return;
        }
    }
}

static bool names_grow(SpingalettGraph *g) {
    const uint32_t cap = g->name_cap ? 2u * g->name_cap : 1024u;
    uint32_t *names = (uint32_t *)calloc(cap, sizeof(uint32_t));
    if (!names) return false;
    free(g->names);
    g->names = names;
    g->name_cap = cap;
    for (uint32_t v = 0; v < g->value_count; v++) name_insert(g, v);
    return true;
}

uint32_t ir_value(SpingalettGraph *g, const char *name, size_t name_len) {
    if (2u * (g->value_count + 1u) > g->name_cap && !names_grow(g)) {
        set_error(SPINGALETT_ERR_ALLOC, "graph: out of memory");
        return IR_NONE;
    }
    if (g->value_count == g->value_cap) {
        uint32_t cap = g->value_cap ? 2u * g->value_cap : 256u;
        IrValue *grown = (IrValue *)realloc(g->values, cap * sizeof(IrValue));
        if (!grown) {
            set_error(SPINGALETT_ERR_ALLOC, "graph: out of memory");
            return IR_NONE;
        }
        g->values = grown;
        g->value_cap = cap;
    }
    IrValue *v = &g->values[g->value_count];
    memset(v, 0, sizeof *v);
    v->name = ir_strdup(g, name ? name : "", name ? name_len : 0);
    if (!v->name) return IR_NONE;
    v->kind = VALUE_NODE;
    v->last = -1;
    for (int k = 0; k < IR_MAX_RANK; k++) v->dims[k] = -1;
    name_insert(g, g->value_count);
    return g->value_count++;
}

uint32_t ir_find_value(const SpingalettGraph *g, const char *name, size_t name_len) {
    /* (the last of equal names: a later definition shadows, as in ONNX's subgraphs, which graphs do not have) */
    if (!g->name_cap) return IR_NONE;
    for (uint32_t at = name_hash(name, name_len) & (g->name_cap - 1u);; at = (at + 1u) & (g->name_cap - 1u)) {
        const uint32_t e = g->names[at];
        if (!e) return IR_NONE;
        const char *n = g->values[e - 1u].name;
        if (strlen(n) == name_len && !memcmp(n, name, name_len)) return e - 1u;
    }
}

uint32_t ir_node(SpingalettGraph *g, IrOp op) {
    if (g->node_count == g->node_cap) {
        uint32_t cap = g->node_cap ? 2u * g->node_cap : 256u;
        IrNode *grown = (IrNode *)realloc(g->nodes, cap * sizeof(IrNode));
        if (!grown) {
            set_error(SPINGALETT_ERR_ALLOC, "graph: out of memory");
            return IR_NONE;
        }
        g->nodes = grown;
        g->node_cap = cap;
    }
    IrNode *n = &g->nodes[g->node_count];
    memset(n, 0, sizeof *n);
    n->op = (uint16_t)op;
    return g->node_count++;
}

uint16_t ir_symbol(SpingalettGraph *g, const char *name, size_t len) {
    for (uint32_t s = 0; s < g->symbol_count; s++)
        if (strlen(g->symbols[s]) == len && !memcmp(g->symbols[s], name, len)) return (uint16_t)(s + 1u);
    if (g->symbol_count >= 65535u) return 0;
    char **grown = (char **)realloc(g->symbols, (g->symbol_count + 1u) * sizeof(char *));
    if (!grown) return 0;
    g->symbols = grown;
    g->symbols[g->symbol_count] = ir_strdup(g, name, len);
    if (!g->symbols[g->symbol_count]) return 0;
    return (uint16_t)(++g->symbol_count);
}

const IrAttr *ir_attr(const IrNode *n, const char *name) {
    for (uint32_t a = 0; a < n->attr_count; a++)
        if (!strcmp(n->attrs[a].name, name)) return &n->attrs[a];
    return NULL;
}

int64_t ir_attr_int(const IrNode *n, const char *name, int64_t fallback) {
    const IrAttr *a = ir_attr(n, name);
    return a && a->type == ATTR_INT ? a->i : a && a->type == ATTR_FLOAT ? (int64_t)a->f : fallback;
}

float ir_attr_float(const IrNode *n, const char *name, float fallback) {
    const IrAttr *a = ir_attr(n, name);
    return a && a->type == ATTR_FLOAT ? a->f : a && a->type == ATTR_INT ? (float)a->i : fallback;
}

/* ------------------------------------------------------------------------- kernels' helpers */

bool ir_fail(IrRun *r, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vsnprintf(r->error, sizeof r->error, format, args);
    va_end(args);
    return false;
}

/* (the scratches are made for the most threads OpenMP runs, once; runs take the graph's lock, folding happens before
   any run) */
static bool scratches(IrRun *r) {
    SpingalettGraph *g = r->g;
    if (g->scratch) return true;
    int threads = 1;
#if defined(_OPENMP)
    threads = omp_get_max_threads();
#endif
    if (threads < 1) threads = 1;
    g->scratch = spingalett_gemm_scratch_create(threads);
    g->thread_scratch = (SpingalettGemmScratch **)calloc((size_t)threads, sizeof(SpingalettGemmScratch *));
    if (!g->scratch || !g->thread_scratch) return ir_fail(r, "out of memory for products");
    g->scratch_threads = threads;
    return true;
}

SpingalettGemmScratch *ir_scratch(IrRun *r) {
    return scratches(r) ? r->g->scratch : NULL;
}

SpingalettGemmScratch *ir_thread_scratch(IrRun *r, int t) {
    if (!scratches(r) || t < 0 || t >= r->g->scratch_threads) return NULL;
    if (!r->g->thread_scratch[t]) r->g->thread_scratch[t] = spingalett_gemm_scratch_create(1);
    return r->g->thread_scratch[t];
}

bool ir_output(IrRun *r, IrTensor *t, uint8_t dtype, uint32_t rank, const int64_t *dims) {
    if (!ir_output_raw(r, t, dtype, rank, dims)) return false;
    memset(t->data, 0, (size_t)t->count * ir_dtype_size(dtype));
    return true;
}

bool ir_output_raw(IrRun *r, IrTensor *t, uint8_t dtype, uint32_t rank, const int64_t *dims) {
    if (rank > IR_MAX_RANK) return ir_fail(r, "an output of more than %d axes", IR_MAX_RANK);
    uint64_t count = 1;
    for (uint32_t k = 0; k < rank; k++) {
        if (dims[k] < 0) return ir_fail(r, "an output of a negative size");
        if (dims[k] && count > UINT64_MAX / (uint64_t)dims[k]) return ir_fail(r, "an output too large");
        count *= (uint64_t)dims[k];
    }
    const size_t size = ir_dtype_size(dtype);
    if (!size) return ir_fail(r, "an output of %s", ir_dtype_name(dtype));
    if (count > ((uint64_t)1 << 40) / size) return ir_fail(r, "an output of %llu values", (unsigned long long)count);
    ir_release(r->g, t);
    t->data = ir_pool_take(r->g, count * size, &t->capacity);
    if (!t->data) return ir_fail(r, "out of memory for %llu values", (unsigned long long)count);
    t->owned = true;
    t->constant = false;
    t->dtype = dtype;
    t->rank = (uint8_t)rank;
    for (uint32_t k = 0; k < IR_MAX_RANK; k++) t->dims[k] = k < rank ? dims[k] : 0;
    t->count = count;
    return true;
}

int64_t ir_get_int(const IrTensor *t, uint64_t i) {
    switch (t->dtype) {
        case SPINGALETT_DTYPE_INT64: return ((const int64_t *)t->data)[i];
        case SPINGALETT_DTYPE_INT32: return ((const int32_t *)t->data)[i];
        case SPINGALETT_DTYPE_INT8: return ((const int8_t *)t->data)[i];
        case SPINGALETT_DTYPE_UINT8: case SPINGALETT_DTYPE_BOOL: return ((const uint8_t *)t->data)[i];
        case SPINGALETT_DTYPE_FLOAT32: return (int64_t)((const float *)t->data)[i];
        default: return 0;
    }
}

double ir_get_float(const IrTensor *t, uint64_t i) {
    if (t->dtype == SPINGALETT_DTYPE_FLOAT32) return ((const float *)t->data)[i];
    return (double)ir_get_int(t, i);
}

/* ------------------------------------------------------------------------- opening */

/* A constant of float16 or bfloat16 widened to float32 (in the graph's memory). */
static bool widen(SpingalettGraph *g, IrValue *v) {
    uint64_t count = 1;
    for (uint32_t k = 0; k < v->rank; k++) count *= (uint64_t)v->dims[k];
    float *f = (float *)ir_alloc(g, (size_t)count * 4u);
    if (!f) return false;
    const uint16_t *h = (const uint16_t *)v->data;
    for (uint64_t i = 0; i < count; i++)
        f[i] = v->dtype == SPINGALETT_DTYPE_FLOAT16 ? spingalett_fp16_to_float(h[i]) : spingalett_bf16_to_float(h[i]);
    v->source = v->data;
    v->source_dtype = v->dtype;
    v->data = f;
    v->dtype = SPINGALETT_DTYPE_FLOAT32;
    return true;
}

static uint64_t value_count_of(const IrValue *v) {
    uint64_t count = 1;
    for (uint32_t k = 0; k < v->rank; k++) count *= (uint64_t)(v->dims[k] < 0 ? 0 : v->dims[k]);
    return count;
}

static bool fail_open(const char *format, ...) {
    char message[300];
    va_list args;
    va_start(args, format);
    vsnprintf(message, sizeof message, format, args);
    va_end(args);
    set_error(SPINGALETT_ERR_INVALID, message);
    return false;
}

/* The nodes in an order that runs each after the nodes whose outputs it reads (ONNX's files are in one; a graph
   that has a cycle is refused). */
static bool order_nodes(SpingalettGraph *g) {
    const uint32_t N = g->node_count;
    int32_t *producer = (int32_t *)malloc((size_t)(g->value_count ? g->value_count : 1u) * sizeof(int32_t));
    uint32_t *order = (uint32_t *)malloc((size_t)(N ? N : 1u) * sizeof(uint32_t));
    uint8_t *state = (uint8_t *)calloc(N ? N : 1u, 1);
    bool ok = producer && order && state;
    for (uint32_t v = 0; ok && v < g->value_count; v++) producer[v] = -1;
    for (uint32_t n = 0; ok && n < N; n++)
        for (uint32_t o = 0; o < g->nodes[n].output_count; o++) {
            const uint32_t v = g->nodes[n].outputs[o];
            if (v == IR_NONE) continue;
            if (producer[v] >= 0 || g->values[v].kind != VALUE_NODE) {
                ok = fail_open("graph: the value '%s' is computed twice", g->values[v].name);
                break;
            }
            producer[v] = (int32_t)n;
        }
    /* the order: a node once every producer of its inputs is placed (stable for files already in order) */
    uint32_t placed = 0;
    while (ok && placed < N) {
        bool progress = false;
        for (uint32_t n = 0; n < N; n++) {
            if (state[n]) continue;
            bool ready = true;
            for (uint32_t i = 0; i < g->nodes[n].input_count && ready; i++) {
                const uint32_t v = g->nodes[n].inputs[i];
                if (v == IR_NONE) continue;
                if (g->values[v].kind == VALUE_NODE && producer[v] < 0) {
                    ok = fail_open("graph: the node '%s' (%s) reads '%s', which nothing computes", g->nodes[n].name,
                                   ir_op_name((IrOp)g->nodes[n].op), g->values[v].name);
                    break;
                }
                ready = g->values[v].kind != VALUE_NODE || state[producer[v]];
            }
            if (!ok) break;
            if (ready) {
                state[n] = 1;
                order[placed++] = n;
                progress = true;
            }
        }
        if (ok && !progress) ok = fail_open("graph: its nodes read each other in a cycle");
    }
    if (ok) {
        IrNode *sorted = (IrNode *)malloc((size_t)(N ? N : 1u) * sizeof(IrNode));
        if (!sorted) ok = fail_open("graph: out of memory");
        else {
            for (uint32_t k = 0; k < N; k++) sorted[k] = g->nodes[order[k]];
            free(g->nodes);
            g->nodes = sorted;
            g->node_cap = N ? N : 1u;
        }
    }
    free(producer);
    free(order);
    free(state);
    return ok;
}

/* The kernel of one node on tensors; false with the error set. */
static bool run_node(SpingalettGraph *g, uint32_t index, IrTensor *tensors, bool parallel) {
    const IrNode *n = &g->nodes[index];
    IrTensor *in[64], *out[64];
    if (n->input_count > 64 || n->output_count > 64)
        return fail_open("graph: the node '%s' (%s) has more than 64 inputs or outputs", n->name, ir_op_name((IrOp)n->op));
    for (uint32_t i = 0; i < n->input_count; i++) in[i] = n->inputs[i] == IR_NONE ? NULL : &tensors[n->inputs[i]];
    for (uint32_t o = 0; o < n->output_count; o++) out[o] = n->outputs[o] == IR_NONE ? NULL : &tensors[n->outputs[o]];
    IrRun r;
    memset(&r, 0, sizeof r);
    r.g = g;
    r.node = n;
    r.index = index;
    r.parallel = parallel;
    if (!ir_kernels[n->op](&r, in, n->input_count, out, n->output_count)) {
        char message[400];
        snprintf(message, sizeof message, "graph: the node '%s' (%s): %s", n->name, ir_op_name((IrOp)n->op), r.error);
        set_error(SPINGALETT_ERR_INVALID, message);
        return false;
    }
    return true;
}

/* A constant's tensor (a view of its data). */
static IrTensor const_tensor(const IrValue *v) {
    IrTensor t;
    memset(&t, 0, sizeof t);
    t.dtype = v->dtype;
    t.rank = v->rank;
    for (uint32_t k = 0; k < v->rank; k++) t.dims[k] = v->dims[k];
    t.count = value_count_of(v);
    t.data = v->data;
    t.constant = true;
    return t;
}

/* Nodes whose inputs are all constants computed now, their outputs made constants (small ones: a folded tensor of
   more than 2^26 values, an Expand or a ConstantOfShape of an activation's size, stays a node). */
static bool fold_constants(SpingalettGraph *g) {
    const uint32_t total = g->node_count;
    uint32_t kept = 0;
    for (uint32_t k = 0; k < g->node_count; k++) {
        IrNode *n = &g->nodes[k];
        bool constant = true;
        for (uint32_t i = 0; i < n->input_count && constant; i++)
            constant = n->inputs[i] == IR_NONE || g->values[n->inputs[i]].kind == VALUE_CONST;
        bool folded = false;
        if (constant) {
            IrTensor *tensors = (IrTensor *)calloc(g->value_count, sizeof(IrTensor));
            if (!tensors) return fail_open("graph: out of memory");
            for (uint32_t i = 0; i < n->input_count; i++)
                if (n->inputs[i] != IR_NONE) tensors[n->inputs[i]] = const_tensor(&g->values[n->inputs[i]]);
            if (!run_node(g, k, tensors, false)) {
                /* (a node that fails on constants would fail at every run: the graph is refused now) */
                for (uint32_t v = 0; v < g->value_count; v++)
                    ir_release(g, &tensors[v]);
                free(tensors);
                return false;
            }
            folded = true;
            for (uint32_t o = 0; folded && o < n->output_count; o++) {
                const uint32_t v = n->outputs[o];
                if (v == IR_NONE) continue;
                IrTensor *t = &tensors[v];
                folded = t->count <= ((uint64_t)1 << 26);
            }
            if (folded) {
                for (uint32_t o = 0; o < n->output_count; o++) {
                    const uint32_t v = n->outputs[o];
                    if (v == IR_NONE) continue;
                    IrTensor *t = &tensors[v];
                    IrValue *value = &g->values[v];
                    const size_t bytes = (size_t)t->count * ir_dtype_size(t->dtype);
                    value->data = ir_alloc(g, bytes);
                    if (!value->data) {
                        folded = false;
                        break;
                    }
                    if (bytes) memcpy(value->data, t->data, bytes);
                    value->kind = VALUE_CONST;
                    value->dtype = t->dtype;
                    value->rank = t->rank;
                    value->declared = true;
                    for (uint32_t d = 0; d < IR_MAX_RANK; d++) value->dims[d] = d < t->rank ? t->dims[d] : -1;
                }
            }
            for (uint32_t v = 0; v < g->value_count; v++)
                ir_release(g, &tensors[v]);
            free(tensors);
        }
        if (!folded) g->nodes[kept++] = *n;
    }
    g->node_count = kept;
    /* (what kernels prepared while folding belongs to nodes' old places) */
    for (uint32_t k = 0; g->prepared && k < total; k++)
        if (g->prepared[k]) g->prepared_free[k](g->prepared[k]);
    free(g->prepared);
    free(g->prepared_free);
    g->prepared = NULL;
    g->prepared_free = NULL;
    return true;
}

/* Nodes none of whose outputs an output of the graph needs left out. */
static bool drop_dead(SpingalettGraph *g) {
    uint8_t *needed = (uint8_t *)calloc(g->value_count ? g->value_count : 1u, 1);
    if (!needed) return fail_open("graph: out of memory");
    for (uint32_t o = 0; o < g->output_count; o++) needed[g->outputs[o]] = 1;
    uint8_t *live = (uint8_t *)calloc(g->node_count ? g->node_count : 1u, 1);
    if (!live) {
        free(needed);
        return fail_open("graph: out of memory");
    }
    for (uint32_t k = g->node_count; k-- > 0;) {
        const IrNode *n = &g->nodes[k];
        for (uint32_t o = 0; o < n->output_count && !live[k]; o++) live[k] = n->outputs[o] != IR_NONE && needed[n->outputs[o]];
        if (live[k])
            for (uint32_t i = 0; i < n->input_count; i++)
                if (n->inputs[i] != IR_NONE) needed[n->inputs[i]] = 1;
    }
    uint32_t kept = 0;
    for (uint32_t k = 0; k < g->node_count; k++)
        if (live[k]) g->nodes[kept++] = g->nodes[k];
    g->node_count = kept;
    free(live);
    free(needed);
    return true;
}

bool ir_finish(SpingalettGraph *g) {
    for (uint32_t v = 0; v < g->value_count; v++) {
        IrValue *value = &g->values[v];
        if (value->kind == VALUE_CONST &&
            (value->dtype == SPINGALETT_DTYPE_FLOAT16 || value->dtype == SPINGALETT_DTYPE_BFLOAT16) && !widen(g, value))
            return false;
    }
    for (uint32_t o = 0; o < g->output_count; o++)
        if (g->outputs[o] == IR_NONE) return fail_open("graph: an output nothing computes");
    if (!order_nodes(g) || !fold_constants(g) || !drop_dead(g) || !ir_fuse(g)) return false;
    for (uint32_t v = 0; v < g->value_count; v++) g->values[v].last = -1;
    for (uint32_t k = 0; k < g->node_count; k++)
        for (uint32_t i = 0; i < g->nodes[k].input_count; i++)
            if (g->nodes[k].inputs[i] != IR_NONE) g->values[g->nodes[k].inputs[i]].last = (int32_t)k;
    for (uint32_t o = 0; o < g->output_count; o++) g->values[g->outputs[o]].last = INT32_MAX;
    return true;
}

/* ------------------------------------------------------------------------- the public functions */

SpingalettGraph *spingalett_graph_open_args(SpingalettGraphOpenArgs args) {
    SpgFileView file;
    memset(&file, 0, sizeof file);
    const uint8_t *data = (const uint8_t *)args.data;
    size_t size = args.size;
    char *dir = NULL;
    if (args.path) {
        if (!spingalett_file_open(&file, args.path)) return NULL;
        data = file.data;
        size = file.size;
        /* the directory of ONNX's external data files */
        const char *slash = strrchr(args.path, '/');
#if defined(_WIN32)
        const char *back = strrchr(args.path, '\\');
        if (back && (!slash || back > slash)) slash = back;
#endif
        const size_t n = slash ? (size_t)(slash - args.path) : 1u;
        dir = (char *)malloc(n + 1);
        if (dir) {
            memcpy(dir, slash ? args.path : ".", n);
            dir[n] = '\0';
        }
    } else if (!data || !size) {
        set_error(SPINGALETT_ERR_INVALID, "graph: no path, and no data");
        return NULL;
    }
    /* (bytes given in memory are copied: the graph's constants may lie in them, and they need not outlive the call) */
    void *copy = NULL;
    if (!file.data) {
        copy = spingalett_aligned_alloc(size);
        if (!copy) {
            set_error(SPINGALETT_ERR_ALLOC, "graph: out of memory");
            return NULL;
        }
        memcpy(copy, data, size);
        data = (const uint8_t *)copy;
    }
    SpingalettGraph *g;
    if (size >= 8 && !memcmp(data, "SLETTM", 6)) {
        g = ir_read_slett(data, size);
    } else {
        g = ir_read_onnx(data, size, dir);
    }
    free(dir);
    if (g && file.data) {
        g->file = file;             /* (the graph's constants may lie in it) */
    } else if (file.data) {
        spingalett_file_close(&file);
    }
    if (g && copy) {
        g->copy = copy;
    } else {
        spingalett_aligned_free(copy);
    }
    return g;
}

void spingalett_graph_free(SpingalettGraph *graph) {
    ir_graph_free(graph);
}

uint32_t spingalett_graph_input_count(const SpingalettGraph *graph) {
    return graph ? graph->input_count : 0;
}

uint32_t spingalett_graph_output_count(const SpingalettGraph *graph) {
    return graph ? graph->output_count : 0;
}

static SpingalettTensorInfo info_of(const SpingalettGraph *g, uint32_t v) {
    SpingalettTensorInfo info;
    memset(&info, 0, sizeof info);
    const IrValue *value = &g->values[v];
    info.name = value->name;
    info.dtype = (SpingalettDType)value->dtype;
    info.rank = value->rank;
    for (uint32_t k = 0; k < value->rank; k++) {
        info.shape[k] = value->dims[k];
        info.symbols[k] = value->sym[k] ? g->symbols[value->sym[k] - 1u] : NULL;
    }
    return info;
}

SpingalettTensorInfo spingalett_graph_input(const SpingalettGraph *graph, uint32_t i) {
    if (!graph || i >= graph->input_count) {
        SpingalettTensorInfo none;
        memset(&none, 0, sizeof none);
        return none;
    }
    return info_of(graph, graph->inputs[i]);
}

SpingalettTensorInfo spingalett_graph_output(const SpingalettGraph *graph, uint32_t i) {
    if (!graph || i >= graph->output_count) {
        SpingalettTensorInfo none;
        memset(&none, 0, sizeof none);
        return none;
    }
    return info_of(graph, graph->outputs[i]);
}

uint32_t spingalett_graph_node_count(const SpingalettGraph *graph) {
    return graph ? graph->node_count : 0;
}

const char *spingalett_graph_node_op(const SpingalettGraph *graph, uint32_t i) {
    return graph && i < graph->node_count ? ir_op_name((IrOp)graph->nodes[i].op) : NULL;
}

bool spingalett_graph_save(const SpingalettGraph *graph, const char *path) {
    if (!graph || !path) {
        set_error(SPINGALETT_ERR_INVALID, "graph save: no graph or no path");
        return false;
    }
    return ir_write_slett(graph, path);
}

/* Input i's tensor of a run from what the caller gives: its shape against the declared one (symbols the same size
   everywhere), float16 and bfloat16 widened. */
static bool bind_input(SpingalettGraph *g, uint32_t v, const SpingalettTensor *t, int64_t *symbol_sizes) {
    const IrValue *value = &g->values[v];
    IrTensor *x = &g->tensors[v];
    if (t->rank > IR_MAX_RANK || (value->declared && t->rank != value->rank))
        return fail_open("graph run: the input '%s' has %u axes, not %u", value->name, t->rank, value->rank);
    uint64_t count = 1;
    for (uint32_t k = 0; k < t->rank; k++) {
        if (t->shape[k] < 0) return fail_open("graph run: the input '%s' has a negative size", value->name);
        if (value->declared && value->dims[k] >= 0 && value->dims[k] != t->shape[k])
            return fail_open("graph run: the input '%s' has size %lld on axis %u, not %lld", value->name,
                             (long long)t->shape[k], k, (long long)value->dims[k]);
        if (value->declared && value->sym[k]) {
            int64_t *s = &symbol_sizes[value->sym[k] - 1u];
            if (*s >= 0 && *s != t->shape[k])
                return fail_open("graph run: the input '%s' gives '%s' the size %lld, %lld elsewhere", value->name,
                                 g->symbols[value->sym[k] - 1u], (long long)t->shape[k], (long long)*s);
            *s = t->shape[k];
        }
        count *= (uint64_t)t->shape[k];
    }
    if (count && !t->data) return fail_open("graph run: the input '%s' has no data", value->name);
    const uint8_t want = value->dtype;
    x->rank = (uint8_t)t->rank;
    for (uint32_t k = 0; k < IR_MAX_RANK; k++) x->dims[k] = k < t->rank ? t->shape[k] : 0;
    x->count = count;
    if (want == SPINGALETT_DTYPE_FLOAT16 || want == SPINGALETT_DTYPE_BFLOAT16) {
        /* float32 as it is, or the declared type's bits widened */
        if (t->dtype == SPINGALETT_DTYPE_FLOAT32) {
            x->dtype = SPINGALETT_DTYPE_FLOAT32;
            x->data = t->data;
            return true;
        }
        if (t->dtype != want)
            return fail_open("graph run: the input '%s' is %s, not %s or float32", value->name, ir_dtype_name(t->dtype),
                             ir_dtype_name(want));
        float *f = (float *)ir_pool_take(g, count * 4u, &x->capacity);
        if (!f) return fail_open("graph run: out of memory");
        const uint16_t *h = (const uint16_t *)t->data;
        for (uint64_t i = 0; i < count; i++)
            f[i] = want == SPINGALETT_DTYPE_FLOAT16 ? spingalett_fp16_to_float(h[i]) : spingalett_bf16_to_float(h[i]);
        x->dtype = SPINGALETT_DTYPE_FLOAT32;
        x->data = f;
        x->owned = true;
        return true;
    }
    if (value->declared && t->dtype != want)
        return fail_open("graph run: the input '%s' is %s, not %s", value->name, ir_dtype_name(t->dtype),
                         ir_dtype_name(want));
    if (!ir_dtype_size((uint8_t)t->dtype)) return fail_open("graph run: the input '%s' is of %s", value->name,
                                                            ir_dtype_name((uint8_t)t->dtype));
    x->dtype = (uint8_t)t->dtype;
    x->data = t->data;
    return true;
}

static uint32_t named(const SpingalettGraph *g, const uint32_t *list, uint32_t count, const char *name) {
    for (uint32_t k = 0; k < count; k++)
        if (!strcmp(g->values[list[k]].name, name)) return list[k];
    return IR_NONE;
}

bool spingalett_graph_run_args(SpingalettGraphRunArgs args) {
    SpingalettGraph *g = args.graph;
    if (!g || (args.input_count && !args.inputs) || (args.output_count && !args.outputs)) {
        set_error(SPINGALETT_ERR_INVALID, "graph run: no graph, or no inputs or outputs where their counts are given");
        return false;
    }
    spg_lock((SpgSignal *)g->lock);
    free_run(g);
    g->runs++;
    bool ok = true;
    g->tensors = (IrTensor *)calloc(g->value_count ? g->value_count : 1u, sizeof(IrTensor));
    int64_t *symbol_sizes = (int64_t *)malloc((size_t)(g->symbol_count + 1u) * sizeof(int64_t));
    uint8_t *bound = (uint8_t *)calloc(g->input_count + 1u, 1);
    if (!g->tensors || !symbol_sizes || !bound) ok = fail_open("graph run: out of memory");
    for (uint32_t s = 0; ok && s < g->symbol_count; s++) symbol_sizes[s] = -1;
    /* the inputs, by name or in order */
    for (uint32_t k = 0; ok && k < args.input_count; k++) {
        const SpingalettTensor *t = &args.inputs[k];
        uint32_t v = t->name ? named(g, g->inputs, g->input_count, t->name) : k < g->input_count ? g->inputs[k] : IR_NONE;
        if (v == IR_NONE) {
            ok = fail_open("graph run: the graph has no input '%s'", t->name ? t->name : "(by order: too many inputs)");
            break;
        }
        for (uint32_t i = 0; i < g->input_count; i++)
            if (g->inputs[i] == v) bound[i] = 1;
        ok = bind_input(g, v, t, symbol_sizes);
    }
    for (uint32_t i = 0; ok && i < g->input_count; i++)
        if (!bound[i] && g->values[g->inputs[i]].last >= 0)
            ok = fail_open("graph run: the input '%s' is not given", g->values[g->inputs[i]].name);
    /* constants as views */
    for (uint32_t v = 0; ok && v < g->value_count; v++)
        if (g->values[v].kind == VALUE_CONST) g->tensors[v] = const_tensor(&g->values[v]);
    /* the nodes, each value freed after its last reader */
    const bool parallel = spingalett_get_compute_mode() == SPINGALETT_COMPUTE_OPENMP && spingalett_get_num_threads() > 1u;
    if (!g->profile && getenv("SPINGALETT_GRAPH_PROFILE") && *getenv("SPINGALETT_GRAPH_PROFILE") != '0') {
        g->profile = (double *)calloc(IR_OP_COUNT, sizeof(double));
        g->profile_nodes = (uint64_t *)calloc(IR_OP_COUNT, sizeof(uint64_t));
        if (!g->profile || !g->profile_nodes) {
            free(g->profile);
            free(g->profile_nodes);
            g->profile = NULL;
            g->profile_nodes = NULL;
        }
    }
    for (uint32_t k = 0; ok && k < g->node_count; k++) {
        const IrNode *n = &g->nodes[k];
        /* (the first run, which also prepares constants, is left out) */
        const bool timed = g->profile && g->runs > 1;
        const double t0 = timed ? seconds() : 0.0;
        ok = run_node(g, k, g->tensors, parallel);
        if (timed) {
            g->profile[n->op] += seconds() - t0;
            g->profile_nodes[n->op]++;
        }
        for (uint32_t i = 0; ok && i < n->input_count; i++) {
            const uint32_t v = n->inputs[i];
            if (v != IR_NONE && g->values[v].last == (int32_t)k) ir_release(g, &g->tensors[v]);
        }
    }
    /* the outputs, by name or in order */
    for (uint32_t k = 0; ok && k < args.output_count; k++) {
        SpingalettTensor *t = &args.outputs[k];
        uint32_t v = t->name ? named(g, g->outputs, g->output_count, t->name)
                             : k < g->output_count ? g->outputs[k] : IR_NONE;
        if (v == IR_NONE) {
            ok = fail_open("graph run: the graph has no output '%s'", t->name ? t->name : "(by order: too many outputs)");
            break;
        }
        IrTensor *x = &g->tensors[v];
        const uint8_t declared = g->values[v].dtype;
        if (!x->owned && x->data && x->count) {
            /* an output that is a constant or a graph's input (or a view of one): a copy in the graph's memory */
            size_t capacity;
            const size_t bytes = (size_t)x->count * ir_dtype_size(x->dtype);
            void *copy = ir_pool_take(g, bytes, &capacity);
            if (!copy) {
                ok = fail_open("graph run: out of memory");
                break;
            }
            memcpy(copy, x->data, bytes);
            x->data = copy;
            x->capacity = capacity;
            x->owned = true;
        }
        if (x->dtype == SPINGALETT_DTYPE_FLOAT32 && x->data &&
            (declared == SPINGALETT_DTYPE_FLOAT16 || declared == SPINGALETT_DTYPE_BFLOAT16)) {
            /* a float16 or bfloat16 output narrowed back (rounded to nearest even) */
            size_t capacity;
            uint16_t *h = (uint16_t *)ir_pool_take(g, x->count * 2u, &capacity);
            if (!h) {
                ok = fail_open("graph run: out of memory");
                break;
            }
            const float *f = (const float *)x->data;
            for (uint64_t i = 0; i < x->count; i++)
                h[i] = declared == SPINGALETT_DTYPE_FLOAT16 ? spingalett_float_to_fp16(f[i]) : spingalett_float_to_bf16(f[i]);
            ir_release(g, x);
            x->data = h;
            x->capacity = capacity;
            x->owned = true;
            x->dtype = declared;
        }
        t->dtype = (SpingalettDType)x->dtype;
        t->rank = x->rank;
        for (uint32_t d = 0; d < IR_MAX_RANK; d++) t->shape[d] = d < x->rank ? x->dims[d] : 0;
        t->data = x->data;
    }
    free(symbol_sizes);
    free(bound);
    pool_trim(g);
    spg_unlock((SpgSignal *)g->lock);
    return ok;
}
