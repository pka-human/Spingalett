/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Graphs of operators (the IR): what the graph's sources share. A graph is values (its inputs, its constants and
 * what its nodes compute), nodes in an order that runs every node after the nodes it reads, and typed attributes.
 * Spingalett.IR.c builds, checks, simplifies and runs graphs and holds the public functions; Spingalett.IROps.c the
 * operators' kernels; Spingalett.IROnnx.c reads ONNX files and Spingalett.IRFile.c `.slett` files of format 11.
 *
 * Floats compute in single precision: constants of float16 and bfloat16 become float32 when a graph opens, inputs of
 * those types when a run takes them, outputs return to them. Integers (int64, int32, int8, uint8) and booleans keep
 * their types (shapes, indices, masks).
 */

#pragma once

#include "Spingalett.Private.h"

#define IR_MAX_RANK SPINGALETT_MAX_RANK
#define IR_NONE UINT32_MAX                  /* an optional input left out */

/* The operators: their names as ONNX writes them (the library's own in the domain "spingalett"). */
typedef enum {
#define IR_OP(name, kernel) IR_##name,
#include "Spingalett.IROps.def"
#undef IR_OP
    IR_OP_COUNT
} IrOp;

typedef enum { ATTR_INT, ATTR_FLOAT, ATTR_INTS, ATTR_FLOATS, ATTR_STRING, ATTR_TENSOR } IrAttrType;

typedef struct IrConst IrConst;

typedef struct {
    char *name;
    uint8_t type;                   /* IrAttrType */
    int64_t i;
    float f;
    int64_t *ints;
    float *floats;
    uint32_t count;                 /* ints or floats */
    char *s;
    uint32_t tensor;                /* ATTR_TENSOR: the value holding it */
} IrAttr;

typedef struct {
    uint16_t op;                    /* IrOp */
    uint32_t *inputs, *outputs;     /* value indices (inputs: IR_NONE for an optional one left out) */
    uint32_t input_count, output_count;
    IrAttr *attrs;
    uint32_t attr_count;
    char *name;
} IrNode;

enum { VALUE_INPUT, VALUE_CONST, VALUE_NODE };

typedef struct {
    char *name;
    uint8_t dtype;                  /* SpingalettDType */
    uint8_t rank;
    uint8_t kind;                   /* VALUE_INPUT, VALUE_CONST, VALUE_NODE */
    bool declared;                  /* dtype and shape known (inputs, constants, declared outputs) */
    int64_t dims[IR_MAX_RANK];      /* declared sizes; -1 for a symbol or an unknown size */
    uint16_t sym[IR_MAX_RANK];      /* a symbol's index + 1, 0 for none */
    void *data;                     /* constants: their values in dtype, row-major (the graph's memory) */
    const void *source;             /* constants widened to float32: their values as the source held them, */
    uint8_t source_dtype;           /* which files of format 11 keep (NULL: none) */
    int32_t last;                   /* the last node that reads it (-1: none); graph outputs keep theirs to the end */
} IrValue;

/* A tensor of a run. */
typedef struct {
    uint8_t dtype;
    uint8_t rank;
    int64_t dims[IR_MAX_RANK];
    uint64_t count;                 /* values */
    void *data;
    bool owned;                     /* the run's: its block of the graph's pool (ir_release()) */
    bool constant;                  /* a constant of the graph (kernels may prepare it once, by node) */
    bool staged;                    /* its values in memory of the GPU's plan (Spingalett.GpuGraph.c), valid until the
                                       next run */
    size_t capacity;                /* owned: the bytes of its block */
} IrTensor;

/* A block of memory of the graph's pool: tensors of runs take blocks and give them back, so that runs after the
   first allocate nothing (blocks no run used in the last one are freed after it). */
typedef struct {
    void *data;
    size_t bytes;
    uint64_t used;                  /* the run that last took it */
    uint32_t refs;                  /* tensors holding it (a view holds its source's block too); 0: free */
} IrBlock;

struct SpingalettGraph {
    IrValue *values;
    uint32_t value_count, value_cap;
    uint32_t *names;                /* a hash table of the values' names: value index + 1, 0 for an empty slot */
    uint32_t name_cap;              /* (a power of two, at least twice the values) */
    IrNode *nodes;
    uint32_t node_count, node_cap;
    uint32_t *inputs, *outputs;     /* value indices */
    uint32_t input_count, output_count;
    char **symbols;
    uint32_t symbol_count;
    int64_t opset;                  /* ONNX's default domain's version */
    char *producer;
    /* memory the values' data and names live in */
    void **blocks;
    uint32_t block_count, block_cap;
    SpgFileView file;               /* a mapped file whose bytes constants may point into */
    void *copy;                     /* or the copy of bytes given in memory */
    SpgFileView *externals;         /* ONNX's external data files, and their paths */
    char **external_paths;
    uint32_t external_count;
    /* runs */
    void *lock;
    IrTensor *tensors;              /* [value_count] of the last run */
    IrBlock *pool;
    uint32_t pool_count, pool_cap;
    uint64_t runs;                  /* runs begun (folding constants is run 0) */
    double *profile;                /* SPINGALETT_GRAPH_PROFILE: seconds in each operator, and its nodes run */
    uint64_t *profile_nodes;
    void **prepared;                /* [node_count]: what a node's kernel made once from its constant inputs */
    void (**prepared_free)(void *); /* and how to free it */
    SpingalettGemmScratch *scratch; /* products on all the threads */
    SpingalettGemmScratch **thread_scratch;     /* products of one thread each, side by side (batched MatMul) */
    int scratch_threads;
    void *gpu;                      /* the graph on the GPU (Spingalett.GpuGraph.c), NULL until a run there */
    char *error;
};

/* Memory of the graph's own, freed with it; NULL with the error set when it runs out. */
void *ir_alloc(SpingalettGraph *g, size_t bytes);
char *ir_strdup(SpingalettGraph *g, const char *s, size_t n);
/* A new value (its name copied), its index; IR_NONE when memory runs out. */
uint32_t ir_value(SpingalettGraph *g, const char *name, size_t name_len);
/* The value of a name, IR_NONE when none has it. */
uint32_t ir_find_value(const SpingalettGraph *g, const char *name, size_t name_len);
/* A new node of op (inputs and outputs given later), its index; IR_NONE when memory runs out. */
uint32_t ir_node(SpingalettGraph *g, IrOp op);
/* A symbol's index + 1. */
uint16_t ir_symbol(SpingalettGraph *g, const char *name, size_t len);
/* The operator of an ONNX op_type in a domain ("" and "ai.onnx" are ONNX's), IR_OP_COUNT when the library has none. */
IrOp ir_op_of(const char *op_type, size_t len, const char *domain, size_t domain_len);
const char *ir_op_name(IrOp op);
/* Bytes of a value of dtype; 0 for a dtype the graphs do not hold. */
size_t ir_dtype_size(uint8_t dtype);
const char *ir_dtype_name(uint8_t dtype);

/* After the sources built it: values of float16 and bfloat16 as float32, the order of the nodes checked (each
   after what it reads), constants folded, nodes no output needs left out, the values' last readers found. false
   with the error set. */
bool ir_finish(SpingalettGraph *g);
/* The fusions of Spingalett.IRFuse.c; false (out of memory) with the error set. */
bool ir_fuse(SpingalettGraph *g);

/* An attribute of a node, NULL when it has none of that name. */
const IrAttr *ir_attr(const IrNode *n, const char *name);
int64_t ir_attr_int(const IrNode *n, const char *name, int64_t fallback);
float ir_attr_float(const IrNode *n, const char *name, float fallback);

/* A run's state for kernels: the graph, its node (and the slot of what the node prepares once), its threads, and
   where errors go. */
typedef struct {
    SpingalettGraph *g;
    const IrNode *node;
    uint32_t index;                 /* the node's */
    bool parallel;                  /* OpenMP threads for large loops */
    char error[256];
} IrRun;

/* A kernel: the node's outputs (their shapes from its inputs' shapes and, for shapes held in tensors, values; their
   memory from ir_output()) from its inputs (NULL for an optional input left out). false with r->error set. */
typedef bool (*IrKernel)(IrRun *r, IrTensor *const *in, uint32_t in_count, IrTensor **out, uint32_t out_count);

/* The scratch of products on all the threads, or of thread t's products of one thread; NULL with the error set. */
SpingalettGemmScratch *ir_scratch(IrRun *r);
SpingalettGemmScratch *ir_thread_scratch(IrRun *r, int t);
/* Output t made of dtype and shape, its memory a block of the graph's pool, of zeros (ir_output_raw: of whatever
   the block held, for kernels that write every value); false with the error set. A tensor that already owns a block
   gives it back first. */
bool ir_output(IrRun *r, IrTensor *t, uint8_t dtype, uint32_t rank, const int64_t *dims);
bool ir_output_raw(IrRun *r, IrTensor *t, uint8_t dtype, uint32_t rank, const int64_t *dims);
/* A tensor's block back to the pool (nothing for one that owns none). */
void ir_release(SpingalettGraph *g, IrTensor *t);
/* Output t made a view of x's values in another shape of as many values (a constant's view owns nothing). */
void ir_view(IrRun *r, IrTensor *t, const IrTensor *x, uint32_t rank, const int64_t *dims);
void *ir_pool_take(SpingalettGraph *g, size_t bytes, size_t *capacity);
void ir_pool_give(SpingalettGraph *g, void *data);
/* What node r->index made once from its constants (NULL before it made it), and the keeping of it: freed with the
   graph by free_fn. */
void *ir_prepared(IrRun *r);
void ir_prepare(IrRun *r, void *data, void (*free_fn)(void *));
bool ir_fail(IrRun *r, const char *format, ...);
/* A tensor's i-th value of an integer or boolean type (or float32) as int64, and of any type as float. */
int64_t ir_get_int(const IrTensor *t, uint64_t i);
double ir_get_float(const IrTensor *t, uint64_t i);
/* The table of kernels, by IrOp. */
extern const IrKernel ir_kernels[IR_OP_COUNT];
/* Node `index` run by its kernel on tensors[] (the run's, by value), on the OpenMP threads with parallel; false with
   the error set (the node named). */
bool ir_run_node(SpingalettGraph *g, uint32_t index, IrTensor *tensors, bool parallel);
/* A constant's tensor: a view of its values. */
IrTensor ir_const_tensor(const IrValue *v);

/* The windows of a convolution or pooling over one to three spatial axes (as three, the first ones of size 1 for
   fewer): sizes, kernel, strides, padding at the start and end, dilations, outputs. */
typedef struct {
    uint32_t spatial;               /* the input's spatial axes */
    int64_t in[3], k[3], s[3], pb[3], pe[3], dil[3], out[3];
} IrGeometry;
/* The windows of node r->node over x (kernel_shape, or the weights' given), its strides, dilations, pads or auto_pad;
   outputs by ceil_mode for pooling. false with the error set. */
bool ir_geometry(IrRun *r, const IrTensor *x, const int64_t *kernel, IrGeometry *g, bool ceil_mode);

/* Resize's (and Upsample's) output shape and taps: for every axis d, T[d] taps of each output index (their input
   indices, -1 for the extrapolation value, and their weights), in memory ir_resize_free() frees. */
typedef struct {
    int64_t dims[IR_MAX_RANK];
    int T[IR_MAX_RANK];
    int64_t *index[IR_MAX_RANK];
    float *weight[IR_MAX_RANK];
    float extrapolation;
    void *index_memory, *weight_memory;
} IrResizeTaps;
bool ir_resize_taps(IrRun *r, IrTensor *const *in, uint32_t n, IrResizeTaps *plan);
void ir_resize_free(IrResizeTaps *taps);

/* Readers of the sources: a graph from ONNX bytes (dir: the directory of external data files, or NULL), and from a
   `.slett` image of format 11. NULL with the error set. */
SpingalettGraph *ir_read_onnx(const uint8_t *data, size_t size, const char *dir);
SpingalettGraph *ir_read_slett(const uint8_t *data, size_t size);
bool ir_write_slett(const SpingalettGraph *g, const char *path);
SpingalettGraph *ir_graph_new(void);
void ir_graph_free(SpingalettGraph *g);
