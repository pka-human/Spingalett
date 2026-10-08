/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#pragma once

#include "Spingalett/Spingalett.h"
#include "Spingalett.Engine.h"
#include "Spingalett.Network.h"
#include <stdint.h>

#if defined(SPINGALETT_HAS_OPENBLAS)
#include <cblas.h>
#endif

#define SPINGALETT_ERRMSG_MAX 256
#define SPINGALETT_ALIGNMENT  64

/* Multiply-adds below which forking an OpenMP team costs more than it saves. */
#define SPINGALETT_OMP_MIN_WORK 32768u

static inline bool spingalett_use_omp(ComputeMode mode, uint64_t work) {
    return mode == COMPUTE_OPENMP && work >= SPINGALETT_OMP_MIN_WORK;
}

/* SPINGALETT_PARALLEL_FOR(cond, for (...) body) runs the loop on the OpenMP threads (static
   schedule) when cond holds, and as plain code otherwise: a parallel region whose if clause is
   false still costs about 0.2 us to enter, more than many of the loops it would guard. The _THREADS
   form also names the number of threads. Bodies cannot hold preprocessor lines; they get their
   thread's number from spingalett_thread_num(). */
#define SPINGALETT_PRAGMA(x) _Pragma(#x)
#if defined(_OPENMP)
#include <omp.h>
#define SPINGALETT_PARALLEL_FOR(cond, ...) \
    do { if (cond) { SPINGALETT_PRAGMA(omp parallel for schedule(static)) __VA_ARGS__ } else { __VA_ARGS__ } } while (0)
#define SPINGALETT_PARALLEL_FOR_THREADS(threads, cond, ...) \
    do { if (cond) { SPINGALETT_PRAGMA(omp parallel for schedule(static) num_threads(threads)) __VA_ARGS__ } \
         else { __VA_ARGS__ } } while (0)
static inline int spingalett_thread_num(void) { return omp_get_thread_num(); }
#else
#define SPINGALETT_PARALLEL_FOR(cond, ...) do { (void)(cond); __VA_ARGS__ } while (0)
#define SPINGALETT_PARALLEL_FOR_THREADS(threads, cond, ...) do { (void)(threads); (void)(cond); __VA_ARGS__ } while (0)
static inline int spingalett_thread_num(void) { return 0; }
#endif

#define SPINGALETT_NEURON(net, l, j)        ((net)->neurons[(net)->neuron_offsets[l] + (uint64_t)(j)])
#define SPINGALETT_LAYER_PTR(net, l)        ((net)->neurons + (net)->neuron_offsets[l])


#define SPINGALETT_WEIGHT_MTX_PTR(net, l)   ((net)->weights + (net)->weight_offsets[l])
#define SPINGALETT_GRAD_W_MTX_PTR(net, l)   ((net)->grad_weights + (net)->weight_offsets[l])

/* Error paths are rare: GCC and Clang keep them out of line, so that the thread-local error state
   is not touched in the bodies of the functions that report errors (GCC 16 with LTO dropped the
   stack realignment of a function that inlined it and still read its arguments through it). */
#if defined(__GNUC__)
#define SPINGALETT_COLD __attribute__((cold, noinline))
#else
#define SPINGALETT_COLD
#endif

SPINGALETT_COLD void set_error(int code, const char *msg);


/* Training-time dropout state. Masks are a hash of (seed, step, position, layer, unit), so every
   backend and thread count draws identical masks for the same sample. */
typedef struct {
    uint64_t seed;          /* drawn once per train() call */
    uint64_t step;          /* optimizer step the sample belongs to */
    uint32_t position;      /* sample index within that step */
    float   *dmask;         /* per-unit mask * f'(a), laid out like net->neurons */
} DropoutContext;

/* Samples per batch-path chunk: bounds workspace memory (full-batch training over a large dataset
   runs in chunks whose gradients are accumulated) while keeping the GEMMs large. Networks with more
   than SPINGALETT_BATCH_FLOATS / SPINGALETT_BATCH_CHUNK activations per sample (convolutions) take
   fewer samples per chunk, so that one chunk's activations stay near SPINGALETT_BATCH_FLOATS. */
#define SPINGALETT_BATCH_CHUNK 2048u
#define SPINGALETT_BATCH_FLOATS (1u << 24)

typedef struct SpingalettGemmScratch SpingalettGemmScratch;

/* Activations (and, when training, deltas and dropout masks) for up to `capacity` samples. */
typedef struct BatchWorkspace {
    float **act;            /* act[0]: inputs of the chunk; act[l]: [capacity x topology[l]]. For
                               inference, outputs that are never alive at the same time share
                               memory (spingalett_plan_outputs), and act[l] of a layer whose output
                               is never stored (a product with its normalization fused) is NULL */
    float **delta;          /* training only: delta[l] for l >= 1 */
    uint32_t *uses;         /* per layer: the inputs of later layers that name it */
    uint32_t *pending;      /* training: uses whose gradient has not arrived yet (backward pass) */
    float *gtmp;            /* training: a gradient contribution to a layer that feeds several
                               (capacity x its outputs), added to its delta */
    float **dmask;          /* training only: dropout mask * f'(a) per hidden layer with dropout */
    float *flat;            /* storage of act[1..] (and delta[1..]) */
    float *dmask_flat;
    float *inputs;          /* gather buffers for shuffled mini-batches, or NULL */
    float *targets;
    SpingalettGemmScratch *gemm;
    float *conv;            /* convolution windows (spingalett_conv_scratch_floats), or NULL */
    /* batch normalization: per normalizing layer l, training only: bn_stats[l] (4 x channels, see
       spingalett_bn_forward_train) and bn_sums[l] (2 x channels: the parameter gradients' sums);
       and scratch for any of them */
    float **bn_stats;
    double **bn_sums;
    double *bn_scratch;
    float *bn_coef;         /* 3 x the most channels */
    float *bn_flat;
    double *bn_sums_flat;
    bool training;          /* normalize with batch statistics */
    uint32_t capacity;
} BatchWorkspace;

BatchWorkspace *spingalett_batch_workspace_create(const NeuralNetwork *net, uint32_t capacity,
                                                  bool training, bool gather, ComputeMode mode);
void spingalett_batch_workspace_free(BatchWorkspace *ws);
/* Forward pass over N samples (ws->act[0] must point at them). `dropout` (or NULL) masks hidden
   layers; samples are numbered position_offset + s for the dropout hash. */
/* Samples per chunk for `count` samples: at most count, SPINGALETT_BATCH_CHUNK and what fits
   SPINGALETT_BATCH_FLOATS activations, at least 1. */
uint32_t spingalett_batch_capacity(const NeuralNetwork *net, uint32_t count);
void spingalett_batch_forward(NeuralNetwork *net, BatchWorkspace *ws, uint32_t N,
                              const DropoutContext *dropout, uint32_t position_offset, ComputeMode mode);
/* Inference: whether layer l (dense or convolution, no activation) feeds only the batch
   normalization l + 1, which then runs in its product's epilogue and l's output is never stored. */
bool spingalett_fused_norm(const NeuralNetwork *net, const uint32_t *uses, uint32_t l);

/* Graphs (Spingalett.Graph.c). Offsets of the outputs of an inference pass, per sample in floats
   (multiples of 16), so that outputs alive at the same time do not overlap: an output lives from
   the step that computes it to the last step that reads it. Layer 0 and the last layer (the
   caller's buffers) and the products whose normalization is fused take none (UINT64_MAX). Returns
   the floats per sample all outputs need. */
uint64_t spingalett_plan_outputs(const NeuralNetwork *net, const uint32_t *uses, uint64_t *offsets);
/* The same over `count` steps from a description of each: steps[s] is the output it computes,
   reads[s] the outputs it reads (read_count[s] of them), sizes[t] the floats of output t (0: not
   stored); output t is computed by step `producer[t]` or never. Generic so that deployment models
   and model files share it. offsets of outputs that are not stored are UINT64_MAX. */
uint64_t spingalett_plan_buffers(uint32_t outputs, const uint64_t *sizes, uint32_t count, const uint32_t *steps,
                                 const uint32_t *const *reads, const uint32_t *read_count, uint64_t align,
                                 uint64_t *offsets);
/* y = act(x[0] + x[1] + ...) over n samples of `size` floats, the inputs added in their order. */
void spingalett_add_forward(const float *const *x, uint32_t count, float *y, uint32_t n, uint64_t size,
                            ActivationFunction act, ComputeMode mode);
/* y = act(concatenation of x[k], channels[k] each) over n samples of `cells` cells. */
void spingalett_concat_forward(const float *const *x, const uint32_t *channels, uint32_t count, float *y, uint32_t n,
                               uint64_t cells, ActivationFunction act, ComputeMode mode);
/* y[c] = act(mean over the cells of x[., c]) for n samples of `cells` x C. */
void spingalett_global_pool_forward(const float *x, float *y, uint32_t n, uint64_t cells, uint32_t C,
                                    ActivationFunction act, ComputeMode mode);
/* dx (= dx + when accumulate) = the part of dy (rows of C channels) at channels [c0, c0 + ck), over
   n samples of `cells` cells: the gradient of a concatenation's input; with act, times the
   derivative read from x (that input's output; not with accumulate). An addition's input is the
   whole of dy (c0 = 0, ck = C). */
void spingalett_slice_backward(const float *dy, uint32_t C, uint32_t c0, uint32_t ck, float *dx, const float *x,
                               uint32_t n, uint64_t cells, ActivationFunction act, bool accumulate, ComputeMode mode);
/* dx (= dx + when accumulate) = dy[c] / cells at every cell, the gradient of a global pooling's
   input; with act, times the derivative read from x (not with accumulate). */
void spingalett_global_pool_backward(const float *dy, float *dx, const float *x, uint32_t n, uint64_t cells, uint32_t C,
                                     ActivationFunction act, bool accumulate, ComputeMode mode);
/* delta (rows of `size`) += add over n samples, then, with last, times dmask (when set) or the
   derivative of act read from y. */
void spingalett_gradient_sum(float *delta, const float *add, const float *y, const float *dmask, uint32_t n,
                             uint64_t size, ActivationFunction act, bool last, ComputeMode mode);
/* Summed loss and number of correctly classified samples (see EvalMetrics) over n samples, in
   chunks of ws->capacity. ws is an inference workspace; out_buf holds [capacity x output size]. */
void spingalett_batch_evaluate(NeuralNetwork *net, BatchWorkspace *ws, float *out_buf,
                               const float *inputs, const float *targets, uint32_t n, ComputeMode mode,
                               double *loss_sum, uint32_t *correct);

/* Convolution and pooling over batches of n channels-last samples (Spingalett.Conv.c); weight
   layer l connects layer l to l + 1. scratch holds spingalett_conv_scratch_floats() floats. */
size_t spingalett_conv_scratch_floats(const NeuralNetwork *net, uint32_t capacity, bool training, ComputeMode mode);
/* y = act(conv(x) + bias), act element-wise (ACT_SOFTMAX is left to the caller, like ACT_NONE) */
void spingalett_conv_forward(const NeuralNetwork *net, uint32_t l, const float *x, float *y, uint32_t n,
                             ActivationFunction act, float *scratch, SpingalettGemmScratch *gemm, ComputeMode mode);
/* dx = dL/dx * act'(x) from dy = dL/d(pre-activation of layer l + 1), x being layer l's output and
   act its activation (ACT_NONE: dL/dx itself); dx is overwritten */
void spingalett_conv_backward_data(const NeuralNetwork *net, uint32_t l, const float *dy, float *dx, uint32_t n,
                                   const float *x, ActivationFunction act, float *scratch,
                                   SpingalettGemmScratch *gemm, ComputeMode mode);
/* weight and bias gradients: g = scale * (sum over the batch) + beta * g */
void spingalett_conv_backward_weights(NeuralNetwork *net, uint32_t l, const float *x, const float *dy, uint32_t n,
                                      float scale, float beta, float *scratch, SpingalettGemmScratch *gemm,
                                      ComputeMode mode);
void spingalett_pool_forward(const NeuralNetwork *net, uint32_t l, const float *x, float *y, uint32_t n,
                             ComputeMode mode);
/* The forward passes from shapes and parameters alone (batched inference of models): W holds
   out->channels filters of kernel_h x kernel_w x in->channels weights; scratch holds
   spingalett_conv_forward_scratch() floats. */
size_t spingalett_conv_forward_scratch(const LayerShape *in, const LayerShape *out, uint32_t capacity,
                                       ComputeMode mode);
void spingalett_conv_forward_shapes(const LayerShape *in, const LayerShape *out, const float *W, const float *bias,
                                    const float *x, float *y, uint32_t n, ActivationFunction act, float *scratch,
                                    SpingalettGemmScratch *gemm, ComputeMode mode);
/* The same with the outputs scaled per filter before the bias (inference with a batch
   normalization folded in). */
void spingalett_conv_forward_scaled(const LayerShape *in, const LayerShape *out, const float *W, const float *scale,
                                    const float *bias, const float *x, float *y, uint32_t n, ActivationFunction act,
                                    float *scratch, SpingalettGemmScratch *gemm, ComputeMode mode);
void spingalett_pool_forward_shapes(const LayerShape *in, const LayerShape *out, const float *x, float *y, uint32_t n,
                                    ComputeMode mode);
/* dx = dL/dx * act'(x) from dy = dL/dy (x: the pooling layer's input, the output of layer l, whose
   activation is act); dx is overwritten */
void spingalett_pool_backward(const NeuralNetwork *net, uint32_t l, const float *x, const float *dy, float *dx,
                              uint32_t n, ActivationFunction act, ComputeMode mode);

/* Batch normalization of layer l + 1 over n samples of layer l's output x (Spingalett.Norm.c);
   gamma and beta are weight layer l's weights and biases. scratch holds
   spingalett_bn_scratch_doubles() doubles. */
size_t spingalett_bn_scratch_doubles(const NeuralNetwork *net, uint32_t capacity);
/* Training: y = act(gamma * xhat + beta) with the batch's statistics, kept in stats [4 x channels:
   mean, 1 / standard deviation, and the coefficients y = act(a x + b)] for the backward pass; the
   running statistics move towards them. */
void spingalett_bn_forward_train(NeuralNetwork *net, uint32_t l, const float *x, float *y, uint32_t n,
                                 ActivationFunction act, float *stats, double *scratch, ComputeMode mode);
/* Inference, with the running statistics (coef: 2 x channels floats of scratch). */
void spingalett_bn_forward(const NeuralNetwork *net, uint32_t l, const float *x, float *y, uint32_t n,
                           ActivationFunction act, float *coef, ComputeMode mode);
/* Per channel, from dy = dL/d(pre-activation of layer l + 1): sums [0, C) of dy (beta's gradient)
   and [C, 2C) of dy * xhat (gamma's) */
void spingalett_bn_backward_sums(const NeuralNetwork *net, uint32_t l, const float *x, const float *dy, uint32_t n,
                                 const float *stats, double *sums, double *scratch, ComputeMode mode);
/* dx = dL/dx * act'(x), act being layer l's activation (ACT_NONE: dL/dx itself); coef: 3 x channels
   floats of scratch */
void spingalett_bn_backward_data(const NeuralNetwork *net, uint32_t l, const float *x, const float *dy, float *dx,
                                 uint32_t n, const float *stats, const double *sums, ActivationFunction act,
                                 float *coef, ComputeMode mode);

bool spingalett_add_layer(LayerArgs args);
/* Whether every layer of net but the last feeds a later one; sets the error (naming `who`) when not. */
bool spingalett_check_graph(const NeuralNetwork *net, const char *who);
/* Gives an empty network room for these totals (zeroed), so that adding its layers moves nothing. */
bool spingalett_network_reserve(NeuralNetwork *net, uint64_t neurons, uint64_t weights, uint64_t biases);
/* Allocates the gradients and optimizer state (zero) unless they exist: networks get them when they
   first train, so that those used for inference only hold their parameters once. Sets the error. */
bool spingalett_training_state(NeuralNetwork *net);
/* The arguments that add layer l of net again (to another network: set .net), parameters aside. */
LayerArgs spingalett_layer_args(NeuralNetwork *net, uint32_t l);
/* The image of a deployment model: net with every batch normalization that directly follows a dense
   or convolution layer without activation folded into that layer, without optimizer state. */
void *spingalett_save_deployment(const NeuralNetwork *net, PrecisionMode precision, size_t *size);
bool spingalett_has_dropout(const NeuralNetwork *net);
/* dropout == NULL: inference. Otherwise hidden layers with a dropout rate are masked. */
float *spingalett_forward_pass(NeuralNetwork *net, const float *input, ComputeMode mode,
                               const DropoutContext *dropout);
/* Masks n activations y (outputs of activation act) in place and stores mask * f'(a) in dmask. */
void spingalett_dropout_apply(float *restrict y, float *restrict dmask, uint32_t n,
                              ActivationFunction act, float rate,
                              const DropoutContext *ctx, uint32_t layer, uint32_t position);

void spingalett_log(LogLevel level, const char *fmt, ...);

/* Reads a whole file into a buffer aligned like spingalett_aligned_alloc (release with
   spingalett_aligned_free). NULL on error, with the error set. */
void *spingalett_read_file(const char *path, size_t *size);
/* ------------------------------------------------------------------------- importers (Import.c) */

/* A file's bytes: mapped read-only where the platform allows, read into memory otherwise. */
typedef struct {
    const uint8_t *data;
    size_t size;
    void *handle;                   /* the view, or the buffer read */
    bool mapped;
} SpgFileView;
bool spingalett_file_open(SpgFileView *f, const char *path);       /* false with the error set */
void spingalett_file_close(SpgFileView *f);

/* Element types of imported tensors (little-endian, at any alignment). */
enum { SPG_DTYPE_F32, SPG_DTYPE_F16, SPG_DTYPE_BF16, SPG_DTYPE_F64, SPG_DTYPE_I32, SPG_DTYPE_I64, SPG_DTYPE_OTHER };
size_t spingalett_dtype_size(int dtype);                            /* 0 for SPG_DTYPE_OTHER */
void spingalett_decode(float *dst, const uint8_t *src, int dtype, size_t n);
/* Filters [OC][CG][KH][KW] at src as [OC][KH][KW][CG], with spingalett_filters_scratch() floats of
   scratch. */
size_t spingalett_filters_scratch(uint32_t CG, uint32_t KH, uint32_t KW);
void spingalett_import_filters(float *dst, const uint8_t *src, int dtype, uint32_t OC, uint32_t CG, uint32_t KH,
                               uint32_t KW, float *scratch);
/* Dense weights [out][in] (transposed: [in][out] at src) times alpha, the columns of a map of C
   channels and HW cells read flat reordered from (c, p) to (p, c); spingalett_dense_scratch() floats
   of scratch. */
size_t spingalett_dense_scratch(uint32_t out, uint32_t in, bool transposed);
void spingalett_import_dense(float *dst, const uint8_t *src, int dtype, uint32_t out, uint32_t in, uint32_t C,
                             uint32_t HW, bool transposed, float alpha, float *scratch);

/* count strings copied into one allocation (a NULL-terminated array of pointers; free() releases it). */
char **spingalett_copy_names(const char *const *names, uint32_t count);
/* load_spingalett_from_memory that also reports the precision of the first weight layer. */
NeuralNetwork *spingalett_load_from_memory_ex(const void *data, size_t size, PrecisionMode *precision);
/* Whether a sample counts as correctly classified (see EvalMetrics). */
bool spingalett_sample_correct(const float *output, const float *target, uint32_t n);

void *spingalett_aligned_alloc(size_t size);
void *spingalett_aligned_calloc(size_t count, size_t elem_size);
void  spingalett_aligned_free(void *ptr);

void apply_softmax(float *layer, uint32_t size);
void apply_activation_batch(float *data, uint32_t size, ActivationFunction act);
void apply_activation_bulk(float *data, uint64_t total, ActivationFunction act);
void apply_derivative_batch(float *deriv, const float *act_data, uint64_t total, ActivationFunction act);

void     rng_seed(uint64_t seed);
uint32_t rng_next(void);
uint64_t rng_next64(void);
float    rng_next_float(void);

float random_uniform_weight(void);
float random_normal_weight(void);

float spingalett_dot_product(const float *restrict a, const float *restrict b, uint64_t n);

void spingalett_shuffle_indices(uint32_t *indices, uint32_t n);

void spingalett_sgd_update(float *restrict W, const float *restrict gW, uint64_t n, float lr, float decay);
void spingalett_momentum_update(float *restrict W, float *restrict mW, const float *restrict gW, uint64_t n,
                                float lr, float momentum, float decay);
void spingalett_rmsprop_update(float *restrict W, float *restrict vW, const float *restrict gW, uint64_t n,
                               float lr, float beta2, float epsilon, float decay);
void spingalett_adam_update(float *restrict W, float *restrict mW, float *restrict vW, const float *restrict gW,
                            uint64_t n, float lr, float beta1, float beta2,
                            float m_factor, float v_factor, float epsilon,
                            float decay, float wd_factor);

double spingalett_vec_sumsq(const float *x, uint64_t n);
float  spingalett_vec_l2norm(const float *x, uint64_t n);
/* Native SGEMM (Spingalett.GEMM.c): C = alpha * op(A) * op(B) + beta * C, row-major. The scratch
   holds packing buffers for `threads` threads and may be reused across calls; NULL allocates a
   temporary one. `parallel` allows threads: the product uses them when it is large enough for its
   kind (SPINGALETT_GEMM_PARALLEL_WORK multiply-adds; fewer for products of few columns). */
/* A GEMM operand produced on demand instead of read from memory: fill writes rows [row, row + rows)
   x columns [col, col + cols) of the operand as stored (before op()), rows ld floats apart. */
typedef struct {
    void (*fill)(const void *ctx, uint32_t row, uint32_t rows, uint32_t col, uint32_t cols, float *dst, size_t ld);
    const void *ctx;
} SpingalettGemmSource;

/* Additions to a product: operands given by sources (A or B is then unused), and an epilogue
   called once on every element of C when its value is final, while the tile is still in cache:
   with c at C[row][col] for a rows x cols block, rows ldc floats apart (concurrently on disjoint
   blocks when the product is multi-threaded). */
typedef struct {
    const SpingalettGemmSource *a, *b;
    void (*epilogue)(const void *ctx, uint32_t row, uint32_t rows, uint32_t col, uint32_t cols, float *c, size_t ldc);
    const void *epilogue_ctx;
} SpingalettGemmHooks;

SpingalettGemmScratch *spingalett_gemm_scratch_create(int threads);
void spingalett_gemm_scratch_free(SpingalettGemmScratch *scratch);
/* Bytes a scratch for `threads` threads holds when it is created. */
size_t spingalett_gemm_scratch_bytes(int threads);
void spingalett_gemm_native(SpingalettGemmScratch *scratch, bool trans_a, bool trans_b,
                            uint32_t M, uint32_t N, uint32_t K, float alpha,
                            const float *A, size_t lda, const float *B, size_t ldb,
                            float beta, float *C, size_t ldc, bool parallel);
/* The same with hooks (NULL: none). */
void spingalett_gemm_hooked(SpingalettGemmScratch *scratch, bool trans_a, bool trans_b,
                            uint32_t M, uint32_t N, uint32_t K, float alpha,
                            const float *A, size_t lda, const float *B, size_t ldb,
                            float beta, float *C, size_t ldc, bool parallel, const SpingalettGemmHooks *hooks);
/* Integer weight rows interleaved for batches (Spingalett.Int8Tiles.c). spingalett_i8_interleaved()
   tells whether this processor runs the tile kernel (x86-64 with AVX-512 VNNI or AVX-VNNI, ARM with
   the dot product instructions); elsewhere the tile function does nothing. spingalett_i8_interleave stores `rows` rows of n bytes (stride bytes apart)
   in spingalett_i8_interleaved_rows(rows) * spingalett_i8_interleaved_len(n) bytes, zeros after the
   last row and byte, and their sums in spingalett_i8_interleaved_rows(rows) entries.
   spingalett_i8_interleaved_tile computes acc[p * acc_stride + j] = row j . vector p of x for the
   SPINGALETT_I8_TILE vectors of x (x_stride bytes apart, spingalett_i8_interleaved_len(n) bytes
   each, any initialized values after the first n; it changes all of them) and every j below
   spingalett_i8_interleaved_rows(rows). */
#define SPINGALETT_I8_TILE 12u
bool spingalett_i8_interleaved(void);
static inline uint32_t spingalett_i8_interleaved_rows(uint32_t rows) { return (rows + 15u) & ~15u; }
static inline uint32_t spingalett_i8_interleaved_len(uint32_t n) { return (n + 3u) & ~3u; }
void spingalett_i8_interleave(const int8_t *w, size_t stride, uint32_t rows, uint32_t n, int8_t *out, int32_t *sums);
void spingalett_i8_interleaved_tile(const int8_t *weights, const int32_t *sums, uint32_t rows, uint32_t n, int8_t *x,
                                    size_t x_stride, int32_t *acc, size_t acc_stride);
#if defined(SPINGALETT_INT8_DISPATCH)
/* The kernel compiled for AVX-512 VNNI, for AVX-VNNI, and for the ARM dot product instructions */
#define SPINGALETT_I8_TILE_KERNEL(name) \
    void name(const int8_t *weights, const int32_t *sums, uint32_t rows, uint32_t n, int8_t *x, size_t x_stride, \
              int32_t *acc, size_t acc_stride);
SPINGALETT_I8_TILE_KERNEL(spingalett_i8_tile_avx512)
SPINGALETT_I8_TILE_KERNEL(spingalett_i8_tile_avxvnni)
SPINGALETT_I8_TILE_KERNEL(spingalett_i8_tile_dotprod)
#undef SPINGALETT_I8_TILE_KERNEL
#endif

#if defined(SPINGALETT_GEMM_DISPATCH)
/* The same with one kernel set: the library's baseline flags, AVX2+FMA or AVX-512 (scratch must
   not be NULL; M, N and K must be positive). */
#define SPINGALETT_GEMM_KERNELS(name) \
    void name(SpingalettGemmScratch *scratch, bool trans_a, bool trans_b, uint32_t M, uint32_t N, uint32_t K, \
              float alpha, const float *A, size_t lda, const float *B, size_t ldb, \
              float beta, float *C, size_t ldc, bool parallel, const SpingalettGemmHooks *hooks);
SPINGALETT_GEMM_KERNELS(spingalett_gemm_baseline)
SPINGALETT_GEMM_KERNELS(spingalett_gemm_avx2)
SPINGALETT_GEMM_KERNELS(spingalett_gemm_avx512)
#undef SPINGALETT_GEMM_KERNELS
#endif

/* Convolutions as indirect matrix products (Spingalett.ConvGEMM.c): windows read straight from the
   image through per-tap pointers. They run where spingalett_conv_direct() holds (native kernels,
   not depthwise, at most SPINGALETT_CONV_DIRECT_TAPS taps), the weight gradient where
   spingalett_conv_direct_wgrad() does too, a group at a time; the scratch they take is sized by the
   functions below, from the widest panels and tiles of any kernel set. The epilogue (or NULL) runs
   on every tile of rows of the group's columns. */
#define SPINGALETT_CONV_DIRECT_TAPS 64u
#define SPINGALETT_CONV_NR_MAX 32u
#define SPINGALETT_CONV_MR_MAX 12u
#define SPINGALETT_CONV_WGRAD_MIN 1024u     /* pixels per slot of the weight gradient, at least */
#define SPINGALETT_CONV_WGRAD_SLOTS 32u     /* slots at most */
#define SPINGALETT_CONV_WGRAD_BLOCK 128u    /* pixels whose tap pointers are made at once */

static inline bool spingalett_conv_direct(const LayerShape *in, const LayerShape *out, ComputeMode mode) {
#if defined(SPINGALETT_NO_DIRECT_CONV)      /* for comparisons with the products of gathered windows */
    (void)in; (void)out; (void)mode;
    return false;
#endif
    uint32_t G = out->groups ? out->groups : 1u;
    return mode != COMPUTE_OPENBLAS && !(G > 1 && in->channels == G) &&
           (uint64_t)out->kernel_h * out->kernel_w <= SPINGALETT_CONV_DIRECT_TAPS;
}

static inline bool spingalett_conv_direct_wgrad(const LayerShape *in, const LayerShape *out, ComputeMode mode) {
    uint32_t G = out->groups ? out->groups : 1u;
    return spingalett_conv_direct(in, out, mode) && (in->channels / G) % 16u == 0 &&
           out->channels / G >= SPINGALETT_CONV_MR_MAX;
}

static inline size_t spingalett_conv_direct_packed_floats(uint32_t cols, uint32_t K) {
    return ((((size_t)cols + SPINGALETT_CONV_NR_MAX - 1) / SPINGALETT_CONV_NR_MAX * SPINGALETT_CONV_NR_MAX * K) + 15u) &
           ~(size_t)15u;
}

static inline uint32_t spingalett_conv_wgrad_slots(uint64_t pixels) {
    uint64_t slots = pixels / SPINGALETT_CONV_WGRAD_MIN;
    if (slots > SPINGALETT_CONV_WGRAD_SLOTS) slots = SPINGALETT_CONV_WGRAD_SLOTS;
    return slots < 1 ? 1u : (uint32_t)slots;
}

static inline size_t spingalett_conv_wgrad_partial_floats(uint32_t OG, uint32_t K, uint32_t slots) {
    return (((size_t)OG + SPINGALETT_CONV_MR_MAX - 1) * ((size_t)K + SPINGALETT_CONV_NR_MAX - 1) * slots + 15u) &
           ~(size_t)15u;
}

static inline size_t spingalett_conv_wgrad_zero_floats(uint32_t CG) {
    return ((size_t)CG + SPINGALETT_CONV_NR_MAX + 15u) & ~(size_t)15u;
}

/* Floats of scratch the direct passes of a convolution need: the forward pass and data gradient
   their packed weights and zeros, the weight gradient its partial sums, zeros and tap pointers. */
static inline size_t spingalett_conv_direct_scratch(const LayerShape *in, const LayerShape *out, uint32_t n,
                                                    bool training, int threads) {
    const uint32_t G = out->groups ? out->groups : 1u, CG = in->channels / G, OG = out->channels / G;
    const uint32_t taps = out->kernel_h * out->kernel_w, wide = (CG > OG ? CG : OG) + 32u;
    size_t need = spingalett_conv_direct_packed_floats(OG, taps * CG) + wide;
    if (training) {
        size_t data = spingalett_conv_direct_packed_floats(CG, taps * OG) + wide;
        if (data > need) need = data;
        if (spingalett_conv_direct_wgrad(in, out, COMPUTE_SINGLE_THREADED)) {
            uint64_t pixels = (uint64_t)n * out->height * out->width;
            size_t w = spingalett_conv_wgrad_partial_floats(OG, taps * CG, spingalett_conv_wgrad_slots(pixels)) +
                       spingalett_conv_wgrad_zero_floats(CG) +
                       (size_t)(threads > 0 ? threads : 1) * 2u * SPINGALETT_CONV_WGRAD_BLOCK * 2u + 16u;
            if (w > need) need = w;
        }
    }
    return need;
}

void spingalett_conv_direct_forward(const LayerShape *in, const LayerShape *out, uint32_t g, const float *W,
                                    const float *x, float *y, uint32_t n, float *scratch,
                                    const SpingalettGemmHooks *epilogue, bool parallel, int threads);
void spingalett_conv_direct_backward_data(const LayerShape *in, const LayerShape *out, uint32_t g, const float *W,
                                          const float *dy, float *dx, uint32_t n, float *scratch,
                                          const SpingalettGemmHooks *epilogue, bool parallel, int threads);
/* gW = scale * (sum over the batch) + beta * gW for group g's filters (the bias gradient is the
   caller's) */
void spingalett_conv_direct_backward_weights(const LayerShape *in, const LayerShape *out, uint32_t g, const float *x,
                                             const float *dy, uint32_t n, float scale, float beta, float *gW,
                                             float *scratch, bool parallel, int threads);
/* Threads a GEMM scratch was made for. */
int spingalett_gemm_scratch_threads(const SpingalettGemmScratch *scratch);

/* An epilogue adding a bias per column, after multiplying by a scale per column when scale is set,
   and applying an element-wise activation (not softmax). Rows are activated one by one: tile
   columns start at multiples of 8, so every element takes the vector or scalar path it takes in a
   whole row, whatever the tiling (and the thread count). */
typedef struct {
    const float *bias;
    ActivationFunction act;
    const float *scale;
} SpingalettBiasActivation;
void spingalett_epilogue_bias_activation(const void *ctx, uint32_t row, uint32_t rows, uint32_t col, uint32_t cols,
                                         float *c, size_t ldc);

/* Multiply-adds below which a GEMM runs on one thread (about 10 us of work: below it, a parallel
   region with its barriers costs more than it saves). */
#define SPINGALETT_GEMM_PARALLEL_WORK (1u << 20)

/* GEMM on the selected backend: OpenBLAS when requested and available, the native kernels
   otherwise (multi-threaded in OpenMP mode). */
static inline void spingalett_gemm(SpingalettGemmScratch *scratch, ComputeMode mode, bool trans_a, bool trans_b,
                                   uint32_t M, uint32_t N, uint32_t K, float alpha,
                                   const float *A, size_t lda, const float *B, size_t ldb,
                                   float beta, float *C, size_t ldc) {
#if defined(SPINGALETT_HAS_OPENBLAS)
    if (mode == COMPUTE_OPENBLAS) {
        cblas_sgemm(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
                    (int)M, (int)N, (int)K, alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
        return;
    }
#endif
    spingalett_gemm_native(scratch, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc,
                           mode == COMPUTE_OPENMP);
}

/* The same with hooks: on the native kernels as they are; with OpenBLAS (operands from memory
   only) the epilogue runs over all of C after the product. */
static inline void spingalett_gemm_ex(SpingalettGemmScratch *scratch, ComputeMode mode, bool trans_a, bool trans_b,
                                      uint32_t M, uint32_t N, uint32_t K, float alpha,
                                      const float *A, size_t lda, const float *B, size_t ldb,
                                      float beta, float *C, size_t ldc, const SpingalettGemmHooks *hooks) {
    if (mode == COMPUTE_OPENBLAS) {
        spingalett_gemm(scratch, mode, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        if (hooks && hooks->epilogue && M && N) hooks->epilogue(hooks->epilogue_ctx, 0, M, 0, N, C, ldc);
        return;
    }
    spingalett_gemm_hooked(scratch, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc,
                           mode == COMPUTE_OPENMP, hooks);
}

void spingalett_vec_scale(float *data, uint64_t n, float scale);
void spingalett_vec_mul(float *restrict y, const float *restrict x, uint64_t n);
void spingalett_vec_scaled_copy(float *restrict dst, const float *restrict src, uint64_t n, float alpha);
void spingalett_vec_axpy(float *restrict y, const float *restrict x, uint64_t n, float alpha);
/* dst[i] = spingalett_float_to_fp16(src[i]) for n values: with F16C eight at a time (its rounding
   gives the same halves for every value but NaN, whose blocks take the portable conversion). */
void spingalett_fp16_encode(const float *restrict src, size_t n, uint16_t *restrict dst);

float spingalett_clip_grad_norm(NeuralNetwork *net, float max_norm);

ComputeMode resolve_compute_mode(void);
/* Whether the compute mode is COMPUTE_VULKAN and a device is usable (warns once when it is not). */
bool spingalett_use_gpu(void);
/* A network on the GPU for inference over up to `count` samples a chunk, when COMPUTE_VULKAN is set
   and it fits (NULL otherwise, with a warning when the GPU was usable). */
struct SpgGpuNet *spingalett_gpu_for(NeuralNetwork *net, uint32_t count);
/* Gives a network got from spingalett_gpu_for() back to be kept. */
void spingalett_gpu_done(NeuralNetwork *net, struct SpgGpuNet *gpu);
/* spingalett_batch_evaluate() on the GPU; false when the device failed. */
bool spingalett_gpu_evaluate(struct SpgGpuNet *gpu, NeuralNetwork *net, const float *inputs, const float *targets,
                             uint32_t n, double *loss_sum, uint32_t *correct);

void spingalett_fp_flush_denormals_begin(void);
void spingalett_fp_flush_denormals_end(void);

float compute_sample_loss(const float *output, const float *target,
                          uint32_t output_size, LossFunction loss_func,
                          ActivationFunction output_act);

extern const char * const act_func_names[];
extern const char * const loss_func_names[];
extern const char * const training_strategy_names[];
extern const char * const training_mode_names[];
extern const char * const optimizer_names[];
extern const char * const weight_initialization_names[];
extern const char * const precision_names[];
extern const char * const compute_mode_names[];
