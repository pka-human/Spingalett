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

#define SPINGALETT_NEURON(net, l, j)        ((net)->neurons[(net)->neuron_offsets[l] + (uint64_t)(j)])
#define SPINGALETT_LAYER_PTR(net, l)        ((net)->neurons + (net)->neuron_offsets[l])


#define SPINGALETT_WEIGHT_MTX_PTR(net, l)   ((net)->weights + (net)->weight_offsets[l])
#define SPINGALETT_GRAD_W_MTX_PTR(net, l)   ((net)->grad_weights + (net)->weight_offsets[l])

void set_error(int code, const char *msg);


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
    float **act;            /* act[0]: inputs of the chunk; act[l]: [capacity x topology[l]] */
    float **delta;          /* training only: delta[l] for l >= 1 */
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

float spingalett_clip_grad_norm(NeuralNetwork *net, float max_norm);

ComputeMode resolve_compute_mode(void);

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
