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
   runs in chunks whose gradients are accumulated) while keeping the GEMMs large. */
#define SPINGALETT_BATCH_CHUNK 2048u

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
    uint32_t capacity;
} BatchWorkspace;

BatchWorkspace *spingalett_batch_workspace_create(const NeuralNetwork *net, uint32_t capacity,
                                                  bool training, bool gather, ComputeMode mode);
void spingalett_batch_workspace_free(BatchWorkspace *ws);
/* Forward pass over N samples (ws->act[0] must point at them). `dropout` (or NULL) masks hidden
   layers; samples are numbered position_offset + s for the dropout hash. */
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
/* dx = dL/dx * act'(x) from dy = dL/dy (x: the pooling layer's input, the output of layer l, whose
   activation is act); dx is overwritten */
void spingalett_pool_backward(const NeuralNetwork *net, uint32_t l, const float *x, const float *dy, float *dx,
                              uint32_t n, ActivationFunction act, ComputeMode mode);

bool spingalett_add_layer(LayerArgs args);
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
   temporary one. */
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

/* Multiply-adds below which a GEMM runs on one thread. */
#define SPINGALETT_GEMM_PARALLEL_WORK (1u << 18)

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
    bool parallel = mode == COMPUTE_OPENMP && (uint64_t)M * N * K >= SPINGALETT_GEMM_PARALLEL_WORK;
    spingalett_gemm_native(scratch, trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, parallel);
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
