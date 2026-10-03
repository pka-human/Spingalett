/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* Elements per optimizer work item: per-sample gradient rows are built in an L1-resident
   buffer of this size, and the batch optimizer pass is split into multiples of it. */
#define OPT_CHUNK 256u

typedef struct {
    OptimizerType type;
    float lr;
    float decay;
    float momentum;
    float beta1, beta2, epsilon;
    float m_factor, v_factor;   /* Adam bias corrections of the current step */
} OptimizerStep;

static void optimizer_update(const OptimizerStep *o, float *W, float *mW, float *vW,
                             const float *g, uint64_t n, float decay) {
    switch (o->type) {
        case OPTIMIZER_SGD:
            spingalett_sgd_update(W, g, n, o->lr, decay);
            break;
        case OPTIMIZER_MOMENTUM:
            spingalett_momentum_update(W, mW, g, n, o->lr, o->momentum, decay);
            break;
        case OPTIMIZER_RMSPROP:
            spingalett_rmsprop_update(W, vW, g, n, o->lr, o->beta2, o->epsilon, decay);
            break;
        case OPTIMIZER_ADAM:
            spingalett_adam_update(W, mW, vW, g, n, o->lr, o->beta1, o->beta2,
                                   o->m_factor, o->v_factor, o->epsilon, decay, 1.0f);
            break;
        case OPTIMIZER_ADAMW:
            spingalett_adam_update(W, mW, vW, g, n, o->lr, o->beta1, o->beta2,
                                   o->m_factor, o->v_factor, o->epsilon, 0.0f, 1.0f - o->lr * decay);
            break;
        default:
            break;
    }
}

/* One optimizer pass over a contiguous parameter array, chunked so OpenMP can split it. */
static void optimizer_update_array(const OptimizerStep *o, float *W, float *mW, float *vW,
                                   const float *g, uint64_t n, float decay, ComputeMode mode) {
    const uint64_t chunk = 64u * OPT_CHUNK;
    int64_t chunks = (int64_t)((n + chunk - 1) / chunk);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(mode == COMPUTE_OPENMP && chunks > 1)
#endif
    for (int64_t c = 0; c < chunks; c++) {
        uint64_t off = (uint64_t)c * chunk;
        uint64_t len = (n - off < chunk) ? n - off : chunk;
        optimizer_update(o, W + off, mW + off, vW + off, g + off, len, decay);
    }
    (void)mode;
}

/* Apply the accumulated (already averaged) gradients of all layers. */
static void apply_gradients(NeuralNetwork *net, const OptimizerStep *o, float max_grad_norm, ComputeMode mode) {
    if (max_grad_norm > 0.0f)
        spingalett_clip_grad_norm(net, max_grad_norm);

    optimizer_update_array(o, net->weights, net->opt_m_weights, net->opt_v_weights,
                           net->grad_weights, net->total_weights, o->decay, mode);
    optimizer_update_array(o, net->biases, net->opt_m_biases, net->opt_v_biases,
                           net->grad_biases, net->total_biases, 0.0f, mode);
}

static bool check_nan_inf(const NeuralNetwork *net) {
    for (uint64_t i = 0; i < net->total_weights; i++) {
        if (!isfinite(net->weights[i])) return true;
    }
    for (uint64_t i = 0; i < net->total_biases; i++) {
        if (!isfinite(net->biases[i])) return true;
    }
    return false;
}

/* dL/dz of the output layer for one sample. */
static void output_delta(LossFunction loss, ActivationFunction act,
                         const float *out, const float *target, float *delta, uint32_t n) {
    if (loss == LOSS_MSE && act == ACT_SOFTMAX) {
        float dot = 0.0f;
        for (uint32_t j = 0; j < n; j++)
            dot += (out[j] - target[j]) * out[j];
        for (uint32_t j = 0; j < n; j++)
            delta[j] = out[j] * ((out[j] - target[j]) - dot);
        return;
    }

    for (uint32_t j = 0; j < n; j++)
        delta[j] = out[j] - target[j];

    /* Cross-entropy cancels the derivative of a sigmoid/softmax output. */
    if (!(loss == LOSS_CROSS_ENTROPY && (act == ACT_SOFTMAX || act == ACT_SIGMOID)))
        apply_derivative_batch(delta, out, (uint64_t)n, act);
}

/* out[0..cols) = W^T d for a row-major rows x cols matrix W. Column slabs keep the output
   chunk in L1 and let OpenMP threads own disjoint outputs. Zero deltas (inactive ReLUs) are skipped. */
static void backproject(float *restrict out, const float *restrict W, const float *restrict d,
                        uint32_t rows, uint32_t cols, ComputeMode mode) {
#if defined(SPINGALETT_HAS_OPENBLAS)
    if (mode == COMPUTE_OPENBLAS) {
        cblas_sgemv(CblasRowMajor, CblasTrans, (int)rows, (int)cols,
                    1.0f, W, (int)cols, d, 1, 0.0f, out, 1);
        return;
    }
#endif
    int64_t slabs = (int64_t)((cols + OPT_CHUNK - 1) / OPT_CHUNK);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, (uint64_t)rows * cols) && slabs > 1)
#endif
    for (int64_t s = 0; s < slabs; s++) {
        uint32_t k0 = (uint32_t)s * OPT_CHUNK;
        uint32_t len = (cols - k0 < OPT_CHUNK) ? cols - k0 : OPT_CHUNK;
        float *o = out + k0;
        memset(o, 0, len * sizeof(float));
        for (uint32_t j = 0; j < rows; j++) {
            float dj = d[j];
            if (dj != 0.0f)
                spingalett_vec_axpy(o, W + (uint64_t)j * cols + k0, (uint64_t)len, dj);
        }
    }
    (void)mode;
}

/* Back-propagate one sample whose activations are in net->neurons. deltas[l] has
   topology[l] entries; deltas[0] is never used. dmask (or NULL) holds mask * f'(a) for
   layers with dropout, laid out like net->neurons. */
static void compute_deltas(NeuralNetwork *net, const float *target, float **deltas,
                           const float *dmask, ComputeMode mode) {
    uint32_t last = net->layers - 1;
    output_delta(net->loss_func, net->act_func[last - 1],
                 SPINGALETT_LAYER_PTR(net, last), target, deltas[last], net->topology[last]);

    for (uint32_t l = last - 1; l > 0; l--) {
        backproject(deltas[l], SPINGALETT_WEIGHT_MTX_PTR(net, l), deltas[l + 1],
                    net->topology[l + 1], net->topology[l], mode);
        if (dmask && net->dropout_rates[l] > 0.0f)
            spingalett_vec_mul(deltas[l], dmask + net->neuron_offsets[l], (uint64_t)net->topology[l]);
        else
            apply_derivative_batch(deltas[l], SPINGALETT_LAYER_PTR(net, l),
                                   (uint64_t)net->topology[l], net->act_func[l - 1]);
    }
}

/* grad += scale * (delta (x) activations) for one sample. */
static void accumulate_gradients(NeuralNetwork *net, float **deltas, float scale, ComputeMode mode) {
    for (uint32_t l = 0; l + 1 < net->layers; l++) {
        uint32_t in_sz  = net->topology[l];
        uint32_t out_sz = net->topology[l + 1];
        float *gW = SPINGALETT_GRAD_W_MTX_PTR(net, l);
        const float *x = SPINGALETT_LAYER_PTR(net, l);
        const float *d = deltas[l + 1];

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, (uint64_t)in_sz * out_sz))
#endif
        for (int64_t j = 0; j < (int64_t)out_sz; j++) {
            float dj = d[j] * scale;
            if (dj != 0.0f)
                spingalett_vec_axpy(gW + (uint64_t)j * in_sz, x, (uint64_t)in_sz, dj);
        }
        spingalett_vec_axpy(net->grad_biases + net->bias_offsets[l], d, (uint64_t)out_sz, scale);
    }
    (void)mode;
}

/* Global-norm clipping for one sample without materializing its gradient: layer l's weight
   gradient is the rank-1 product delta[l+1] (x) x[l], so its squared Frobenius norm is
   |delta|^2 * |x|^2, and the bias gradient adds |delta|^2. Scaling the deltas scales both. */
static void clip_sample_gradient(NeuralNetwork *net, float **deltas, float max_norm) {
    double total_sq = 0.0;
    for (uint32_t l = 0; l + 1 < net->layers; l++) {
        double x_sq = spingalett_vec_sumsq(SPINGALETT_LAYER_PTR(net, l), net->topology[l]);
        double d_sq = spingalett_vec_sumsq(deltas[l + 1], net->topology[l + 1]);
        total_sq += d_sq * (x_sq + 1.0);
    }

    double norm = sqrt(total_sq);
    if (isfinite(norm) && norm > (double)max_norm) {
        float scale = (float)((double)max_norm / norm);
        for (uint32_t l = 1; l < net->layers; l++)
            spingalett_vec_scale(deltas[l], net->topology[l], scale);
    }
}

/* Per-sample (online) update: the gradient of each weight row is the rank-1 product
   delta[j] * x, built chunk by chunk in an L1 buffer and fed straight into the optimizer
   kernel, so the full gradient matrix is never materialized. */
static void apply_rank1_updates(NeuralNetwork *net, const OptimizerStep *o, float **deltas, ComputeMode mode) {
    bool plain_sgd = (o->type == OPTIMIZER_SGD && o->decay == 0.0f);

    for (uint32_t l = 0; l + 1 < net->layers; l++) {
        uint32_t in_sz  = net->topology[l];
        uint32_t out_sz = net->topology[l + 1];
        float *W  = SPINGALETT_WEIGHT_MTX_PTR(net, l);
        float *mW = net->opt_m_weights + net->weight_offsets[l];
        float *vW = net->opt_v_weights + net->weight_offsets[l];
        const float *x = SPINGALETT_LAYER_PTR(net, l);
        const float *d = deltas[l + 1];

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, (uint64_t)in_sz * out_sz))
#endif
        for (int64_t j = 0; j < (int64_t)out_sz; j++) {
            float dj = d[j];
            uint64_t row = (uint64_t)j * in_sz;
            if (plain_sgd) {
                if (dj != 0.0f)
                    spingalett_vec_axpy(W + row, x, (uint64_t)in_sz, -o->lr * dj);
                continue;
            }
            float g[OPT_CHUNK];
            for (uint32_t k0 = 0; k0 < in_sz; k0 += OPT_CHUNK) {
                uint32_t len = (in_sz - k0 < OPT_CHUNK) ? in_sz - k0 : OPT_CHUNK;
                spingalett_vec_scaled_copy(g, x + k0, (uint64_t)len, dj);
                optimizer_update(o, W + row + k0, mW + row + k0, vW + row + k0, g, (uint64_t)len, o->decay);
            }
        }

        uint64_t boff = net->bias_offsets[l];
        optimizer_update(o, net->biases + boff, net->opt_m_biases + boff, net->opt_v_biases + boff,
                         d, (uint64_t)out_sz, 0.0f);
    }
    (void)mode;
}

#if defined(SPINGALETT_HAS_OPENBLAS)

typedef struct {
    float **act;        /* act[0]: the batch inputs; act[l]: [capacity x topology[l]] */
    float **delta;      /* delta[l] for l >= 1 */
    float **dmask;      /* dropout mask * f'(a) for hidden layers with dropout, else NULL */
    float *flat;        /* backing storage for act[1..] and delta[1..] */
    float *dmask_flat;
    float *inputs;      /* gather buffer for mini-batches (NULL for full batch) */
    float *targets;     /* gather buffer for mini-batches (NULL for full batch) */
    float *ones;
    uint32_t capacity;
} BatchWorkspace;

static void batch_workspace_free(BatchWorkspace *ws) {
    if (!ws) return;
    spingalett_aligned_free(ws->flat);
    spingalett_aligned_free(ws->dmask_flat);
    free(ws->dmask);
    spingalett_aligned_free(ws->inputs);
    spingalett_aligned_free(ws->targets);
    spingalett_aligned_free(ws->ones);
    free(ws->act);
    free(ws->delta);
    free(ws);
}

static BatchWorkspace *batch_workspace_create(const NeuralNetwork *net, uint32_t max_batch, bool gather) {
    BatchWorkspace *ws = (BatchWorkspace *)calloc(1, sizeof(BatchWorkspace));
    if (!ws) return NULL;

    ws->capacity = max_batch;
    ws->act   = (float **)calloc(net->layers, sizeof(float *));
    ws->delta = (float **)calloc(net->layers, sizeof(float *));
    ws->dmask = (float **)calloc(net->layers, sizeof(float *));

    size_t per_layer_total = 0, dropout_total = 0;
    for (uint32_t l = 1; l < net->layers; l++) {
        per_layer_total += (size_t)max_batch * (size_t)net->topology[l];
        if (l + 1 < net->layers && net->dropout_rates[l] > 0.0f)
            dropout_total += (size_t)max_batch * (size_t)net->topology[l];
    }

    ws->flat = (float *)spingalett_aligned_calloc(2 * per_layer_total, sizeof(float));
    if (dropout_total > 0) {
        ws->dmask_flat = (float *)spingalett_aligned_alloc(dropout_total * sizeof(float));
        if (!ws->dmask_flat) { batch_workspace_free(ws); return NULL; }
        size_t doff = 0;
        for (uint32_t l = 1; l + 1 < net->layers; l++) {
            if (net->dropout_rates[l] > 0.0f) {
                ws->dmask[l] = ws->dmask_flat + doff;
                doff += (size_t)max_batch * (size_t)net->topology[l];
            }
        }
    }
    ws->ones = (float *)spingalett_aligned_alloc((size_t)max_batch * sizeof(float));
    if (gather) {
        ws->inputs  = (float *)spingalett_aligned_alloc((size_t)max_batch * net->topology[0] * sizeof(float));
        ws->targets = (float *)spingalett_aligned_alloc((size_t)max_batch * net->topology[net->layers - 1] * sizeof(float));
    }

    if (!ws->act || !ws->delta || !ws->dmask || !ws->flat || !ws->ones || (gather && (!ws->inputs || !ws->targets))) {
        batch_workspace_free(ws);
        return NULL;
    }

    for (uint32_t i = 0; i < max_batch; i++)
        ws->ones[i] = 1.0f;

    size_t off = 0;
    for (uint32_t l = 1; l < net->layers; l++) {
        size_t sz = (size_t)max_batch * (size_t)net->topology[l];
        ws->act[l]   = ws->flat + off;
        ws->delta[l] = ws->flat + per_layer_total + off;
        off += sz;
    }
    ws->act[0] = ws->inputs;

    return ws;
}

static void batch_forward(NeuralNetwork *net, BatchWorkspace *ws, uint32_t N, const DropoutContext *dropout) {
    for (uint32_t l = 1; l < net->layers; l++) {
        uint32_t prev_size = net->topology[l - 1];
        uint32_t curr_size = net->topology[l];
        const float *W = SPINGALETT_WEIGHT_MTX_PTR(net, l - 1);
        const float *bias = net->biases + net->bias_offsets[l - 1];
        float *C = ws->act[l];
        ActivationFunction act = net->act_func[l - 1];

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)N, (int)curr_size, (int)prev_size,
                    1.0f,
                    ws->act[l - 1], (int)prev_size,
                    W, (int)prev_size,
                    0.0f,
                    C, (int)curr_size);

        cblas_sger(CblasRowMajor, (int)N, (int)curr_size,
                   1.0f, ws->ones, 1, bias, 1, C, (int)curr_size);

        if (act == ACT_SOFTMAX) {
            for (uint32_t s = 0; s < N; s++)
                apply_softmax(C + (size_t)s * curr_size, curr_size);
        } else {
            apply_activation_bulk(C, (uint64_t)N * (uint64_t)curr_size, act);
        }

        if (dropout && ws->dmask[l]) {
            for (uint32_t s = 0; s < N; s++) {
                size_t off = (size_t)s * curr_size;
                spingalett_dropout_apply(C + off, ws->dmask[l] + off, curr_size, act,
                                         net->dropout_rates[l], dropout, l, s);
            }
        }
    }
}

static void batch_compute_deltas(NeuralNetwork *net, BatchWorkspace *ws, const float *targets, uint32_t N) {
    uint32_t last = net->layers - 1;
    ActivationFunction last_act = net->act_func[last - 1];
    uint32_t n_out = net->topology[last];

    for (uint32_t s = 0; s < N; s++) {
        size_t off = (size_t)s * n_out;
        output_delta(net->loss_func, last_act, ws->act[last] + off, targets + off, ws->delta[last] + off, n_out);
    }

    for (uint32_t l = last - 1; l > 0; l--) {
        uint32_t cur_sz  = net->topology[l];
        uint32_t next_sz = net->topology[l + 1];

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)N, (int)cur_sz, (int)next_sz,
                    1.0f,
                    ws->delta[l + 1], (int)next_sz,
                    SPINGALETT_WEIGHT_MTX_PTR(net, l), (int)cur_sz,
                    0.0f,
                    ws->delta[l], (int)cur_sz);

        if (ws->dmask[l])
            spingalett_vec_mul(ws->delta[l], ws->dmask[l], (uint64_t)N * (uint64_t)cur_sz);
        else
            apply_derivative_batch(ws->delta[l], ws->act[l], (uint64_t)N * (uint64_t)cur_sz, net->act_func[l - 1]);
    }
}

/* grad = grad_scale * sum over the batch; overwrites the gradient buffers. */
static void batch_accumulate_gradients(NeuralNetwork *net, BatchWorkspace *ws, uint32_t N, float grad_scale) {
    for (uint32_t l = 0; l + 1 < net->layers; l++) {
        uint32_t in_sz  = net->topology[l];
        uint32_t out_sz = net->topology[l + 1];

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    (int)out_sz, (int)in_sz, (int)N,
                    grad_scale,
                    ws->delta[l + 1], (int)out_sz,
                    ws->act[l], (int)in_sz,
                    0.0f,
                    SPINGALETT_GRAD_W_MTX_PTR(net, l), (int)in_sz);

        cblas_sgemv(CblasRowMajor, CblasTrans,
                    (int)N, (int)out_sz,
                    grad_scale, ws->delta[l + 1], (int)out_sz,
                    ws->ones, 1,
                    0.0f, net->grad_biases + net->bias_offsets[l], 1);
    }
}

static float batch_compute_loss(const NeuralNetwork *net, const BatchWorkspace *ws,
                                const float *targets, uint32_t N) {
    uint32_t last = net->layers - 1;
    uint32_t n_out = net->topology[last];
    ActivationFunction output_act = net->act_func[last - 1];
    float total_error = 0.0f;

    for (uint32_t s = 0; s < N; s++) {
        size_t off = (size_t)s * (size_t)n_out;
        total_error += compute_sample_loss(ws->act[last] + off, targets + off, n_out, net->loss_func, output_act);
    }

    return total_error;
}

#endif /* SPINGALETT_HAS_OPENBLAS */

static void handle_autosave(NeuralNetwork *net, const TrainArgs *args, size_t epoch) {
    if (!args || args->autosave_mode == AUTOSAVE_OFF) return;
    if (args->autosave_interval == 0 || !args->autosave_path) return;

    SaveArgs save_args = {0};
    save_args.net = net;
    save_args.do_not_save_optimizer = args->autosave_do_not_save_optimizer;
    save_args.precision = args->autosave_precision;

    if (args->autosave_mode == AUTOSAVE_OVERWRITE) {
        save_args.filename = args->autosave_path;
        save_spingalett_struct_arguments(save_args);
    } else if (args->autosave_mode == AUTOSAVE_NEW_FILES) {
        const char *path = args->autosave_path;
        size_t path_len = strlen(path);

        const char *dot = strrchr(path, '.');
        const char *slash = strrchr(path, '/');
        const char *backslash = strrchr(path, '\\');
        const char *last_sep = NULL;
        if (slash && backslash)
            last_sep = (slash > backslash) ? slash : backslash;
        else
            last_sep = slash ? slash : backslash;

        bool has_ext = (dot && (!last_sep || dot > last_sep));

        size_t buf_len = path_len + 64;
        char *buf = (char *)malloc(buf_len);
        if (!buf) return;

        if (has_ext) {
            size_t base_len = (size_t)(dot - path);
            snprintf(buf, buf_len, "%.*s_epoch_%zu%s", (int)base_len, path, epoch, dot);
        } else {
            snprintf(buf, buf_len, "%s_epoch_%zu", path, epoch);
        }

        save_args.filename = buf;
        save_spingalett_struct_arguments(save_args);
        free(buf);
    }
}

#if defined(SPINGALETT_OPENBLAS_THREAD_CONTROL)
/* Multiply-adds per BLAS call below which OpenBLAS's thread pool costs more than it saves
   (measured: per-sample training of a 784-512-1000-10 net runs 1.5x faster on one thread,
   a 784-64-64-10 net 3.8x; mini-batches of 32 over 256-wide layers break even). */
#define BLAS_SINGLE_THREAD_WORK (4u << 20)

static int choose_blas_threads(const NeuralNetwork *net, const TrainArgs *args, uint32_t batch) {
    if (args->blas_num_threads > 0)
        return args->blas_num_threads;

    uint64_t widest = 0;
    for (uint32_t l = 0; l + 1 < net->layers; l++) {
        uint64_t w = (uint64_t)net->topology[l] * net->topology[l + 1];
        if (w > widest) widest = w;
    }
    if ((uint64_t)batch * widest < BLAS_SINGLE_THREAD_WORK)
        return 1;
    return (int)spingalett_get_num_threads();   /* 0 keeps OpenBLAS's own setting */
}
#endif

/* Flush-to-zero on the calling thread and, in OpenMP mode, on every worker of the team. */
static void flush_denormals_begin(ComputeMode mode) {
    spingalett_fp_flush_denormals_begin();
#if defined(_OPENMP)
    if (mode == COMPUTE_OPENMP) {
#pragma omp parallel
        if (omp_get_thread_num() != 0) spingalett_fp_flush_denormals_begin();
    }
#endif
    (void)mode;
}

static void flush_denormals_end(ComputeMode mode) {
#if defined(_OPENMP)
    if (mode == COMPUTE_OPENMP) {
#pragma omp parallel
        if (omp_get_thread_num() != 0) spingalett_fp_flush_denormals_end();
    }
#endif
    spingalett_fp_flush_denormals_end();
    (void)mode;
}

static inline bool should_report(size_t epoch, size_t epochs, size_t interval) {
    return interval > 0 && ((epoch % interval == 0) || (epoch == epochs));
}

typedef struct {
    NeuralNetwork *net;
    TrainArgs *args;
    ComputeMode mode;
    OptimizerStep opt;
    float beta1_pow, beta2_pow;

    bool use_blas_batch;
    uint32_t batch_size;        /* samples per optimizer step */
    uint32_t *order;            /* sample order (shuffled per epoch for mini-batches) */

    float **deltas;             /* per-sample path */
    float *deltas_flat;

    bool use_dropout;
    DropoutContext dropout;     /* dropout.dmask: per-sample path buffer */
#if defined(SPINGALETT_HAS_OPENBLAS)
    BatchWorkspace *ws;         /* BLAS batch path */
#endif
} Trainer;

static void trainer_free(Trainer *t) {
    free(t->order);
    free(t->deltas);
    spingalett_aligned_free(t->deltas_flat);
    spingalett_aligned_free(t->dropout.dmask);
#if defined(SPINGALETT_HAS_OPENBLAS)
    batch_workspace_free(t->ws);
#endif
}

static bool trainer_alloc(Trainer *t) {
    NeuralNetwork *net = t->net;
    uint32_t sample_count = t->args->sample_count;

    if (t->args->training_strategy == STRATEGY_SMALL_BATCH) {
        t->order = (uint32_t *)malloc((size_t)sample_count * sizeof(uint32_t));
        if (!t->order) return false;
        for (uint32_t i = 0; i < sample_count; i++)
            t->order[i] = i;
    }

#if defined(SPINGALETT_HAS_OPENBLAS)
    if (t->use_blas_batch) {
        t->ws = batch_workspace_create(net, t->batch_size, t->args->training_strategy == STRATEGY_SMALL_BATCH);
        return t->ws != NULL;
    }
#endif

    t->deltas = (float **)calloc(net->layers, sizeof(float *));
    t->deltas_flat = (float *)spingalett_aligned_calloc(net->total_neurons, sizeof(float));
    if (!t->deltas || !t->deltas_flat) return false;
    if (t->use_dropout) {
        t->dropout.dmask = (float *)spingalett_aligned_calloc(net->total_neurons, sizeof(float));
        if (!t->dropout.dmask) return false;
    }
    for (uint32_t l = 0; l < net->layers; l++)
        t->deltas[l] = t->deltas_flat + net->neuron_offsets[l];
    return true;
}

static void trainer_begin_step(Trainer *t) {
    t->net->time_step++;
    t->beta1_pow *= t->opt.beta1;
    t->beta2_pow *= t->opt.beta2;
    t->opt.m_factor = 1.0f / (1.0f - t->beta1_pow);
    t->opt.v_factor = 1.0f / (1.0f - t->beta2_pow);
}

/* Trains on the samples order[start .. start+count) (or start.. directly when order is NULL)
   and performs the optimizer step(s). Returns the summed loss when need_loss is set. */
static float trainer_step(Trainer *t, uint32_t start, uint32_t count, bool need_loss) {
    NeuralNetwork *net = t->net;
    const TrainArgs *args = t->args;
    uint32_t in_sz  = net->topology[0];
    uint32_t out_sz = net->topology[net->layers - 1];
    ActivationFunction out_act = net->act_func[net->layers - 2];
    float loss = 0.0f;

#if defined(SPINGALETT_HAS_OPENBLAS)
    if (t->use_blas_batch) {
        BatchWorkspace *ws = t->ws;
        const float *targets;
        if (t->order) {
            for (uint32_t s = 0; s < count; s++) {
                uint32_t idx = t->order[start + s];
                memcpy(ws->inputs + (size_t)s * in_sz, args->inputs + (size_t)idx * in_sz, in_sz * sizeof(float));
                memcpy(ws->targets + (size_t)s * out_sz, args->targets + (size_t)idx * out_sz, out_sz * sizeof(float));
            }
            targets = ws->targets;
        } else {
            /* Full batch reads the caller's arrays in place; act[0] is never written. */
            ws->act[0] = (float *)(args->inputs + (size_t)start * in_sz);
            targets = args->targets + (size_t)start * out_sz;
        }

        t->dropout.step = net->time_step;
        batch_forward(net, ws, count, t->use_dropout ? &t->dropout : NULL);
        if (need_loss)
            loss = batch_compute_loss(net, ws, targets, count);
        batch_compute_deltas(net, ws, targets, count);
        batch_accumulate_gradients(net, ws, count, 1.0f / (float)count);

        trainer_begin_step(t);
        apply_gradients(net, &t->opt, args->max_grad_norm, t->mode);
        return loss;
    }
#endif

    bool online = (args->training_strategy == STRATEGY_SAMPLE);
    float scale = 1.0f / (float)count;

    const DropoutContext *dropout = t->use_dropout ? &t->dropout : NULL;

    for (uint32_t s = 0; s < count; s++) {
        uint32_t idx = t->order ? t->order[start + s] : start + s;
        const float *target = args->targets + (size_t)idx * out_sz;

        /* Online steps hold one sample each, so its position within the step is 0. */
        t->dropout.step = net->time_step;
        t->dropout.position = online ? 0 : s;
        const float *out = spingalett_forward_pass(net, args->inputs + (size_t)idx * in_sz, t->mode, dropout);

        if (need_loss)
            loss += compute_sample_loss(out, target, out_sz, net->loss_func, out_act);

        compute_deltas(net, target, t->deltas, dropout ? t->dropout.dmask : NULL, t->mode);

        if (online) {
            if (args->max_grad_norm > 0.0f)
                clip_sample_gradient(net, t->deltas, args->max_grad_norm);
            trainer_begin_step(t);
            apply_rank1_updates(net, &t->opt, t->deltas, t->mode);
        } else {
            accumulate_gradients(net, t->deltas, scale, t->mode);
        }
    }

    if (!online) {
        trainer_begin_step(t);
        apply_gradients(net, &t->opt, args->max_grad_norm, t->mode);
        memset(net->grad_weights, 0, net->total_weights * sizeof(float));
        memset(net->grad_biases,  0, net->total_biases  * sizeof(float));
    }
    return loss;
}

void train_struct_arguments(TrainArgs args) {
    NeuralNetwork *net = args.net;
    TrainingMode training_mode = args.training_mode;
    TrainingStrategy training_strategy = args.training_strategy;
    uint32_t sample_count = args.sample_count;
    size_t epochs = args.epochs;

    if (args.learning_rate <= 0.0f) args.learning_rate = 0.01f;
    if (args.momentum <= 0.0f) args.momentum = 0.9f;
    if (args.beta1 <= 0.0f) args.beta1 = 0.9f;
    if (args.beta2 <= 0.0f) args.beta2 = 0.999f;
    if (args.epsilon <= 0.0f) args.epsilon = 1e-8f;

    if (args.callback && args.callback_interval == 0)
        args.callback_interval = 1;

    if ((unsigned)training_mode >= MODE_COUNT ||
        (unsigned)training_strategy >= STRATEGY_COUNT ||
        (unsigned)args.optimizer_type >= OPTIMIZER_COUNT) {
        set_error(SPINGALETT_ERR_INVALID, "Invalid training mode, strategy or optimizer");
        spingalett_log(LOG_ERROR, "Invalid training mode, strategy or optimizer");
        return;
    }

    if (training_mode == MODE_GENERATOR_FUNCTION) {
        set_error(SPINGALETT_ERR_INVALID, "Generator function training mode is not implemented yet");
        spingalett_log(LOG_ERROR, "Generator function training mode is not implemented yet");
        return;
    }

    if (!net || sample_count == 0 || epochs == 0 || !args.inputs || !args.targets) {
        set_error(SPINGALETT_ERR_INVALID, "Invalid training arguments (NULL net/inputs/targets or zero count/epochs)");
        spingalett_log(LOG_ERROR, "Invalid training arguments");
        return;
    }

    if (net->layers < 2) {
        set_error(SPINGALETT_ERR_INVALID, "Network must have at least 2 layers for training");
        spingalett_log(LOG_ERROR, "Network must have at least 2 layers for training");
        return;
    }

    for (uint32_t l = 1; l < net->layers - 1; l++) {
        if (net->act_func[l - 1] == ACT_SOFTMAX) {
            set_error(SPINGALETT_ERR_INVALID, "Softmax is not supported in hidden layers");
            spingalett_log(LOG_ERROR, "Softmax in hidden layer %u is not supported. Use softmax only in the output layer.", l);
            return;
        }
    }

    if (training_strategy == STRATEGY_SMALL_BATCH && args.batch_size == 0)
        args.batch_size = 32;
    if (training_strategy == STRATEGY_SMALL_BATCH && args.batch_size > sample_count)
        args.batch_size = sample_count;

    if (args.reset_optimizer) {
        memset(net->opt_m_weights, 0, net->total_weights * sizeof(float));
        memset(net->opt_v_weights, 0, net->total_weights * sizeof(float));
        memset(net->opt_m_biases,  0, net->total_biases  * sizeof(float));
        memset(net->opt_v_biases,  0, net->total_biases  * sizeof(float));
        net->time_step = 0;
        spingalett_log(LOG_INFO, "Optimizer state reset");
    }

    ComputeMode effective_mode = resolve_compute_mode();

#if defined(SPINGALETT_HAS_OPENBLAS)
    if (effective_mode == COMPUTE_OPENBLAS) {
        for (uint32_t l = 0; l < net->layers; l++) {
            if (net->topology[l] > (uint32_t)INT32_MAX) {
                spingalett_log(LOG_ERROR, "Layer %u size %u exceeds BLAS int limit", l, net->topology[l]);
                set_error(SPINGALETT_ERR_INVALID, "Layer size exceeds BLAS int limit");
                return;
            }
        }
        if (sample_count > (uint32_t)INT32_MAX) {
            spingalett_log(LOG_ERROR, "Sample count %u exceeds BLAS int limit", sample_count);
            set_error(SPINGALETT_ERR_INVALID, "Sample count exceeds BLAS int limit");
            return;
        }
    }
#endif

#if defined(_OPENMP)
    if (spingalett_get_num_threads() > 0) {
        omp_set_num_threads((int)spingalett_get_num_threads());
        if (effective_mode == COMPUTE_OPENMP && (int)spingalett_get_num_threads() > omp_get_num_procs())
            spingalett_log(LOG_WARNING, "%u threads requested but only %d processors are available; "
                           "oversubscription usually slows training down",
                           spingalett_get_num_threads(), omp_get_num_procs());
    }
#endif

    if (training_strategy == STRATEGY_SMALL_BATCH) {
        spingalett_log(LOG_INFO, "Starting training: loss=%s, strategy=%s, mode=%s, optimizer=%s, compute=%s, batch_size=%u",
            loss_func_names[net->loss_func],
            training_strategy_names[training_strategy],
            training_mode_names[training_mode],
            optimizer_names[args.optimizer_type],
            compute_mode_names[effective_mode < COMPUTE_COUNT ? effective_mode : 0],
            args.batch_size);
    } else {
        spingalett_log(LOG_INFO, "Starting training: loss=%s, strategy=%s, mode=%s, optimizer=%s, compute=%s",
            loss_func_names[net->loss_func],
            training_strategy_names[training_strategy],
            training_mode_names[training_mode],
            optimizer_names[args.optimizer_type],
            compute_mode_names[effective_mode < COMPUTE_COUNT ? effective_mode : 0]);
    }

    Trainer t = {0};
    t.net = net;
    t.args = &args;
    t.mode = effective_mode;
    t.opt = (OptimizerStep){
        .type = args.optimizer_type, .lr = args.learning_rate, .decay = args.weight_decay,
        .momentum = args.momentum, .beta1 = args.beta1, .beta2 = args.beta2, .epsilon = args.epsilon,
    };
    // Adam bias correction must continue from the persisted step count; restarting it at 1
    // on every train() call (or after loading a checkpoint) inflates the first updates ~10x.
    t.beta1_pow = powf(args.beta1, (float)net->time_step);
    t.beta2_pow = powf(args.beta2, (float)net->time_step);

    switch (training_strategy) {
        case STRATEGY_SAMPLE:      t.batch_size = 1; break;
        case STRATEGY_SMALL_BATCH: t.batch_size = args.batch_size; break;
        default:                   t.batch_size = sample_count; break;
    }
#if defined(SPINGALETT_HAS_OPENBLAS)
    t.use_blas_batch = (effective_mode == COMPUTE_OPENBLAS && training_strategy != STRATEGY_SAMPLE);
#endif

    if (net->dropout_rates[net->layers - 1] > 0.0f)
        spingalett_log(LOG_WARNING, "Dropout is not applied to the output layer; ignoring rate %g",
                       (double)net->dropout_rates[net->layers - 1]);
    t.use_dropout = spingalett_has_dropout(net);
    if (t.use_dropout)
        t.dropout.seed = rng_next64();

    if (!trainer_alloc(&t)) {
        trainer_free(&t);
        set_error(SPINGALETT_ERR_ALLOC, "Failed to allocate training buffers");
        spingalett_log(LOG_ERROR, "Failed to allocate training buffers");
        return;
    }

    // The accumulating paths add into grad_* and expect it to start at zero; a previous run
    // on another backend may have left the last batch's gradients there.
    memset(net->grad_weights, 0, net->total_weights * sizeof(float));
    memset(net->grad_biases,  0, net->total_biases  * sizeof(float));

    /* STRATEGY_SAMPLE walks the whole set in one call and steps after every sample. */
    uint32_t step_size = (training_strategy == STRATEGY_SAMPLE) ? sample_count : t.batch_size;
    uint32_t steps_per_epoch = (sample_count + step_size - 1) / step_size;

    flush_denormals_begin(effective_mode);

#if defined(SPINGALETT_OPENBLAS_THREAD_CONTROL)
    int saved_blas_threads = 0;
    if (effective_mode == COMPUTE_OPENBLAS) {
        int blas_threads = choose_blas_threads(net, &args, t.batch_size);
        if (blas_threads > 0) {
            saved_blas_threads = openblas_get_num_threads();
            openblas_set_num_threads(blas_threads);
            spingalett_log(LOG_DEBUG, "OpenBLAS threads: %d", blas_threads);
        }
    }
#endif

    bool lr_warned = false;

    for (size_t epoch = 1; epoch <= epochs; epoch++) {
        bool need_loss = should_report(epoch, epochs, args.report_interval) ||
                         (args.callback && should_report(epoch, epochs, args.callback_interval));
        float total_error = 0.0f;

        if (args.lr_scheduler) {
            float lr = args.lr_scheduler(epoch - 1, epochs, args.learning_rate, args.lr_scheduler_data);
            if (lr >= 0.0f && isfinite(lr)) {
                t.opt.lr = lr;
            } else if (!lr_warned) {
                lr_warned = true;
                spingalett_log(LOG_WARNING, "LR scheduler returned %g at epoch %zu; keeping lr=%g",
                               (double)lr, epoch, (double)t.opt.lr);
            }
        }

        if (t.order)
            spingalett_shuffle_indices(t.order, sample_count);

        for (uint32_t bi = 0; bi < steps_per_epoch; bi++) {
            uint32_t start = bi * step_size;
            uint32_t count = (sample_count - start < step_size) ? sample_count - start : step_size;
            total_error += trainer_step(&t, start, count, need_loss);
        }

        float current_error = total_error / (float)sample_count;

        if (should_report(epoch, epochs, args.report_interval)) {
            if (args.lr_scheduler)
                spingalett_log(LOG_INFO, "Epoch: %zu/%zu, Error: %f, LR: %g", epoch, epochs, (double)current_error, (double)t.opt.lr);
            else
                spingalett_log(LOG_INFO, "Epoch: %zu/%zu, Error: %f", epoch, epochs, (double)current_error);
        }

        if (args.nan_check_interval > 0 && (epoch % args.nan_check_interval == 0)) {
            if (check_nan_inf(net)) {
                spingalett_log(LOG_ERROR, "NaN/Inf detected in weights at epoch %zu, stopping training", epoch);
                break;
            }
        }

        if (should_report(epoch, epochs, args.autosave_interval))
            handle_autosave(net, &args, epoch);

        if (args.callback && should_report(epoch, epochs, args.callback_interval)) {
            if (args.callback(net, epoch, current_error)) {
                spingalett_log(LOG_INFO, "Training interrupted by callback at epoch %zu", epoch);
                break;
            }
        }
    }

#if defined(SPINGALETT_OPENBLAS_THREAD_CONTROL)
    if (saved_blas_threads > 0)
        openblas_set_num_threads(saved_blas_threads);
#endif

    flush_denormals_end(effective_mode);
    trainer_free(&t);
    spingalett_log(LOG_INFO, "Training completed.");
}
