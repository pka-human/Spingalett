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

static bool has_batch_norm(const NeuralNetwork *net) {
    for (uint32_t l = 1; l < net->layers; l++)
        if (net->shapes[l].type == LAYER_BATCH_NORM) return true;
    return false;
}

/* Weight decay of weight layer l: none for batch normalization's gamma (nor for any bias). */
static inline float layer_decay(const NeuralNetwork *net, uint32_t l, float decay) {
    return net->shapes[l + 1].type == LAYER_BATCH_NORM ? 0.0f : decay;
}

/* Apply the accumulated (already averaged) gradients of all layers. */
static void apply_gradients(NeuralNetwork *net, const OptimizerStep *o, float max_grad_norm, ComputeMode mode) {
    if (max_grad_norm > 0.0f)
        spingalett_clip_grad_norm(net, max_grad_norm);

    if (o->decay != 0.0f && has_batch_norm(net)) {
        for (uint32_t l = 0; l + 1 < net->layers; l++) {
            uint64_t w = net->weight_offsets[l];
            optimizer_update_array(o, net->weights + w, net->opt_m_weights + w, net->opt_v_weights + w,
                                   net->grad_weights + w,
                                   (uint64_t)spingalett_weight_rows(net, l) * spingalett_weight_row_len(net, l),
                                   layer_decay(net, l, o->decay), mode);
        }
    } else {
        optimizer_update_array(o, net->weights, net->opt_m_weights, net->opt_v_weights,
                               net->grad_weights, net->total_weights, o->decay, mode);
    }
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

/* dL/dz of the output layer from a caller-supplied dL/d(output): the vector-Jacobian product of
   the output activation, y * (g - g.y) for softmax. */
static void output_delta_from_grad(ActivationFunction act, const float *out, const float *grad,
                                   float *delta, uint32_t n) {
    if (act == ACT_SOFTMAX) {
        float dot = 0.0f;
        for (uint32_t j = 0; j < n; j++)
            dot += grad[j] * out[j];
        for (uint32_t j = 0; j < n; j++)
            delta[j] = out[j] * (grad[j] - dot);
        return;
    }
    memcpy(delta, grad, n * sizeof(float));
    apply_derivative_batch(delta, out, (uint64_t)n, act);
}

/* Smoothed targets of n samples: dst = (1 - eps) t + eps / k, k the outputs (2 for sigmoid outputs,
   each its own pair of classes). */
static void smooth_targets(const float *t, float *dst, uint32_t n, uint32_t outputs, ActivationFunction act, float eps,
                           ComputeMode mode) {
    const float keep = 1.0f - eps, share = eps / (act == ACT_SIGMOID ? 2.0f : (float)outputs);
    const uint64_t total = (uint64_t)n * outputs;
    SPINGALETT_PARALLEL_FOR(spingalett_use_omp(mode, total) && n > 1,
        for (int64_t i = 0; i < (int64_t)total; i++) dst[i] = keep * t[i] + share;
    );
    (void)mode;
}

/* ---- batch path (all backends): see Spingalett.Batch.c for the workspace and forward pass ---- */

/* Output deltas of N samples from targets (the network's loss) or from dL/d(output) rows. */
static void batch_output_deltas(NeuralNetwork *net, BatchWorkspace *ws, const float *targets,
                                const float *output_grads, uint32_t N) {
    uint32_t last = net->layers - 1;
    ActivationFunction last_act = net->act_func[last - 1];
    uint32_t n_out = net->topology[last];

    for (uint32_t s = 0; s < N; s++) {
        size_t off = (size_t)s * n_out;
        if (output_grads)
            output_delta_from_grad(last_act, ws->act[last] + off, output_grads + off, ws->delta[last] + off, n_out);
        else
            output_delta(net->loss_func, last_act, ws->act[last] + off, targets + off, ws->delta[last] + off, n_out);
    }
}

/* The gradient that input k of layer c gets from it (delta[c] = dL/d(c's pre-activation)): written
   to dst, or added to it with accumulate (dense, adding, concatenating and global pooling layers
   only), times the derivative of the input's activation `fused` (ACT_NONE: none; never with
   accumulate) when the kernel can apply it while the result is in cache. Returns whether it did. */
static bool input_gradient(NeuralNetwork *net, BatchWorkspace *ws, uint32_t c, uint32_t k, float *dst, uint32_t N,
                           ActivationFunction fused, bool accumulate, ComputeMode mode) {
    const uint32_t *in = spingalett_inputs(net, c), i = in[k];
    const uint32_t cur_sz = net->topology[i], next_sz = net->topology[c];
    const LayerShape *s = &net->shapes[c];
    switch (s->type) {
        case LAYER_DENSE:           /* dst = delta[c] * W, W stored [next x cur] */
            spingalett_gemm(ws->gemm, mode, false, false, N, cur_sz, next_sz, 1.0f, ws->delta[c], next_sz,
                            SPINGALETT_WEIGHT_MTX_PTR(net, c - 1), cur_sz, accumulate ? 1.0f : 0.0f, dst, cur_sz);
            return false;
        case LAYER_CONV2D:
            spingalett_conv_backward_data(net, c - 1, ws->delta[c], dst, N, ws->act[i], fused, ws->conv, ws->gemm, mode);
            return true;
        case LAYER_BATCH_NORM:      /* the sums are left by the backward pass */
            spingalett_bn_backward_data(net, c - 1, ws->act[i], ws->delta[c], dst, N, ws->bn_stats[c], ws->bn_sums[c],
                                        fused, ws->bn_coef, mode);
            return true;
        case LAYER_ADD:
        case LAYER_CONCAT: {        /* the input's channels of delta[c] */
            uint32_t c0 = 0;
            for (uint32_t j = 0; s->type == LAYER_CONCAT && j < k; j++) c0 += net->shapes[in[j]].channels;
            spingalett_slice_backward(ws->delta[c], s->channels, c0, net->shapes[i].channels, dst, ws->act[i], N,
                                      (uint64_t)s->height * s->width, fused, accumulate, mode);
            return true;
        }
        case LAYER_GLOBAL_AVG_POOL: {
            const LayerShape *x = &net->shapes[i];
            spingalett_global_pool_backward(ws->delta[c], dst, ws->act[i], N, (uint64_t)x->height * x->width, x->channels,
                                            fused, accumulate, mode);
            return true;
        }
        default:                    /* pooling has no activation: delta[c] is dL/d(its output) */
            spingalett_pool_backward(net, c - 1, ws->act[i], ws->delta[c], dst, N, fused, mode);
            return true;
    }
}

/* Propagates the output deltas back through the hidden layers, from the last layer down: a layer's
   delta is complete once every later layer that reads it has run. A layer that feeds several gets
   their gradients in that fixed order (the first written, the others added), so the sums do not
   depend on the thread count, and the derivative of its activation after the last. */
static void batch_backprop_hidden(NeuralNetwork *net, BatchWorkspace *ws, uint32_t N, ComputeMode mode) {
    const uint32_t last = net->layers - 1;
    memcpy(ws->pending, ws->uses, net->layers * sizeof(uint32_t));
    for (uint32_t c = last; c > 0; c--) {
        const uint32_t count = spingalett_input_count(net, c), *in = spingalett_inputs(net, c);
        const LayerType type = net->shapes[c].type;
        if (type == LAYER_BATCH_NORM)       /* also the parameters' gradients, whatever the input */
            spingalett_bn_backward_sums(net, c - 1, ws->act[in[0]], ws->delta[c], N, ws->bn_stats[c], ws->bn_sums[c],
                                        ws->bn_scratch, mode);
        for (uint32_t k = 0; k < count; k++) {
            const uint32_t l = in[k], cur_sz = net->topology[l];
            if (l == 0) continue;           /* the network's inputs need no gradient */
            ActivationFunction act = net->act_func[l - 1];
            if (ws->uses[l] > 1) {
                const bool first = ws->pending[l] == ws->uses[l], done = --ws->pending[l] == 0;
                const bool direct = first || type == LAYER_DENSE || type == LAYER_ADD || type == LAYER_CONCAT ||
                                    type == LAYER_GLOBAL_AVG_POOL;
                input_gradient(net, ws, c, k, direct ? ws->delta[l] : ws->gtmp, N, ACT_NONE, !first, mode);
                if (!direct || done)
                    spingalett_gradient_sum(ws->delta[l], direct ? NULL : ws->gtmp, ws->act[l], ws->dmask[l], N, cur_sz,
                                            act, done, mode);
                continue;
            }
            /* the derivative of layer l's activation is applied by the convolution and pooling
               kernels while their output is in cache; dropout masks hold it already */
            if (input_gradient(net, ws, c, k, ws->delta[l], N, ws->dmask[l] ? ACT_NONE : act, false, mode) &&
                !ws->dmask[l])
                continue;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, (uint64_t)N * cur_sz))
#endif
            for (int64_t s = 0; s < (int64_t)N; s++) {
                size_t off = (size_t)s * cur_sz;
                if (ws->dmask[l])
                    spingalett_vec_mul(ws->delta[l] + off, ws->dmask[l] + off, cur_sz);
                else
                    apply_derivative_batch(ws->delta[l] + off, ws->act[l] + off, cur_sz, act);
            }
        }
    }
}

/* grad = scale * (sum over the chunk) + beta * grad, so chunks of one step accumulate. With an
   optimizer step `o`, each layer is updated as soon as its gradient is complete, while the
   gradient is still in cache (the gradient must then be the whole step's, unclipped). */
static void batch_accumulate_gradients(NeuralNetwork *net, BatchWorkspace *ws, uint32_t N,
                                       float scale, float beta, ComputeMode mode, const OptimizerStep *o) {
    for (uint32_t l = 0; l + 1 < net->layers; l++) {
        const uint32_t src = spingalett_source(net, l + 1);
        uint32_t in_sz  = net->topology[src];
        uint32_t out_sz = net->topology[l + 1];
        LayerType type = net->shapes[l + 1].type;

        if (type == LAYER_DENSE) {
            /* gW[out x in] = delta^T[out x N] * act[N x in] */
            spingalett_gemm(ws->gemm, mode, true, false, out_sz, in_sz, N, scale,
                            ws->delta[l + 1], out_sz, ws->act[src], in_sz,
                            beta, SPINGALETT_GRAD_W_MTX_PTR(net, l), in_sz);

            float *gB = net->grad_biases + net->bias_offsets[l];
            if (beta == 0.0f)
                memset(gB, 0, out_sz * sizeof(float));
            for (uint32_t s = 0; s < N; s++)
                spingalett_vec_axpy(gB, ws->delta[l + 1] + (size_t)s * out_sz, out_sz, scale);
        } else if (type == LAYER_CONV2D) {
            spingalett_conv_backward_weights(net, l, ws->act[src], ws->delta[l + 1], N, scale, beta, ws->conv,
                                             ws->gemm, mode);
        } else if (type == LAYER_BATCH_NORM) {
            /* gamma: sum of dy * xhat, beta: sum of dy (left by the backward pass) */
            const double *sums = ws->bn_sums[l + 1];
            const uint32_t C = net->shapes[l + 1].channels;
            float *gW = net->grad_weights + net->weight_offsets[l], *gB = net->grad_biases + net->bias_offsets[l];
            for (uint32_t c = 0; c < C; c++) {
                float dg = (float)sums[C + c] * scale, db = (float)sums[c] * scale;
                gW[c] = beta == 0.0f ? dg : dg + beta * gW[c];
                gB[c] = beta == 0.0f ? db : db + beta * gB[c];
            }
        } else {
            continue;               /* pooling, adding and concatenating layers have no parameters */
        }

        if (o) {
            uint64_t w = net->weight_offsets[l], b = net->bias_offsets[l];
            uint64_t rows = spingalett_weight_rows(net, l);
            optimizer_update_array(o, net->weights + w, net->opt_m_weights + w, net->opt_v_weights + w,
                                   net->grad_weights + w, rows * spingalett_weight_row_len(net, l),
                                   layer_decay(net, l, o->decay), mode);
            optimizer_update_array(o, net->biases + b, net->opt_m_biases + b, net->opt_v_biases + b,
                                   net->grad_biases + b, rows, 0.0f, mode);
        }
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

    bool use_batch_path;        /* every strategy except per-sample training */
    uint32_t batch_size;        /* samples per optimizer step */
    uint32_t *order;            /* sample order, reshuffled every epoch (NULL = in order) */

    bool use_generator;
    uint32_t gen_capacity;      /* samples requested per generator call */
    float *gen_inputs;          /* generator buffers [gen_capacity rows] */
    float *gen_targets;

    float **deltas;             /* per-sample path */
    float *deltas_flat;

    bool use_dropout;
    DropoutContext dropout;     /* dropout.dmask: per-sample path buffer */
    BatchWorkspace *ws;         /* batch path */

    bool augment;               /* augment_shift or augment_flip */
    uint64_t augment_seed;      /* drawn once per train() call */
    float *augmented;           /* per-sample path: one augmented input */
    float *smoothed;            /* label smoothing: the targets of a chunk (or a sample), smoothed */

    BatchWorkspace *val_ws;     /* validation: inference workspace and output rows */
    float *val_out;
    float *best_params;         /* restore_best_weights: weights then biases of the best epoch */
} Trainer;

static void trainer_free(Trainer *t) {
    spingalett_aligned_free(t->augmented);
    spingalett_aligned_free(t->smoothed);
    spingalett_aligned_free(t->gen_inputs);
    spingalett_aligned_free(t->gen_targets);
    free(t->order);
    free(t->deltas);
    spingalett_aligned_free(t->deltas_flat);
    spingalett_aligned_free(t->dropout.dmask);
    spingalett_batch_workspace_free(t->ws);
    spingalett_batch_workspace_free(t->val_ws);
    spingalett_aligned_free(t->val_out);
    free(t->best_params);
}

static bool trainer_alloc(Trainer *t) {
    NeuralNetwork *net = t->net;
    uint32_t sample_count = t->args->sample_count;

    TrainingStrategy strategy = t->args->training_strategy;
    if ((strategy == STRATEGY_SMALL_BATCH || strategy == STRATEGY_SAMPLE) &&
        !t->use_generator && !t->args->do_not_shuffle) {
        t->order = (uint32_t *)malloc((size_t)sample_count * sizeof(uint32_t));
        if (!t->order) return false;
        for (uint32_t i = 0; i < sample_count; i++)
            t->order[i] = i;
    }

    if (t->args->val_count > 0) {
        uint32_t capacity = spingalett_batch_capacity(net, t->args->val_count);
        t->val_ws = spingalett_batch_workspace_create(net, capacity, false, false, t->mode);
        t->val_out = (float *)spingalett_aligned_alloc((size_t)capacity * net->topology[net->layers - 1] * sizeof(float));
        if (!t->val_ws || !t->val_out) return false;
    }

    if (t->args->restore_best_weights) {
        /* weights, biases and the running statistics of batch normalization */
        t->best_params = (float *)malloc((size_t)(net->total_weights + 3u * net->total_biases) * sizeof(float));
        if (!t->best_params) return false;
    }

    if (t->use_generator) {
        t->gen_inputs  = (float *)spingalett_aligned_alloc((size_t)t->gen_capacity * net->topology[0] * sizeof(float));
        t->gen_targets = (float *)spingalett_aligned_alloc((size_t)t->gen_capacity * net->topology[net->layers - 1] * sizeof(float));
        if (!t->gen_inputs || !t->gen_targets) return false;
    }

    uint32_t out_sz = net->topology[net->layers - 1];
    if (t->use_batch_path) {
        uint32_t capacity = spingalett_batch_capacity(net, t->batch_size);
        t->ws = spingalett_batch_workspace_create(net, capacity, true, t->order != NULL || t->augment, t->mode);
        if (t->args->label_smoothing > 0.0f)
            t->smoothed = (float *)spingalett_aligned_alloc((size_t)capacity * out_sz * sizeof(float));
        return t->ws != NULL && (t->args->label_smoothing <= 0.0f || t->smoothed);
    }
    if (t->args->label_smoothing > 0.0f) {
        t->smoothed = (float *)spingalett_aligned_alloc((size_t)out_sz * sizeof(float));
        if (!t->smoothed) return false;
    }
    if (t->augment) {
        t->augmented = (float *)spingalett_aligned_alloc((size_t)net->topology[0] * sizeof(float));
        if (!t->augmented) return false;
    }

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

static inline uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* dst = the image src (shape s) shifted and possibly mirrored as drawn for sample `position` of
   step `step`: dst(y, x) = src(y + dy, x' + dx) with x' = x, or W - 1 - x when mirrored, and 0
   where that falls outside the image. */
static void augment_image(const Trainer *t, const float *src, float *dst, uint64_t step, uint32_t position) {
    const LayerShape *s = &t->net->shapes[0];
    const uint32_t H = s->height, W = s->width, C = s->channels, k = t->args->augment_shift;
    uint64_t h = mix64(t->augment_seed ^ mix64(step * 0x9E3779B97F4A7C15ull + position));
    int64_t dy = k ? (int64_t)(h % (2u * k + 1u)) - k : 0, dx = k ? (int64_t)((h >> 21) % (2u * k + 1u)) - k : 0;
    bool mirror = t->args->augment_flip && ((h >> 42) & 1u);
    for (uint32_t y = 0; y < H; y++) {
        float *row = dst + (size_t)y * W * C;
        int64_t sy = (int64_t)y + dy;
        if (sy < 0 || sy >= (int64_t)H) { memset(row, 0, (size_t)W * C * sizeof(float)); continue; }
        const float *srow = src + (size_t)sy * W * C;
        if (!mirror) {
            /* columns [a, b) of dst read columns [a + dx, b + dx) of src */
            int64_t a = dx < 0 ? -dx : 0, b = dx > 0 ? (int64_t)W - dx : (int64_t)W;
            if (b < a) b = a;
            memset(row, 0, (size_t)a * C * sizeof(float));
            memcpy(row + (size_t)a * C, srow + (size_t)(a + dx) * C, (size_t)(b - a) * C * sizeof(float));
            memset(row + (size_t)b * C, 0, (size_t)((int64_t)W - b) * C * sizeof(float));
            continue;
        }
        for (uint32_t x = 0; x < W; x++) {             /* a pixel at a time: a few channels each */
            int64_t sx = (int64_t)(W - 1u - x) + dx;
            float *d = row + (size_t)x * C;
            if (sx < 0 || sx >= (int64_t)W) { for (uint32_t c = 0; c < C; c++) d[c] = 0.0f; continue; }
            const float *v = srow + (size_t)sx * C;
            for (uint32_t c = 0; c < C; c++) d[c] = v[c];
        }
    }
}

/* Trains on rows order[start .. start+count) of inputs/targets (rows start.. directly when order
   is NULL) and performs the optimizer step(s). Returns the summed loss. */
static float trainer_step(Trainer *t, const float *inputs, const float *targets_in, const uint32_t *order,
                          uint32_t start, uint32_t count) {
    NeuralNetwork *net = t->net;
    const TrainArgs *args = t->args;
    uint32_t in_sz  = net->topology[0];
    uint32_t out_sz = net->topology[net->layers - 1];
    ActivationFunction out_act = net->act_func[net->layers - 2];
    float loss = 0.0f;

    if (t->use_batch_path) {
        BatchWorkspace *ws = t->ws;
        const DropoutContext *dropout = t->use_dropout ? &t->dropout : NULL;
        t->dropout.step = net->time_step;

        /* The step's gradient is accumulated over chunks of at most ws->capacity samples. */
        for (uint32_t c0 = 0; c0 < count; c0 += ws->capacity) {
            uint32_t n = (count - c0 < ws->capacity) ? count - c0 : ws->capacity;
            const float *targets;
            if (order || t->augment) {
                /* the step's samples, gathered (and augmented) side by side */
                SPINGALETT_PARALLEL_FOR(spingalett_use_omp(t->mode, (uint64_t)n * in_sz * (t->augment ? 4u : 1u)) && n > 1,
                    for (int64_t s = 0; s < (int64_t)n; s++) {
                        uint32_t idx = order ? order[start + c0 + s] : start + c0 + (uint32_t)s;
                        const float *src = inputs + (size_t)idx * in_sz;
                        if (t->augment)
                            augment_image(t, src, ws->inputs + (size_t)s * in_sz, net->time_step, c0 + (uint32_t)s);
                        else
                            memcpy(ws->inputs + (size_t)s * in_sz, src, in_sz * sizeof(float));
                        if (order)
                            memcpy(ws->targets + (size_t)s * out_sz, targets_in + (size_t)idx * out_sz, out_sz * sizeof(float));
                    }
                );
                ws->act[0] = ws->inputs;
                targets = order ? ws->targets : targets_in + (size_t)(start + c0) * out_sz;
            } else {
                /* Contiguous rows are read in place; act[0] is never written. */
                ws->act[0] = (float *)(inputs + (size_t)(start + c0) * in_sz);
                targets = targets_in + (size_t)(start + c0) * out_sz;
            }

            if (t->smoothed) {
                smooth_targets(targets, t->smoothed, n, out_sz, out_act, args->label_smoothing, t->mode);
                targets = t->smoothed;
            }
            spingalett_batch_forward(net, ws, n, dropout, c0, t->mode);
            loss += batch_compute_loss(net, ws, targets, n);
            batch_output_deltas(net, ws, targets, NULL, n);
            batch_backprop_hidden(net, ws, n, t->mode);
            /* the last chunk of an unclipped step updates each layer right after its gradient */
            bool last = c0 + n == count, fused = last && args->max_grad_norm <= 0.0f;
            if (fused) trainer_begin_step(t);
            batch_accumulate_gradients(net, ws, n, 1.0f / (float)count, c0 == 0 ? 0.0f : 1.0f, t->mode,
                                       fused ? &t->opt : NULL);
            if (last && !fused) {
                trainer_begin_step(t);
                apply_gradients(net, &t->opt, args->max_grad_norm, t->mode);
            }
        }
        return loss;
    }

    /* Per-sample (online) training: one optimizer step per sample. */
    const DropoutContext *dropout = t->use_dropout ? &t->dropout : NULL;

    for (uint32_t s = 0; s < count; s++) {
        uint32_t idx = order ? order[start + s] : start + s;
        const float *target = targets_in + (size_t)idx * out_sz;
        if (t->smoothed) {
            smooth_targets(target, t->smoothed, 1, out_sz, out_act, args->label_smoothing, COMPUTE_SINGLE_THREADED);
            target = t->smoothed;
        }

        /* Every online step holds one sample, so its position within the step is 0. */
        t->dropout.step = net->time_step;
        t->dropout.position = 0;
        const float *x = inputs + (size_t)idx * in_sz;
        if (t->augment) {
            augment_image(t, x, t->augmented, net->time_step, 0);
            x = t->augmented;
        }
        const float *out = spingalett_forward_pass(net, x, t->mode, dropout);

        loss += compute_sample_loss(out, target, out_sz, net->loss_func, out_act);

        compute_deltas(net, target, t->deltas, dropout ? t->dropout.dmask : NULL, t->mode);

        if (args->max_grad_norm > 0.0f)
            clip_sample_gradient(net, t->deltas, args->max_grad_norm);
        trainer_begin_step(t);
        apply_rank1_updates(net, &t->opt, t->deltas, t->mode);
    }
    return loss;
}

/* Checks that a network can be trained; logs and sets the error otherwise. */
static bool check_trainable(const NeuralNetwork *net) {
    if (!net || net->layers < 2) {
        set_error(SPINGALETT_ERR_INVALID, "Network must have at least 2 layers for training");
        spingalett_log(LOG_ERROR, "Network must have at least 2 layers for training");
        return false;
    }

    ActivationFunction output_act = net->act_func[net->layers - 2];
    if (net->loss_func == LOSS_CROSS_ENTROPY && output_act != ACT_SOFTMAX && output_act != ACT_SIGMOID) {
        set_error(SPINGALETT_ERR_INVALID, "Cross-entropy loss needs a softmax or sigmoid output layer");
        spingalett_log(LOG_ERROR, "Cross-entropy loss needs a softmax or sigmoid output layer (got %s)",
                       act_func_names[output_act]);
        return false;
    }

    for (uint32_t l = 1; l < net->layers - 1; l++) {
        if (net->act_func[l - 1] == ACT_SOFTMAX) {
            set_error(SPINGALETT_ERR_INVALID, "Softmax is not supported in hidden layers");
            spingalett_log(LOG_ERROR, "Softmax in hidden layer %u is not supported. Use softmax only in the output layer.", l);
            return false;
        }
    }
    return spingalett_check_graph(net, "train");
}

static TrainReport train_failed(const char *message) {
    set_error(SPINGALETT_ERR_INVALID, message);
    spingalett_log(LOG_ERROR, "%s", message);
    return (TrainReport){.status = TRAIN_FAILED};
}

/* The parameters to restore: weights, biases, running statistics, back to back. */
static void save_params(float *dst, const NeuralNetwork *net) {
    uint64_t w = net->total_weights, b = net->total_biases;
    memcpy(dst, net->weights, w * sizeof(float));
    memcpy(dst + w, net->biases, b * sizeof(float));
    memcpy(dst + w + b, net->running_mean, b * sizeof(float));
    memcpy(dst + w + 2u * b, net->running_var, b * sizeof(float));
}

static void restore_params(NeuralNetwork *net, const float *src) {
    uint64_t w = net->total_weights, b = net->total_biases;
    memcpy(net->weights, src, w * sizeof(float));
    memcpy(net->biases, src + w, b * sizeof(float));
    memcpy(net->running_mean, src + w + b, b * sizeof(float));
    memcpy(net->running_var, src + w + 2u * b, b * sizeof(float));
}

TrainReport train_struct_arguments(TrainArgs args) {
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
        (unsigned)args.optimizer_type >= OPTIMIZER_COUNT ||
        (unsigned)args.monitor >= MONITOR_COUNT)
        return train_failed("Invalid training mode, strategy, optimizer or monitor");

    bool use_generator = (training_mode == MODE_GENERATOR_FUNCTION);

    if (!net || epochs == 0)
        return train_failed("Invalid training arguments (NULL net or zero epochs)");

    if (use_generator) {
        if (!args.generator || (training_strategy == STRATEGY_FULL_BATCH && sample_count == 0))
            return train_failed("Generator mode needs a generator (and sample_count for full batch)");
    } else if (sample_count == 0 || !args.inputs || !args.targets) {
        return train_failed("Invalid training arguments (NULL inputs/targets or zero sample count)");
    }

    bool has_validation = args.val_count > 0;
    if (has_validation && (!args.val_inputs || !args.val_targets))
        return train_failed("val_count is set but val_inputs or val_targets is NULL");

    MonitorMetric monitor = args.monitor;
    if (monitor == MONITOR_AUTO)
        monitor = has_validation ? MONITOR_VAL_LOSS : MONITOR_TRAIN_LOSS;
    if ((monitor == MONITOR_VAL_LOSS || monitor == MONITOR_VAL_ACCURACY) && !has_validation)
        return train_failed("Monitoring a validation metric needs validation data (val_inputs, val_targets, val_count)");
    const bool plateau = args.lr_plateau_patience > 0;
    if (plateau && !(args.lr_plateau_factor > 0.0f && args.lr_plateau_factor < 1.0f))
        return train_failed("lr_plateau_factor must be in (0, 1) when lr_plateau_patience is set");
    bool monitoring = has_validation || args.early_stopping_patience > 0 || args.restore_best_weights || plateau;
    bool higher_is_better = (monitor == MONITOR_VAL_ACCURACY);
    float min_delta = fabsf(args.early_stopping_min_delta);

    if (!check_trainable(net))
        return (TrainReport){.status = TRAIN_FAILED};

    /* Per-sample training of networks with convolution or pooling layers runs as mini-batches of
       one sample: the same steps, through the batch kernels. */
    if (training_strategy == STRATEGY_SAMPLE && !spingalett_all_dense(net)) {
        training_strategy = STRATEGY_SMALL_BATCH;
        args.batch_size = 1;
    }
    if (training_strategy == STRATEGY_SMALL_BATCH && args.batch_size == 0)
        args.batch_size = 32;
    if (training_strategy == STRATEGY_SMALL_BATCH && sample_count > 0 && args.batch_size > sample_count)
        args.batch_size = sample_count;
    if ((args.augment_shift > 0 || args.augment_flip) && net->shapes[0].height * net->shapes[0].width == 1)
        return train_failed("Augmentation needs an input layer with a height and width (an image)");
    if (!(args.label_smoothing >= 0.0f && args.label_smoothing < 1.0f))
        return train_failed("label_smoothing must be in [0, 1)");
    uint32_t step_samples = training_strategy == STRATEGY_SMALL_BATCH ? args.batch_size : sample_count;
    for (uint32_t l = 1; step_samples == 1 && l < net->layers; l++)
        if (net->shapes[l].type == LAYER_BATCH_NORM && net->shapes[l].height * net->shapes[l].width == 1)
            return train_failed("Batch normalization of a dense layer needs batches of at least 2 samples");

    ComputeMode effective_mode = resolve_compute_mode();

#if defined(SPINGALETT_HAS_OPENBLAS)
    if (effective_mode == COMPUTE_OPENBLAS) {
        for (uint32_t l = 0; l < net->layers; l++) {
            if (net->topology[l] > (uint32_t)INT32_MAX)
                return train_failed("Layer size exceeds BLAS int limit");
        }
        if (sample_count > (uint32_t)INT32_MAX)
            return train_failed("Sample count exceeds BLAS int limit");
    }
#endif

    if (args.reset_optimizer) {
        memset(net->opt_m_weights, 0, net->total_weights * sizeof(float));
        memset(net->opt_v_weights, 0, net->total_weights * sizeof(float));
        memset(net->opt_m_biases,  0, net->total_biases  * sizeof(float));
        memset(net->opt_v_biases,  0, net->total_biases  * sizeof(float));
        net->time_step = 0;
        spingalett_log(LOG_INFO, "Optimizer state reset");
    }

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

    t.use_generator = use_generator;
    if (use_generator) {
        /* Per-sample training pulls chunks of samples and steps after each one. */
        t.gen_capacity = (training_strategy == STRATEGY_SAMPLE)
                         ? ((sample_count > 0 && sample_count < 256u) ? sample_count : 256u)
                         : t.batch_size;
    }
    t.use_batch_path = (training_strategy != STRATEGY_SAMPLE);

    if (net->dropout_rates[net->layers - 1] > 0.0f)
        spingalett_log(LOG_WARNING, "Dropout is not applied to the output layer; ignoring rate %g",
                       (double)net->dropout_rates[net->layers - 1]);
    t.use_dropout = spingalett_has_dropout(net);
    if (t.use_dropout)
        t.dropout.seed = rng_next64();
    t.augment = args.augment_shift > 0 || args.augment_flip;
    if (t.augment)
        t.augment_seed = rng_next64();

    if (!trainer_alloc(&t)) {
        trainer_free(&t);
        set_error(SPINGALETT_ERR_ALLOC, "Failed to allocate training buffers");
        spingalett_log(LOG_ERROR, "Failed to allocate training buffers");
        return (TrainReport){.status = TRAIN_FAILED};
    }

    // The accumulating paths add into grad_* and expect it to start at zero; a previous run
    // on another backend may have left the last batch's gradients there.
    memset(net->grad_weights, 0, net->total_weights * sizeof(float));
    memset(net->grad_biases,  0, net->total_biases  * sizeof(float));

    /* STRATEGY_SAMPLE walks the whole set in one call and steps after every sample. */
    uint32_t step_size = (training_strategy == STRATEGY_SAMPLE) ? sample_count : t.batch_size;
    uint32_t steps_per_epoch = step_size ? (sample_count + step_size - 1) / step_size : 0;

    flush_denormals_begin(effective_mode);

#if defined(SPINGALETT_OPENBLAS_THREAD_CONTROL)
    int saved_blas_threads = 0;
    if (effective_mode == COMPUTE_OPENBLAS) {
        int blas_threads = choose_blas_threads(net, &args, t.batch_size < SPINGALETT_BATCH_CHUNK ? t.batch_size : SPINGALETT_BATCH_CHUNK);
        if (blas_threads > 0) {
            saved_blas_threads = openblas_get_num_threads();
            openblas_set_num_threads(blas_threads);
            spingalett_log(LOG_DEBUG, "OpenBLAS threads: %d", blas_threads);
        }
    }
#endif

    TrainReport report = {
        .status = TRAIN_COMPLETED, .train_loss = NAN, .has_validation = has_validation,
        .validation = {NAN, NAN}, .monitor = monitor, .best_value = NAN,
    };
    float best = higher_is_better ? -INFINITY : INFINITY;
    size_t epochs_without_improvement = 0, stale = 0;
    bool lr_warned = false;
    float lr_scale = 1.0f, base_lr = args.learning_rate;   /* reduce on plateau: the scale it reached */

    for (size_t epoch = 1; epoch <= epochs; epoch++) {
        float total_error = 0.0f;

        if (args.lr_scheduler) {
            float lr = args.lr_scheduler(epoch - 1, epochs, args.learning_rate, args.lr_scheduler_data);
            if (lr >= 0.0f && isfinite(lr)) {
                base_lr = lr;
            } else if (!lr_warned) {
                lr_warned = true;
                spingalett_log(LOG_WARNING, "LR scheduler returned %g at epoch %zu; keeping lr=%g",
                               (double)lr, epoch, (double)base_lr);
            }
        }
        /* reductions on plateau scale the base rate, down to lr_plateau_min_lr (never above the base) */
        t.opt.lr = lr_scale < 1.0f ? fmaxf(base_lr * lr_scale, fminf(args.lr_plateau_min_lr, base_lr)) : base_lr;

        uint64_t epoch_samples = 0;

        if (use_generator) {
            bool failed = false;
            for (;;) {
                uint32_t want = t.gen_capacity;
                if (sample_count > 0) {
                    if (epoch_samples >= sample_count) break;
                    if (sample_count - epoch_samples < want) want = (uint32_t)(sample_count - epoch_samples);
                }
                uint32_t got = args.generator(t.gen_inputs, t.gen_targets, want, args.generator_data);
                /* A generator that ends its passes with a 0 (such as spingalett_dataset_generator) answers
                   0 first when the previous epoch, cut by sample_count, never asked past its last sample. */
                if (got == 0 && epoch_samples == 0)
                    got = args.generator(t.gen_inputs, t.gen_targets, want, args.generator_data);
                if (got == 0) break;
                if (got > want) {
                    set_error(SPINGALETT_ERR_INVALID, "Generator returned more samples than requested");
                    spingalett_log(LOG_ERROR, "Generator returned %u samples for a request of %u; stopping training", got, want);
                    failed = true;
                    break;
                }
                total_error += trainer_step(&t, t.gen_inputs, t.gen_targets, NULL, 0, got);
                epoch_samples += got;
                if (training_strategy == STRATEGY_FULL_BATCH) break;
            }
            if (failed) { report.status = TRAIN_FAILED; break; }
            if (epoch_samples == 0) {
                spingalett_log(LOG_WARNING, "Generator produced no samples in epoch %zu; stopping training", epoch);
                report.status = TRAIN_NO_DATA;
                break;
            }
        } else {
            if (t.order)
                spingalett_shuffle_indices(t.order, sample_count);

            for (uint32_t bi = 0; bi < steps_per_epoch; bi++) {
                uint32_t start = bi * step_size;
                uint32_t count = (sample_count - start < step_size) ? sample_count - start : step_size;
                total_error += trainer_step(&t, args.inputs, args.targets, t.order, start, count);
            }
            epoch_samples = sample_count;
        }

        report.epochs_run = epoch;
        report.train_loss = total_error / (float)epoch_samples;

        if (args.nan_check_interval > 0 && (epoch % args.nan_check_interval == 0) && check_nan_inf(net)) {
            spingalett_log(LOG_ERROR, "NaN/Inf detected in weights at epoch %zu, stopping training", epoch);
            report.status = TRAIN_DIVERGED;
            break;
        }

        if (has_validation) {
            double loss;
            uint32_t correct;
            spingalett_batch_evaluate(net, t.val_ws, t.val_out, args.val_inputs, args.val_targets, args.val_count,
                                      effective_mode, &loss, &correct);
            report.validation.loss = (float)(loss / args.val_count);
            report.validation.accuracy = (float)correct / (float)args.val_count;
        }

        bool improved = false;
        if (monitoring) {
            float value = monitor == MONITOR_TRAIN_LOSS ? report.train_loss
                        : monitor == MONITOR_VAL_LOSS   ? report.validation.loss
                                                        : report.validation.accuracy;
            improved = higher_is_better ? value > best + min_delta : value < best - min_delta;
            if (report.best_epoch == 0 && isfinite(value))
                improved = true;    /* the first finite value is the first best, whatever min_delta */
            if (improved) {
                best = value;
                report.best_epoch = epoch;
                report.best_value = value;
                epochs_without_improvement = 0;
                if (t.best_params)
                    save_params(t.best_params, net);
            } else {
                epochs_without_improvement++;
            }
            stale = improved ? 0 : stale + 1;
            if (plateau && stale >= args.lr_plateau_patience) {
                lr_scale *= args.lr_plateau_factor;
                stale = 0;
                spingalett_log(LOG_INFO, "No improvement for %zu epochs: learning rate scaled to %g of the schedule's",
                               args.lr_plateau_patience, (double)lr_scale);
            }
        }

        if (should_report(epoch, epochs, args.report_interval)) {
            char lr_text[32] = "", val_text[64] = "";
            if (args.lr_scheduler || lr_scale < 1.0f)
                snprintf(lr_text, sizeof lr_text, ", LR: %g", (double)t.opt.lr);
            if (has_validation)
                snprintf(val_text, sizeof val_text, ", Val loss: %f, Val accuracy: %.2f%%",
                         (double)report.validation.loss, 100.0 * (double)report.validation.accuracy);
            spingalett_log(LOG_INFO, "Epoch: %zu/%zu, Error: %f%s%s%s", epoch, epochs, (double)report.train_loss,
                           val_text, lr_text, improved && monitoring ? " (best)" : "");
        }

        if (should_report(epoch, epochs, args.autosave_interval))
            handle_autosave(net, &args, epoch);

        if (args.callback && should_report(epoch, epochs, args.callback_interval)) {
            TrainProgress progress = {
                .epoch = epoch, .epochs = epochs, .train_loss = report.train_loss, .learning_rate = t.opt.lr,
                .has_validation = has_validation, .validation = report.validation, .monitor = monitor,
                .best_epoch = report.best_epoch, .best_value = report.best_value, .improved = improved,
            };
            if (args.callback(net, &progress, args.callback_data)) {
                spingalett_log(LOG_INFO, "Training interrupted by callback at epoch %zu", epoch);
                report.status = TRAIN_INTERRUPTED;
                break;
            }
        }

        if (args.early_stopping_patience > 0 && epochs_without_improvement >= args.early_stopping_patience) {
            spingalett_log(LOG_INFO, "Early stopping at epoch %zu: no improvement since epoch %zu",
                           epoch, report.best_epoch);
            report.status = TRAIN_EARLY_STOPPED;
            break;
        }
    }

    if (t.best_params && report.best_epoch > 0 && report.best_epoch != report.epochs_run) {
        restore_params(net, t.best_params);
        report.restored_best = true;
        spingalett_log(LOG_INFO, "Restored the weights of epoch %zu", report.best_epoch);
    }

#if defined(SPINGALETT_OPENBLAS_THREAD_CONTROL)
    if (saved_blas_threads > 0)
        openblas_set_num_threads(saved_blas_threads);
#endif

    flush_denormals_end(effective_mode);
    trainer_free(&t);
    spingalett_log(LOG_INFO, "Training completed.");
    return report;
}

/* ---- low-level training API ---- */

struct SpingalettTrainer {
    NeuralNetwork *net;
    ComputeMode mode;           /* resolved when the trainer was created */
    BatchWorkspace *ws;         /* training workspace for max_batch samples (inputs copied in) */
    bool use_dropout;
    DropoutContext dropout;
    uint32_t pending;           /* samples of the last forward pass, until it is back-propagated */
    uint32_t accumulated;       /* samples in the network's gradient since the last step */
    float label_smoothing;
    float *smoothed;            /* the smoothed targets of a batch */
};

SpingalettTrainer *spingalett_trainer_new(NeuralNetwork *net, uint32_t max_batch) {
    if (max_batch == 0) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_trainer_new: max_batch is 0");
        return NULL;
    }
    if (!check_trainable(net))
        return NULL;
    SpingalettTrainer *tr = (SpingalettTrainer *)calloc(1, sizeof(SpingalettTrainer));
    if (!tr) {
        set_error(SPINGALETT_ERR_ALLOC, "spingalett_trainer_new: allocation failed");
        return NULL;
    }
    tr->net = net;
    tr->mode = resolve_compute_mode();
    tr->use_dropout = spingalett_has_dropout(net);
    if (tr->use_dropout)
        tr->dropout.seed = rng_next64();
    tr->ws = spingalett_batch_workspace_create(net, max_batch, true, true, tr->mode);
    if (!tr->ws) {
        free(tr);
        set_error(SPINGALETT_ERR_ALLOC, "spingalett_trainer_new: workspace allocation failed");
        return NULL;
    }
    return tr;
}

void spingalett_trainer_free(SpingalettTrainer *tr) {
    if (!tr) return;
    spingalett_batch_workspace_free(tr->ws);
    spingalett_aligned_free(tr->smoothed);
    free(tr);
}

bool spingalett_trainer_set_label_smoothing(SpingalettTrainer *tr, float label_smoothing) {
    if (!tr || !(label_smoothing >= 0.0f && label_smoothing < 1.0f)) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_trainer_set_label_smoothing: NULL trainer or a value outside [0, 1)");
        return false;
    }
    if (label_smoothing > 0.0f && !tr->smoothed) {
        const NeuralNetwork *net = tr->net;
        tr->smoothed = (float *)spingalett_aligned_alloc((size_t)tr->ws->capacity * net->topology[net->layers - 1] *
                                                         sizeof(float));
        if (!tr->smoothed) {
            set_error(SPINGALETT_ERR_ALLOC, "spingalett_trainer_set_label_smoothing: allocation failed");
            return false;
        }
    }
    tr->label_smoothing = label_smoothing;
    return true;
}

const float *spingalett_trainer_forward(SpingalettTrainer *tr, const float *inputs, uint32_t count) {
    if (!tr || !inputs || count == 0 || count > tr->ws->capacity) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_trainer_forward: NULL argument, or count is 0 or above max_batch");
        return NULL;
    }
    NeuralNetwork *net = tr->net;
    BatchWorkspace *ws = tr->ws;
    memcpy(ws->inputs, inputs, (size_t)count * net->topology[0] * sizeof(float));
    ws->act[0] = ws->inputs;

    /* Masks depend on the step and on the sample's position within it, as in train(). */
    tr->dropout.step = net->time_step;
    flush_denormals_begin(tr->mode);
    spingalett_batch_forward(net, ws, count, tr->use_dropout ? &tr->dropout : NULL, tr->accumulated, tr->mode);
    flush_denormals_end(tr->mode);
    tr->pending = count;
    return ws->act[net->layers - 1];
}

static bool trainer_backward(SpingalettTrainer *tr, const float *targets, const float *output_grads, float *loss) {
    if (!tr || !(targets || output_grads)) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_trainer_backward: NULL argument");
        return false;
    }
    if (tr->pending == 0) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_trainer_backward: no forward pass to back-propagate");
        return false;
    }
    NeuralNetwork *net = tr->net;
    uint32_t n = tr->pending;
    if (targets && tr->label_smoothing > 0.0f) {
        smooth_targets(targets, tr->smoothed, n, net->topology[net->layers - 1], net->act_func[net->layers - 2],
                       tr->label_smoothing, tr->mode);
        targets = tr->smoothed;
    }
    if (loss)
        *loss = batch_compute_loss(net, tr->ws, targets, n);

    flush_denormals_begin(tr->mode);
    batch_output_deltas(net, tr->ws, targets, output_grads, n);
    batch_backprop_hidden(net, tr->ws, n, tr->mode);
    batch_accumulate_gradients(net, tr->ws, n, 1.0f, tr->accumulated == 0 ? 0.0f : 1.0f, tr->mode, NULL);
    flush_denormals_end(tr->mode);

    tr->accumulated += n;
    tr->pending = 0;
    return true;
}

float spingalett_trainer_backward(SpingalettTrainer *tr, const float *targets) {
    float loss = NAN;
    return trainer_backward(tr, targets, NULL, &loss) ? loss : NAN;
}

bool spingalett_trainer_backward_output_grads(SpingalettTrainer *tr, const float *output_grads) {
    return trainer_backward(tr, NULL, output_grads, NULL);
}

void spingalett_trainer_zero_grad(SpingalettTrainer *tr) {
    if (!tr) return;
    memset(tr->net->grad_weights, 0, tr->net->total_weights * sizeof(float));
    memset(tr->net->grad_biases, 0, tr->net->total_biases * sizeof(float));
    tr->accumulated = 0;
    tr->pending = 0;
}

bool spingalett_trainer_step(SpingalettTrainer *tr, const OptimizerArgs *optimizer) {
    if (!tr || !optimizer || (unsigned)optimizer->type >= OPTIMIZER_COUNT) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_trainer_step: NULL argument or invalid optimizer");
        return false;
    }
    if (tr->accumulated == 0) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_trainer_step: no gradient accumulated since the last step");
        return false;
    }
    NeuralNetwork *net = tr->net;
    OptimizerStep o = {
        .type = optimizer->type,
        .lr = optimizer->learning_rate > 0.0f ? optimizer->learning_rate : 0.01f,
        .decay = optimizer->weight_decay,
        .momentum = optimizer->momentum > 0.0f ? optimizer->momentum : 0.9f,
        .beta1 = optimizer->beta1 > 0.0f ? optimizer->beta1 : 0.9f,
        .beta2 = optimizer->beta2 > 0.0f ? optimizer->beta2 : 0.999f,
        .epsilon = optimizer->epsilon > 0.0f ? optimizer->epsilon : 1e-8f,
    };
    net->time_step++;
    o.m_factor = 1.0f / (1.0f - powf(o.beta1, (float)net->time_step));
    o.v_factor = 1.0f / (1.0f - powf(o.beta2, (float)net->time_step));

    flush_denormals_begin(tr->mode);
    float scale = 1.0f / (float)tr->accumulated;
    spingalett_vec_scale(net->grad_weights, net->total_weights, scale);
    spingalett_vec_scale(net->grad_biases, net->total_biases, scale);
    apply_gradients(net, &o, optimizer->max_grad_norm, tr->mode);
    flush_denormals_end(tr->mode);

    tr->accumulated = 0;
    tr->pending = 0;
    return true;
}

float spingalett_train_on_batch(SpingalettTrainer *tr, const float *inputs, const float *targets,
                                uint32_t count, const OptimizerArgs *optimizer) {
    if (!targets) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_train_on_batch: targets is NULL");
        return NAN;
    }
    if (!spingalett_trainer_forward(tr, inputs, count))
        return NAN;
    float loss = spingalett_trainer_backward(tr, targets);
    if (isnan(loss) || !spingalett_trainer_step(tr, optimizer))
        return NAN;
    return loss / (float)count;
}
