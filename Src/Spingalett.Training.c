#include "Spingalett.Private.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

static bool check_nan_inf(NeuralNetwork *net) {
    for (uint64_t i = 0; i < net->total_weights; i++) {
        if (!isfinite(net->weights[i])) return true;
    }
    for (uint64_t i = 0; i < net->total_biases; i++) {
        if (!isfinite(net->biases[i])) return true;
    }
    return false;
}

static void compute_deltas(NeuralNetwork *net, float *target, float **deltas, ComputeMode mode) {
    uint32_t last = net->layers - 1;
    ActivationFunction last_act = net->act_func[last - 1];
    uint32_t n_out = net->topology[last];

    if (net->loss_func == LOSS_MSE && last_act == ACT_SOFTMAX) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < n_out; j++) {
            float y = SPINGALETT_NEURON(net, last, j);
            sum += (y - target[j]) * y;
        }
        for (uint32_t i = 0; i < n_out; i++) {
            float y = SPINGALETT_NEURON(net, last, i);
            deltas[last][i] = y * ((y - target[i]) - sum);
        }
    } else {
        for (uint32_t i = 0; i < n_out; i++) {
            float output = SPINGALETT_NEURON(net, last, i);
            if (net->loss_func == LOSS_CROSS_ENTROPY &&
                (last_act == ACT_SOFTMAX || last_act == ACT_SIGMOID))
                deltas[last][i] = output - target[i];
            else
                deltas[last][i] = (output - target[i]) * derivative(output, last_act);
        }
    }

    for (uint32_t l = last - 1; l > 0; l--) {
        uint32_t cur_sz  = net->topology[l];
        uint32_t next_sz = net->topology[l + 1];
        ActivationFunction prev_act = net->act_func[l - 1];
        float *W = SPINGALETT_WEIGHT_MTX_PTR(net, l);

#if defined(SPINGALETT_HAS_OPENBLAS)
        if (mode == COMPUTE_OPENBLAS) {
            cblas_sgemv(CblasRowMajor, CblasTrans,
                        (int)next_sz, (int)cur_sz,
                        1.0f, W, (int)cur_sz,
                        deltas[l + 1], 1,
                        0.0f, deltas[l], 1);
        } else
#endif
        {
            memset(deltas[l], 0, cur_sz * sizeof(float));
            for (uint32_t j = 0; j < next_sz; j++) {
                float d = deltas[l + 1][j];
                float *w_row = W + (uint64_t)j * (uint64_t)cur_sz;
                spingalett_vec_axpy(deltas[l], w_row, (uint64_t)cur_sz, d);
            }
        }

        for (uint32_t k = 0; k < cur_sz; k++)
            deltas[l][k] *= derivative(SPINGALETT_NEURON(net, l, k), prev_act);
    }

    (void)mode;
}

static void backpropagation(NeuralNetwork *net, float *target, TrainArgs *args, float **deltas,
                             float *beta1_pow_p, float *beta2_pow_p, ComputeMode mode) {
    uint32_t last = net->layers - 1;
    float lr = args->learning_rate;
    float decay = args->weight_decay;
    float momentum = args->momentum;
    float beta1 = args->beta1;
    float beta2 = args->beta2;
    float epsilon = args->epsilon;

    compute_deltas(net, target, deltas, mode);

    net->time_step++;
    *beta1_pow_p *= beta1;
    *beta2_pow_p *= beta2;

    for (uint32_t l = 0; l < last; l++) {
        uint32_t in_sz  = net->topology[l];
        uint32_t out_sz = net->topology[l + 1];
        float *W   = SPINGALETT_WEIGHT_MTX_PTR(net, l);
        float *b   = net->biases + net->bias_offsets[l];
        float *mW  = net->opt_m_weights + net->weight_offsets[l];
        float *vW  = net->opt_v_weights + net->weight_offsets[l];
        float *mB  = net->opt_m_biases  + net->bias_offsets[l];
        float *vB  = net->opt_v_biases  + net->bias_offsets[l];
        const float *x = SPINGALETT_LAYER_PTR(net, l);
        float *delta = deltas[l + 1];

        if (args->optimizer_type == OPTIMIZER_SGD) {
#if defined(SPINGALETT_HAS_OPENBLAS)
            if (mode == COMPUTE_OPENBLAS) {
                if (decay > 0.0f)
                    cblas_sscal((int)((uint64_t)out_sz * (uint64_t)in_sz),
                                1.0f - lr * decay, W, 1);
                cblas_sger(CblasRowMajor, (int)out_sz, (int)in_sz,
                           -lr, delta, 1, x, 1, W, (int)in_sz);
            } else
#endif
            {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(mode == COMPUTE_OPENMP)
#endif
                for (uint32_t j = 0; j < out_sz; j++) {
                    float dj = delta[j];
                    float *Wj = W + (uint64_t)j * (uint64_t)in_sz;
                    for (uint32_t k = 0; k < in_sz; k++) {
                        float grad = dj * x[k];
                        if (decay > 0.0f) grad += decay * Wj[k];
                        Wj[k] -= lr * grad;
                    }
                }
            }
            for (uint32_t j = 0; j < out_sz; j++)
                b[j] -= lr * delta[j];

        } else if (args->optimizer_type == OPTIMIZER_MOMENTUM) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(mode == COMPUTE_OPENMP)
#endif
            for (uint32_t j = 0; j < out_sz; j++) {
                float dj = delta[j];
                uint64_t row_off = (uint64_t)j * (uint64_t)in_sz;
                for (uint32_t k = 0; k < in_sz; k++) {
                    uint64_t idx = row_off + k;
                    float grad = dj * x[k];
                    if (decay > 0.0f) grad += decay * W[idx];
                    mW[idx] = momentum * mW[idx] + grad;
                    W[idx] -= lr * mW[idx];
                }
            }
            for (uint32_t j = 0; j < out_sz; j++) {
                mB[j] = momentum * mB[j] + delta[j];
                b[j] -= lr * mB[j];
            }

        } else if (args->optimizer_type == OPTIMIZER_ADAM || args->optimizer_type == OPTIMIZER_ADAMW) {
            float m_factor = 1.0f / (1.0f - *beta1_pow_p);
            float v_factor = 1.0f / (1.0f - *beta2_pow_p);
            bool is_adamw = (args->optimizer_type == OPTIMIZER_ADAMW);
            float one_minus_b1 = 1.0f - beta1;
            float one_minus_b2 = 1.0f - beta2;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(mode == COMPUTE_OPENMP)
#endif
            for (uint32_t j = 0; j < out_sz; j++) {
                float dj = delta[j];
                uint64_t row_off = (uint64_t)j * (uint64_t)in_sz;
                for (uint32_t k = 0; k < in_sz; k++) {
                    uint64_t idx = row_off + k;
                    float grad = dj * x[k];
                    if (is_adamw) {
                        if (decay > 0.0f) W[idx] *= (1.0f - lr * decay);
                    } else {
                        if (decay > 0.0f) grad += decay * W[idx];
                    }
                    mW[idx] = beta1 * mW[idx] + one_minus_b1 * grad;
                    vW[idx] = beta2 * vW[idx] + one_minus_b2 * (grad * grad);
                    float m_hat = mW[idx] * m_factor;
                    float v_hat = vW[idx] * v_factor;
                    W[idx] -= lr * m_hat / sqrtf(v_hat + epsilon);
                }
            }
            for (uint32_t j = 0; j < out_sz; j++) {
                float grad = delta[j];
                mB[j] = beta1 * mB[j] + one_minus_b1 * grad;
                vB[j] = beta2 * vB[j] + one_minus_b2 * (grad * grad);
                float m_hat = mB[j] * m_factor;
                float v_hat = vB[j] * v_factor;
                b[j] -= lr * m_hat / sqrtf(v_hat + epsilon);
            }

        } else if (args->optimizer_type == OPTIMIZER_RMSPROP) {
            float one_minus_b2 = 1.0f - beta2;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(mode == COMPUTE_OPENMP)
#endif
            for (uint32_t j = 0; j < out_sz; j++) {
                float dj = delta[j];
                uint64_t row_off = (uint64_t)j * (uint64_t)in_sz;
                for (uint32_t k = 0; k < in_sz; k++) {
                    uint64_t idx = row_off + k;
                    float grad = dj * x[k];
                    if (decay > 0.0f) grad += decay * W[idx];
                    vW[idx] = beta2 * vW[idx] + one_minus_b2 * (grad * grad);
                    W[idx] -= lr * grad / sqrtf(vW[idx] + epsilon);
                }
            }
            for (uint32_t j = 0; j < out_sz; j++) {
                float grad = delta[j];
                vB[j] = beta2 * vB[j] + one_minus_b2 * (grad * grad);
                b[j] -= lr * grad / sqrtf(vB[j] + epsilon);
            }
        }
    }
}

static void accumulate_gradients_single(NeuralNetwork *net, float *target, TrainArgs *args, float **deltas, ComputeMode mode) {
    uint32_t last = net->layers - 1;
    (void)args;

    compute_deltas(net, target, deltas, mode);

    for (uint32_t l = 0; l < last; l++) {
        uint32_t in_sz  = net->topology[l];
        uint32_t out_sz = net->topology[l + 1];
        float *gW = SPINGALETT_GRAD_W_MTX_PTR(net, l);
        float *gB = net->grad_biases + net->bias_offsets[l];
        const float *x = SPINGALETT_LAYER_PTR(net, l);
        float *delta = deltas[l + 1];

#if defined(SPINGALETT_HAS_OPENBLAS)
        if (mode == COMPUTE_OPENBLAS) {
            cblas_sger(CblasRowMajor, (int)out_sz, (int)in_sz,
                       1.0f, delta, 1, x, 1, gW, (int)in_sz);
        } else
#endif
        {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(mode == COMPUTE_OPENMP)
#endif
            for (uint32_t j = 0; j < out_sz; j++) {
                float dj = delta[j];
                float *gWj = gW + (uint64_t)j * (uint64_t)in_sz;
                for (uint32_t k = 0; k < in_sz; k++)
                    gWj[k] += dj * x[k];
            }
        }

        for (uint32_t j = 0; j < out_sz; j++)
            gB[j] += delta[j];
    }
}

#if defined(SPINGALETT_HAS_OPENBLAS)

typedef struct {
    float **act;
    float **delta;
    float *act_flat;
    float *delta_flat;
    float *ones;
    uint32_t capacity;
} BatchWorkspace;

static BatchWorkspace *batch_workspace_create(NeuralNetwork *net, uint32_t max_batch) {
    BatchWorkspace *ws = (BatchWorkspace *)calloc(1, sizeof(BatchWorkspace));
    if (!ws) return NULL;

    ws->capacity = max_batch;
    ws->act   = (float **)malloc(net->layers * sizeof(float *));
    ws->delta = (float **)malloc(net->layers * sizeof(float *));

    size_t total = 0;
    for (uint32_t l = 0; l < net->layers; l++)
        total += (size_t)max_batch * (size_t)net->topology[l];

    ws->act_flat   = (float *)spingalett_aligned_calloc(total, sizeof(float));
    ws->delta_flat = (float *)spingalett_aligned_calloc(total, sizeof(float));
    ws->ones       = (float *)spingalett_aligned_alloc(max_batch * sizeof(float));

    if (!ws->act || !ws->delta || !ws->act_flat || !ws->delta_flat || !ws->ones) {
        spingalett_aligned_free(ws->act_flat);
        spingalett_aligned_free(ws->delta_flat);
        spingalett_aligned_free(ws->ones);
        free(ws->act);
        free(ws->delta);
        free(ws);
        return NULL;
    }

    for (uint32_t i = 0; i < max_batch; i++)
        ws->ones[i] = 1.0f;

    size_t off = 0;
    for (uint32_t l = 0; l < net->layers; l++) {
        size_t sz = (size_t)max_batch * (size_t)net->topology[l];
        ws->act[l]   = ws->act_flat + off;
        ws->delta[l] = ws->delta_flat + off;
        off += sz;
    }

    return ws;
}

static void batch_workspace_free(BatchWorkspace *ws) {
    if (!ws) return;
    spingalett_aligned_free(ws->act_flat);
    spingalett_aligned_free(ws->delta_flat);
    spingalett_aligned_free(ws->ones);
    free(ws->act);
    free(ws->delta);
    free(ws);
}

static void batch_forward(NeuralNetwork *net, BatchWorkspace *ws,
                          const float *inputs, uint32_t N) {
    uint32_t in_size = net->topology[0];
    memcpy(ws->act[0], inputs, (size_t)N * (size_t)in_size * sizeof(float));

    for (uint32_t l = 1; l < net->layers; l++) {
        uint32_t prev_size = net->topology[l - 1];
        uint32_t curr_size = net->topology[l];
        float *W = SPINGALETT_WEIGHT_MTX_PTR(net, l - 1);
        float *bias = net->biases + net->bias_offsets[l - 1];
        float *A = ws->act[l - 1];
        float *C = ws->act[l];
        ActivationFunction act = net->act_func[l - 1];

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)N, (int)curr_size, (int)prev_size,
                    1.0f,
                    A, (int)prev_size,
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
    }
}

static void batch_compute_deltas(NeuralNetwork *net, BatchWorkspace *ws,
                                  const float *targets, uint32_t N) {
    uint32_t last = net->layers - 1;
    ActivationFunction last_act = net->act_func[last - 1];
    uint32_t n_out = net->topology[last];
    size_t out_total = (size_t)N * (size_t)n_out;

    float *act_last = ws->act[last];
    float *delta_last = ws->delta[last];

    if (net->loss_func == LOSS_MSE && last_act == ACT_SOFTMAX) {
        for (uint32_t s = 0; s < N; s++) {
            float *a = act_last + (size_t)s * n_out;
            float *d = delta_last + (size_t)s * n_out;
            const float *t = targets + (size_t)s * n_out;
            float sum = 0.0f;
            for (uint32_t j = 0; j < n_out; j++)
                sum += (a[j] - t[j]) * a[j];
            for (uint32_t j = 0; j < n_out; j++)
                d[j] = a[j] * ((a[j] - t[j]) - sum);
        }
    } else if (net->loss_func == LOSS_CROSS_ENTROPY &&
               (last_act == ACT_SOFTMAX || last_act == ACT_SIGMOID)) {
        memcpy(delta_last, act_last, out_total * sizeof(float));
        cblas_saxpy((int)out_total, -1.0f, targets, 1, delta_last, 1);
    } else {
        for (size_t i = 0; i < out_total; i++)
            delta_last[i] = (act_last[i] - targets[i]) * derivative(act_last[i], last_act);
    }

    for (uint32_t l = last - 1; l > 0; l--) {
        uint32_t cur_sz  = net->topology[l];
        uint32_t next_sz = net->topology[l + 1];
        ActivationFunction prev_act = net->act_func[l - 1];
        float *W = SPINGALETT_WEIGHT_MTX_PTR(net, l);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)N, (int)cur_sz, (int)next_sz,
                    1.0f,
                    ws->delta[l + 1], (int)next_sz,
                    W, (int)cur_sz,
                    0.0f,
                    ws->delta[l], (int)cur_sz);

        uint64_t total_elems = (uint64_t)N * (uint64_t)cur_sz;
        apply_derivative_batch(ws->delta[l], ws->act[l], total_elems, prev_act);
    }
}

static void batch_accumulate_gradients(NeuralNetwork *net, BatchWorkspace *ws, uint32_t N, float grad_scale) {
    uint32_t last = net->layers - 1;

    for (uint32_t l = 0; l < last; l++) {
        uint32_t in_sz  = net->topology[l];
        uint32_t out_sz = net->topology[l + 1];
        float *gW = SPINGALETT_GRAD_W_MTX_PTR(net, l);
        float *gB = net->grad_biases + net->bias_offsets[l];

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    (int)out_sz, (int)in_sz, (int)N,
                    grad_scale,
                    ws->delta[l + 1], (int)out_sz,
                    ws->act[l], (int)in_sz,
                    0.0f,
                    gW, (int)in_sz);

        cblas_sgemv(CblasRowMajor, CblasTrans,
                    (int)N, (int)out_sz,
                    grad_scale, ws->delta[l + 1], (int)out_sz,
                    ws->ones, 1,
                    0.0f, gB, 1);
    }
}

static float batch_compute_loss(NeuralNetwork *net, BatchWorkspace *ws,
                                 const float *targets, uint32_t N) {
    uint32_t last = net->layers - 1;
    uint32_t n_out = net->topology[last];
    ActivationFunction output_act = net->act_func[last - 1];
    float total_error = 0.0f;

    for (uint32_t s = 0; s < N; s++) {
        const float *out = ws->act[last] + (size_t)s * (size_t)n_out;
        const float *tgt = targets + (size_t)s * (size_t)n_out;
        total_error += compute_sample_loss(out, tgt, n_out, net->loss_func, output_act);
    }

    return total_error;
}

#endif

static void apply_gradients_batch_prescaled(NeuralNetwork *net, TrainArgs *args,
                                             float beta1_pow_t, float beta2_pow_t) {
    uint32_t last = net->layers - 1;
    float lr = args->learning_rate;
    float decay = args->weight_decay;
    float momentum = args->momentum;
    float beta1 = args->beta1;
    float beta2 = args->beta2;
    float epsilon = args->epsilon;
    float one_minus_b1 = 1.0f - beta1;
    float one_minus_b2 = 1.0f - beta2;

    if (args->max_grad_norm > 0.0f)
        spingalett_clip_grad_norm(net, args->max_grad_norm);

    for (uint32_t l = 0; l < last; l++) {
        uint32_t in_sz  = net->topology[l];
        uint32_t out_sz = net->topology[l + 1];
        uint64_t total_w = (uint64_t)in_sz * (uint64_t)out_sz;

        float *W   = SPINGALETT_WEIGHT_MTX_PTR(net, l);
        float *gW  = SPINGALETT_GRAD_W_MTX_PTR(net, l);
        float *b   = net->biases       + net->bias_offsets[l];
        float *gB  = net->grad_biases  + net->bias_offsets[l];
        float *mW  = net->opt_m_weights + net->weight_offsets[l];
        float *vW  = net->opt_v_weights + net->weight_offsets[l];
        float *mB  = net->opt_m_biases  + net->bias_offsets[l];
        float *vB  = net->opt_v_biases  + net->bias_offsets[l];

        if (args->optimizer_type == OPTIMIZER_SGD) {
            if (decay > 0.0f) {
                spingalett_sgd_decay_update_avx(W, gW, total_w, lr, decay);
            } else {
                spingalett_sgd_update_avx(W, gW, total_w, lr);
            }
            spingalett_sgd_update_avx(b, gB, (uint64_t)out_sz, lr);

        } else if (args->optimizer_type == OPTIMIZER_MOMENTUM) {
            spingalett_momentum_update_avx(W, mW, gW, total_w, lr, momentum, decay);
            spingalett_momentum_update_avx(b, mB, gB, (uint64_t)out_sz, lr, momentum, 0.0f);

        } else if (args->optimizer_type == OPTIMIZER_ADAM || args->optimizer_type == OPTIMIZER_ADAMW) {
            float m_factor = 1.0f / (1.0f - beta1_pow_t);
            float v_factor = 1.0f / (1.0f - beta2_pow_t);
            bool is_adamw = (args->optimizer_type == OPTIMIZER_ADAMW);

            if (is_adamw && decay > 0.0f) {
                float wd_factor = 1.0f - lr * decay;
                spingalett_adamw_update_avx(W, mW, vW, gW, total_w,
                                            lr, beta1, beta2, one_minus_b1, one_minus_b2,
                                            m_factor, v_factor, epsilon, wd_factor);
            } else {
                if (!is_adamw && decay > 0.0f) {
                    spingalett_vec_axpy(gW, W, total_w, decay);
                }
                spingalett_adam_update_avx(W, mW, vW, gW, total_w,
                                           lr, beta1, beta2, one_minus_b1, one_minus_b2,
                                           m_factor, v_factor, epsilon);
            }
            spingalett_adam_update_avx(b, mB, vB, gB, (uint64_t)out_sz,
                                       lr, beta1, beta2, one_minus_b1, one_minus_b2,
                                       m_factor, v_factor, epsilon);

        } else if (args->optimizer_type == OPTIMIZER_RMSPROP) {
            spingalett_rmsprop_update_avx(W, vW, gW, total_w, lr, beta2, one_minus_b2, epsilon, decay);
            spingalett_rmsprop_update_avx(b, vB, gB, (uint64_t)out_sz, lr, beta2, one_minus_b2, epsilon, 0.0f);
        }
    }
}

static void apply_gradients_batch(NeuralNetwork *net, TrainArgs *args, uint32_t effective_count,
                                   float beta1_pow_t, float beta2_pow_t) {
    uint32_t last = net->layers - 1;
    float scaler = 1.0f / (float)effective_count;

    for (uint32_t l = 0; l < last; l++) {
        uint32_t in_sz  = net->topology[l];
        uint32_t out_sz = net->topology[l + 1];
        uint64_t total_w = (uint64_t)in_sz * (uint64_t)out_sz;
        float *gW = SPINGALETT_GRAD_W_MTX_PTR(net, l);
        float *gB = net->grad_biases + net->bias_offsets[l];

        spingalett_vec_scale(gW, total_w, scaler);
        spingalett_vec_scale(gB, (uint64_t)out_sz, scaler);
    }

    apply_gradients_batch_prescaled(net, args, beta1_pow_t, beta2_pow_t);

    for (uint32_t l = 0; l < last; l++) {
        uint32_t in_sz  = net->topology[l];
        uint32_t out_sz = net->topology[l + 1];
        uint64_t total_w = (uint64_t)in_sz * (uint64_t)out_sz;
        memset(SPINGALETT_GRAD_W_MTX_PTR(net, l), 0, total_w * sizeof(float));
        memset(net->grad_biases + net->bias_offsets[l], 0, out_sz * sizeof(float));
    }
}

static void handle_autosave(NeuralNetwork *net, TrainArgs *args, size_t epoch) {
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

static inline bool should_report(size_t epoch, size_t epochs, size_t interval) {
    return interval > 0 && ((epoch % interval == 0) || (epoch == epochs));
}

void train_struct_arguments(TrainArgs args) {
    NeuralNetwork *net = args.net;
    TrainingMode training_mode = args.training_mode;
    TrainingStrategy training_strategy = args.training_strategy;
    float *inputs = args.inputs;
    float *targets = args.targets;
    uint32_t sample_count = args.sample_count;
    size_t epochs = args.epochs;

    if (args.learning_rate <= 0.0f) args.learning_rate = 0.01f;
    if (args.momentum <= 0.0f) args.momentum = 0.9f;
    if (args.beta1 <= 0.0f) args.beta1 = 0.9f;
    if (args.beta2 <= 0.0f) args.beta2 = 0.999f;
    if (args.epsilon <= 0.0f) args.epsilon = 1e-8f;

    if (args.callback && args.callback_interval == 0)
        args.callback_interval = 1;

    if (training_mode == MODE_GENERATOR_FUNCTION) {
        set_error(SPINGALETT_ERR_INVALID, "Generator function training mode is not implemented yet");
        spingalett_log(LOG_ERROR, "Generator function training mode is not implemented yet");
        return;
    }

    if (!net || sample_count == 0 || epochs == 0 || !inputs || !targets) {
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
    if (spingalett_get_num_threads() > 0)
        omp_set_num_threads((int)spingalett_get_num_threads());
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

    uint32_t input_node_count = net->topology[0];
    uint32_t output_node_count = net->topology[net->layers - 1];
    ActivationFunction output_act = net->act_func[net->layers - 2];

    float beta1_pow = 1.0f;
    float beta2_pow = 1.0f;

#if defined(SPINGALETT_HAS_OPENBLAS)
    if (effective_mode == COMPUTE_OPENBLAS &&
        (training_strategy == STRATEGY_FULL_BATCH || training_strategy == STRATEGY_SMALL_BATCH)) {

        uint32_t effective_batch = (training_strategy == STRATEGY_SMALL_BATCH)
                                   ? args.batch_size : sample_count;

        BatchWorkspace *ws = batch_workspace_create(net, effective_batch);
        if (!ws) {
            set_error(SPINGALETT_ERR_ALLOC, "Failed to allocate batch workspace");
            spingalett_log(LOG_ERROR, "Failed to allocate batch workspace");
            return;
        }

        uint32_t *shuffle_idx = NULL;
        float *batch_inputs = NULL;
        float *batch_targets = NULL;

        if (training_strategy == STRATEGY_SMALL_BATCH) {
            shuffle_idx = (uint32_t *)malloc(sample_count * sizeof(uint32_t));
            batch_inputs = (float *)spingalett_aligned_alloc(
                (size_t)effective_batch * (size_t)input_node_count * sizeof(float));
            batch_targets = (float *)spingalett_aligned_alloc(
                (size_t)effective_batch * (size_t)output_node_count * sizeof(float));
            if (!shuffle_idx || !batch_inputs || !batch_targets) {
                set_error(SPINGALETT_ERR_ALLOC, "Mini-batch allocation failed");
                spingalett_log(LOG_ERROR, "Mini-batch allocation failed");
                free(shuffle_idx);
                spingalett_aligned_free(batch_inputs);
                spingalett_aligned_free(batch_targets);
                batch_workspace_free(ws);
                return;
            }
            for (uint32_t i = 0; i < sample_count; i++)
                shuffle_idx[i] = i;
        }

        for (size_t epoch = 1; epoch <= epochs; epoch++) {
            bool need_loss = should_report(epoch, epochs, args.report_interval) ||
                             (args.callback && should_report(epoch, epochs, args.callback_interval));
            float total_error = 0.0f;

            if (training_strategy == STRATEGY_FULL_BATCH) {
                float grad_scale = 1.0f / (float)sample_count;

                batch_forward(net, ws, inputs, sample_count);

                if (need_loss)
                    total_error = batch_compute_loss(net, ws, targets, sample_count);

                batch_compute_deltas(net, ws, targets, sample_count);
                batch_accumulate_gradients(net, ws, sample_count, grad_scale);

                net->time_step++;
                beta1_pow *= args.beta1;
                beta2_pow *= args.beta2;
                apply_gradients_batch_prescaled(net, &args, beta1_pow, beta2_pow);
            } else {
                spingalett_shuffle_indices(shuffle_idx, sample_count);

                uint32_t num_batches = (sample_count + effective_batch - 1) / effective_batch;

                for (uint32_t bi = 0; bi < num_batches; bi++) {
                    uint32_t start = bi * effective_batch;
                    uint32_t end = start + effective_batch;
                    if (end > sample_count) end = sample_count;
                    uint32_t cur_batch = end - start;
                    float grad_scale = 1.0f / (float)cur_batch;

                    for (uint32_t s = 0; s < cur_batch; s++) {
                        uint32_t idx = shuffle_idx[start + s];
                        memcpy(batch_inputs + (size_t)s * input_node_count,
                               inputs + (size_t)idx * input_node_count,
                               input_node_count * sizeof(float));
                        memcpy(batch_targets + (size_t)s * output_node_count,
                               targets + (size_t)idx * output_node_count,
                               output_node_count * sizeof(float));
                    }

                    batch_forward(net, ws, batch_inputs, cur_batch);

                    if (need_loss)
                        total_error += batch_compute_loss(net, ws, batch_targets, cur_batch);

                    batch_compute_deltas(net, ws, batch_targets, cur_batch);
                    batch_accumulate_gradients(net, ws, cur_batch, grad_scale);

                    net->time_step++;
                    beta1_pow *= args.beta1;
                    beta2_pow *= args.beta2;
                    apply_gradients_batch_prescaled(net, &args, beta1_pow, beta2_pow);
                }
            }

            float current_error = total_error / (float)sample_count;

            if (should_report(epoch, epochs, args.report_interval))
                spingalett_log(LOG_INFO, "Epoch: %zu/%zu, Error: %f", epoch, epochs, (double)current_error);

            if (args.nan_check_interval > 0 && (epoch % args.nan_check_interval == 0)) {
                if (check_nan_inf(net)) {
                    spingalett_log(LOG_ERROR, "NaN/Inf detected in weights at epoch %zu, stopping training", epoch);
                    break;
                }
            }

            if (args.autosave_mode != AUTOSAVE_OFF &&
                args.autosave_interval > 0 &&
                args.autosave_path &&
                should_report(epoch, epochs, args.autosave_interval))
                handle_autosave(net, &args, epoch);

            if (args.callback && should_report(epoch, epochs, args.callback_interval)) {
                if (args.callback(net, epoch, current_error)) {
                    spingalett_log(LOG_INFO, "Training interrupted by callback at epoch %zu", epoch);
                    break;
                }
            }
        }

        free(shuffle_idx);
        spingalett_aligned_free(batch_inputs);
        spingalett_aligned_free(batch_targets);
        batch_workspace_free(ws);
        spingalett_log(LOG_INFO, "Training completed.");
        return;
    }
#endif

    float **deltas = (float **)malloc(net->layers * sizeof(float *));
    if (!deltas) {
        set_error(SPINGALETT_ERR_ALLOC, "Failed to allocate deltas pointer array");
        spingalett_log(LOG_ERROR, "Failed to allocate deltas");
        return;
    }
    float *deltas_flat = (float *)malloc(net->total_neurons * sizeof(float));
    if (!deltas_flat) {
        free(deltas);
        set_error(SPINGALETT_ERR_ALLOC, "Failed to allocate deltas flat buffer");
        spingalett_log(LOG_ERROR, "Failed to allocate deltas");
        return;
    }

    size_t d_off = 0;
    for (uint32_t i = 0; i < net->layers; i++) {
        deltas[i] = deltas_flat + d_off;
        d_off += net->topology[i];
    }

    uint32_t *shuffle_idx = NULL;
    if (training_strategy == STRATEGY_SMALL_BATCH) {
        shuffle_idx = (uint32_t *)malloc(sample_count * sizeof(uint32_t));
        if (!shuffle_idx) {
            free(deltas_flat);
            free(deltas);
            set_error(SPINGALETT_ERR_ALLOC, "Failed to allocate shuffle indices");
            spingalett_log(LOG_ERROR, "Failed to allocate shuffle indices");
            return;
        }
        for (uint32_t i = 0; i < sample_count; i++)
            shuffle_idx[i] = i;
    }

    for (size_t epoch = 1; epoch <= epochs; epoch++) {
        float total_error = 0.0f;
        bool need_loss = should_report(epoch, epochs, args.report_interval) ||
                         (args.callback && should_report(epoch, epochs, args.callback_interval));

        if (training_strategy == STRATEGY_SMALL_BATCH) {
            spingalett_shuffle_indices(shuffle_idx, sample_count);

            uint32_t bs = args.batch_size;
            uint32_t num_batches = (sample_count + bs - 1) / bs;

            for (uint32_t bi = 0; bi < num_batches; bi++) {
                uint32_t start = bi * bs;
                uint32_t end = start + bs;
                if (end > sample_count) end = sample_count;
                uint32_t cur_batch = end - start;

                for (uint32_t s = 0; s < cur_batch; s++) {
                    uint32_t idx = shuffle_idx[start + s];
                    float *current_input  = inputs  + (size_t)idx * input_node_count;
                    float *current_target = targets + (size_t)idx * output_node_count;

                    ForwardArgs fwd_args = { .net = net, .input = current_input };
                    forward_struct_arguments(fwd_args);

                    if (need_loss) {
                        const float *out_layer = SPINGALETT_LAYER_PTR(net, net->layers - 1);
                        total_error += compute_sample_loss(out_layer, current_target, output_node_count, net->loss_func, output_act);
                    }

                    accumulate_gradients_single(net, current_target, &args, deltas, effective_mode);
                }

                net->time_step++;
                beta1_pow *= args.beta1;
                beta2_pow *= args.beta2;
                apply_gradients_batch(net, &args, cur_batch, beta1_pow, beta2_pow);
            }
        } else {
            for (uint32_t s = 0; s < sample_count; s++) {
                float *current_input  = inputs  + (size_t)s * input_node_count;
                float *current_target = targets + (size_t)s * output_node_count;

                ForwardArgs fwd_args = { .net = net, .input = current_input };
                forward_struct_arguments(fwd_args);

                if (need_loss) {
                    const float *out_layer = SPINGALETT_LAYER_PTR(net, net->layers - 1);
                    total_error += compute_sample_loss(out_layer, current_target, output_node_count, net->loss_func, output_act);
                }

                if (training_strategy == STRATEGY_SAMPLE) {
                    backpropagation(net, current_target, &args, deltas, &beta1_pow, &beta2_pow, effective_mode);
                } else {
                    accumulate_gradients_single(net, current_target, &args, deltas, effective_mode);
                }
            }

            if (training_strategy != STRATEGY_SAMPLE) {
                net->time_step++;
                beta1_pow *= args.beta1;
                beta2_pow *= args.beta2;
                apply_gradients_batch(net, &args, sample_count, beta1_pow, beta2_pow);
            }
        }

        float current_error = total_error / (float)sample_count;

        if (should_report(epoch, epochs, args.report_interval))
            spingalett_log(LOG_INFO, "Epoch: %zu/%zu, Error: %f", epoch, epochs, (double)current_error);

        if (args.nan_check_interval > 0 && (epoch % args.nan_check_interval == 0)) {
            if (check_nan_inf(net)) {
                spingalett_log(LOG_ERROR, "NaN/Inf detected in weights at epoch %zu, stopping training", epoch);
                break;
            }
        }

        if (args.autosave_mode != AUTOSAVE_OFF &&
            args.autosave_interval > 0 &&
            args.autosave_path &&
            should_report(epoch, epochs, args.autosave_interval))
            handle_autosave(net, &args, epoch);

        if (args.callback && should_report(epoch, epochs, args.callback_interval)) {
            if (args.callback(net, epoch, current_error)) {
                spingalett_log(LOG_INFO, "Training interrupted by callback at epoch %zu", epoch);
                break;
            }
        }
    }

    free(shuffle_idx);
    free(deltas_flat);
    free(deltas);
    spingalett_log(LOG_INFO, "Training completed.");
}
