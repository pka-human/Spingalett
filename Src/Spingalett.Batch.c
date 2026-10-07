/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Batch execution shared by training and predict(): every layer is one GEMM over a chunk of
   samples, followed by a per-row pass for bias, activation and dropout. */

#include "Spingalett.Private.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

void spingalett_batch_workspace_free(BatchWorkspace *ws) {
    if (!ws) return;
    spingalett_aligned_free(ws->flat);
    spingalett_aligned_free(ws->dmask_flat);
    spingalett_aligned_free(ws->inputs);
    spingalett_aligned_free(ws->targets);
    spingalett_aligned_free(ws->conv);
    spingalett_gemm_scratch_free(ws->gemm);
    free(ws->act);
    free(ws->delta);
    free(ws->dmask);
    free(ws);
}

uint32_t spingalett_batch_capacity(const NeuralNetwork *net, uint32_t count) {
    uint64_t per_sample = net->total_neurons ? net->total_neurons : 1u;
    uint64_t cap = SPINGALETT_BATCH_FLOATS / per_sample;
    if (cap > SPINGALETT_BATCH_CHUNK) cap = SPINGALETT_BATCH_CHUNK;
    if (cap < 1) cap = 1;
    return count < cap ? count : (uint32_t)cap;
}

BatchWorkspace *spingalett_batch_workspace_create(const NeuralNetwork *net, uint32_t capacity,
                                                  bool training, bool gather, ComputeMode mode) {
    BatchWorkspace *ws = (BatchWorkspace *)calloc(1, sizeof(BatchWorkspace));
    if (!ws) return NULL;
    ws->capacity = capacity;
    ws->act   = (float **)calloc(net->layers, sizeof(float *));
    ws->delta = (float **)calloc(net->layers, sizeof(float *));
    ws->dmask = (float **)calloc(net->layers, sizeof(float *));
    if (!ws->act || !ws->delta || !ws->dmask) goto fail;

    /* Inference writes the last layer straight into the caller's output buffer. */
    uint32_t act_layers = training ? net->layers : net->layers - 1;
    size_t act_total = 0, dropout_total = 0;
    for (uint32_t l = 1; l < act_layers; l++)
        act_total += (size_t)capacity * net->topology[l];
    for (uint32_t l = 1; training && l + 1 < net->layers; l++)
        if (net->dropout_rates[l] > 0.0f)
            dropout_total += (size_t)capacity * net->topology[l];

    size_t delta_total = training ? act_total : 0;
    if (act_total + delta_total > 0) {
        ws->flat = (float *)spingalett_aligned_alloc((act_total + delta_total) * sizeof(float));
        if (!ws->flat) goto fail;
    }
    size_t off = 0;
    for (uint32_t l = 1; l < act_layers; l++) {
        ws->act[l] = ws->flat + off;
        if (training) ws->delta[l] = ws->flat + act_total + off;
        off += (size_t)capacity * net->topology[l];
    }

    if (dropout_total > 0) {
        ws->dmask_flat = (float *)spingalett_aligned_alloc(dropout_total * sizeof(float));
        if (!ws->dmask_flat) goto fail;
        size_t doff = 0;
        for (uint32_t l = 1; l + 1 < net->layers; l++)
            if (net->dropout_rates[l] > 0.0f) {
                ws->dmask[l] = ws->dmask_flat + doff;
                doff += (size_t)capacity * net->topology[l];
            }
    }

    if (gather) {
        ws->inputs  = (float *)spingalett_aligned_alloc((size_t)capacity * net->topology[0] * sizeof(float));
        ws->targets = (float *)spingalett_aligned_alloc((size_t)capacity * net->topology[net->layers - 1] * sizeof(float));
        if (!ws->inputs || !ws->targets) goto fail;
    }

    size_t conv_floats = spingalett_conv_scratch_floats(net, capacity, training, mode);
    if (conv_floats > 0) {
        ws->conv = (float *)spingalett_aligned_alloc(conv_floats * sizeof(float));
        if (!ws->conv) goto fail;
    }

    if (mode != COMPUTE_OPENBLAS) {
        int threads = 1;
#if defined(_OPENMP)
        if (mode == COMPUTE_OPENMP) threads = omp_get_max_threads();
#endif
        ws->gemm = spingalett_gemm_scratch_create(threads);
        if (!ws->gemm) goto fail;
    }
    return ws;

fail:
    spingalett_batch_workspace_free(ws);
    return NULL;
}

void spingalett_batch_forward(NeuralNetwork *net, BatchWorkspace *ws, uint32_t N,
                              const DropoutContext *dropout, uint32_t position_offset, ComputeMode mode) {
    for (uint32_t l = 1; l < net->layers; l++) {
        uint32_t prev_size = net->topology[l - 1];
        uint32_t curr_size = net->topology[l];
        const float *bias = net->biases + net->bias_offsets[l - 1];
        ActivationFunction act = net->act_func[l - 1];
        float *C = ws->act[l];
        bool masked = dropout && ws->dmask[l];
        LayerType type = net->shapes[l].type;
        bool done = false;          /* bias and activation applied */

        if (type == LAYER_DENSE) {
            /* act[l] = act[l-1] * W^T, W stored [curr x prev]; the bias follows per row */
            spingalett_gemm(ws->gemm, mode, false, true, N, curr_size, prev_size, 1.0f,
                            ws->act[l - 1], prev_size, SPINGALETT_WEIGHT_MTX_PTR(net, l - 1), prev_size,
                            0.0f, C, curr_size);
        } else if (type == LAYER_CONV2D) {
            /* the bias, and an element-wise activation, are applied as the product's tiles complete */
            done = act != ACT_SOFTMAX && !masked;
            spingalett_conv_forward(net, l - 1, ws->act[l - 1], C, N, done ? act : ACT_NONE, ws->conv, ws->gemm, mode);
        } else {
            spingalett_pool_forward(net, l - 1, ws->act[l - 1], C, N, mode);
            done = act == ACT_NONE && !masked;
        }
        if (done)
            continue;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, (uint64_t)N * curr_size))
#endif
        for (int64_t s = 0; s < (int64_t)N; s++) {
            float *row = C + (size_t)s * curr_size;
            if (type == LAYER_DENSE)
                spingalett_vec_axpy(row, bias, curr_size, 1.0f);
            apply_activation_batch(row, curr_size, act);
            if (masked)
                spingalett_dropout_apply(row, ws->dmask[l] + (size_t)s * curr_size, curr_size, act,
                                         net->dropout_rates[l], dropout, l, position_offset + (uint32_t)s);
        }
    }
}

bool predict_struct_arguments(PredictArgs args) {
    NeuralNetwork *net = args.net;
    if (!net || !args.inputs || !args.outputs) {
        set_error(SPINGALETT_ERR_INVALID, "predict: net, inputs or outputs is NULL");
        return false;
    }
    if (net->layers < 2) {
        set_error(SPINGALETT_ERR_INVALID, "predict: network must have at least 2 layers");
        return false;
    }
    if (args.sample_count == 0)
        return true;

    ComputeMode mode = resolve_compute_mode();
    uint32_t capacity = spingalett_batch_capacity(net, args.sample_count);
    BatchWorkspace *ws = spingalett_batch_workspace_create(net, capacity, false, false, mode);
    if (!ws) {
        set_error(SPINGALETT_ERR_ALLOC, "predict: workspace allocation failed");
        return false;
    }

    uint32_t in_sz = net->topology[0], out_sz = net->topology[net->layers - 1];
    for (uint32_t start = 0; start < args.sample_count; start += capacity) {
        uint32_t n = args.sample_count - start < capacity ? args.sample_count - start : capacity;
        ws->act[0] = (float *)(args.inputs + (size_t)start * in_sz);           /* read only */
        ws->act[net->layers - 1] = args.outputs + (size_t)start * out_sz;
        spingalett_batch_forward(net, ws, n, NULL, 0, mode);
    }

    spingalett_batch_workspace_free(ws);
    return true;
}

bool spingalett_sample_correct(const float *out, const float *target, uint32_t n) {
    if (n == 1)
        return (out[0] >= 0.5f) == (target[0] >= 0.5f);
    uint32_t best_out = 0, best_target = 0;
    for (uint32_t k = 1; k < n; k++) {
        if (out[k] > out[best_out]) best_out = k;
        if (target[k] > target[best_target]) best_target = k;
    }
    return best_out == best_target;
}

void spingalett_batch_evaluate(NeuralNetwork *net, BatchWorkspace *ws, float *out_buf,
                               const float *inputs, const float *targets, uint32_t n, ComputeMode mode,
                               double *loss_sum, uint32_t *correct) {
    uint32_t in_sz = net->topology[0], out_sz = net->topology[net->layers - 1];
    ActivationFunction out_act = net->act_func[net->layers - 2];
    double loss = 0.0;
    uint32_t hits = 0;
    for (uint32_t start = 0; start < n; start += ws->capacity) {
        uint32_t count = n - start < ws->capacity ? n - start : ws->capacity;
        ws->act[0] = (float *)(inputs + (size_t)start * in_sz);              /* read only */
        ws->act[net->layers - 1] = out_buf;
        spingalett_batch_forward(net, ws, count, NULL, 0, mode);
        for (uint32_t s = 0; s < count; s++) {
            const float *o = out_buf + (size_t)s * out_sz, *t = targets + ((size_t)start + s) * out_sz;
            loss += compute_sample_loss(o, t, out_sz, net->loss_func, out_act);
            hits += spingalett_sample_correct(o, t, out_sz);
        }
    }
    *loss_sum = loss;
    *correct = hits;
}

EvalMetrics evaluate_struct_arguments(EvaluateArgs args) {
    EvalMetrics m = {NAN, NAN};
    NeuralNetwork *net = args.net;
    if (!net || !args.inputs || !args.targets || args.sample_count == 0) {
        set_error(SPINGALETT_ERR_INVALID, "evaluate: net, inputs or targets is NULL, or sample_count is 0");
        return m;
    }
    if (net->layers < 2) {
        set_error(SPINGALETT_ERR_INVALID, "evaluate: network must have at least 2 layers");
        return m;
    }

    ComputeMode mode = resolve_compute_mode();
    uint32_t capacity = spingalett_batch_capacity(net, args.sample_count);
    BatchWorkspace *ws = spingalett_batch_workspace_create(net, capacity, false, false, mode);
    float *out = (float *)spingalett_aligned_alloc((size_t)capacity * net->topology[net->layers - 1] * sizeof(float));
    if (ws && out) {
        double loss;
        uint32_t correct;
        spingalett_batch_evaluate(net, ws, out, args.inputs, args.targets, args.sample_count, mode, &loss, &correct);
        m.loss = (float)(loss / args.sample_count);
        m.accuracy = (float)correct / (float)args.sample_count;
    } else {
        set_error(SPINGALETT_ERR_ALLOC, "evaluate: workspace allocation failed");
    }
    spingalett_aligned_free(out);
    spingalett_batch_workspace_free(ws);
    return m;
}
