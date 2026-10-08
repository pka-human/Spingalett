/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Batch execution shared by training and predict(): every layer is one GEMM over a chunk of
   samples (or a kernel of its own), followed by a per-row pass for bias, activation and dropout.
   Layers run in index order, which puts every layer after the layers it reads. */

#include "Spingalett.Private.h"
#include "Spingalett.Gpu.h"
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
    spingalett_aligned_free(ws->bn_flat);
    spingalett_aligned_free(ws->bn_sums_flat);
    spingalett_aligned_free(ws->bn_scratch);
    spingalett_aligned_free(ws->bn_coef);
    spingalett_aligned_free(ws->gtmp);
    spingalett_gemm_scratch_free(ws->gemm);
    free(ws->uses);
    free(ws->pending);
    free(ws->act);
    free(ws->delta);
    free(ws->dmask);
    free(ws->bn_stats);
    free(ws->bn_sums);
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
    ws->training = training;
    ws->act   = (float **)calloc(net->layers, sizeof(float *));
    ws->delta = (float **)calloc(net->layers, sizeof(float *));
    ws->dmask = (float **)calloc(net->layers, sizeof(float *));
    ws->bn_stats = (float **)calloc(net->layers, sizeof(float *));
    ws->bn_sums = (double **)calloc(net->layers, sizeof(double *));
    ws->uses = (uint32_t *)calloc(net->layers, sizeof(uint32_t));
    ws->pending = (uint32_t *)calloc(net->layers, sizeof(uint32_t));
    if (!ws->act || !ws->delta || !ws->dmask || !ws->bn_stats || !ws->bn_sums || !ws->uses || !ws->pending) goto fail;
    for (uint32_t l = 1; l < net->layers; l++)
        for (uint32_t k = 0; k < spingalett_input_count(net, l); k++) ws->uses[spingalett_inputs(net, l)[k]]++;

    size_t dropout_total = 0;
    for (uint32_t l = 1; training && l + 1 < net->layers; l++)
        if (net->dropout_rates[l] > 0.0f)
            dropout_total += (size_t)capacity * net->topology[l];

    if (training) {
        /* every layer's output and delta, kept for the backward pass */
        size_t act_total = 0, widest_shared = 0;
        for (uint32_t l = 1; l < net->layers; l++) {
            act_total += (size_t)capacity * net->topology[l];
            if (ws->uses[l] > 1 && net->topology[l] > widest_shared) widest_shared = net->topology[l];
        }
        ws->flat = (float *)spingalett_aligned_alloc(2u * act_total * sizeof(float));
        if (!ws->flat) goto fail;
        size_t off = 0;
        for (uint32_t l = 1; l < net->layers; l++) {
            ws->act[l] = ws->flat + off;
            ws->delta[l] = ws->flat + act_total + off;
            off += (size_t)capacity * net->topology[l];
        }
        if (widest_shared > 0) {
            ws->gtmp = (float *)spingalett_aligned_alloc((size_t)capacity * widest_shared * sizeof(float));
            if (!ws->gtmp) goto fail;
        }
    } else if (net->layers > 2) {
        /* Inference writes the last layer straight into the caller's output buffer; the others
           share memory where their lives do not overlap. */
        uint64_t *offsets = (uint64_t *)malloc((size_t)net->layers * sizeof(uint64_t));
        uint64_t per_sample = offsets ? spingalett_plan_outputs(net, ws->uses, offsets) : UINT64_MAX;
        if (per_sample != UINT64_MAX && per_sample > 0)
            ws->flat = (float *)spingalett_aligned_alloc((size_t)(per_sample * capacity) * sizeof(float));
        if (per_sample == UINT64_MAX || (per_sample > 0 && !ws->flat)) {
            free(offsets);
            goto fail;
        }
        for (uint32_t l = 1; l + 1 < net->layers; l++)
            if (offsets[l] != UINT64_MAX) ws->act[l] = ws->flat + offsets[l] * capacity;
        free(offsets);
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

    size_t bn_channels = 0, widest = 0;
    for (uint32_t l = 1; l < net->layers; l++)
        if (net->shapes[l].type == LAYER_BATCH_NORM) {
            bn_channels += net->shapes[l].channels;
            if (net->shapes[l].channels > widest) widest = net->shapes[l].channels;
        }
    if (bn_channels > 0) {
        ws->bn_coef = (float *)spingalett_aligned_alloc(3u * widest * sizeof(float));
        if (!ws->bn_coef) goto fail;
        if (training) {
            ws->bn_flat = (float *)spingalett_aligned_alloc(4u * bn_channels * sizeof(float));
            ws->bn_sums_flat = (double *)spingalett_aligned_alloc(2u * bn_channels * sizeof(double));
            ws->bn_scratch = (double *)spingalett_aligned_alloc((2u * widest + spingalett_bn_scratch_doubles(net, capacity)) *
                                                                sizeof(double));
            if (!ws->bn_flat || !ws->bn_sums_flat || !ws->bn_scratch) goto fail;
            for (uint32_t l = 1, off = 0; l < net->layers; l++)
                if (net->shapes[l].type == LAYER_BATCH_NORM) {
                    ws->bn_stats[l] = ws->bn_flat + 4u * off;
                    ws->bn_sums[l] = ws->bn_sums_flat + 2u * off;
                    off += net->shapes[l].channels;
                }
        }
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

/* The output of a layer that adds or concatenates others, or pools globally (act: applied, or
   ACT_NONE). */
static void combine_forward(const NeuralNetwork *net, const BatchWorkspace *ws, uint32_t l, float *y, uint32_t N,
                            ActivationFunction act, ComputeMode mode) {
    const uint32_t count = spingalett_input_count(net, l), *in = spingalett_inputs(net, l);
    const LayerShape *s = &net->shapes[l];
    const float *x[SPINGALETT_MAX_INPUTS];
    uint32_t channels[SPINGALETT_MAX_INPUTS];
    for (uint32_t k = 0; k < count; k++) {
        x[k] = ws->act[in[k]];
        channels[k] = net->shapes[in[k]].channels;
    }
    if (s->type == LAYER_ADD) {
        spingalett_add_forward(x, count, y, N, net->topology[l], act, mode);
    } else if (s->type == LAYER_CONCAT) {
        spingalett_concat_forward(x, channels, count, y, N, (uint64_t)s->height * s->width, act, mode);
    } else {
        const LayerShape *i = &net->shapes[in[0]];
        spingalett_global_pool_forward(x[0], y, N, (uint64_t)i->height * i->width, i->channels, act, mode);
    }
}

void spingalett_batch_forward(NeuralNetwork *net, BatchWorkspace *ws, uint32_t N,
                              const DropoutContext *dropout, uint32_t position_offset, ComputeMode mode) {
    for (uint32_t l = 1; l < net->layers; l++) {
        const uint32_t src = spingalett_source(net, l);
        uint32_t prev_size = net->topology[src];
        uint32_t curr_size = net->topology[l];
        const float *bias = net->biases + net->bias_offsets[l - 1];
        ActivationFunction act = net->act_func[l - 1];
        float *C = ws->act[l];
        const float *X = ws->act[src];
        bool masked = dropout && ws->dmask[l];
        LayerType type = net->shapes[l].type;
        bool done = false;          /* bias and activation applied */

        /* inference: a batch normalization after a dense or convolution layer without activation
           scales its outputs per channel in the product's epilogue, straight into its own output */
        if (!ws->training && spingalett_fused_norm(net, ws->uses, l)) {
            const uint64_t b0 = net->bias_offsets[l];
            uint32_t channels = net->shapes[l].channels;
            float *scale = ws->bn_coef, *shift = ws->bn_coef + channels;
            spingalett_bn_coefficients(net->weights + net->weight_offsets[l], net->biases + b0, net->running_mean + b0,
                                       net->running_var + b0, net->shapes[l + 1].eps, channels, scale, shift);
            for (uint32_t c = 0; c < channels; c++) shift[c] += bias[c] * scale[c];
            if (type == LAYER_DENSE) {
                SpingalettBiasActivation epilogue = {shift, net->act_func[l], scale};
                SpingalettGemmHooks hooks = {NULL, NULL, spingalett_epilogue_bias_activation, &epilogue};
                spingalett_gemm_ex(ws->gemm, mode, false, true, N, curr_size, prev_size, 1.0f, X, prev_size,
                                   SPINGALETT_WEIGHT_MTX_PTR(net, l - 1), prev_size, 0.0f, ws->act[l + 1], curr_size,
                                   &hooks);
            } else {
                spingalett_conv_forward_scaled(&net->shapes[src], &net->shapes[l], SPINGALETT_WEIGHT_MTX_PTR(net, l - 1),
                                               scale, shift, X, ws->act[l + 1], N, net->act_func[l], ws->conv,
                                               ws->gemm, mode);
            }
            l++;
            continue;
        }

        if (type == LAYER_DENSE) {
            /* act[l] = act[l-1] * W^T, W stored [curr x prev]; the bias and an element-wise
               activation follow as the product's tiles complete */
            done = act != ACT_SOFTMAX && !masked;
            SpingalettBiasActivation epilogue = {bias, act};
            SpingalettGemmHooks hooks = {NULL, NULL, spingalett_epilogue_bias_activation, &epilogue};
            spingalett_gemm_ex(ws->gemm, mode, false, true, N, curr_size, prev_size, 1.0f,
                               X, prev_size, SPINGALETT_WEIGHT_MTX_PTR(net, l - 1), prev_size,
                               0.0f, C, curr_size, done ? &hooks : NULL);
        } else if (type == LAYER_CONV2D) {
            /* the bias, and an element-wise activation, are applied as the product's tiles complete */
            done = act != ACT_SOFTMAX && !masked;
            spingalett_conv_forward(net, l - 1, X, C, N, done ? act : ACT_NONE, ws->conv, ws->gemm, mode);
        } else if (type == LAYER_BATCH_NORM) {
            done = act != ACT_SOFTMAX && !masked;
            if (ws->training)
                spingalett_bn_forward_train(net, l - 1, X, C, N, done ? act : ACT_NONE, ws->bn_stats[l],
                                            ws->bn_scratch, mode);
            else
                spingalett_bn_forward(net, l - 1, X, C, N, done ? act : ACT_NONE, ws->bn_coef, mode);
        } else if (type == LAYER_ADD || type == LAYER_CONCAT || type == LAYER_GLOBAL_AVG_POOL) {
            done = !masked;
            combine_forward(net, ws, l, C, N, done ? act : ACT_NONE, mode);
        } else {
            spingalett_pool_forward(net, l - 1, X, C, N, mode);
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
    if (!spingalett_check_graph(net, "predict"))
        return false;
    if (args.sample_count == 0)
        return true;

    SpgGpuNet *gpu = spingalett_gpu_for(net, args.sample_count);
    if (gpu) {
        const uint32_t cap = spingalett_gpu_net_capacity(gpu), in_sz = net->topology[0];
        const uint32_t out_sz = net->topology[net->layers - 1];
        bool ok = true;
        for (uint32_t start = 0; ok && start < args.sample_count; start += cap) {
            uint32_t n = args.sample_count - start < cap ? args.sample_count - start : cap;
            ok = spingalett_gpu_predict(gpu, args.inputs + (size_t)start * in_sz, args.outputs + (size_t)start * out_sz, n);
        }
        spingalett_gpu_done(net, gpu);
        if (ok) return true;
        spingalett_log(LOG_WARNING, "predict: the GPU failed; predicting on the CPU");
    }

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

SpgGpuNet *spingalett_gpu_for(NeuralNetwork *net, uint32_t count) {
    const char *why = NULL;
    if (!spingalett_use_gpu()) return NULL;
    if (!spingalett_gpu_supports(net, &why)) {
        spingalett_log(LOG_WARNING, "The GPU cannot run this network (%s); running it on the CPU", why);
        return NULL;
    }
    /* chunks of a power of two (at least 64) samples, so that calls of nearby sizes share one */
    uint32_t want = 64;
    while (want < count && want < SPINGALETT_BATCH_CHUNK) want *= 2;
    uint32_t capacity = spingalett_gpu_capacity(net, want, false);
    SpgGpuNet *gpu = atomic_exchange(&net->gpu_predict, NULL);
    if (gpu && spingalett_gpu_net_capacity(gpu) >= capacity) {
        if (spingalett_gpu_upload(gpu)) return gpu;     /* the parameters as they are now */
    }
    spingalett_gpu_net_free(gpu);
    gpu = capacity ? spingalett_gpu_net_create(net, capacity, NULL) : NULL;
    if (!gpu) spingalett_log(LOG_WARNING, "Not enough GPU memory for the network; running it on the CPU");
    return gpu;
}

void spingalett_gpu_done(NeuralNetwork *net, SpgGpuNet *gpu) {
    SpgGpuNet *empty = NULL;
    if (!atomic_compare_exchange_strong(&net->gpu_predict, &empty, gpu))
        spingalett_gpu_net_free(gpu);                   /* another call put one back first */
}

bool spingalett_gpu_evaluate(SpgGpuNet *gpu, NeuralNetwork *net, float *out_buf, const float *inputs,
                             const float *targets, uint32_t n, double *loss_sum, uint32_t *correct) {
    const uint32_t in_sz = net->topology[0], out_sz = net->topology[net->layers - 1];
    const uint32_t cap = spingalett_gpu_net_capacity(gpu);
    const ActivationFunction out_act = net->act_func[net->layers - 2];
    double loss = 0.0;
    uint32_t hits = 0;
    for (uint32_t start = 0; start < n; start += cap) {
        uint32_t count = n - start < cap ? n - start : cap;
        if (!spingalett_gpu_predict(gpu, inputs + (size_t)start * in_sz, out_buf, count)) return false;
        for (uint32_t s = 0; s < count; s++) {
            const float *o = out_buf + (size_t)s * out_sz, *t = targets + ((size_t)start + s) * out_sz;
            loss += compute_sample_loss(o, t, out_sz, net->loss_func, out_act);
            hits += spingalett_sample_correct(o, t, out_sz);
        }
    }
    *loss_sum = loss;
    *correct = hits;
    return true;
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
    if (!spingalett_check_graph(net, "evaluate"))
        return m;

    SpgGpuNet *gpu = spingalett_gpu_for(net, args.sample_count);
    if (gpu) {
        float *buf = (float *)spingalett_aligned_alloc((size_t)spingalett_gpu_net_capacity(gpu) *
                                                       net->topology[net->layers - 1] * sizeof(float));
        double loss;
        uint32_t correct;
        bool ok = buf && spingalett_gpu_evaluate(gpu, net, buf, args.inputs, args.targets, args.sample_count, &loss,
                                                 &correct);
        spingalett_aligned_free(buf);
        spingalett_gpu_done(net, gpu);
        if (ok) {
            m.loss = (float)(loss / args.sample_count);
            m.accuracy = (float)correct / (float)args.sample_count;
            return m;
        }
        spingalett_log(LOG_WARNING, "evaluate: the GPU failed; evaluating on the CPU");
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
