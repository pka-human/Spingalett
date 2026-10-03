/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Batch execution shared by training and predict(): every layer is one GEMM over a chunk of
   samples, followed by a per-row pass for bias, activation and dropout. */

#include "Spingalett.Private.h"
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

void spingalett_batch_workspace_free(BatchWorkspace *ws) {
    if (!ws) return;
    spingalett_aligned_free(ws->flat);
    spingalett_aligned_free(ws->dmask_flat);
    spingalett_aligned_free(ws->inputs);
    spingalett_aligned_free(ws->targets);
    spingalett_gemm_scratch_free(ws->gemm);
    free(ws->act);
    free(ws->delta);
    free(ws->dmask);
    free(ws);
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

        /* act[l] = act[l-1] * W^T, W stored [curr x prev] */
        spingalett_gemm(ws->gemm, mode, false, true, N, curr_size, prev_size, 1.0f,
                        ws->act[l - 1], prev_size, SPINGALETT_WEIGHT_MTX_PTR(net, l - 1), prev_size,
                        0.0f, C, curr_size);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(spingalett_use_omp(mode, (uint64_t)N * curr_size))
#endif
        for (int64_t s = 0; s < (int64_t)N; s++) {
            float *row = C + (size_t)s * curr_size;
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
    uint32_t capacity = args.sample_count < SPINGALETT_BATCH_CHUNK ? args.sample_count : SPINGALETT_BATCH_CHUNK;
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
