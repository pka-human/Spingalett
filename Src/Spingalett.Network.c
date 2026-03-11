/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

static void compute_offsets(NeuralNetwork *net) {
    uint64_t n = 0, w = 0, b = 0;
    net->neuron_offsets[0] = 0;
    for (uint32_t l = 0; l < net->layers; l++) {
        n += (uint64_t)net->topology[l];
        net->neuron_offsets[l + 1] = n;
        if (l + 1 < net->layers) {
            net->weight_offsets[l] = w;
            net->bias_offsets[l] = b;
            w += (uint64_t)net->topology[l] * (uint64_t)net->topology[l + 1];
            b += (uint64_t)net->topology[l + 1];
        }
    }
    net->total_neurons = n;
    net->total_weights = w;
    net->total_biases = b;
}

NeuralNetwork *new_spingalett_struct_arguments(NeuralNetworkArgs args) {
    LossFunction loss_func = args.loss_func;
    if ((unsigned)loss_func >= LOSS_COUNT) {
        set_error(SPINGALETT_ERR_INVALID, "Invalid loss function");
        return NULL;
    }
    spingalett_log(LOG_INFO, "Creating new network with %s loss function", loss_func_names[loss_func]);
    NeuralNetwork *net = (NeuralNetwork *)calloc(1, sizeof(NeuralNetwork));
    if (!net) { set_error(SPINGALETT_ERR_ALLOC, "NeuralNetwork calloc failed"); return NULL; }
    net->loss_func = loss_func;
    return net;
}

void layer_struct_arguments(LayerArgs args) {
    NeuralNetwork *net = args.net;
    uint32_t neurons_amount = args.neurons_amount;
    ActivationFunction act_func = args.act_func;
    WeightInitialization wi = args.weight_initialization;

    if (!net || neurons_amount == 0) {
        set_error(SPINGALETT_ERR_INVALID, "Net is NULL or neurons amount is 0");
        return;
    }

    uint32_t nl = net->layers + 1;

    if (nl == 1) {
        spingalett_log(LOG_INFO, "Layer #0: Neurons amount: %u", neurons_amount);
    } else {
        if ((unsigned)act_func >= ACT_COUNT) {
            set_error(SPINGALETT_ERR_INVALID, "Invalid activation function");
            return;
        }
        if ((unsigned)wi >= WEIGHT_INITIALIZATION_COUNT) {
            set_error(SPINGALETT_ERR_INVALID, "Invalid weight initialization");
            return;
        }
        spingalett_log(LOG_INFO, "Layer #%u: Neurons amount: %u, Activation function: %s, Weight initialization: %s",
            nl - 1, neurons_amount, act_func_names[act_func], weight_initialization_names[wi]);
    }

    uint32_t prev_neurons = (net->layers > 0) ? net->topology[net->layers - 1] : 0;
    uint64_t add_w = (nl > 1) ? (uint64_t)prev_neurons * (uint64_t)neurons_amount : 0;
    uint64_t add_b = (nl > 1) ? (uint64_t)neurons_amount : 0;

    uint64_t new_tn = net->total_neurons + (uint64_t)neurons_amount;
    uint64_t new_tw = net->total_weights + add_w;
    uint64_t new_tb = net->total_biases  + add_b;

    uint32_t           *t_topo    = (uint32_t *)           malloc(nl * sizeof(uint32_t));
    uint64_t           *t_noff    = (uint64_t *)           calloc(nl + 1, sizeof(uint64_t));
    uint64_t           *t_woff    = (uint64_t *)           calloc(nl, sizeof(uint64_t));
    uint64_t           *t_boff    = (uint64_t *)           calloc(nl, sizeof(uint64_t));
    float              *t_neurons = (float *)              spingalett_aligned_calloc(new_tn, sizeof(float));

    ActivationFunction *t_act     = NULL;
    float *t_w = NULL, *t_b = NULL;
    float *t_gw = NULL, *t_gb = NULL;
    float *t_mw = NULL, *t_mb = NULL;
    float *t_vw = NULL, *t_vb = NULL;

    if (nl > 1) {
        t_act = (ActivationFunction *)malloc((nl - 1) * sizeof(ActivationFunction));
        t_w   = (float *)spingalett_aligned_calloc(new_tw, sizeof(float));
        t_b   = (float *)spingalett_aligned_calloc(new_tb, sizeof(float));
        t_gw  = (float *)spingalett_aligned_calloc(new_tw, sizeof(float));
        t_gb  = (float *)spingalett_aligned_calloc(new_tb, sizeof(float));
        t_mw  = (float *)spingalett_aligned_calloc(new_tw, sizeof(float));
        t_mb  = (float *)spingalett_aligned_calloc(new_tb, sizeof(float));
        t_vw  = (float *)spingalett_aligned_calloc(new_tw, sizeof(float));
        t_vb  = (float *)spingalett_aligned_calloc(new_tb, sizeof(float));
    }

    bool ok = t_topo && t_noff && t_woff && t_boff && t_neurons;
    if (nl > 1)
        ok = ok && t_act && t_w && t_b && t_gw && t_gb && t_mw && t_mb && t_vw && t_vb;

    if (!ok) {
        free(t_topo); free(t_noff); free(t_woff); free(t_boff);
        spingalett_aligned_free(t_neurons);
        if (nl > 1) {
            free(t_act);
            spingalett_aligned_free(t_w);  spingalett_aligned_free(t_b);
            spingalett_aligned_free(t_gw); spingalett_aligned_free(t_gb);
            spingalett_aligned_free(t_mw); spingalett_aligned_free(t_mb);
            spingalett_aligned_free(t_vw); spingalett_aligned_free(t_vb);
        }
        set_error(SPINGALETT_ERR_ALLOC, "Layer allocation failed");
        return;
    }

    if (net->layers > 0)
        memcpy(t_topo, net->topology, net->layers * sizeof(uint32_t));
    t_topo[net->layers] = neurons_amount;

    if (net->total_neurons > 0)
        memcpy(t_neurons, net->neurons, net->total_neurons * sizeof(float));

    if (nl > 1) {
        if (net->layers > 1)
            memcpy(t_act, net->act_func, (net->layers - 1) * sizeof(ActivationFunction));
        t_act[net->layers - 1] = act_func;

        if (net->total_weights > 0) {
            memcpy(t_w,  net->weights,       net->total_weights * sizeof(float));
            memcpy(t_gw, net->grad_weights,  net->total_weights * sizeof(float));
            memcpy(t_mw, net->opt_m_weights, net->total_weights * sizeof(float));
            memcpy(t_vw, net->opt_v_weights, net->total_weights * sizeof(float));
        }
        if (net->total_biases > 0) {
            memcpy(t_b,  net->biases,        net->total_biases * sizeof(float));
            memcpy(t_gb, net->grad_biases,   net->total_biases * sizeof(float));
            memcpy(t_mb, net->opt_m_biases,  net->total_biases * sizeof(float));
            memcpy(t_vb, net->opt_v_biases,  net->total_biases * sizeof(float));
        }

        float scale = 1.0f;
        if (wi == WEIGHT_INITIALIZATION_XAVIER)
            scale = sqrtf(1.0f / (float)prev_neurons);
        else if (wi == WEIGHT_INITIALIZATION_HE)
            scale = sqrtf(2.0f / (float)prev_neurons);

        for (uint64_t idx = 0; idx < add_w; idx++) {
            uint64_t pos = net->total_weights + idx;
            if (wi == WEIGHT_INITIALIZATION_RANDOM)
                t_w[pos] = random_uniform_weight();
            else if (wi == WEIGHT_INITIALIZATION_XAVIER || wi == WEIGHT_INITIALIZATION_HE)
                t_w[pos] = random_normal_weight() * scale;
        }
    }

    free(net->topology);
    free(net->act_func);
    free(net->neuron_offsets);
    free(net->weight_offsets);
    free(net->bias_offsets);
    spingalett_aligned_free(net->neurons);
    spingalett_aligned_free(net->weights);
    spingalett_aligned_free(net->biases);
    spingalett_aligned_free(net->grad_weights);
    spingalett_aligned_free(net->grad_biases);
    spingalett_aligned_free(net->opt_m_weights);
    spingalett_aligned_free(net->opt_m_biases);
    spingalett_aligned_free(net->opt_v_weights);
    spingalett_aligned_free(net->opt_v_biases);

    net->layers         = nl;
    net->topology       = t_topo;
    net->act_func       = t_act;
    net->neuron_offsets  = t_noff;
    net->weight_offsets  = t_woff;
    net->bias_offsets    = t_boff;
    net->neurons         = t_neurons;
    net->weights         = t_w;
    net->biases          = t_b;
    net->grad_weights    = t_gw;
    net->grad_biases     = t_gb;
    net->opt_m_weights   = t_mw;
    net->opt_m_biases    = t_mb;
    net->opt_v_weights   = t_vw;
    net->opt_v_biases    = t_vb;
    net->total_neurons   = new_tn;
    net->total_weights   = new_tw;
    net->total_biases    = new_tb;

    compute_offsets(net);
}

float *forward_struct_arguments(ForwardArgs args) {
    NeuralNetwork *net = args.net;
    const float *input = args.input;
    ComputeMode mode = resolve_compute_mode();

    if (!net || !input) return NULL;

    if (net->layers < 2) {
        set_error(SPINGALETT_ERR_INVALID, "Network must have at least 2 layers for forward pass");
        return NULL;
    }

    uint32_t in_size = net->topology[0];
    float *layer0 = SPINGALETT_LAYER_PTR(net, 0);
    memcpy(layer0, input, in_size * sizeof(float));

    for (uint32_t l = 1; l < net->layers; l++) {
        uint32_t prev_size = net->topology[l - 1];
        uint32_t curr_size = net->topology[l];
        float *W = SPINGALETT_WEIGHT_MTX_PTR(net, l - 1);
        float *b = net->biases + net->bias_offsets[l - 1];
        const float *x = SPINGALETT_LAYER_PTR(net, l - 1);
        float *y = SPINGALETT_LAYER_PTR(net, l);
        ActivationFunction act = net->act_func[l - 1];

        memcpy(y, b, curr_size * sizeof(float));

#if defined(SPINGALETT_HAS_OPENBLAS)
        if (mode == COMPUTE_OPENBLAS) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        (int)curr_size, (int)prev_size,
                        1.0f, W, (int)prev_size,
                        x, 1, 1.0f, y, 1);
        } else
#endif
        {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(mode == COMPUTE_OPENMP)
#endif
            for (uint32_t j = 0; j < curr_size; j++) {
                const float *row = W + (uint64_t)j * (uint64_t)prev_size;
                y[j] += spingalett_dot_product(x, row, (uint64_t)prev_size);
            }
        }

        apply_activation_batch(y, curr_size, act);
    }

    return SPINGALETT_LAYER_PTR(net, net->layers - 1);
}

void print_parameters(NeuralNetwork *net) {
    spingalett_log(LOG_INFO, "========== DEBUG NETWORK PARAMETERS ==========");
    spingalett_log(LOG_INFO, "Loss Function: %s", loss_func_names[net->loss_func]);

    for (uint32_t i = 0; i < net->layers - 1; i++) {
        uint32_t from_layer = i;
        uint32_t to_layer = i + 1;
        spingalett_log(LOG_INFO, "[Connection: Layer %u (%u neurons) -> Layer %u (%u neurons), activation function: %s]",
            from_layer, net->topology[from_layer],
            to_layer, net->topology[to_layer],
            act_func_names[net->act_func[i]]);
        spingalett_log(LOG_INFO, "  Biases (Thresholds) for Layer %u:", to_layer);
        for (uint32_t k = 0; k < net->topology[to_layer]; k++) {
            spingalett_log(LOG_INFO, "    Neuron %u bias: %12.6g", k, (double)SPINGALETT_BIAS(net, i, k));
        }

        uint64_t weights_count = (uint64_t)net->topology[from_layer] * (uint64_t)net->topology[to_layer];
        spingalett_log(LOG_INFO, "  Weights: (total: %llu)", (unsigned long long)weights_count);

        for (uint32_t k = 0; k < net->topology[from_layer]; k++) {
            for (uint32_t j = 0; j < net->topology[to_layer]; j++) {
                spingalett_log(LOG_INFO, "    L%u N%u -> L%u N%u: %12.6g",
                    from_layer, k, to_layer, j, (double)SPINGALETT_WEIGHT(net, i, j, k));
            }
        }
    }

    spingalett_log(LOG_INFO, "================================================");
}

void free_network(NeuralNetwork *net) {
    if (!net) return;
    spingalett_aligned_free(net->neurons);
    spingalett_aligned_free(net->weights);
    spingalett_aligned_free(net->biases);
    spingalett_aligned_free(net->grad_weights);
    spingalett_aligned_free(net->grad_biases);
    spingalett_aligned_free(net->opt_m_weights);
    spingalett_aligned_free(net->opt_m_biases);
    spingalett_aligned_free(net->opt_v_weights);
    spingalett_aligned_free(net->opt_v_biases);
    free(net->neuron_offsets);
    free(net->weight_offsets);
    free(net->bias_offsets);
    free(net->topology);
    free(net->act_func);
    free(net);
    spingalett_log(LOG_DEBUG, "Memory freed.");
}
