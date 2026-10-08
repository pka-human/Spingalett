/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Spingalett test suite. Uses only the public API, so the same checks run against every build
 * configuration; backends that are not compiled in fall back to single-threaded and the
 * cross-backend comparisons then pass trivially.
 *
 *   Spingalett.Tests [group]     groups: grad conv norm graph onnx equiv cont optim sched dropout gen predict valid
 *                                step data io model xor
 *                                (default: all)
 *
 * Numerical gradients come from central differences of an independently computed loss; analytic
 * gradients from a single SGD step with lr = 1 (W_before - W_after). Everything is seeded, so
 * results are deterministic.
 */

#include <Spingalett/Spingalett.h>
#include "Spingalett.Network.h"      /* white-box: the network's arrays */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef SPINGALETT_TEST_DATA_DIR
#define SPINGALETT_TEST_DATA_DIR "Data"
#endif

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static unsigned lcg_state = 12345;
static float frand(void) { lcg_state = lcg_state * 1103515245u + 12345u; return (float)((lcg_state >> 8) & 0xFFFFFF) / 16777216.0f; }

typedef struct { uint32_t n; ActivationFunction act; } L;

static NeuralNetwork *build(LossFunction loss, const L *ls, int nl, const float *w0, const float *b0) {
    NeuralNetwork *net = new_spingalett(.loss_func = loss);
    for (int i = 0; i < nl; i++)
        layer(.net = net, .neurons_amount = ls[i].n, .act_func = ls[i].act, .weight_initialization = WEIGHT_INITIALIZATION_XAVIER);
    if (w0) memcpy(net->weights, w0, net->total_weights * sizeof(float));
    else for (uint64_t i = 0; i < net->total_weights; i++) net->weights[i] = frand() * 1.2f - 0.6f;
    if (b0) memcpy(net->biases, b0, net->total_biases * sizeof(float));
    return net;
}

static double sample_loss(LossFunction loss, ActivationFunction out_act, const float *o, const float *t, uint32_t n) {
    double l = 0;
    for (uint32_t k = 0; k < n; k++) {
        if (loss == LOSS_MSE) l += 0.5 * ((double)o[k] - t[k]) * ((double)o[k] - t[k]);
        else if (out_act == ACT_SIGMOID) l -= t[k] * log(o[k]) + (1 - t[k]) * log(1 - o[k]);
        else l -= t[k] * log(o[k]);
    }
    return l;
}

static double dataset_loss(NeuralNetwork *net, const float *x, const float *y, uint32_t N) {
    uint32_t in = net->topology[0], out = net->topology[net->layers - 1];
    ActivationFunction oa = net->act_func[net->layers - 2];
    double s = 0;
    for (uint32_t i = 0; i < N; i++) {
        float *o = forward(.net = net, .input = x + (size_t)i * in);
        s += sample_loss(net->loss_func, oa, o, y + (size_t)i * out, out);
    }
    return s / N;
}

static void make_data(uint32_t N, uint32_t in, uint32_t out, LossFunction loss, ActivationFunction oa, float **x, float **y) {
    *x = malloc((size_t)N * in * sizeof(float));
    *y = malloc((size_t)N * out * sizeof(float));
    for (size_t i = 0; i < (size_t)N * in; i++) (*x)[i] = frand() * 2 - 1;
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t k = 0; k < out; k++) (*y)[i * out + k] = (loss == LOSS_MSE && oa != ACT_SOFTMAX) ? frand() * 1.6f - 0.8f : 0.f;
        if (loss == LOSS_CROSS_ENTROPY && oa == ACT_SIGMOID) for (uint32_t k = 0; k < out; k++) (*y)[i * out + k] = frand() < 0.5f ? 0.f : 1.f;
        else if (oa == ACT_SOFTMAX) (*y)[i * out + (lcg_state >> 10) % out] = 1.f, frand();
    }
}

/* Weight rows longer than this (when set) are scaled by sqrt(limit / length), so that wide windows do
   not saturate their activations at the check's weights. */
static uint32_t gradcheck_fan_in_limit = 0;

// One full-batch SGD step with lr=1 => analytic grad = W_before - W_after.
static void gradcheck_net(const char *name, NeuralNetwork *net, ComputeMode mode, TrainingStrategy strat) {
    uint32_t N = strat == STRATEGY_SAMPLE ? 1 : 6;
    uint32_t in = net->topology[0], out = net->topology[net->layers - 1];
    float *x, *y;
    make_data(N, in, out, net->loss_func, net->act_func[net->layers - 2], &x, &y);
    spingalett_set_compute_mode(mode);
    for (uint64_t i = 0; i < net->total_weights; i++) net->weights[i] = frand() * 1.2f - 0.6f;
    for (uint64_t i = 0; i < net->total_biases; i++) net->biases[i] = frand() * 0.2f - 0.1f;
    for (uint32_t l = 1; gradcheck_fan_in_limit && l < net->layers; l++) {
        SpingalettNetworkLayer d;
        spingalett_network_layer(net, l, &d);
        uint64_t len = d.bias_count ? d.weight_count / d.bias_count : 0;
        if (len > gradcheck_fan_in_limit)
            for (uint64_t i = 0; i < d.weight_count; i++)
                net->weights[net->weight_offsets[l - 1] + i] *= sqrtf((float)gradcheck_fan_in_limit / (float)len);
    }
    uint64_t nw = net->total_weights, nb = net->total_biases;
    float *w0 = malloc(nw * 4), *b0 = malloc(nb * 4);
    memcpy(w0, net->weights, nw * 4); memcpy(b0, net->biases, nb * 4);

    train(.net = net, .inputs = x, .targets = y, .sample_count = N, .epochs = 1, .learning_rate = 1.0f,
          .optimizer_type = OPTIMIZER_SGD, .training_strategy = strat, .batch_size = N);
    float *ga = malloc((nw + nb) * 4);
    for (uint64_t i = 0; i < nw; i++) ga[i] = w0[i] - net->weights[i];
    for (uint64_t i = 0; i < nb; i++) ga[nw + i] = b0[i] - net->biases[i];
    memcpy(net->weights, w0, nw * 4); memcpy(net->biases, b0, nb * 4);

    int bad = 0, kinks = 0; double maxrel = 0;
    for (uint64_t i = 0; i < nw + nb; i++) {
        float *p = i < nw ? &net->weights[i] : &net->biases[i - nw];
        float orig = *p, h = 1e-3f, h2 = 2.5e-4f;
        *p = orig + h; double lp = dataset_loss(net, x, y, N);
        *p = orig - h; double lm = dataset_loss(net, x, y, N);
        *p = orig + h2; double lp2 = dataset_loss(net, x, y, N);
        *p = orig - h2; double lm2 = dataset_loss(net, x, y, N);
        *p = orig;
        double gn = (lp - lm) / (2.0 * h), gn2 = (lp2 - lm2) / (2.0 * h2);
        if (fabs(gn - gn2) > 0.1 * (fabs(gn) + fabs(gn2)) + 3e-4) { kinks++; continue; } // perturbation crosses a kink
        /* the closer of the two estimates: a max pooling switch can lie within the larger step
           and still pass the kink test; a tiny gradient, lost in the loss's rounding at both
           steps, gets a third, larger one */
        double rel = fmin(fabs(gn - ga[i]) / fmax(1e-2, fabs(gn) + fabs(ga[i])),
                          fabs(gn2 - ga[i]) / fmax(1e-2, fabs(gn2) + fabs(ga[i])));
        if (rel > 3e-2 && fabs(ga[i]) < 1e-2) {
            float h3 = 1e-2f;
            *p = orig + h3; double lp3 = dataset_loss(net, x, y, N);
            *p = orig - h3; double lm3 = dataset_loss(net, x, y, N);
            *p = orig;
            double gn3 = (lp3 - lm3) / (2.0 * h3);
            rel = fmin(rel, fabs(gn3 - ga[i]) / fmax(1e-2, fabs(gn3) + fabs(ga[i])));
        }
        if (rel > maxrel) maxrel = rel;
        if (rel > 3e-2) bad++;
    }
    printf("  gradcheck %-28s mode=%d strat=%d  maxrel=%.2e  outliers=%d/%llu kinks=%d\n", name, mode, strat, maxrel, bad, (unsigned long long)(nw + nb), kinks);
    CHECK(bad == 0 && kinks * 10 < (int)(nw + nb), "gradcheck %s mode %d strat %d: %d outliers", name, mode, strat, bad);
    free(ga); free(w0); free(b0); free(x); free(y);
    free_network(net);
}

static void gradcheck(const char *name, LossFunction loss, const L *ls, int nl, ComputeMode mode, TrainingStrategy strat) {
    lcg_state = 777;
    gradcheck_net(name, build(loss, ls, nl, NULL, NULL), mode, strat);
}

/* Convolutional networks for the gradient checks: every window option of conv2d and pooling. */
static NeuralNetwork *conv_net(int which) {
    NeuralNetwork *net;
    switch (which) {
        case 0:     /* same-size 3x3 conv, max pool, dense softmax (cross-entropy) */
            net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
            layer(.net = net, .height = 6, .width = 6, .channels = 2);
            conv2d(.net = net, .filters = 3, .kernel = 3, .padding = 1, .act_func = ACT_TANH);
            max_pool2d(.net = net, .kernel = 2);
            layer(.net = net, .neurons_amount = 4, .act_func = ACT_SOFTMAX);
            return net;
        case 1:     /* strided conv, overlapping padded average pool, pointwise conv, dense (MSE) */
            net = new_spingalett(.loss_func = LOSS_MSE);
            layer(.net = net, .height = 7, .width = 5, .channels = 3);
            conv2d(.net = net, .filters = 4, .kernel = 3, .stride = 2, .act_func = ACT_TANH);
            avg_pool2d(.net = net, .kernel = 2, .stride = 1, .padding = 1);
            conv2d(.net = net, .filters = 2, .kernel = 1, .act_func = ACT_SIGMOID);
            layer(.net = net, .neurons_amount = 3, .act_func = ACT_NONE);
            return net;
        case 2:     /* rectangular windows and strides, overlapping max pool, conv output layer */
            net = new_spingalett(.loss_func = LOSS_MSE);
            layer(.net = net, .height = 5, .width = 8, .channels = 2);
            conv2d(.net = net, .filters = 3, .kernel_h = 2, .kernel_w = 3, .stride_w = 2, .padding_w = 1,
                   .act_func = ACT_LEAKY_RELU);
            max_pool2d(.net = net, .kernel = 3, .stride = 2, .padding = 1);
            conv2d(.net = net, .filters = 2, .kernel = 2, .act_func = ACT_NONE);
            return net;
        case 3:     /* dropout-free deep stack: conv after dense input reshaped by a 1x1 view */
            net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
            layer(.net = net, .height = 4, .width = 4, .channels = 3);
            conv2d(.net = net, .filters = 5, .kernel = 2, .act_func = ACT_RELU);
            conv2d(.net = net, .filters = 4, .kernel = 2, .padding = 1, .act_func = ACT_FOO52);
            avg_pool2d(.net = net, .kernel = 4);
            layer(.net = net, .neurons_amount = 3, .act_func = ACT_SIGMOID);
            return net;
        case 5:     /* depthwise with a multiplier of 2, then a pointwise convolution */
            net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
            layer(.net = net, .height = 6, .width = 6, .channels = 3);
            conv2d(.net = net, .filters = 6, .kernel = 3, .padding = 1, .groups = 3, .act_func = ACT_TANH);
            conv2d(.net = net, .filters = 4, .kernel = 1, .act_func = ACT_RELU);
            layer(.net = net, .neurons_amount = 3, .act_func = ACT_SOFTMAX);
            return net;
        case 6:     /* two groups with rectangular strided windows, a grouped pointwise convolution */
            net = new_spingalett(.loss_func = LOSS_MSE);
            layer(.net = net, .height = 5, .width = 6, .channels = 4);
            conv2d(.net = net, .filters = 6, .kernel_h = 2, .kernel_w = 3, .stride = 2, .padding_w = 1, .groups = 2,
                   .act_func = ACT_TANH);
            avg_pool2d(.net = net, .kernel = 2, .stride = 1, .padding = 1);
            conv2d(.net = net, .filters = 4, .kernel = 1, .groups = 2, .act_func = ACT_SIGMOID);
            layer(.net = net, .neurons_amount = 2, .act_func = ACT_NONE);
            return net;
        case 7:     /* depthwise, strided, padded, after the input and before max pooling */
            net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
            layer(.net = net, .height = 7, .width = 7, .channels = 5);
            conv2d(.net = net, .filters = 5, .kernel = 3, .stride = 2, .padding = 1, .groups = 5,
                   .act_func = ACT_LEAKY_RELU);
            max_pool2d(.net = net, .kernel = 2);
            conv2d(.net = net, .filters = 5, .kernel = 1, .groups = 5, .act_func = ACT_FOO52);
            layer(.net = net, .neurons_amount = 3, .act_func = ACT_SIGMOID);
            return net;
        case 8:     /* 16 input channels per group and at least 12 filters: the indirect weight
                       gradient, grouped and strided, then a pointwise one */
            net = new_spingalett(.loss_func = LOSS_MSE);
            layer(.net = net, .height = 4, .width = 4, .channels = 32);
            conv2d(.net = net, .filters = 24, .kernel = 3, .stride = 2, .padding = 1, .groups = 2, .act_func = ACT_TANH);
            conv2d(.net = net, .filters = 12, .kernel = 1, .groups = 1, .act_func = ACT_NONE);
            avg_pool2d(.net = net, .kernel = 2);
            layer(.net = net, .neurons_amount = 2, .act_func = ACT_SIGMOID);
            return net;
        case 9:     /* the same in a 3 x 3 convolution of stride 1 over 16 channels, padded */
            net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
            layer(.net = net, .height = 3, .width = 3, .channels = 16);
            conv2d(.net = net, .filters = 16, .kernel = 3, .padding = 1, .act_func = ACT_SIGMOID);
            conv2d(.net = net, .filters = 13, .kernel = 2, .act_func = ACT_TANH);
            layer(.net = net, .neurons_amount = 3, .act_func = ACT_SOFTMAX);
            return net;
        default:    /* one input channel, more filters than window weights (the weight gradient is
                       multiplied the other way round), ReLU into max pooling */
            net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
            layer(.net = net, .height = 6, .width = 6, .channels = 1);
            conv2d(.net = net, .filters = 12, .kernel = 3, .padding = 1, .act_func = ACT_RELU);
            max_pool2d(.net = net, .kernel = 2);
            layer(.net = net, .neurons_amount = 3, .act_func = ACT_SOFTMAX);
            return net;
    }
}

static float max_abs_diff(const float *a, const float *b, uint64_t n) {
    float m = 0; for (uint64_t i = 0; i < n; i++) { float d = fabsf(a[i] - b[i]); if (d > m) m = d; } return m;
}

/* Batches whose gathered windows exceed one chunk (2^21 floats) must compute what single samples
   compute: outputs, and gradients summed over the batch. */
static void conv_chunks(ComputeMode mode) {
    lcg_state = 4711;
    spingalett_set_compute_mode(mode);
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(.net = net, .height = 48, .width = 48, .channels = 8);
    conv2d(.net = net, .filters = 16, .kernel = 3, .padding = 1, .act_func = ACT_RELU, .weight_initialization = WEIGHT_INITIALIZATION_HE);
    conv2d(.net = net, .filters = 16, .kernel = 3, .stride = 2, .act_func = ACT_TANH, .weight_initialization = WEIGHT_INITIALIZATION_HE);
    max_pool2d(.net = net, .kernel = 2);
    layer(.net = net, .neurons_amount = 5, .act_func = ACT_SOFTMAX, .weight_initialization = WEIGHT_INITIALIZATION_XAVIER);
    const uint32_t N = 16, in = net->topology[0], out = 5;
    float *x, *y;
    make_data(N, in, out, LOSS_CROSS_ENTROPY, ACT_SOFTMAX, &x, &y);
    float *batch = malloc((size_t)N * out * 4), *single = malloc((size_t)N * out * 4);
    predict(.net = net, .inputs = x, .sample_count = N, .outputs = batch);
    for (uint32_t s = 0; s < N; s++) memcpy(single + s * out, forward(.net = net, .input = x + (size_t)s * in), out * 4);
    float dout = max_abs_diff(batch, single, (size_t)N * out);

    uint64_t nw = net->total_weights, nb = net->total_biases;
    float *g_batch = malloc((nw + nb) * 4);
    SpingalettTrainer *t = spingalett_trainer_new(net, N);
    spingalett_trainer_forward(t, x, N);
    spingalett_trainer_backward(t, y);
    memcpy(g_batch, net->grad_weights, nw * 4); memcpy(g_batch + nw, net->grad_biases, nb * 4);
    spingalett_trainer_free(t);
    t = spingalett_trainer_new(net, 1);
    for (uint32_t s = 0; s < N; s++) {
        spingalett_trainer_forward(t, x + (size_t)s * in, 1);
        spingalett_trainer_backward(t, y + (size_t)s * out);
    }
    double worst = 0, scale = 0;
    for (uint64_t i = 0; i < nw + nb; i++) {
        float g = i < nw ? net->grad_weights[i] : net->grad_biases[i - nw];
        if (fabs(g_batch[i]) > scale) scale = fabs(g_batch[i]);
        if (fabs(g - g_batch[i]) > worst) worst = fabs(g - g_batch[i]);
    }
    spingalett_trainer_free(t);
    printf("  conv chunks mode=%d: |batch - single| outputs %.2e, gradients %.2e (largest %.2e)\n", mode, dout, worst, scale);
    CHECK(dout < 1e-5f && worst < 1e-4 * scale, "conv chunks mode %d: outputs %.2e gradients %.2e", mode, dout, worst);
    free(batch); free(single); free(g_batch); free(x); free(y);
    free_network(net);
}

/* Layer arguments that cannot be built are refused, leaving the network as it was. */
static void conv_api(void) {
    spingalett_set_verbose(false);
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_MSE);
    conv2d(.net = net, .filters = 2, .kernel = 3);
    CHECK(spingalett_layer_count(net) == 0 && spingalett_last_error_code() == SPINGALETT_ERR_INVALID, "conv input layer must be refused");
    layer(.net = net, .height = 5, .width = 4, .channels = 3, .neurons_amount = 61);
    CHECK(spingalett_layer_count(net) == 0, "input size disagreeing with the shape must be refused");
    layer(.net = net, .height = 5, .width = 4, .channels = 3);
    CHECK(spingalett_layer_count(net) == 1 && spingalett_input_size(net) == 60, "image input layer");
    struct { LayerArgs a; const char *what; } bad[] = {
        {{.type = LAYER_CONV2D, .filters = 2}, "conv without kernel"},
        {{.type = LAYER_CONV2D, .kernel = 3}, "conv without filters"},
        {{.type = LAYER_CONV2D, .filters = 2, .kernel = 3, .padding = 3}, "padding >= kernel"},
        {{.type = LAYER_CONV2D, .filters = 2, .kernel = 6}, "kernel larger than the input"},
        {{.type = LAYER_MAX_POOL2D, .kernel = 5, .padding = 5}, "pool padding >= kernel"},
        {{.type = (LayerType)9, .neurons_amount = 3}, "unknown layer type"},
    };
    for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) {
        bad[i].a.net = net;
        layer_struct_arguments(bad[i].a);
        CHECK(spingalett_layer_count(net) == 1 && spingalett_last_error_code() == SPINGALETT_ERR_INVALID, "%s must be refused", bad[i].what);
    }
    conv2d(.net = net, .filters = 7, .kernel_h = 3, .kernel_w = 2, .stride_w = 2, .padding = 1, .act_func = ACT_RELU);
    max_pool2d(.net = net, .kernel = 2, .act_func = ACT_TANH);          /* pooling has no activation */
    layer(.net = net, .neurons_amount = 4, .act_func = ACT_NONE);
    SpingalettNetworkLayer c, p, d;
    bool ok = spingalett_network_layer(net, 1, &c) && spingalett_network_layer(net, 2, &p) && spingalett_network_layer(net, 3, &d);
    /* conv: (5 + 2 - 3) / 1 + 1 = 5 rows, (4 + 2 - 2) / 2 + 1 = 3 columns; pool: 2 x 1 */
    CHECK(ok && c.type == LAYER_CONV2D && c.height == 5 && c.width == 3 && c.channels == 7 && c.outputs == 105 &&
          c.kernel_h == 3 && c.kernel_w == 2 && c.stride_h == 1 && c.stride_w == 2 && c.padding_h == 1 && c.padding_w == 1 &&
          c.weight_count == 7 * 3 * 2 * 3 && c.bias_count == 7 && c.activation == ACT_RELU, "conv layer description");
    CHECK(ok && p.type == LAYER_MAX_POOL2D && p.height == 2 && p.width == 1 && p.channels == 7 && p.stride_h == 2 &&
          p.activation == ACT_NONE && p.weight_count == 0 && p.bias_count == 0, "pool layer description");
    CHECK(ok && d.type == LAYER_DENSE && d.weight_count == 4 * 14 && d.bias_count == 4, "dense after pooling flattens");
    CHECK(spingalett_parameter_count(net) == 7 * 18 + 7 + 4 * 14 + 4, "parameter count");
    float w[126], back[126];
    for (int i = 0; i < 126; i++) w[i] = (float)i * 0.01f;
    CHECK(spingalett_set_parameters(net, 1, PARAM_WEIGHTS, w, 126) && spingalett_get_parameters(net, 1, PARAM_WEIGHTS, back, 126) &&
          !memcmp(w, back, sizeof w), "conv weights round-trip");
    CHECK(!spingalett_set_parameters(net, 1, PARAM_WEIGHTS, w, 125) && !spingalett_get_parameters(net, 2, PARAM_BIASES, back, 1) &&
          spingalett_get_parameters(net, 2, PARAM_BIASES, back, 0) && !spingalett_get_parameters(net, 0, PARAM_WEIGHTS, back, 0),
          "parameter counts and layer indices are checked");
    free_network(net);
}

/* A small CNN tells vertical from horizontal bars in noisy 8 x 8 images. */
/* Training gives the same bits on one thread and many, in either compute mode: products split
   their work by tiles of C (and k slots fixed by the shape), epilogues and activations run row by
   row, reductions in fixed blocks. Odd sizes put elements into vector tails. */
static void deterministic_threads(void) {
    static const ComputeMode modes[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENMP, COMPUTE_OPENMP};
    static const unsigned threads[] = {1, 1, 3, 4};
    for (int kind = 0; kind < 3; kind++) {
        float *ref = NULL;
        uint64_t count = 0;
        bool same = true;
        for (int v = 0; v < 4; v++) {
            spingalett_set_num_threads(threads[v]);
            spingalett_set_compute_mode(modes[v]);
            lcg_state = 77;
            spingalett_seed(5);
            NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
            uint32_t n = 96, in;
            if (kind == 0) {        /* convolutions with tanh and 70 or 5 filters (few pixels: tiles of
                                       the product split the filters between threads), pooling, dense */
                layer(.net = net, .height = 5, .width = 4, .channels = 16);
                conv2d(.net = net, .filters = 70, .kernel = 3, .padding = 1, .act_func = ACT_TANH);
                avg_pool2d(.net = net, .kernel = 2, .stride = 1);
                conv2d(.net = net, .filters = 5, .kernel = 3, .stride = 2, .act_func = ACT_SIGMOID);
                layer(.net = net, .neurons_amount = 13, .act_func = ACT_TANH);
                layer(.net = net, .neurons_amount = 3, .act_func = ACT_SOFTMAX);
            } else if (kind == 2) { /* grouped and depthwise convolutions (multiplier 2) */
                layer(.net = net, .height = 9, .width = 8, .channels = 8);
                conv2d(.net = net, .filters = 16, .kernel = 3, .padding = 1, .groups = 4, .act_func = ACT_TANH);
                conv2d(.net = net, .filters = 32, .kernel = 3, .stride = 2, .groups = 16, .act_func = ACT_RELU);
                avg_pool2d(.net = net, .kernel = 2);
                layer(.net = net, .neurons_amount = 3, .act_func = ACT_SOFTMAX);
            } else {                /* wide dense layers of odd sizes */
                layer(.net = net, .neurons_amount = 300);
                layer(.net = net, .neurons_amount = 517, .act_func = ACT_TANH);
                layer(.net = net, .neurons_amount = 77, .act_func = ACT_SIGMOID);
                layer(.net = net, .neurons_amount = 3, .act_func = ACT_SOFTMAX);
            }
            in = net->topology[0];
            float *x = malloc((size_t)n * in * sizeof(float)), *y = calloc((size_t)n * 3, sizeof(float));
            for (size_t i = 0; i < (size_t)n * in; i++) x[i] = frand() * 2 - 1;
            for (uint32_t s = 0; s < n; s++) y[s * 3 + s % 3] = 1.0f;
            train(.net = net, .inputs = x, .targets = y, .sample_count = n, .epochs = 2, .learning_rate = 0.01f,
                  .optimizer_type = OPTIMIZER_ADAM, .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 32,
                  .do_not_shuffle = true);
            if (!ref) {
                count = net->total_weights;
                ref = malloc(count * sizeof(float));
                memcpy(ref, net->weights, count * sizeof(float));
            } else if (memcmp(ref, net->weights, count * sizeof(float)) != 0) {
                same = false;
            }
            free(x); free(y); free_network(net);
        }
        static const char *const kinds[] = {"convolution", "dense", "grouped convolution"};
        CHECK(same, "%s training differs between thread counts or compute modes", kinds[kind]);
        printf("  %s training: identical on 1, 3 and 4 threads and single-threaded%s\n", kinds[kind], same ? "" : " -- NO");
        free(ref);
    }
    spingalett_set_num_threads(4);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
}

/* Augmentation: refused without an image input; flips mirror the images (with full batches of
   half-flipped samples, which half is bright tells almost nothing); deterministic on any backend. */
static void augmentation(void) {
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_MSE);
    layer(.net = net, .neurons_amount = 4);
    layer(.net = net, .neurons_amount = 1, .act_func = ACT_NONE);
    float x[8] = {0}, y[2] = {0};
    TrainReport r = train(.net = net, .inputs = x, .targets = y, .sample_count = 2, .epochs = 1, .augment_flip = true);
    CHECK(r.status == TRAIN_FAILED, "augmentation without an image input must be refused");
    free_network(net);

    const uint32_t N = 2000, T = 400;               /* training samples, then held-out ones */
    float *xs = malloc((size_t)(N + T) * 64 * 4), *ys = calloc((size_t)(N + T) * 2, 4);
    lcg_state = 4040;
    for (uint32_t s = 0; s < N + T; s++) {          /* the bright half: left or right */
        bool right = s % 2;
        for (uint32_t i = 0; i < 64; i++)
            xs[s * 64 + i] = (frand() - 0.5f) * 0.3f + (((i % 8) >= 4) == right ? 1.0f : 0.0f);
        ys[s * 2 + right] = 1.0f;
    }
    for (int flip = 0; flip < 2; flip++) {
        spingalett_seed(8);
        net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
        layer(.net = net, .height = 8, .width = 8, .channels = 1);
        layer(.net = net, .neurons_amount = 2, .act_func = ACT_SOFTMAX, .weight_initialization = WEIGHT_INITIALIZATION_NONE);
        train(.net = net, .inputs = xs, .targets = ys, .sample_count = N, .epochs = 30, .learning_rate = 0.05f,
              .optimizer_type = OPTIMIZER_SGD, .training_strategy = STRATEGY_FULL_BATCH, .augment_flip = flip);
        /* flipped, the halves carry no information: the outputs stay near 1/2, the loss near ln 2 */
        EvalMetrics m = evaluate(.net = net, .inputs = xs + (size_t)N * 64, .targets = ys + (size_t)N * 2, .sample_count = T);
        printf("  augmentation%s: held-out loss %.3f\n", flip ? " with flips" : "", m.loss);
        CHECK(flip ? m.loss > 0.6f : m.loss < 0.1f, "augmentation%s: held-out loss %.3f", flip ? " with flips" : "", m.loss);
        free_network(net);
    }

    /* shifts and flips through a convolution network: identical weights on every backend */
    float *ref = NULL;
    uint64_t count = 0;
    bool same = true;
    for (int mode = 0; mode < 3; mode++) {
        spingalett_seed(9);
        spingalett_set_compute_mode(mode == 0 ? COMPUTE_SINGLE_THREADED : COMPUTE_OPENMP);
        spingalett_set_num_threads(mode == 2 ? 3 : 4);
        net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
        layer(.net = net, .height = 8, .width = 8, .channels = 1);
        conv2d(.net = net, .filters = 4, .kernel = 3, .padding = 1, .act_func = ACT_RELU);
        max_pool2d(.net = net, .kernel = 2);
        layer(.net = net, .neurons_amount = 2, .act_func = ACT_SOFTMAX);
        train(.net = net, .inputs = xs, .targets = ys, .sample_count = 256, .epochs = 2, .learning_rate = 0.01f,
              .optimizer_type = OPTIMIZER_ADAM, .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 32,
              .augment_shift = 2, .augment_flip = true);
        if (!ref) {
            count = net->total_weights;
            ref = malloc(count * sizeof(float));
            memcpy(ref, net->weights, count * sizeof(float));
        } else {
            same = same && !memcmp(ref, net->weights, count * sizeof(float));
        }
        free_network(net);
    }
    CHECK(same, "augmented training differs between compute modes or thread counts");
    free(ref);
    free(xs); free(ys);
    spingalett_set_num_threads(4);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
}

/* The weight and bias gradients of a convolution output layer for given output gradients (over
   40 samples of 8 x 8 cells: several slots of pixels), against sums in double, and the same bits on
   1, 3 and 4 threads. */
static void conv_weight_gradient(void) {
    static const struct { uint32_t c, f, k, stride, pad, groups; } cases[] = {
        {32, 24, 3, 1, 1, 2}, {16, 16, 3, 2, 1, 1}, {16, 12, 1, 1, 0, 1}, {48, 18, 2, 1, 0, 1},
    };
    for (size_t t = 0; t < sizeof cases / sizeof cases[0]; t++) {
        const uint32_t n = 40, C = cases[t].c, F = cases[t].f, K = cases[t].k, G = cases[t].groups;
        float *ref = NULL;
        bool same = true;
        double worst = 0.0;
        for (int v = 0; v < 4; v++) {
            static const unsigned threads[] = {1, 1, 3, 4};
            spingalett_set_num_threads(threads[v]);
            spingalett_set_compute_mode(v ? COMPUTE_OPENMP : COMPUTE_SINGLE_THREADED);
            lcg_state = 90 + (unsigned)t;
            NeuralNetwork *net = new_spingalett(.loss_func = LOSS_MSE);
            layer(.net = net, .height = 8, .width = 8, .channels = C);
            conv2d(.net = net, .filters = F, .kernel = K, .stride = cases[t].stride, .padding = cases[t].pad, .groups = G,
                   .act_func = ACT_NONE);
            SpingalettNetworkLayer o;
            spingalett_network_layer(net, 1, &o);
            const uint32_t in = 64 * C, out = o.outputs;
            float *x = malloc((size_t)n * in * 4), *dy = malloc((size_t)n * out * 4);
            for (size_t i = 0; i < (size_t)n * in; i++) x[i] = frand() * 2 - 1;
            for (size_t i = 0; i < (size_t)n * out; i++) dy[i] = frand() * 2 - 1;
            for (uint64_t i = 0; i < net->total_weights; i++) net->weights[i] = frand() - 0.5f;
            SpingalettTrainer *tr = spingalett_trainer_new(net, n);
            spingalett_trainer_forward(tr, x, n);
            spingalett_trainer_backward_output_grads(tr, dy);
            float *gw = malloc(o.weight_count * 4), gb[64];
            spingalett_get_parameters(net, 1, PARAM_WEIGHT_GRADIENTS, gw, o.weight_count);
            spingalett_get_parameters(net, 1, PARAM_BIAS_GRADIENTS, gb, o.bias_count);
            if (v == 0) {
                /* gW[f][kh][kw][c] = sum over samples and output cells of dy * the input cell it covers */
                const uint32_t CG = C / G, FG = F / G;
                for (uint32_t f = 0; f < F; f++) {
                    double db = 0.0;
                    for (uint32_t s = 0; s < n; s++)
                        for (uint32_t p = 0; p < o.height * o.width; p++) db += dy[((size_t)s * o.height * o.width + p) * F + f];
                    worst = fmax(worst, fabs(db - gb[f]) / (1.0 + fabs(db)));
                    for (uint32_t kh = 0; kh < K; kh++)
                        for (uint32_t kw = 0; kw < K; kw++)
                            for (uint32_t c = 0; c < CG; c++) {
                                double sum = 0.0;
                                for (uint32_t s = 0; s < n; s++)
                                    for (uint32_t oh = 0; oh < o.height; oh++)
                                        for (uint32_t ow = 0; ow < o.width; ow++) {
                                            int ih = (int)(oh * cases[t].stride + kh) - (int)cases[t].pad;
                                            int iw = (int)(ow * cases[t].stride + kw) - (int)cases[t].pad;
                                            if (ih < 0 || iw < 0 || ih >= 8 || iw >= 8) continue;
                                            sum += (double)dy[(((size_t)s * o.height + oh) * o.width + ow) * F + f] *
                                                   x[(((size_t)s * 8 + ih) * 8 + iw) * C + (f / FG) * CG + c];
                                        }
                                float got = gw[(((size_t)f * K + kh) * K + kw) * CG + c];
                                worst = fmax(worst, fabs(sum - got) / (1.0 + fabs(sum)));
                            }
                }
                ref = malloc(o.weight_count * 4);
                memcpy(ref, gw, o.weight_count * 4);
            } else {
                same = same && !memcmp(ref, gw, o.weight_count * 4);
            }
            free(gw); free(x); free(dy);
            spingalett_trainer_free(tr);
            free_network(net);
        }
        CHECK(worst < 1e-4 && same, "conv weight gradient %zu: relative error %.2e%s", t, worst,
              same ? "" : ", differs between thread counts");
        printf("  conv weight gradient %ux%u, %u -> %u channels, %u groups: error %.1e, identical on 1, 3, 4 threads%s\n",
               K, K, C, F, G, worst, same ? "" : " -- NO");
        free(ref);
    }
    spingalett_set_num_threads(4);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
}

static void conv_learns(ComputeMode mode) {
    lcg_state = 31337;
    spingalett_set_compute_mode(mode);
    const uint32_t N = 400;
    float *x = malloc((size_t)N * 64 * 4), *y = calloc((size_t)N * 2, 4);
    for (uint32_t s = 0; s < N; s++) {
        bool vertical = s % 2;
        uint32_t pos = (uint32_t)(frand() * 6) + 1;
        for (uint32_t i = 0; i < 64; i++) {
            uint32_t r = i / 8, c = i % 8;
            x[s * 64 + i] = (frand() - 0.5f) * 0.6f + ((vertical ? c : r) == pos ? 1.0f : 0.0f);
        }
        y[s * 2 + vertical] = 1.0f;
    }
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(.net = net, .height = 8, .width = 8, .channels = 1);
    conv2d(.net = net, .filters = 4, .kernel = 3, .padding = 1, .act_func = ACT_RELU, .weight_initialization = WEIGHT_INITIALIZATION_HE);
    max_pool2d(.net = net, .kernel = 2);
    layer(.net = net, .neurons_amount = 2, .act_func = ACT_SOFTMAX, .weight_initialization = WEIGHT_INITIALIZATION_XAVIER);
    train(.net = net, .inputs = x, .targets = y, .sample_count = N, .epochs = 15, .learning_rate = 0.01f,
          .optimizer_type = OPTIMIZER_ADAM, .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 32);
    EvalMetrics m = evaluate(.net = net, .inputs = x, .targets = y, .sample_count = N);
    printf("  conv learns bars mode=%d: accuracy %.1f%%, loss %.4f\n", mode, 100.0 * m.accuracy, m.loss);
    CHECK(m.accuracy > 0.95f, "conv net must learn vertical vs horizontal bars (mode %d): %.3f", mode, m.accuracy);
    free(x); free(y);
    free_network(net);
}

/* RMSProp divides every gradient by its own running magnitude (sqrt(v) + eps with eps = 1e-8, as
   in PyTorch): a gradient at rounding-noise level still gets a sizeable step whose sign is set by
   rounding, so backends that sum in a different order drift apart by ~1e-4 relative to the total
   weight movement (~1 here). The other optimizers agree to ~1e-6. */
static float tolerance_for(OptimizerType opt, float usual) {
    return opt == OPTIMIZER_RMSPROP ? 2e-3f : usual;
}

// Train identical nets in every backend; weights must agree.
static void equivalence(OptimizerType opt, TrainingStrategy strat, float decay, float clip) {
    L ls[] = {{12, ACT_NONE}, {16, ACT_TANH}, {9, ACT_SIGMOID}, {4, ACT_SOFTMAX}};
    uint32_t N = 40; lcg_state = 4242;
    float *x, *y;
    make_data(N, 12, 4, LOSS_CROSS_ENTROPY, ACT_SOFTMAX, &x, &y);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    NeuralNetwork *ref = build(LOSS_CROSS_ENTROPY, ls, 4, NULL, NULL);
    ComputeMode modes[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS};
    float *res[3];
    for (int m = 0; m < 3; m++) {
        spingalett_set_compute_mode(modes[m]);
        NeuralNetwork *net = build(LOSS_CROSS_ENTROPY, ls, 4, ref->weights, ref->biases);
        spingalett_seed(21);
        train(.net = net, .inputs = x, .targets = y, .sample_count = N, .epochs = 20, .learning_rate = 0.01f,
              .optimizer_type = opt, .training_strategy = strat, .batch_size = N, .weight_decay = decay, .max_grad_norm = clip);
        res[m] = malloc(net->total_weights * 4);
        memcpy(res[m], net->weights, net->total_weights * 4);
        free_network(net);
    }
    float d1 = max_abs_diff(res[0], res[1], ref->total_weights), d2 = max_abs_diff(res[0], res[2], ref->total_weights);
    printf("  equivalence opt=%d strat=%d decay=%.3f clip=%.2f: |st-omp|=%.2e |st-blas|=%.2e\n", opt, strat, decay, clip, d1, d2);
    CHECK(d1 < tolerance_for(opt, 1e-4f) && d2 < tolerance_for(opt, 1e-4f), "equivalence opt=%d strat=%d", opt, strat);
    for (int m = 0; m < 3; m++) free(res[m]);
    free_network(ref); free(x); free(y);
}

// With one sample, online and full-batch training are the same algorithm: weights must match.
static void strategy_consistency(OptimizerType opt, float decay, float clip) {
    L ls[] = {{6, ACT_NONE}, {9, ACT_TANH}, {4, ACT_SIGMOID}};
    float *x, *y; lcg_state = 99;
    make_data(1, 6, 4, LOSS_MSE, ACT_SIGMOID, &x, &y);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    NeuralNetwork *a = build(LOSS_MSE, ls, 3, NULL, NULL);
    NeuralNetwork *b = build(LOSS_MSE, ls, 3, a->weights, a->biases);
    train(.net = a, .inputs = x, .targets = y, .sample_count = 1, .epochs = 15, .learning_rate = 0.01f, .optimizer_type = opt, .weight_decay = decay, .max_grad_norm = clip, .training_strategy = STRATEGY_SAMPLE);
    train(.net = b, .inputs = x, .targets = y, .sample_count = 1, .epochs = 15, .learning_rate = 0.01f, .optimizer_type = opt, .weight_decay = decay, .max_grad_norm = clip, .training_strategy = STRATEGY_FULL_BATCH);
    float d = max_abs_diff(a->weights, b->weights, a->total_weights);
    printf("  sample vs full-batch opt=%d decay=%.2f clip=%.2f: %.2e\n", opt, decay, clip, d);
    CHECK(d < tolerance_for(opt, 1e-6f), "strategy consistency opt=%d decay=%.2f clip=%.2f diff %.3e", opt, decay, clip, d);
    free_network(a); free_network(b); free(x); free(y);
}

// One SGD step with lr = 1 must move the parameters by exactly max_grad_norm.
static void clip_norm(TrainingStrategy strat, ComputeMode mode) {
    L ls[] = {{6, ACT_NONE}, {9, ACT_TANH}, {4, ACT_SIGMOID}};
    float *x, *y; lcg_state = 5;
    make_data(1, 6, 4, LOSS_MSE, ACT_SIGMOID, &x, &y);
    spingalett_set_compute_mode(mode);
    NeuralNetwork *a = build(LOSS_MSE, ls, 3, NULL, NULL);
    uint64_t nw = a->total_weights, nb = a->total_biases;
    float *w0 = malloc(nw * 4), *b0 = malloc(nb * 4);
    memcpy(w0, a->weights, nw * 4); memcpy(b0, a->biases, nb * 4);
    train(.net = a, .inputs = x, .targets = y, .sample_count = 1, .epochs = 1, .learning_rate = 1.0f,
          .optimizer_type = OPTIMIZER_SGD, .max_grad_norm = 0.01f, .training_strategy = strat);
    double sq = 0;
    for (uint64_t i = 0; i < nw; i++) sq += ((double)a->weights[i] - w0[i]) * ((double)a->weights[i] - w0[i]);
    for (uint64_t i = 0; i < nb; i++) sq += ((double)a->biases[i] - b0[i]) * ((double)a->biases[i] - b0[i]);
    printf("  clip strat=%d mode=%d: |step| = %.6f (expected 0.010000)\n", strat, mode, sqrt(sq));
    CHECK(fabs(sqrt(sq) - 0.01) < 1e-5, "clip strat=%d mode=%d step norm %.6f", strat, mode, sqrt(sq));
    free_network(a); free(x); free(y); free(w0); free(b0);
}

/* Batches larger than the internal chunk (2048 samples) accumulate their gradient over chunks:
   one full-batch SGD step on 5000 samples must equal the sample-weighted mean of the gradients of
   its three chunks computed separately. */
static void chunked_batches(ComputeMode mode) {
    L ls[] = {{7, ACT_NONE}, {11, ACT_TANH}, {3, ACT_SOFTMAX}};
    const uint32_t N = 5000, parts[3][2] = {{0, 2048}, {2048, 2048}, {4096, 904}};
    float *x, *y; lcg_state = 9001;
    make_data(N, 7, 3, LOSS_CROSS_ENTROPY, ACT_SOFTMAX, &x, &y);
    spingalett_set_compute_mode(mode);
    NeuralNetwork *ref = build(LOSS_CROSS_ENTROPY, ls, 3, NULL, NULL);
    uint64_t nw = ref->total_weights;
    double *expected = calloc(nw, sizeof(double));
    for (int p = 0; p < 3; p++) {
        NeuralNetwork *n = build(LOSS_CROSS_ENTROPY, ls, 3, ref->weights, ref->biases);
        train(.net = n, .inputs = x + (size_t)parts[p][0] * 7, .targets = y + (size_t)parts[p][0] * 3,
              .sample_count = parts[p][1], .epochs = 1, .learning_rate = 1.0f, .optimizer_type = OPTIMIZER_SGD,
              .training_strategy = STRATEGY_FULL_BATCH);
        for (uint64_t i = 0; i < nw; i++)
            expected[i] += ((double)ref->weights[i] - n->weights[i]) * parts[p][1] / N;
        free_network(n);
    }
    NeuralNetwork *full = build(LOSS_CROSS_ENTROPY, ls, 3, ref->weights, ref->biases);
    train(.net = full, .inputs = x, .targets = y, .sample_count = N, .epochs = 1, .learning_rate = 1.0f,
          .optimizer_type = OPTIMIZER_SGD, .training_strategy = STRATEGY_FULL_BATCH);
    double worst = 0;
    for (uint64_t i = 0; i < nw; i++)
        worst = fmax(worst, fabs(((double)ref->weights[i] - full->weights[i]) - expected[i]));
    printf("  chunked full batch (5000 = 2048+2048+904) mode=%d: max |diff| %.2e\n", mode, worst);
    CHECK(worst < 1e-6, "chunked batch accumulation mode=%d (%.3e)", mode, worst);
    free(expected); free_network(full); free_network(ref); free(x); free(y);
}

// Training 10 epochs at once must equal 5 + 5 (optimizer state persists in the net).
static void continuation(OptimizerType opt) {
    L ls[] = {{5, ACT_NONE}, {7, ACT_TANH}, {3, ACT_SIGMOID}};
    uint32_t N = 16; lcg_state = 16;
    float *x, *y;
    make_data(N, 5, 3, LOSS_MSE, ACT_SIGMOID, &x, &y);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    NeuralNetwork *a = build(LOSS_MSE, ls, 3, NULL, NULL);
    NeuralNetwork *b = build(LOSS_MSE, ls, 3, a->weights, a->biases);
    train(.net = a, .inputs = x, .targets = y, .sample_count = N, .epochs = 10, .learning_rate = 0.01f, .optimizer_type = opt, .training_strategy = STRATEGY_FULL_BATCH);
    train(.net = b, .inputs = x, .targets = y, .sample_count = N, .epochs = 5, .learning_rate = 0.01f, .optimizer_type = opt, .training_strategy = STRATEGY_FULL_BATCH);
    train(.net = b, .inputs = x, .targets = y, .sample_count = N, .epochs = 5, .learning_rate = 0.01f, .optimizer_type = opt, .training_strategy = STRATEGY_FULL_BATCH);
    float d = max_abs_diff(a->weights, b->weights, a->total_weights);
    printf("  continuation opt=%d: |10 - (5+5)| = %.2e\n", opt, d);
    CHECK(d < 1e-6f, "continuation opt=%d diff %.3e", opt, d);
    free_network(a); free_network(b); free(x); free(y);
}

static void roundtrip(PrecisionMode p, float tol) {
    L ls[] = {{6, ACT_NONE}, {10, ACT_RELU}, {3, ACT_SIGMOID}};
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    NeuralNetwork *a = build(LOSS_MSE, ls, 3, NULL, NULL);
    for (uint64_t i = 0; i < a->total_biases; i++) a->biases[i] = frand() - 0.5f;
    save_spingalett(.net = a, .filename = "spingalett_test_roundtrip.slett", .precision = p);
    NeuralNetwork *b = load_spingalett("spingalett_test_roundtrip.slett");
    remove("spingalett_test_roundtrip.slett");
    if (p == PRECISION_FLOAT32) {   /* a name without extension gets SPINGALETT_MODEL_EXTENSION */
        save_spingalett(.net = a, .filename = "spingalett_test_noext");
        FILE *f = fopen("spingalett_test_noext" SPINGALETT_MODEL_EXTENSION, "rb");
        CHECK(f != NULL, "save without extension should write spingalett_test_noext.slett");
        if (f) fclose(f);
        remove("spingalett_test_noext" SPINGALETT_MODEL_EXTENSION);
    }
    CHECK(b != NULL, "load returned NULL for precision %d", p);
    if (b) {
        float dw = max_abs_diff(a->weights, b->weights, a->total_weights);
        float db = max_abs_diff(a->biases, b->biases, a->total_biases);
        printf("  roundtrip precision=%d: max|dw|=%.2e max|db|=%.2e\n", p, dw, db);
        CHECK(dw <= tol && db <= tol, "roundtrip precision %d", p);
        free_network(b);
    }
    free_network(a);
}

/* Rewrites `path` without its last `drop` bytes. */
static bool truncate_file(const char *path, long drop) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)size);
    bool ok = buf && fread(buf, 1, (size_t)size, f) == (size_t)size;
    fclose(f);
    if (ok && (f = fopen(path, "wb")) != NULL) {
        ok = fwrite(buf, 1, (size_t)(size - drop), f) == (size_t)(size - drop);
        fclose(f);
    }
    free(buf);
    return ok;
}

static void error_codes(void) {
    char expected[32];
    snprintf(expected, sizeof expected, "%d.%d.%d", SPINGALETT_VERSION_MAJOR, SPINGALETT_VERSION_MINOR, SPINGALETT_VERSION_PATCH);
    CHECK(strcmp(spingalett_version(), expected) == 0 && strcmp(spingalett_version(), SPINGALETT_VERSION_STRING) == 0,
          "library version %s vs headers %s", spingalett_version(), expected);

    spingalett_clear_error();
    CHECK(load_spingalett("spingalett_test_missing.slett") == NULL && spingalett_last_error_code() == SPINGALETT_ERR_FILE_IO,
          "missing file: code %d", spingalett_last_error_code());

    FILE *f = fopen("spingalett_test_future.slett", "wb");
    uint16_t future = SPINGALETT_FORMAT_VERSION + 1;
    fwrite(&future, sizeof future, 1, f);
    fclose(f);
    spingalett_clear_error();
    CHECK(load_spingalett("spingalett_test_future.slett") == NULL && spingalett_last_error_code() == SPINGALETT_ERR_FORMAT_VERSION,
          "future format version: code %d", spingalett_last_error_code());
    remove("spingalett_test_future.slett");

    /* cross-entropy needs a softmax or sigmoid output; otherwise the gradient would be wrong */
    float x[2] = {0.5f, -0.5f}, y[2] = {1.0f, 0.0f};
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(net, 2); layer(net, 2, ACT_TANH);
    spingalett_clear_error();
    train(.net = net, .inputs = x, .targets = y, .sample_count = 1, .epochs = 1);
    CHECK(spingalett_last_error_code() == SPINGALETT_ERR_INVALID && net->time_step == 0, "CE + tanh output must be rejected");
    free_network(net);
    printf("  error codes checked (library %s)\n", spingalett_version());
}

static void load_robustness(void) {
    L ls[] = {{6, ACT_NONE}, {10, ACT_RELU}, {3, ACT_SIGMOID}};
    NeuralNetwork *a = build(LOSS_MSE, ls, 3, NULL, NULL);
    save_spingalett(.net = a, .filename = "spingalett_test_robust.slett");
    layer(.net = NULL, .neurons_amount = 3);              /* leaves a sticky error behind */
    NeuralNetwork *b = load_spingalett("spingalett_test_robust.slett");
    CHECK(b != NULL, "load failed after an unrelated earlier error");
    if (b) free_network(b);
    CHECK(truncate_file("spingalett_test_robust.slett", 7), "could not truncate test file");
    spingalett_clear_error();
    NeuralNetwork *c = load_spingalett("spingalett_test_robust.slett");
    CHECK(c == NULL && spingalett_last_error_code() != SPINGALETT_OK, "truncated file must fail to load");
    if (c) free_network(c);
    remove("spingalett_test_robust.slett");
    printf("  load robustness checked\n");
    free_network(a);
}

static size_t sched_calls, sched_last_epoch, sched_total;
static float one_shot(size_t epoch, size_t total, float lr, void *ud) {
    (void)ud; sched_calls++; sched_last_epoch = epoch; sched_total = total;
    return epoch == 0 ? lr : (epoch == 1 ? -1.0f : 0.0f);   /* -1 must be ignored (keeps lr) */
}

static void schedulers(void) {
    #define NEAR(a, b) (fabsf((a) - (b)) < 1e-6f)
    LRScheduleParams p = {.warmup_epochs = 4, .step_size = 10, .gamma = 0.5f, .min_lr = 0.1f};
    CHECK(NEAR(spingalett_lr_cosine_decay(0, 100, 1.0f, NULL), 1.0f), "cosine start");
    CHECK(NEAR(spingalett_lr_cosine_decay(50, 100, 1.0f, NULL), 0.5f), "cosine mid");
    CHECK(NEAR(spingalett_lr_cosine_decay(50, 100, 1.0f, &p), 0.55f), "cosine mid with floor");
    CHECK(spingalett_lr_cosine_decay(99, 100, 1.0f, NULL) > 0.0f, "cosine last epoch still trains");
    CHECK(NEAR(spingalett_lr_linear_warmup(0, 100, 1.0f, &p), 0.25f), "warmup first");
    CHECK(NEAR(spingalett_lr_linear_warmup(3, 100, 1.0f, &p), 1.0f), "warmup end");
    CHECK(NEAR(spingalett_lr_linear_warmup(50, 100, 1.0f, &p), 1.0f), "warmup after");
    CHECK(NEAR(spingalett_lr_linear_warmup(0, 100, 1.0f, NULL), 0.2f), "warmup default 5%%");
    CHECK(NEAR(spingalett_lr_step_decay(9, 100, 1.0f, &p), 1.0f), "step before");
    CHECK(NEAR(spingalett_lr_step_decay(10, 100, 1.0f, &p), 0.5f), "step one");
    CHECK(NEAR(spingalett_lr_step_decay(25, 100, 1.0f, &p), 0.25f), "step two");
    CHECK(NEAR(spingalett_lr_step_decay(34, 100, 1.0f, NULL), 0.1f), "step default");
    CHECK(NEAR(spingalett_lr_warmup_cosine(1, 104, 1.0f, &p), 0.5f), "warmup_cosine warm");
    CHECK(NEAR(spingalett_lr_warmup_cosine(4, 104, 1.0f, &p), 1.0f), "warmup_cosine peak");
    CHECK(NEAR(spingalett_lr_warmup_cosine(54, 104, 1.0f, &p), 0.55f), "warmup_cosine mid");

    /* integration: lr only on epoch 0 must equal one plain epoch */
    L ls[] = {{5, ACT_NONE}, {7, ACT_TANH}, {3, ACT_SIGMOID}};
    float *x, *y; lcg_state = 31;
    make_data(8, 5, 3, LOSS_MSE, ACT_SIGMOID, &x, &y);
    TrainingStrategy strats[] = {STRATEGY_FULL_BATCH, STRATEGY_SAMPLE};
    for (int st = 0; st < 2; st++) {
        NeuralNetwork *a = build(LOSS_MSE, ls, 3, NULL, NULL);
        NeuralNetwork *b = build(LOSS_MSE, ls, 3, a->weights, a->biases);
        sched_calls = 0;
        train(.net = a, .inputs = x, .targets = y, .sample_count = 8, .epochs = 6, .learning_rate = 0.3f,
              .optimizer_type = OPTIMIZER_SGD, .training_strategy = strats[st], .lr_scheduler = one_shot, .do_not_shuffle = true);
        train(.net = b, .inputs = x, .targets = y, .sample_count = 8, .epochs = 1, .learning_rate = 0.3f,
              .optimizer_type = OPTIMIZER_SGD, .training_strategy = strats[st], .do_not_shuffle = true);
        /* epoch 1 returns -1 (ignored => lr stays 0.3), so a ran 2 effective epochs; b one more */
        train(.net = b, .inputs = x, .targets = y, .sample_count = 8, .epochs = 1, .learning_rate = 0.3f,
              .optimizer_type = OPTIMIZER_SGD, .training_strategy = strats[st], .do_not_shuffle = true);
        float d = max_abs_diff(a->weights, b->weights, a->total_weights);
        printf("  scheduler integration strat=%d: calls=%zu last_epoch=%zu total=%zu diff=%.2e\n", strats[st], sched_calls, sched_last_epoch, sched_total, d);
        CHECK(sched_calls == 6 && sched_last_epoch == 5 && sched_total == 6, "scheduler call sequence");
        CHECK(d < 1e-6f, "scheduler lr not applied (diff %.3e)", d);
        free_network(a); free_network(b);
    }
    free(x); free(y);
}

/* ---------------- dropout ---------------- */
static float cb_loss;
static bool capture_loss(NeuralNetwork *n, const TrainProgress *p, void *loss) { (void)n; *(float *)loss = p->train_loss; return false; }

/* Same seed + same time_step => same dropout masks, so train() with a negligible lr evaluates the
   masked loss, and train() with lr = 1 and SGD yields the masked gradient. */
static void dropout_run(NeuralNetwork *net, const float *x, const float *y, uint32_t N, float lr, TrainingStrategy st) {
    net->time_step = 0;
    spingalett_seed(1234);
    train(.net = net, .inputs = x, .targets = y, .sample_count = N, .epochs = 1, .learning_rate = lr,
          .optimizer_type = OPTIMIZER_SGD, .training_strategy = st, .batch_size = N,
          .callback = capture_loss, .callback_interval = 1, .callback_data = &cb_loss);
}

static void dropout_gradcheck(ComputeMode mode, TrainingStrategy st) {
    uint32_t N = st == STRATEGY_SAMPLE ? 1 : 5;
    float *x, *y; lcg_state = 2024;
    make_data(N, 4, 3, LOSS_CROSS_ENTROPY, ACT_SOFTMAX, &x, &y);
    spingalett_set_compute_mode(mode);
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(net, 4);
    layer(net, 12, ACT_TANH, WEIGHT_INITIALIZATION_XAVIER, 0.4f);
    layer(net, 10, ACT_SIGMOID, WEIGHT_INITIALIZATION_XAVIER, 0.25f);
    layer(net, 9, ACT_FOO52, WEIGHT_INITIALIZATION_XAVIER, 0.3f);
    layer(net, 3, ACT_SOFTMAX, WEIGHT_INITIALIZATION_XAVIER);
    for (uint64_t i = 0; i < net->total_weights; i++) net->weights[i] = frand() * 1.2f - 0.6f;
    for (uint64_t i = 0; i < net->total_biases; i++) net->biases[i] = frand() * 0.4f - 0.2f + 0.5f * (i >= 22 && i < 31);
    uint64_t nw = net->total_weights, nb = net->total_biases;
    float *w0 = malloc(nw * 4), *b0 = malloc(nb * 4), *ga = malloc((nw + nb) * 4);
    memcpy(w0, net->weights, nw * 4); memcpy(b0, net->biases, nb * 4);

    dropout_run(net, x, y, N, 1.0f, st);
    for (uint64_t i = 0; i < nw; i++) ga[i] = w0[i] - net->weights[i];
    for (uint64_t i = 0; i < nb; i++) ga[nw + i] = b0[i] - net->biases[i];

    int bad = 0, kinks = 0, zero_rows = 0;
    for (uint64_t i = 0; i < nw + nb; i++) {
        double l[4]; float hs[4] = {1e-3f, -1e-3f, 2.5e-4f, -2.5e-4f};
        for (int k = 0; k < 4; k++) {
            memcpy(net->weights, w0, nw * 4); memcpy(net->biases, b0, nb * 4);
            if (i < nw) net->weights[i] += hs[k]; else net->biases[i - nw] += hs[k];
            dropout_run(net, x, y, N, 1e-30f, st);
            l[k] = cb_loss;
        }
        double gn = (l[0] - l[1]) / 2e-3, gn2 = (l[2] - l[3]) / 5e-4;
        if (ga[i] == 0.0f && gn == 0.0) zero_rows++;
        if (fabs(gn - gn2) > 0.1 * (fabs(gn) + fabs(gn2)) + 3e-4) { kinks++; continue; }
        if (fabs(gn - ga[i]) / fmax(1e-2, fabs(gn) + fabs(ga[i])) > 3e-2)
            bad++;
    }
    printf("  dropout gradcheck mode=%d strat=%d: outliers=%d/%llu kinks=%d dropped-param-grads=%d\n",
           mode, st, bad, (unsigned long long)(nw + nb), kinks, zero_rows);
    CHECK(bad == 0 && kinks * 10 < (int)(nw + nb), "dropout gradcheck mode=%d strat=%d", mode, st);
    CHECK(zero_rows > 0, "dropout gradcheck: no parameter had a dropped unit (mask inactive?)");
    memcpy(net->weights, w0, nw * 4); memcpy(net->biases, b0, nb * 4);
    free_network(net); free(x); free(y); free(w0); free(b0); free(ga);
}

static void dropout_mask_stats(void) {
    const uint32_t H = 4000; const float p = 0.3f;
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    spingalett_seed(11);
    float x[3] = {0.3f, -0.7f, 0.9f}, y[2] = {1.0f, 0.0f};
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(net, 3);
    layer(.net = net, .neurons_amount = H, .act_func = ACT_TANH, .weight_initialization = WEIGHT_INITIALIZATION_XAVIER, .dropout_rate = p);
    layer(net, 2, ACT_SOFTMAX);
    train(.net = net, .inputs = x, .targets = y, .sample_count = 1, .epochs = 1, .learning_rate = 1e-30f, .optimizer_type = OPTIMIZER_SGD);
    float *masked = malloc(H * 4);
    memcpy(masked, net->neurons + net->neuron_offsets[1], H * 4);
    forward(net, x);                                       /* inference: no dropout */
    const float *plain = net->neurons + net->neuron_offsets[1];
    uint32_t dropped = 0, bad_scale = 0;
    for (uint32_t j = 0; j < H; j++) {
        if (masked[j] == 0.0f && plain[j] != 0.0f) dropped++;
        else if (fabsf(masked[j] - plain[j] / (1.0f - p)) > 1e-6f * fabsf(masked[j]) + 1e-12f) bad_scale++;
    }
    double expect = H * p, sd = sqrt(H * p * (1 - p));
    printf("  dropout mask: dropped %u of %u (expected %.0f +- %.0f), mis-scaled kept units: %u\n", dropped, H, expect, sd, bad_scale);
    CHECK(fabs(dropped - expect) < 5 * sd, "dropout rate off: %u dropped", dropped);
    CHECK(bad_scale == 0, "kept units must be scaled by 1/(1-p)");
    free(masked); free_network(net);
}

static void dropout_equivalence(TrainingStrategy st, uint32_t batch) {
    uint32_t N = 24;
    float *x, *y; lcg_state = 77;
    make_data(N, 6, 3, LOSS_CROSS_ENTROPY, ACT_SOFTMAX, &x, &y);
    float *ref_w = NULL, *ref_b = NULL, *trained0 = NULL; uint64_t nw = 0;
    ComputeMode modes[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS};
    float diff = 0;
    for (int m = 0; m < 3; m++) {
        spingalett_set_compute_mode(modes[m]);
        NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
        layer(net, 6);
        layer(net, 40, ACT_RELU, WEIGHT_INITIALIZATION_HE, 0.5f);
        layer(net, 33, ACT_TANH, WEIGHT_INITIALIZATION_XAVIER, 0.2f);
        layer(net, 3, ACT_SOFTMAX, WEIGHT_INITIALIZATION_XAVIER);
        if (!ref_w) { nw = net->total_weights; ref_w = malloc(nw * 4); ref_b = malloc(net->total_biases * 4);
                      for (uint64_t i = 0; i < nw; i++) ref_w[i] = frand() - 0.5f;
                      for (uint64_t i = 0; i < net->total_biases; i++) ref_b[i] = 0.1f * (frand() - 0.5f); }
        memcpy(net->weights, ref_w, nw * 4); memcpy(net->biases, ref_b, net->total_biases * 4);
        spingalett_seed(99);
        train(.net = net, .inputs = x, .targets = y, .sample_count = N, .epochs = 15, .learning_rate = 0.01f,
              .optimizer_type = OPTIMIZER_ADAM, .training_strategy = st, .batch_size = batch);
        if (m == 0) { trained0 = malloc(nw * 4); memcpy(trained0, net->weights, nw * 4); }
        else { float d = max_abs_diff(trained0, net->weights, nw); if (d > diff) diff = d; }
        free_network(net);
    }
    free(trained0);
    printf("  dropout backend equivalence strat=%d batch=%u: max |diff| = %.2e\n", st, batch, diff);
    CHECK(diff < 1e-4f, "dropout equivalence strat=%d", st);
    free(ref_w); free(ref_b); free(x); free(y);
}

static void dropout_misc(void) {
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    spingalett_seed(12);
    /* inference is unaffected by dropout settings */
    NeuralNetwork *a = new_spingalett(.loss_func = LOSS_MSE);
    layer(a, 3, .dropout_rate = 0.5f);                    /* input layer: ignored */
    layer(a, 16, ACT_RELU, WEIGHT_INITIALIZATION_HE, 0.5f);
    layer(a, 2, ACT_SIGMOID, WEIGHT_INITIALIZATION_XAVIER, 0.5f);   /* output layer: ignored in training */
    CHECK(a->dropout_rates[0] == 0.0f, "input-layer dropout must be ignored");
    NeuralNetwork *b = new_spingalett(.loss_func = LOSS_MSE);
    layer(b, 3); layer(b, 16, ACT_RELU); layer(b, 2, ACT_SIGMOID);
    memcpy(b->weights, a->weights, a->total_weights * 4);
    float in[3] = {0.5f, -0.25f, 1.0f};
    float oa[2], ob[2];
    memcpy(oa, forward(a, in), 8); memcpy(ob, forward(b, in), 8);
    CHECK(oa[0] == ob[0] && oa[1] == ob[1], "dropout must not affect inference");

    /* invalid rates */
    spingalett_clear_error();
    layer(b, 4, ACT_RELU, WEIGHT_INITIALIZATION_HE, 1.0f);
    CHECK(spingalett_last_error_code() == SPINGALETT_ERR_INVALID && b->layers == 3, "dropout 1.0 must be rejected");
    layer(b, 4, ACT_RELU, WEIGHT_INITIALIZATION_HE, -0.1f);
    CHECK(b->layers == 3, "negative dropout must be rejected");

    /* v2 roundtrip keeps the rates */
    save_spingalett(.net = a, .filename = "spingalett_test_dropout.slett");
    NeuralNetwork *c = load_spingalett("spingalett_test_dropout.slett");
    remove("spingalett_test_dropout.slett");
    CHECK(c && c->dropout_rates[1] == 0.5f && c->dropout_rates[2] == 0.5f && c->dropout_rates[0] == 0.0f, "dropout rates not persisted");
    if (c) free_network(c);

    /* v1 files written by the original release still load (rates = 0) */
    NeuralNetwork *d = load_spingalett(SPINGALETT_TEST_DATA_DIR "/xor_v1.nn");
    CHECK(d != NULL, "v1 model failed to load");
    if (d) {
        float xin[4][2] = {{0,0},{0,1},{1,0},{1,1}};
        const float expected[4] = {0.0013f, 0.9978f, 0.9989f, 0.0022f};   /* printed by the original release */
        printf("  v1 model outputs:");
        for (int i = 0; i < 4; i++) {
            float o = forward(d, xin[i])[0];
            printf(" %.4f", o);
            CHECK(fabsf(o - expected[i]) < 5e-5f, "v1 model output %d: %.5f", i, o);
        }
        printf("\n");
        CHECK(d->layers == 3 && d->time_step > 0 && d->dropout_rates[1] == 0.0f, "v1 model metadata");
        free_network(d);
    }
    printf("  dropout misc checked\n");
    free_network(a); free_network(b);
}

static void dropout_xor(void) {
    float x[] = {0,0, 0,1, 1,0, 1,1}, y[] = {0,1,1,0};
    spingalett_set_compute_mode(COMPUTE_OPENBLAS);
    spingalett_seed(7);
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_MSE);
    layer(net, 2);
    layer(net, 64, ACT_TANH, WEIGHT_INITIALIZATION_XAVIER, 0.2f);
    layer(net, 1, ACT_SIGMOID, WEIGHT_INITIALIZATION_XAVIER);
    train(.net = net, .inputs = x, .targets = y, .sample_count = 4, .epochs = 3000, .learning_rate = 0.01f,
          .optimizer_type = OPTIMIZER_ADAM, .training_strategy = STRATEGY_FULL_BATCH);
    int ok = 0;
    for (int i = 0; i < 4; i++) ok += fabsf(forward(net, &x[i * 2])[0] - y[i]) < 0.2f;
    printf("  xor with dropout: %d/4\n", ok);
    CHECK(ok == 4, "xor with dropout");
    free_network(net);
}

/* ---------------- initialization, optimizer formulas, shuffling ---------------- */
static double weight_std(WeightInitialization wi, uint32_t in, uint32_t out) {
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_MSE);
    layer(net, in);
    layer(net, out, ACT_TANH, wi);
    double s = 0, s2 = 0; uint64_t n = net->total_weights;
    for (uint64_t i = 0; i < n; i++) { s += net->weights[i]; s2 += (double)net->weights[i] * net->weights[i]; }
    free_network(net);
    return sqrt(s2 / n - (s / n) * (s / n));
}

static void initialization(void) {
    spingalett_seed(8);
    struct { WeightInitialization wi; double expected; const char *name; } cases[] = {
        {WEIGHT_INITIALIZATION_XAVIER, sqrt(2.0 / (400 + 600)), "xavier/glorot"},
        {WEIGHT_INITIALIZATION_HE,     sqrt(2.0 / 400),         "he"},
        {WEIGHT_INITIALIZATION_LECUN,  sqrt(1.0 / 400),         "lecun"},
        {WEIGHT_INITIALIZATION_RANDOM, sqrt(1.0 / 3.0),         "uniform[-1,1]"},
    };
    for (int c = 0; c < 4; c++) {
        double sd = weight_std(cases[c].wi, 400, 600);
        printf("  init %-14s std %.5f (expected %.5f)\n", cases[c].name, sd, cases[c].expected);
        CHECK(fabs(sd / cases[c].expected - 1.0) < 0.01, "init %s std %.5f", cases[c].name, sd);
    }
}

/* First step from zero moments: Adam moves by lr * g / (|g| + eps), RMSProp by
   lr * g / (sqrt(1 - beta2) * |g| + eps). The gradient comes from an SGD step with lr = 1024
   (a power of two, so dividing it out is exact and small gradients keep their precision). */
static void optimizer_first_step(OptimizerType opt) {
    L ls[] = {{5, ACT_NONE}, {7, ACT_TANH}, {3, ACT_SIGMOID}};
    float *x, *y; lcg_state = 404;
    make_data(4, 5, 3, LOSS_MSE, ACT_SIGMOID, &x, &y);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    NeuralNetwork *g = build(LOSS_MSE, ls, 3, NULL, NULL);
    for (uint64_t i = 0; i < g->total_weights; i++) g->weights[i] *= (i % 7 == 0) ? 1e-4f : 1.0f;  /* include tiny gradients */
    NeuralNetwork *a = build(LOSS_MSE, ls, 3, g->weights, g->biases);
    uint64_t nw = g->total_weights;
    float *w0 = malloc(nw * 4); memcpy(w0, g->weights, nw * 4);
    train(.net = g, .inputs = x, .targets = y, .sample_count = 4, .epochs = 1, .learning_rate = 1024.0f,
          .optimizer_type = OPTIMIZER_SGD, .training_strategy = STRATEGY_FULL_BATCH);
    const float lr = 0.01f, eps = 1e-8f;
    train(.net = a, .inputs = x, .targets = y, .sample_count = 4, .epochs = 1, .learning_rate = lr,
          .optimizer_type = opt, .training_strategy = STRATEGY_FULL_BATCH);
    double worst = 0, tiny_ratio = 0; int tiny = 0;
    for (uint64_t i = 0; i < nw; i++) {
        double grad = ((double)w0[i] - g->weights[i]) / 1024.0;
        double denom = (opt == OPTIMIZER_RMSPROP ? sqrt(1.0 - 0.999) : 1.0) * fabs(grad) + eps;
        double expect = lr * grad / denom, got = (double)w0[i] - a->weights[i];
        double err = fabs(got - expect) / fmax(fabs(expect), 1e-7);
        if (err > worst) worst = err;
        if (fabs(grad) < 1e-4 && fabs(grad) > 1e-7) { tiny++; tiny_ratio = fmax(tiny_ratio, fabs(got) / (lr * (opt == OPTIMIZER_RMSPROP ? 31.6 : 1.0))); }
    }
    printf("  first %s step: max rel. error %.2e (%d tiny gradients, max |step|/expected %.3f)\n",
           opt == OPTIMIZER_RMSPROP ? "RMSProp" : opt == OPTIMIZER_ADAM ? "Adam" : "AdamW", worst, tiny, tiny_ratio);
    CHECK(worst < 2e-4, "first-step formula opt=%d (rel. error %.3e)", opt, worst);
    free(w0); free(x); free(y); free_network(g); free_network(a);
}

static void shuffling(void) {
    L ls[] = {{5, ACT_NONE}, {9, ACT_TANH}, {3, ACT_SIGMOID}};
    float *x, *y; lcg_state = 55;
    make_data(12, 5, 3, LOSS_MSE, ACT_SIGMOID, &x, &y);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    NeuralNetwork *ref = build(LOSS_MSE, ls, 3, NULL, NULL);
    TrainingStrategy strats[] = {STRATEGY_SAMPLE, STRATEGY_SMALL_BATCH};
    for (int s = 0; s < 2; s++) {
        NeuralNetwork *n[3];
        for (int k = 0; k < 3; k++) {
            n[k] = build(LOSS_MSE, ls, 3, ref->weights, ref->biases);
            spingalett_seed(k == 2 ? 2 : 1);
            train(.net = n[k], .inputs = x, .targets = y, .sample_count = 12, .epochs = 3, .batch_size = 4,
                  .learning_rate = 0.05f, .optimizer_type = OPTIMIZER_SGD, .training_strategy = strats[s]);
        }
        NeuralNetwork *fixed = build(LOSS_MSE, ls, 3, ref->weights, ref->biases);
        train(.net = fixed, .inputs = x, .targets = y, .sample_count = 12, .epochs = 3, .batch_size = 4,
              .learning_rate = 0.05f, .optimizer_type = OPTIMIZER_SGD, .training_strategy = strats[s], .do_not_shuffle = true);
        uint64_t nw = ref->total_weights;
        float same = max_abs_diff(n[0]->weights, n[1]->weights, nw);
        float other_seed = max_abs_diff(n[0]->weights, n[2]->weights, nw);
        float vs_fixed = max_abs_diff(n[0]->weights, fixed->weights, nw);
        printf("  shuffle strat=%d: same seed %.1e, other seed %.1e, vs fixed order %.1e\n", strats[s], same, other_seed, vs_fixed);
        CHECK(same == 0.0f && other_seed > 1e-5f && vs_fixed > 1e-5f, "shuffling strat=%d", strats[s]);
        for (int k = 0; k < 3; k++) free_network(n[k]);
        free_network(fixed);
    }
    free_network(ref); free(x); free(y);
}

/* ---------------- batched inference ---------------- */
static void predict_matches_forward(ComputeMode mode) {
    const uint32_t N = 4500;            /* > one internal chunk */
    float *x, *y; lcg_state = 1234;
    make_data(N, 9, 4, LOSS_CROSS_ENTROPY, ACT_SOFTMAX, &x, &y);
    spingalett_set_compute_mode(mode);
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(net, 9);
    layer(net, 37, ACT_RELU, WEIGHT_INITIALIZATION_HE, 0.5f);     /* dropout must not apply */
    layer(net, 21, ACT_TANH, WEIGHT_INITIALIZATION_XAVIER);
    layer(net, 4, ACT_SOFTMAX, WEIGHT_INITIALIZATION_XAVIER);
    for (uint64_t i = 0; i < net->total_biases; i++) net->biases[i] = frand() - 0.5f;
    float *out = malloc((size_t)N * 4 * sizeof(float));
    for (size_t i = 0; i < (size_t)N * 4; i++) out[i] = NAN;
    spingalett_clear_error();
    bool ok = predict(.net = net, .inputs = x, .sample_count = N, .outputs = out);
    float worst = 0;
    for (uint32_t s = 0; s < N; s++) {
        const float *o = forward(net, x + (size_t)s * 9);
        for (int k = 0; k < 4; k++) {
            float d = fabsf(o[k] - out[(size_t)s * 4 + k]);
            if (!(d <= worst)) worst = d;
        }
    }
    printf("  predict == forward, mode=%d, %u samples: max |diff| %.2e\n", mode, N, worst);
    CHECK(ok && worst < 1e-5f, "predict mode=%d diff %.3e", mode, worst);
    CHECK(predict(.net = net, .inputs = x, .sample_count = 0, .outputs = out), "empty predict");
    CHECK(!predict(.net = net, .inputs = NULL, .sample_count = 3, .outputs = out) &&
          spingalett_last_error_code() == SPINGALETT_ERR_INVALID, "predict without inputs must fail");
    free(out); free(x); free(y); free_network(net);
}

/* ---------------- generator mode ---------------- */
typedef struct {
    const float *x, *y;
    uint32_t n, in, out, pos;
    uint32_t calls, requested[64];
    int mode;                       /* 0 = dataset once per epoch (then return 0), 1 = endless,
                                       2 = overflow, 3 = empty, 4 = restart on every call (full batch) */
} GenState;

static uint32_t serve(float *inputs, float *targets, uint32_t requested, void *ud) {
    GenState *g = ud;
    if (g->calls < 64) g->requested[g->calls] = requested;
    g->calls++;
    if (g->mode == 3) return 0;
    if (g->mode == 2) return requested + 1;
    if (g->mode == 4) g->pos = 0;
    uint32_t count = 0;
    while (count < requested) {
        if (g->pos == g->n) {
            if (g->mode == 0) break;
            g->pos = 0;
        }
        memcpy(inputs + (size_t)count * g->in, g->x + (size_t)g->pos * g->in, g->in * sizeof(float));
        memcpy(targets + (size_t)count * g->out, g->y + (size_t)g->pos * g->out, g->out * sizeof(float));
        g->pos++; count++;
    }
    if (g->mode == 0 && count == 0) g->pos = 0;   /* epoch end reported; start over next time */
    return count;
}

static NeuralNetwork *gen_net(const float *w, const float *b) {
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(net, 6);
    layer(net, 20, ACT_TANH, WEIGHT_INITIALIZATION_XAVIER, 0.3f);
    layer(net, 11, ACT_RELU, WEIGHT_INITIALIZATION_HE);
    layer(net, 3, ACT_SOFTMAX, WEIGHT_INITIALIZATION_XAVIER);
    if (w) { memcpy(net->weights, w, net->total_weights * 4); memcpy(net->biases, b, net->total_biases * 4); }
    else for (uint64_t i = 0; i < net->total_weights; i++) net->weights[i] = frand() - 0.5f;
    return net;
}

static void generator_mode(ComputeMode mode) {
    const uint32_t N = 22, B = 8;
    float *x, *y; lcg_state = 314;
    make_data(N, 6, 3, LOSS_CROSS_ENTROPY, ACT_SOFTMAX, &x, &y);
    spingalett_set_compute_mode(mode);
    NeuralNetwork *ref = gen_net(NULL, NULL);

    /* full batch and per-sample: generator == array mode (dropout masks included) */
    TrainingStrategy strats[] = {STRATEGY_FULL_BATCH, STRATEGY_SAMPLE};
    for (int s = 0; s < 2; s++) {
        NeuralNetwork *a = gen_net(ref->weights, ref->biases), *b = gen_net(ref->weights, ref->biases);
        GenState g = {.x = x, .y = y, .n = N, .in = 6, .out = 3, .mode = strats[s] == STRATEGY_FULL_BATCH ? 4 : 0};
        spingalett_seed(5);
        train(.net = a, .inputs = x, .targets = y, .sample_count = N, .epochs = 7, .learning_rate = 0.01f,
              .optimizer_type = OPTIMIZER_ADAM, .training_strategy = strats[s], .do_not_shuffle = true);
        spingalett_seed(5);
        train(.net = b, .training_mode = MODE_GENERATOR_FUNCTION, .generator = serve, .generator_data = &g,
              .sample_count = strats[s] == STRATEGY_FULL_BATCH ? N : 0, .epochs = 7, .learning_rate = 0.01f,
              .optimizer_type = OPTIMIZER_ADAM, .training_strategy = strats[s]);
        float d = max_abs_diff(a->weights, b->weights, a->total_weights);
        printf("  generator == array, mode=%d strat=%d: diff %.2e, steps %llu/%llu, calls %u\n", mode, strats[s], d,
               (unsigned long long)a->time_step, (unsigned long long)b->time_step, g.calls);
        CHECK(d < 1e-6f && a->time_step == b->time_step, "generator vs array mode=%d strat=%d", mode, strats[s]);
        free_network(a); free_network(b);
    }

    /* mini-batches: E epochs from the generator == one full-batch train() per consecutive chunk */
    {
        NeuralNetwork *a = gen_net(ref->weights, ref->biases), *b = gen_net(ref->weights, ref->biases);
        GenState g = {.x = x, .y = y, .n = N, .in = 6, .out = 3};
        spingalett_seed(6);
        train(.net = b, .training_mode = MODE_GENERATOR_FUNCTION, .generator = serve, .generator_data = &g,
              .epochs = 3, .batch_size = B, .learning_rate = 0.01f, .optimizer_type = OPTIMIZER_ADAMW,
              .weight_decay = 0.01f, .training_strategy = STRATEGY_SMALL_BATCH);
        /* the dropout seed is drawn once per train() call, so compare without dropout */
        a->dropout_rates[1] = 0.0f; NeuralNetwork *c = gen_net(ref->weights, ref->biases); c->dropout_rates[1] = 0.0f;
        GenState g2 = {.x = x, .y = y, .n = N, .in = 6, .out = 3};
        train(.net = c, .training_mode = MODE_GENERATOR_FUNCTION, .generator = serve, .generator_data = &g2,
              .epochs = 3, .batch_size = B, .learning_rate = 0.01f, .optimizer_type = OPTIMIZER_ADAMW,
              .weight_decay = 0.01f, .training_strategy = STRATEGY_SMALL_BATCH);
        for (int e = 0; e < 3; e++)
            for (uint32_t s = 0; s < N; s += B) {
                uint32_t cnt = N - s < B ? N - s : B;
                train(.net = a, .inputs = x + (size_t)s * 6, .targets = y + (size_t)s * 3, .sample_count = cnt, .epochs = 1,
                      .learning_rate = 0.01f, .optimizer_type = OPTIMIZER_ADAMW, .weight_decay = 0.01f,
                      .training_strategy = STRATEGY_FULL_BATCH);
            }
        float d = max_abs_diff(a->weights, c->weights, a->total_weights);
        printf("  generator mini-batches == chunked full batches, mode=%d: diff %.2e, steps %llu (expected 9), requests %u,%u,%u,%u\n",
               mode, d, (unsigned long long)c->time_step, g2.requested[0], g2.requested[1], g2.requested[2], g2.requested[3]);
        CHECK(d < 1e-6f && c->time_step == 9 && b->time_step == 9, "generator mini-batch mode=%d", mode);
        CHECK(g2.requested[0] == B && g2.requested[3] == B, "mini-batch requests");
        free_network(a); free_network(b); free_network(c);
    }

    /* endless generator, epoch length capped by sample_count: requests 4, 4, 2 per epoch */
    {
        NeuralNetwork *a = gen_net(ref->weights, ref->biases);
        GenState g = {.x = x, .y = y, .n = N, .in = 6, .out = 3, .mode = 1};
        train(.net = a, .training_mode = MODE_GENERATOR_FUNCTION, .generator = serve, .generator_data = &g,
              .sample_count = 10, .epochs = 2, .batch_size = 4, .training_strategy = STRATEGY_SMALL_BATCH);
        CHECK(g.calls == 6 && g.requested[0] == 4 && g.requested[1] == 4 && g.requested[2] == 2 && g.requested[5] == 2 && a->time_step == 6,
              "epoch cap: calls %u requests %u,%u,%u steps %llu", g.calls, g.requested[0], g.requested[1], g.requested[2], (unsigned long long)a->time_step);
        free_network(a);
    }

    /* misbehaving generators */
    {
        NeuralNetwork *a = gen_net(ref->weights, ref->biases);
        GenState over = {.mode = 2}, empty = {.mode = 3};
        spingalett_clear_error();
        train(.net = a, .training_mode = MODE_GENERATOR_FUNCTION, .generator = serve, .generator_data = &over,
              .epochs = 5, .batch_size = 4, .training_strategy = STRATEGY_SMALL_BATCH);
        CHECK(spingalett_last_error_code() == SPINGALETT_ERR_INVALID && a->time_step == 0 && over.calls == 1, "overflowing generator must stop training");
        train(.net = a, .training_mode = MODE_GENERATOR_FUNCTION, .generator = serve, .generator_data = &empty,
              .epochs = 5, .training_strategy = STRATEGY_SMALL_BATCH);
        /* a 0 in answer to an epoch's first request is retried once, then training stops */
        CHECK(a->time_step == 0 && empty.calls == 2, "empty generator must stop training (calls %u)", empty.calls);
        spingalett_clear_error();
        train(.net = a, .training_mode = MODE_GENERATOR_FUNCTION, .generator = serve, .generator_data = &empty,
              .epochs = 1, .training_strategy = STRATEGY_FULL_BATCH);
        CHECK(spingalett_last_error_code() == SPINGALETT_ERR_INVALID, "full batch without sample_count must be rejected");
        free_network(a);
    }
    free_network(ref); free(x); free(y);
}

static void xor_converges(OptimizerType opt, TrainingStrategy strat, ComputeMode mode) {
    float x[] = {0,0, 0,1, 1,0, 1,1}, y[] = {0,1,1,0};
    spingalett_set_compute_mode(mode);
    spingalett_seed(2026);
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_MSE);
    layer(net, 2);
    layer(net, 8, ACT_TANH, WEIGHT_INITIALIZATION_XAVIER);
    layer(net, 1, ACT_SIGMOID, WEIGHT_INITIALIZATION_XAVIER);
    float lr = (opt == OPTIMIZER_SGD || opt == OPTIMIZER_MOMENTUM) ? 0.5f : 0.02f;
    train(.net = net, .inputs = x, .targets = y, .sample_count = 4, .epochs = 4000, .learning_rate = lr,
          .optimizer_type = opt, .training_strategy = strat, .batch_size = 2);
    int ok = 0;
    for (int i = 0; i < 4; i++) { float *o = forward(net, &x[i * 2]); ok += fabsf(o[0] - y[i]) < 0.2f; }
    printf("  xor opt=%d strat=%d mode=%d: %d/4\n", opt, strat, mode, ok);
    CHECK(ok == 4, "xor opt=%d strat=%d mode=%d", opt, strat, mode);
    free_network(net);
}


/* ---------------- evaluate, validation, early stopping ---------------- */

/* evaluate() against a direct computation: the loss train() reports and argmax accuracy. */
static void evaluate_matches_manual(void) {
    lcg_state = 4242;
    L ls[] = {{5, ACT_NONE}, {9, ACT_TANH}, {4, ACT_SOFTMAX}};
    float *x, *y;
    make_data(37, 5, 4, LOSS_CROSS_ENTROPY, ACT_SOFTMAX, &x, &y);
    NeuralNetwork *net = build(LOSS_CROSS_ENTROPY, ls, 3, NULL, NULL);
    double loss = 0; uint32_t correct = 0;
    for (uint32_t i = 0; i < 37; i++) {
        float *o = forward(.net = net, .input = x + i * 5);
        int bo = 0, bt = 0;
        for (int k = 0; k < 4; k++) { loss -= y[i * 4 + k] * log(fmax(o[k], 1e-9)); if (o[k] > o[bo]) bo = k; if (y[i * 4 + k] > y[i * 4 + bt]) bt = k; }
        correct += bo == bt;
    }
    EvalMetrics m = evaluate(.net = net, .inputs = x, .targets = y, .sample_count = 37);
    printf("  evaluate: loss %.5f (manual %.5f), accuracy %.4f (manual %.4f)\n", (double)m.loss, loss / 37, (double)m.accuracy, correct / 37.0);
    CHECK(fabs(m.loss - loss / 37) < 1e-4 && fabs(m.accuracy - correct / 37.0) < 1e-6, "evaluate metrics differ from the manual computation");
    EvalMetrics bad = evaluate(.net = net, .inputs = x, .targets = NULL, .sample_count = 37);
    CHECK(isnan(bad.loss) && spingalett_last_error_code() == SPINGALETT_ERR_INVALID, "evaluate without targets should fail");

    /* single-output networks: correct when output and target are on the same side of 0.5 */
    L lb[] = {{2, ACT_NONE}, {1, ACT_SIGMOID}};
    NeuralNetwork *b = build(LOSS_CROSS_ENTROPY, lb, 2, (const float[]){4.0f, 0.0f}, (const float[]){-2.0f});
    float bx[] = {0, 0, 1, 0, 0, 0, 1, 0}, by[] = {0, 1, 1, 0};   /* outputs: <0.5, >0.5, <0.5, >0.5 */
    EvalMetrics mb = evaluate(.net = b, .inputs = bx, .targets = by, .sample_count = 4);
    CHECK(fabsf(mb.accuracy - 0.5f) < 1e-6f, "binary accuracy %.3f, expected 0.5", (double)mb.accuracy);
    free_network(b); free_network(net); free(x); free(y);
}

typedef struct {
    float *weights;             /* parameters after every epoch, [epoch][total_weights] */
    size_t calls, last_epoch, best_epoch;
    bool improved_seen, data_ok;
} History;

static bool record_epoch(NeuralNetwork *net, const TrainProgress *p, void *user) {
    History *h = user;
    h->calls++;
    h->data_ok = h->data_ok && p->epoch == h->last_epoch + 1 && p->has_validation;
    h->last_epoch = p->epoch;
    if (p->improved) { h->improved_seen = true; h->data_ok = h->data_ok && p->best_epoch == p->epoch; }
    h->best_epoch = p->best_epoch;
    memcpy(h->weights + (p->epoch - 1) * net->total_weights, net->weights, net->total_weights * sizeof(float));
    return false;
}

/* Validation targets are the complement of the training targets, so validation loss rises
   while training loss falls: the best epoch is early, early stopping ends the run patience
   epochs later and restore_best_weights brings back exactly that epoch's weights. */
static void early_stopping(ComputeMode mode, TrainingStrategy strat) {
    spingalett_set_compute_mode(mode);
    lcg_state = 99;
    L ls[] = {{3, ACT_NONE}, {8, ACT_TANH}, {1, ACT_SIGMOID}};
    float *x, *y;
    make_data(24, 3, 1, LOSS_CROSS_ENTROPY, ACT_SIGMOID, &x, &y);
    float *yv = malloc(24 * sizeof(float));
    for (int i = 0; i < 24; i++) yv[i] = 1.0f - y[i];
    NeuralNetwork *net = build(LOSS_CROSS_ENTROPY, ls, 3, NULL, NULL);
    History h = {.weights = calloc(50 * net->total_weights, sizeof(float)), .data_ok = true};
    TrainReport r = train(.net = net, .inputs = x, .targets = y, .sample_count = 24, .epochs = 50,
                          .training_strategy = strat, .batch_size = 8, .optimizer_type = OPTIMIZER_ADAM,
                          .learning_rate = 0.05f, .val_inputs = x, .val_targets = yv, .val_count = 24,
                          .early_stopping_patience = 4, .restore_best_weights = true,
                          .callback = record_epoch, .callback_data = &h);
    float d = r.best_epoch ? max_abs_diff(net->weights, h.weights + (r.best_epoch - 1) * net->total_weights, net->total_weights) : 1.0f;
    printf("  early stopping mode=%d strat=%d: status=%d best=%zu run=%zu restored=%d diff=%.1e\n",
           mode, strat, r.status, r.best_epoch, r.epochs_run, r.restored_best, (double)d);
    CHECK(r.status == TRAIN_EARLY_STOPPED && r.epochs_run == r.best_epoch + 4 && r.restored_best && d == 0.0f,
          "early stopping / restore_best_weights (mode %d strat %d)", mode, strat);
    CHECK(h.calls == r.epochs_run && h.data_ok && h.improved_seen && h.best_epoch == r.best_epoch && r.monitor == MONITOR_VAL_LOSS,
          "callback progress fields (mode %d strat %d)", mode, strat);
    EvalMetrics v = evaluate(.net = net, .inputs = x, .targets = yv, .sample_count = 24);
    CHECK(fabsf(v.loss - r.best_value) < 1e-5f, "best value %.6f != validation loss after restore %.6f", (double)r.best_value, (double)v.loss);
    free(h.weights); free(yv); free(x); free(y);
    free_network(net);
}

static bool stop_at_3(NeuralNetwork *n, const TrainProgress *p, void *u) { (void)n; (void)u; return p->epoch == 3; }
static bool poison_at_2(NeuralNetwork *n, const TrainProgress *p, void *u) { (void)u; if (p->epoch == 2) n->weights[0] = NAN; return false; }

static void train_report_misc(void) {
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    lcg_state = 5;
    L ls[] = {{3, ACT_NONE}, {6, ACT_TANH}, {3, ACT_SOFTMAX}};
    float *x, *y;
    make_data(30, 3, 3, LOSS_CROSS_ENTROPY, ACT_SOFTMAX, &x, &y);
    NeuralNetwork *net = build(LOSS_CROSS_ENTROPY, ls, 3, NULL, NULL);

    TrainReport r = train(.net = net, .inputs = x, .targets = y, .sample_count = 30, .epochs = 10,
                          .optimizer_type = OPTIMIZER_ADAM, .learning_rate = 0.02f);
    CHECK(r.status == TRAIN_COMPLETED && r.epochs_run == 10 && r.best_epoch == 0 && !r.has_validation && isfinite(r.train_loss),
          "plain train() report (status %d, epochs %zu)", r.status, r.epochs_run);

    r = train(.net = net, .inputs = x, .targets = y, .sample_count = 30, .epochs = 10, .callback = stop_at_3);
    CHECK(r.status == TRAIN_INTERRUPTED && r.epochs_run == 3, "callback interruption (status %d, epochs %zu)", r.status, r.epochs_run);

    r = train(.net = net, .inputs = x, .targets = y, .sample_count = 30, .epochs = 10, .monitor = MONITOR_VAL_ACCURACY);
    CHECK(r.status == TRAIN_FAILED && spingalett_last_error_code() == SPINGALETT_ERR_INVALID, "validation monitor without validation data");

    /* training-loss monitoring with a min_delta no change can meet: only epoch 1 counts */
    r = train(.net = net, .inputs = x, .targets = y, .sample_count = 30, .epochs = 20, .optimizer_type = OPTIMIZER_ADAM,
              .early_stopping_patience = 3, .early_stopping_min_delta = 1e9f);
    CHECK(r.status == TRAIN_EARLY_STOPPED && r.best_epoch == 1 && r.epochs_run == 4 && r.monitor == MONITOR_TRAIN_LOSS && !r.restored_best,
          "min_delta (status %d best %zu run %zu)", r.status, r.best_epoch, r.epochs_run);

    /* validation accuracy is maximized */
    r = train(.net = net, .inputs = x, .targets = y, .sample_count = 30, .epochs = 15, .optimizer_type = OPTIMIZER_ADAM,
              .learning_rate = 0.02f, .val_inputs = x, .val_targets = y, .val_count = 30, .monitor = MONITOR_VAL_ACCURACY);
    EvalMetrics m = evaluate(.net = net, .inputs = x, .targets = y, .sample_count = 30);
    CHECK(r.status == TRAIN_COMPLETED && r.best_value >= m.accuracy - 1e-6f && fabsf(r.validation.accuracy - m.accuracy) < 1e-6f,
          "val accuracy monitoring (best %.3f, final %.3f)", (double)r.best_value, (double)m.accuracy);

    /* NaN parameters are reported as divergence, and the best (finite) epoch is restored */
    r = train(.net = net, .inputs = x, .targets = y, .sample_count = 30, .epochs = 30, .optimizer_type = OPTIMIZER_SGD,
              .nan_check_interval = 1, .restore_best_weights = true, .callback = poison_at_2);
    bool finite = true;
    for (uint64_t i = 0; i < net->total_weights; i++) finite = finite && isfinite(net->weights[i]);
    printf("  divergence: status=%d epochs=%zu best=%zu restored=%d finite=%d\n", r.status, r.epochs_run, r.best_epoch, r.restored_best, finite);
    CHECK(r.status == TRAIN_DIVERGED && r.epochs_run == 3 && r.best_epoch == 2 && r.restored_best && finite, "divergence report");
    free_network(net); free(x); free(y);
}

/* ---------------- low-level training API ---------------- */

/* Gradient of the mean loss from forward + backward, against central differences. */
static void trainer_gradcheck(const char *name, LossFunction loss, const L *ls, int nl, ComputeMode mode) {
    uint32_t N = 6; lcg_state = 31337;
    float *x, *y;
    make_data(N, ls[0].n, ls[nl - 1].n, loss, ls[nl - 1].act, &x, &y);
    spingalett_set_compute_mode(mode);
    NeuralNetwork *net = build(loss, ls, nl, NULL, NULL);
    for (uint64_t i = 0; i < net->total_biases; i++) net->biases[i] = frand() * 0.2f - 0.1f;
    SpingalettTrainer *tr = spingalett_trainer_new(net, N);
    spingalett_trainer_forward(tr, x, 4);                    /* two backward passes accumulate */
    spingalett_trainer_backward(tr, y);
    spingalett_trainer_forward(tr, x + 4 * ls[0].n, N - 4);
    spingalett_trainer_backward(tr, y + 4 * ls[nl - 1].n);
    uint64_t nw = net->total_weights, nb = net->total_biases;
    int bad = 0; double maxrel = 0;
    for (uint64_t i = 0; i < nw + nb; i++) {
        float *p = i < nw ? &net->weights[i] : &net->biases[i - nw];
        double ga = (i < nw ? net->grad_weights[i] : net->grad_biases[i - nw]) / N;
        float orig = *p, h = 1e-3f, h2 = 2.5e-4f;
        *p = orig + h; double lp = dataset_loss(net, x, y, N);
        *p = orig - h; double lm = dataset_loss(net, x, y, N);
        *p = orig + h2; double lp2 = dataset_loss(net, x, y, N);
        *p = orig - h2; double lm2 = dataset_loss(net, x, y, N);
        *p = orig;
        double gn = (lp - lm) / (2.0 * h), gn2 = (lp2 - lm2) / (2.0 * h2);
        if (fabs(gn - gn2) > 0.1 * (fabs(gn) + fabs(gn2)) + 3e-4) continue;   /* perturbation crosses a kink */
        double rel = fabs(gn - ga) / fmax(1e-2, fabs(gn) + fabs(ga));
        if (rel > maxrel) maxrel = rel;
        if (rel > 3e-2) bad++;
    }
    printf("  trainer gradcheck %-22s mode=%d  maxrel=%.2e  outliers=%d\n", name, mode, maxrel, bad);
    CHECK(bad == 0, "trainer gradcheck %s mode %d", name, mode);
    spingalett_trainer_free(tr);
    free_network(net); free(x); free(y);
}

/* dL/d(output) of the built-in losses, fed through the custom-loss path, gives the same gradient. */
static void trainer_custom_grads(LossFunction loss, ActivationFunction out_act) {
    uint32_t N = 7; lcg_state = 8080;
    L ls[] = {{4, ACT_NONE}, {9, ACT_RELU}, {3, out_act}};
    float *x, *y;
    make_data(N, 4, 3, loss, out_act, &x, &y);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    NeuralNetwork *net = build(loss, ls, 3, NULL, NULL);
    SpingalettTrainer *tr = spingalett_trainer_new(net, N);
    spingalett_trainer_forward(tr, x, N);
    spingalett_trainer_backward(tr, y);
    float *g_builtin = malloc(net->total_weights * sizeof(float));
    memcpy(g_builtin, net->grad_weights, net->total_weights * sizeof(float));
    spingalett_trainer_zero_grad(tr);

    const float *out = spingalett_trainer_forward(tr, x, N);
    float *dy = malloc(N * 3 * sizeof(float));
    for (uint32_t i = 0; i < N * 3; i++) {
        float o = out[i], t = y[i];
        dy[i] = loss == LOSS_MSE ? o - t : out_act == ACT_SIGMOID ? (o - t) / (o * (1 - o)) : -t / o;
    }
    bool ok = spingalett_trainer_backward_output_grads(tr, dy);
    float d = max_abs_diff(g_builtin, net->grad_weights, net->total_weights);
    printf("  custom output gradients loss=%d act=%d: max diff %.2e\n", loss, out_act, (double)d);
    CHECK(ok && d < 1e-4f, "custom output gradients (loss %d act %d): %.3e", loss, out_act, (double)d);
    free(dy); free(g_builtin);
    spingalett_trainer_free(tr);
    free_network(net); free(x); free(y);
}

/* A train_on_batch loop over the same batches equals train() with unshuffled mini-batches,
   dropout included; a step split into two backward passes equals one over the whole batch. */
static void trainer_matches_train(ComputeMode mode) {
    uint32_t N = 40; lcg_state = 1212;
    float *x, *y;
    make_data(N, 6, 4, LOSS_CROSS_ENTROPY, ACT_SOFTMAX, &x, &y);
    spingalett_set_compute_mode(mode);
    NeuralNetwork *a = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(a, 6);
    layer(a, 32, ACT_RELU, WEIGHT_INITIALIZATION_HE, .dropout_rate = 0.3f);
    layer(a, 4, ACT_SOFTMAX, WEIGHT_INITIALIZATION_XAVIER);
    NeuralNetwork *b = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(b, 6);
    layer(b, 32, ACT_RELU, WEIGHT_INITIALIZATION_HE, .dropout_rate = 0.3f);
    layer(b, 4, ACT_SOFTMAX, WEIGHT_INITIALIZATION_XAVIER);
    memcpy(b->weights, a->weights, a->total_weights * sizeof(float));
    memcpy(b->biases, a->biases, a->total_biases * sizeof(float));
    NeuralNetwork *c = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(c, 6);
    layer(c, 32, ACT_RELU, WEIGHT_INITIALIZATION_HE, .dropout_rate = 0.3f);
    layer(c, 4, ACT_SOFTMAX, WEIGHT_INITIALIZATION_XAVIER);
    memcpy(c->weights, a->weights, a->total_weights * sizeof(float));
    memcpy(c->biases, a->biases, a->total_biases * sizeof(float));

    spingalett_seed(77);
    train(.net = a, .inputs = x, .targets = y, .sample_count = N, .epochs = 3, .training_strategy = STRATEGY_SMALL_BATCH,
          .batch_size = 8, .do_not_shuffle = true, .optimizer_type = OPTIMIZER_ADAM, .learning_rate = 0.01f, .max_grad_norm = 2.0f);

    OptimizerArgs opt = {.type = OPTIMIZER_ADAM, .learning_rate = 0.01f, .max_grad_norm = 2.0f};
    spingalett_seed(77);
    SpingalettTrainer *tb = spingalett_trainer_new(b, 8);
    float loss = 0;
    for (int e = 0; e < 3; e++)
        for (uint32_t s = 0; s < N; s += 8)
            loss = spingalett_train_on_batch(tb, x + s * 6, y + s * 4, 8, &opt);

    spingalett_seed(77);
    SpingalettTrainer *tc = spingalett_trainer_new(c, 5);
    for (int e = 0; e < 3; e++)
        for (uint32_t s = 0; s < N; s += 8) {
            spingalett_trainer_forward(tc, x + s * 6, 5);
            spingalett_trainer_backward(tc, y + s * 4);
            spingalett_trainer_forward(tc, x + (s + 5) * 6, 3);
            spingalett_trainer_backward(tc, y + (s + 5) * 4);
            spingalett_trainer_step(tc, &opt);
        }
    float dab = max_abs_diff(a->weights, b->weights, a->total_weights);
    float dbc = max_abs_diff(b->weights, c->weights, b->total_weights);
    printf("  trainer vs train() mode=%d: |a-b| %.2e, split step |b-c| %.2e, last loss %.4f\n", mode, (double)dab, (double)dbc, (double)loss);
    CHECK(dab < 1e-5f && dbc < 1e-5f && a->time_step == b->time_step && isfinite(loss),
          "trainer API diverges from train() (mode %d): %.3e / %.3e", mode, (double)dab, (double)dbc);

    /* misuse is reported, not crashed on */
    CHECK(isnan(spingalett_trainer_backward(tb, y)), "backward without a forward pass should fail");
    CHECK(!spingalett_trainer_step(tb, &opt), "step without gradients should fail");
    CHECK(spingalett_trainer_forward(tb, x, 9) == NULL, "forward above max_batch should fail");
    CHECK(spingalett_trainer_new(a, 0) == NULL, "max_batch 0 should fail");
    spingalett_trainer_free(tb); spingalett_trainer_free(tc);
    free_network(a); free_network(b); free_network(c); free(x); free(y);
}

/* ---------------- data sets ---------------- */

static void write_be32(FILE *f, uint32_t v) {
    unsigned char b[4] = {(unsigned char)(v >> 24), (unsigned char)(v >> 16), (unsigned char)(v >> 8), (unsigned char)v};
    fwrite(b, 1, 4, f);
}

/* CIFAR-10 and CIFAR-100 records: labels, then the image plane by plane, read channels last. */
static void cifar_files(void) {
    const char *paths[] = {"spingalett_test_cifar_a.bin", "spingalett_test_cifar_b.bin"};
    for (int v = 0; v < 2; v++) {               /* CIFAR-10, CIFAR-100 */
        size_t labels = v ? 2 : 1, record = labels + 3072;
        uint8_t *rec = malloc(record);
        for (int file = 0; file < 2; file++) {
            FILE *f = fopen(paths[file], "wb");
            for (int r = 0; r < 2 + file; r++) {          /* 2 records, then 3 */
                rec[0] = (uint8_t)(r + file * 2);         /* CIFAR-10 label, or CIFAR-100 coarse */
                if (v) rec[1] = (uint8_t)(50 + r + file);  /* fine */
                for (int i = 0; i < 3072; i++) rec[labels + i] = (uint8_t)((i * 7 + r * 13 + file) & 0xFF);
                fwrite(rec, 1, record, f);
            }
            fclose(f);
        }
        SpingalettDataset d;
        uint32_t classes = v ? 100 : 10;
        bool ok = spingalett_load_cifar(paths, 2, classes, &d);
        /* sample 3 is the second record of file b: pixel p channel c from plane c */
        uint32_t p = 33, c = 2;
        float want = (float)((((int)(c * 1024 + p) * 7 + 1 * 13 + 1) & 0xFF) / 255.0);
        uint32_t label = v ? 50 + 1 + 1 : 1 + 2;
        CHECK(ok && d.count == 5 && d.input_size == 3072 && d.target_size == classes &&
              d.inputs[3 * 3072 + p * 3 + c] == want && d.targets[3 * classes + label] == 1.0f,
              "cifar%d: records read channels last with their labels", v ? 100 : 10);
        if (ok) spingalett_dataset_free(&d);
        if (v) {
            ok = spingalett_load_cifar(paths, 2, 20, &d);
            CHECK(ok && d.target_size == 20 && d.targets[3 * 20 + 3] == 1.0f, "cifar100 coarse labels");
            if (ok) spingalett_dataset_free(&d);
        }
        free(rec);
    }
    FILE *f = fopen(paths[0], "wb");
    fputc(1, f);
    fclose(f);
    SpingalettDataset d;
    CHECK(!spingalett_load_cifar(paths, 1, 10, &d) && d.count == 0 && spingalett_last_error_code() == SPINGALETT_ERR_INVALID,
          "cifar: a truncated record is refused");
    CHECK(!spingalett_load_cifar(paths, 1, 7, &d), "cifar: num_classes must be 10, 20 or 100");
    remove(paths[0]);
    remove(paths[1]);
}

static void datasets(void) {
    /* IDX: 5 images of 2x3 bytes, labels 0..4 */
    FILE *f = fopen("spingalett_test_images.idx", "wb");
    write_be32(f, 0x00000803); write_be32(f, 5); write_be32(f, 2); write_be32(f, 3);
    for (int i = 0; i < 30; i++) fputc(i * 8, f);
    fclose(f);
    f = fopen("spingalett_test_labels.idx", "wb");
    write_be32(f, 0x00000801); write_be32(f, 5);
    for (int i = 0; i < 5; i++) fputc((i * 3) % 5, f);
    fclose(f);
    SpingalettDataset d, tail;
    bool ok = spingalett_load_idx("spingalett_test_images.idx", "spingalett_test_labels.idx", 0, &d);
    CHECK(ok && d.count == 5 && d.input_size == 6 && d.target_size == 5 && fabsf(d.inputs[29] - 232 / 255.0f) < 1e-7f &&
          d.targets[1 * 5 + 3] == 1.0f && d.targets[1 * 5 + 0] == 0.0f, "IDX reader");
    ok = ok && spingalett_dataset_split(&d, 2, &tail);
    CHECK(ok && d.count == 3 && tail.count == 2 && tail.input_size == 6 && tail.target_size == 5, "dataset split");
    CHECK(ok && fabsf(tail.inputs[0] - 144 / 255.0f) < 1e-7f && tail.targets[0 * 5 + 4] == 1.0f, "dataset split contents");
    spingalett_dataset_free(&tail);
    CHECK(!spingalett_load_idx("spingalett_test_images.idx", "spingalett_test_labels.idx", 3, &tail) &&
          spingalett_last_error_code() == SPINGALETT_ERR_INVALID && tail.count == 0, "IDX label beyond num_classes");
    spingalett_dataset_free(&d);
    remove("spingalett_test_images.idx");
    remove("spingalett_test_labels.idx");

    /* bytes are scaled to exactly v / 255.0f, as other frameworks do */
    f = fopen("spingalett_test_images.idx", "wb");
    write_be32(f, 0x00000802); write_be32(f, 256); write_be32(f, 1);
    for (int i = 0; i < 256; i++) fputc(i, f);
    fclose(f);
    f = fopen("spingalett_test_labels.idx", "wb");
    write_be32(f, 0x00000801); write_be32(f, 256);
    for (int i = 0; i < 256; i++) fputc(0, f);
    fclose(f);
    int inexact = 0;
    ok = spingalett_load_idx("spingalett_test_images.idx", "spingalett_test_labels.idx", 1, &d);
    for (int i = 0; ok && i < 256; i++) inexact += d.inputs[i] != (float)i / 255.0f;
    CHECK(ok && inexact == 0, "IDX byte scaling: %d values differ from v / 255.0f", inexact);
    spingalett_dataset_free(&d);
    remove("spingalett_test_images.idx");
    remove("spingalett_test_labels.idx");

    /* CSV with a header, class labels in the last column */
    f = fopen("spingalett_test.csv", "w");
    fputs("a, b, label\n", f);
    for (int i = 0; i < 50; i++) fprintf(f, "%d, %g, %d\r\n", i, i * 0.5, i % 3);
    fputs("\n", f);
    fclose(f);
    ok = spingalett_load_csv("spingalett_test.csv", 1, 3, &d);
    CHECK(ok && d.count == 50 && d.input_size == 2 && d.target_size == 3 && d.inputs[2 * 7 + 1] == 3.5f &&
          d.targets[7 * 3 + 1] == 1.0f, "CSV reader with classes");
    /* shuffling keeps every input with its target */
    spingalett_seed(3);
    spingalett_dataset_shuffle(&d);
    bool paired = ok, moved = false;
    for (uint32_t i = 0; ok && i < d.count; i++) {
        int v = (int)d.inputs[i * 2];
        paired = paired && d.targets[i * 3 + v % 3] == 1.0f && d.inputs[i * 2 + 1] == v * 0.5f;
        moved = moved || v != (int)i;
    }
    CHECK(paired && moved, "dataset shuffle");
    spingalett_dataset_free(&d);
    ok = spingalett_load_csv("spingalett_test.csv", 2, 0, &d);
    CHECK(ok && d.input_size == 1 && d.target_size == 2 && d.targets[4 * 2 + 0] == 2.0f && d.targets[4 * 2 + 1] == 1.0f, "CSV regression targets");
    spingalett_dataset_free(&d);

    f = fopen("spingalett_test.csv", "w");
    fputs("1, 2, 0\n3, x, 1\n", f);
    fclose(f);
    CHECK(!spingalett_load_csv("spingalett_test.csv", 1, 2, &d) && spingalett_last_error_code() == SPINGALETT_ERR_INVALID, "malformed CSV");
    f = fopen("spingalett_test.csv", "w");
    fputs("1, 2, 0\n3, 4\n", f);
    fclose(f);
    CHECK(!spingalett_load_csv("spingalett_test.csv", 1, 2, &d), "ragged CSV");
    remove("spingalett_test.csv");
    CHECK(!spingalett_load_csv("spingalett_test_missing.csv", 1, 0, &d) && spingalett_last_error_code() == SPINGALETT_ERR_FILE_IO, "missing CSV");
}


/* ---------------- .slettd data set files ---------------- */

static uint32_t test_crc32(const unsigned char *p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c;
}

static unsigned char *read_file(const char *path, long *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); *size = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc((size_t)*size + 1);
    if (b && fread(b, 1, (size_t)*size, f) != (size_t)*size) { free(b); b = NULL; }
    fclose(f);
    return b;
}

static void write_file(const char *path, const unsigned char *b, long size) {
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(b, 1, (size_t)size, f); fclose(f); }
}

static bool same_dataset(const SpingalettDataset *a, const SpingalettDataset *b) {
    return a->count == b->count && a->input_size == b->input_size && a->target_size == b->target_size &&
           !memcmp(a->inputs, b->inputs, (size_t)a->count * a->input_size * sizeof(float)) &&
           !memcmp(a->targets, b->targets, (size_t)a->count * a->target_size * sizeof(float));
}

static SpingalettDatasetInfo file_info(const char *path) {
    SpingalettDatasetReader *r = spingalett_dataset_open(path, false);
    SpingalettDatasetInfo info = spingalett_dataset_info(r);
    spingalett_dataset_close(r);
    return info;
}

/* Saves d, loads it back from the file and from memory; returns the max abs input difference. */
static float ds_roundtrip(const char *name, const SpingalettDataset *d, DatasetEncoding in_enc, DatasetEncoding tg_enc,
                          bool raw, DatasetEncoding want_in, DatasetEncoding want_tg, SpingalettDataset *out) {
    const char *path = "spingalett_test_ds.slettd";
    DatasetSaveOptions o = {.input_encoding = in_enc, .target_encoding = tg_enc, .no_compression = raw};
    bool ok = spingalett_save_dataset(d, path, &o);
    SpingalettDatasetInfo info = file_info(path);
    long size = 0;
    unsigned char *bytes = read_file(path, &size);
    SpingalettDataset m;
    ok = ok && spingalett_load_dataset(path, out) && bytes && spingalett_load_dataset_from_memory(bytes, (size_t)size, &m);
    float diff = INFINITY;
    if (ok) {
        diff = max_abs_diff(d->inputs, out->inputs, (uint64_t)d->count * d->input_size);
        CHECK(same_dataset(out, &m), "%s: memory load differs from file load", name);
        spingalett_dataset_free(&m);
    }
    double raw_bytes = (double)d->count * (d->input_size + d->target_size) * 4;
    printf("  slettd %-24s %8ld bytes (%5.1f%% of float32), encodings %d/%d, chunks %u, max input error %.2e\n",
           name, size, 100.0 * size / raw_bytes, info.input_encoding, info.target_encoding, info.chunk_count, (double)diff);
    CHECK(ok && info.input_encoding == want_in && info.target_encoding == want_tg && (uint64_t)size == info.file_size,
          "%s: save/load failed or wrong encodings (%d/%d): %s", name, info.input_encoding, info.target_encoding,
          spingalett_last_error_message());
    free(bytes);
    remove(path);
    return diff;
}

static void make_ds(SpingalettDataset *d, uint32_t count, uint32_t in, uint32_t out) {
    memset(d, 0, sizeof *d);
    d->count = count; d->input_size = in; d->target_size = out;
    d->inputs = calloc((size_t)count * in, sizeof(float));
    d->targets = calloc((size_t)count * out, sizeof(float));
}

static void dataset_files(void) {
    /* 8-bit "images" scaled to [0, 1] with one-hot labels: stored losslessly as bytes and class indices */
    SpingalettDataset img, back;
    make_ds(&img, 3000, 784, 10);
    lcg_state = 4;
    for (uint32_t i = 0; i < img.count; i++) {
        int cx = 6 + (int)(frand() * 16), cy = 6 + (int)(frand() * 16), r = 3 + (int)(frand() * 5);
        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++) {
                int dd = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                int v = dd < r * r ? 255 - dd * 4 : 0;
                img.inputs[(size_t)i * 784 + y * 28 + x] = (float)(v < 0 ? 0 : v) / 255.0f;
            }
        img.targets[(size_t)i * 10 + i % 10] = 1.0f;
    }
    float e = ds_roundtrip("u8 images, one-hot", &img, DATASET_ENCODING_AUTO, DATASET_ENCODING_AUTO, false,
                           DATASET_ENCODING_U8_UNIT, DATASET_ENCODING_CLASS, &back);
    CHECK(e == 0.0f && same_dataset(&img, &back), "u8 images must round-trip bit for bit");
    spingalett_dataset_free(&back);
    e = ds_roundtrip("u8 images, stored", &img, DATASET_ENCODING_AUTO, DATASET_ENCODING_AUTO, true,
                     DATASET_ENCODING_U8_UNIT, DATASET_ENCODING_CLASS, &back);
    CHECK(e == 0.0f && same_dataset(&img, &back), "stored u8 images must round-trip bit for bit");
    spingalett_dataset_free(&back);

    /* arbitrary floats: FLOAT32 is lossless; FP16, BF16 and U8_AFFINE stay within their precision */
    SpingalettDataset fl;
    make_ds(&fl, 2500, 13, 3);
    for (size_t i = 0; i < (size_t)fl.count * 13; i++) fl.inputs[i] = (frand() - 0.5f) * 40.0f * (float)(1 + i % 13);
    for (size_t i = 0; i < (size_t)fl.count * 3; i++) fl.targets[i] = frand() * 2 - 1;
    e = ds_roundtrip("float32", &fl, DATASET_ENCODING_AUTO, DATASET_ENCODING_AUTO, false,
                     DATASET_ENCODING_FLOAT32, DATASET_ENCODING_FLOAT32, &back);
    CHECK(e == 0.0f && same_dataset(&fl, &back), "float32 must round-trip bit for bit");
    spingalett_dataset_free(&back);
    float worst_rel = 0;
    e = ds_roundtrip("fp16 (lossy)", &fl, DATASET_ENCODING_FP16, DATASET_ENCODING_AUTO, false,
                     DATASET_ENCODING_FP16, DATASET_ENCODING_FLOAT32, &back);
    for (size_t i = 0; i < (size_t)fl.count * 13; i++)
        if (fabsf(fl.inputs[i]) > 1e-3f) worst_rel = fmaxf(worst_rel, fabsf(back.inputs[i] - fl.inputs[i]) / fabsf(fl.inputs[i]));
    CHECK(worst_rel <= 1.0f / 2048 + 1e-7f, "fp16 relative error %.3e", (double)worst_rel);
    spingalett_dataset_free(&back);
    worst_rel = 0;
    ds_roundtrip("bfloat16 (lossy)", &fl, DATASET_ENCODING_BFLOAT16, DATASET_ENCODING_BFLOAT16, false,
                 DATASET_ENCODING_BFLOAT16, DATASET_ENCODING_BFLOAT16, &back);
    for (size_t i = 0; i < (size_t)fl.count * 13; i++)
        if (fabsf(fl.inputs[i]) > 1e-3f) worst_rel = fmaxf(worst_rel, fabsf(back.inputs[i] - fl.inputs[i]) / fabsf(fl.inputs[i]));
    CHECK(worst_rel <= 1.0f / 256 + 1e-7f, "bf16 relative error %.3e", (double)worst_rel);
    spingalett_dataset_free(&back);
    e = ds_roundtrip("u8 affine (lossy)", &fl, DATASET_ENCODING_U8_AFFINE, DATASET_ENCODING_AUTO, false,
                     DATASET_ENCODING_U8_AFFINE, DATASET_ENCODING_FLOAT32, &back);
    bool within = true;
    for (uint32_t f = 0; f < 13; f++) {
        float lo = INFINITY, hi = -INFINITY;
        for (uint32_t r = 0; r < fl.count; r++) { lo = fminf(lo, fl.inputs[r * 13 + f]); hi = fmaxf(hi, fl.inputs[r * 13 + f]); }
        float half_step = (hi - lo) / 255.0f / 2 * 1.001f + 1e-6f * fabsf(hi);
        for (uint32_t r = 0; r < fl.count; r++) within = within && fabsf(back.inputs[r * 13 + f] - fl.inputs[r * 13 + f]) <= half_step;
    }
    CHECK(within, "u8 affine error beyond half a quantization step");
    spingalett_dataset_free(&back);

    /* values that are halves are stored as FP16 without loss; many classes need 2-byte indices */
    SpingalettDataset hv;
    make_ds(&hv, 700, 5, 300);
    for (size_t i = 0; i < (size_t)hv.count * 5; i++) hv.inputs[i] = (float)((int)(frand() * 2000) - 1000) / 8.0f;
    for (uint32_t i = 0; i < hv.count; i++) hv.targets[(size_t)i * 300 + (i * 37) % 300] = 1.0f;
    e = ds_roundtrip("halves, 300 classes", &hv, DATASET_ENCODING_AUTO, DATASET_ENCODING_AUTO, false,
                     DATASET_ENCODING_FP16, DATASET_ENCODING_CLASS, &back);
    CHECK(e == 0.0f && same_dataset(&hv, &back), "fp16-exact data and 300 classes must round-trip exactly");
    spingalett_dataset_free(&back);
    spingalett_dataset_free(&hv);

    /* corruption and truncation are detected */
    CHECK(spingalett_save_dataset(&img, "spingalett_test_ds", NULL), "save without extension");
    long size = 0;
    unsigned char *good = read_file("spingalett_test_ds" SPINGALETT_DATASET_EXTENSION, &size);
    CHECK(good != NULL, "save without extension should write spingalett_test_ds.slettd");
    if (good) {
        SpingalettDataset bad;
        unsigned char *copy = malloc((size_t)size);
        long cases[] = {size / 2, 20, 70};       /* chunk data, header field, index */
        for (int c = 0; c < 3; c++) {
            memcpy(copy, good, (size_t)size);
            copy[cases[c]] ^= 0x10;
            CHECK(!spingalett_load_dataset_from_memory(copy, (size_t)size, &bad) && spingalett_last_error_code() == SPINGALETT_ERR_INVALID &&
                  bad.count == 0, "flipped bit at %ld not detected", cases[c]);
        }
        CHECK(!spingalett_load_dataset_from_memory(good, (size_t)size - 1, &bad) && spingalett_last_error_code() == SPINGALETT_ERR_FILE_IO,
              "truncated data set not detected");
        memcpy(copy, good, (size_t)size);
        copy[6] = 3;                             /* a future format version, with a valid checksum */
        uint32_t crc = test_crc32(copy, 60);
        for (int k = 0; k < 4; k++) copy[60 + k] = (unsigned char)(crc >> (8 * k));
        CHECK(!spingalett_load_dataset_from_memory(copy, (size_t)size, &bad) && spingalett_last_error_code() == SPINGALETT_ERR_FORMAT_VERSION,
              "future format version not reported");
        write_file("spingalett_test_bad.slettd", copy, 40);
        CHECK(!spingalett_load_dataset("spingalett_test_bad.slettd", &bad) && spingalett_dataset_open("spingalett_test_bad.slettd", false) == NULL,
              "file shorter than a header accepted");
        remove("spingalett_test_bad.slettd");
        CHECK(!spingalett_load_dataset("spingalett_test_missing.slettd", &bad) && spingalett_last_error_code() == SPINGALETT_ERR_FILE_IO,
              "missing data set file");
        free(copy);
        free(good);
    }

    /* streaming: every sample exactly once per pass, in file order or shuffled anew each pass */
    for (int shuffled = 0; shuffled < 2; shuffled++) {
        spingalett_seed(11);
        SpingalettDatasetReader *r = spingalett_dataset_open("spingalett_test_ds.slettd", shuffled);
        float *xi = malloc(97 * 784 * sizeof(float)), *yi = malloc(97 * 10 * sizeof(float));
        uint32_t *seen = calloc(img.count, sizeof(uint32_t)), first[2] = {0, 0};
        bool exact = r != NULL, ordered = true;
        for (int pass = 0; pass < 2 && r; pass++) {
            uint32_t total = 0, got;
            while ((got = spingalett_dataset_read(r, xi, yi, 97)) > 0) {
                for (uint32_t k = 0; k < got; k++) {
                    /* find the sample by its label and pixels (labels repeat every 10, pixels identify it) */
                    uint32_t idx = UINT32_MAX, label = 0;
                    for (uint32_t c = 1; c < 10; c++) if (yi[(size_t)k * 10 + c] > yi[(size_t)k * 10 + label]) label = c;
                    for (uint32_t j = label; j < img.count && idx == UINT32_MAX; j += 10)
                        if (!seen[j] || seen[j] == (uint32_t)pass)
                            if (!memcmp(img.inputs + (size_t)j * 784, xi + (size_t)k * 784, 784 * sizeof(float))) idx = j;
                    exact = exact && idx != UINT32_MAX;
                    if (idx != UINT32_MAX) {
                        seen[idx]++;
                        exact = exact && !memcmp(img.targets + (size_t)idx * 10, yi + (size_t)k * 10, 10 * sizeof(float));
                        ordered = ordered && idx == total + k;
                        if (total + k == 0) first[pass] = idx;
                    }
                }
                total += got;
            }
            exact = exact && total == img.count;
        }
        for (uint32_t j = 0; j < img.count; j++) exact = exact && seen[j] == 2;
        printf("  slettd reader shuffle=%d: exact=%d in_order=%d first samples %u, %u\n", shuffled, exact, ordered, first[0], first[1]);
        CHECK(exact && (shuffled ? !ordered && first[0] != first[1] : ordered), "dataset reader (shuffle %d)", shuffled);
        spingalett_dataset_close(r);
        free(xi); free(yi); free(seen);
    }

    /* the reader as a train() generator matches training on the arrays */
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    NeuralNetwork *a = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY), *b = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    for (int n = 0; n < 2; n++) {
        NeuralNetwork *net = n ? b : a;
        layer(net, 784);
        layer(net, 32, ACT_RELU, WEIGHT_INITIALIZATION_HE);
        layer(net, 10, ACT_SOFTMAX, WEIGHT_INITIALIZATION_XAVIER);
    }
    memcpy(b->weights, a->weights, a->total_weights * sizeof(float));
    train(.net = a, .inputs = img.inputs, .targets = img.targets, .sample_count = img.count, .epochs = 2,
          .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 64, .do_not_shuffle = true, .optimizer_type = OPTIMIZER_ADAM);
    SpingalettDatasetReader *r = spingalett_dataset_open("spingalett_test_ds.slettd", false);
    TrainReport rep = train(.net = b, .training_mode = MODE_GENERATOR_FUNCTION, .generator = spingalett_dataset_generator,
                            .generator_data = r, .epochs = 2, .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 64,
                            .optimizer_type = OPTIMIZER_ADAM);
    float dw = max_abs_diff(a->weights, b->weights, a->total_weights);
    printf("  slettd generator training: status %d, steps %llu vs %llu, |a-b| %.1e\n", rep.status,
           (unsigned long long)b->time_step, (unsigned long long)a->time_step, (double)dw);
    CHECK(rep.status == TRAIN_COMPLETED && a->time_step == b->time_step && dw == 0.0f, "training from a .slettd reader");
    /* epochs delimited by sample_count (needed for full batch) never ask past a pass's last sample */
    for (int st = 0; st < 2; st++) {
        uint64_t steps = b->time_step;
        rep = train(.net = b, .training_mode = MODE_GENERATOR_FUNCTION, .generator = spingalett_dataset_generator,
                    .generator_data = r, .sample_count = img.count, .epochs = 3,
                    .training_strategy = st ? STRATEGY_FULL_BATCH : STRATEGY_SMALL_BATCH, .batch_size = 64,
                    .optimizer_type = OPTIMIZER_ADAM);
        uint64_t want = st ? 3 : 3 * ((img.count + 63) / 64);
        CHECK(rep.status == TRAIN_COMPLETED && rep.epochs_run == 3 && b->time_step - steps == want,
              "reader with sample_count (strategy %d): status %d, %llu steps", st, rep.status,
              (unsigned long long)(b->time_step - steps));
    }
    spingalett_dataset_close(r);
    free_network(a); free_network(b);

    remove("spingalett_test_ds.slettd");
    spingalett_dataset_free(&img);
    spingalett_dataset_free(&fl);
}

/* Reads every sample of one pass: the order of the samples (their first input value identifies
   them when it is unique, as below) and whether inputs and targets match the data set. */
static bool read_pass(SpingalettDatasetReader *r, const SpingalettDataset *d, uint32_t *order, uint32_t batch) {
    float *x = malloc((size_t)batch * d->input_size * sizeof(float)), *y = malloc((size_t)batch * d->target_size * sizeof(float));
    uint32_t total = 0, got;
    bool exact = x && y;
    while (exact && (got = spingalett_dataset_read(r, x, y, batch)) > 0)
        for (uint32_t k = 0; k < got && exact; k++, total++) {
            uint32_t idx = (uint32_t)(x[(size_t)k * d->input_size] * 255.0f + 0.5f) |
                           (uint32_t)(x[(size_t)k * d->input_size + 1] * 255.0f + 0.5f) << 8;
            exact = idx < d->count && total < d->count &&
                    !memcmp(d->inputs + (size_t)idx * d->input_size, x + (size_t)k * d->input_size, d->input_size * sizeof(float)) &&
                    !memcmp(d->targets + (size_t)idx * d->target_size, y + (size_t)k * d->target_size, d->target_size * sizeof(float));
            if (exact) order[total] = idx;
        }
    free(x);
    free(y);
    return exact && total == d->count;
}

static uint32_t le32(const unsigned char *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static void put_le32(unsigned char *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (unsigned char)(v >> (8 * i)); }

/* A .slettd image altered after its header: the index checksum made right again, so that the
   reader gets as far as the altered fields. */
static void slettd_fix_index_crc(unsigned char *b) {
    uint32_t pbytes = le32(b + 48), sets = le32(b + 52) ? le32(b + 52) : 1u, mbytes = le32(b + 56), chunks = le32(b + 28);
    size_t meta = 64 + (size_t)pbytes + mbytes + (size_t)chunks * (12 + 4 * (1 + sets)) + 4;
    put_le32(b + meta - 4, test_crc32(b + 64, meta - 68));
}

/* The first metadata record with this tag (its header: 2 bytes of tag, 4 of length), or NULL. */
static unsigned char *slettd_record(unsigned char *b, uint32_t tag) {
    unsigned char *p = b + 64 + le32(b + 48), *end = p + le32(b + 56);
    while (end - p >= 6) {
        if ((uint32_t)(p[0] | p[1] << 8) == tag) return p;
        p += 6 + le32(p + 2);
    }
    return NULL;
}

/* Crafted files whose fields would make the reader index past what it holds are rejected. */
static void dataset_crafted(const char *path, const SpingalettDataset *ph) {
    long size = 0;
    unsigned char *good = read_file(path, &size), *b = good ? malloc((size_t)size) : NULL;
    if (!b) { CHECK(false, "crafted: cannot read %s", path); free(good); return; }
    SpingalettDataset d;
    struct { const char *what; uint32_t tag; int field; } cases[] = {
        {"a name record shorter than its set number", 1, -1},           /* the shape record, made a NAME of length 0 */
        {"a set record for set 2^32 - 1", 2, 0},
        {"a class-name record for set 2^32 - 1", 4, 0},
    };
    for (size_t k = 0; k < sizeof cases / sizeof *cases; k++) {
        memcpy(b, good, (size_t)size);
        unsigned char *r = slettd_record(b, cases[k].tag);
        if (!r) { CHECK(false, "crafted: no record %u", cases[k].tag); continue; }
        if (cases[k].field < 0) { r[0] = 3; r[1] = 0; put_le32(r + 2, 0); }
        else put_le32(r + 6 + 4 * cases[k].field, 0xFFFFFFFFu);
        slettd_fix_index_crc(b);
        CHECK(!spingalett_load_dataset_from_memory(b, (size_t)size, &d) && spingalett_last_error_code() == SPINGALETT_ERR_INVALID,
              "crafted: %s not rejected", cases[k].what);
    }
    /* a chunk offset whose sum with the chunk's size wraps around past zero */
    memcpy(b, good, (size_t)size);
    uint32_t sets = le32(b + 52), streams = 1 + sets;
    unsigned char *entry = b + 64 + le32(b + 48) + le32(b + 56);
    put_le32(entry, 0xFFFF0000u);
    put_le32(entry + 4, 0xFFFFFFFFu);
    put_le32(entry + 8, 0x10000u);
    for (uint32_t s = 1; s < streams; s++) put_le32(entry + 8 + 4 * s, 0x80u);
    slettd_fix_index_crc(b);
    CHECK(!spingalett_load_dataset_from_memory(b, (size_t)size, &d) && spingalett_last_error_code() == SPINGALETT_ERR_INVALID,
          "crafted: a chunk offset that wraps around not rejected");
    free(b);
    free(good);
    /* a set number of 2^32 - 1, which plus one is 0, the inputs */
    CHECK(!spingalett_load_dataset_targets(path, UINT32_MAX, &d) && spingalett_last_error_code() == SPINGALETT_ERR_INVALID,
          "crafted: set 2^32 - 1 loaded");
    DatasetReaderOptions o = {.target_set = UINT32_MAX};
    CHECK(spingalett_dataset_open_ex(path, &o) == NULL, "crafted: a reader of set 2^32 - 1 opened");
    SpingalettDatasetReader *r = spingalett_dataset_open(path, false);
    CHECK(r && !spingalett_dataset_target_set_name(r, UINT32_MAX) && !spingalett_dataset_class_name(r, UINT32_MAX, 0) &&
          spingalett_dataset_target_set_size(r, UINT32_MAX) == 0, "crafted: reader names of set 2^32 - 1");
    spingalett_dataset_close(r);
    /* 2^32 - 1 further sets: their count plus one is 0 */
    SpingalettTargetSet extra = {.size = ph->target_size, .targets = ph->targets};
    DatasetSaveOptions many = {.extra_targets = &extra, .extra_target_count = UINT32_MAX};
    CHECK(!spingalett_save_dataset(ph, "spingalett_test_many.slettd", &many) && spingalett_last_error_code() == SPINGALETT_ERR_INVALID,
          "crafted: 2^32 - 1 further sets of targets accepted");
    remove("spingalett_test_many.slettd");
}

static void dataset_files_v2(void) {
    /* dense 8-bit "photographs" (smooth colour fields with noise), 32 x 32 x 3; the first two
       values of a sample encode its index */
    SpingalettDataset ph, back;
    make_ds(&ph, 1200, 3072, 10);
    lcg_state = 77;
    for (uint32_t i = 0; i < ph.count; i++) {
        float fx = frand() * 0.2f, fy = frand() * 0.2f, base = frand() * 120.0f;
        for (int y = 0; y < 32; y++)
            for (int x = 0; x < 32; x++)
                for (int c = 0; c < 3; c++) {
                    float v = base + 60.0f * sinf(fx * x + c) + 50.0f * cosf(fy * y) + frand() * 24.0f;
                    int q = (int)(v < 0 ? 0 : v > 255 ? 255 : v);
                    ph.inputs[(size_t)i * 3072 + (y * 32 + x) * 3 + c] = (float)q / 255.0f;
                }
        ph.inputs[(size_t)i * 3072] = (float)(i & 255) / 255.0f;
        ph.inputs[(size_t)i * 3072 + 1] = (float)(i >> 8) / 255.0f;
        ph.targets[(size_t)i * 10 + i % 10] = 1.0f;
    }
    const char *path = "spingalett_test_v2.slettd";
    CHECK(spingalett_save_dataset(&ph, path, NULL), "v2: save dense data");
    SpingalettDatasetInfo info = file_info(path);
    SpingalettDataset ld;
    bool ok = spingalett_load_dataset(path, &ld);
    printf("  slettd dense u8 (rANS)        %8llu bytes (%5.1f%% of the bytes), version %u, chunks %u\n",
           (unsigned long long)info.file_size, 100.0 * (double)info.file_size / ((double)ph.count * 3073), info.format_version,
           info.chunk_count);
    /* the writer took the rANS coder (only version 2 files have it) and the data comes back exactly */
    CHECK(ok && info.format_version == 2 && info.input_encoding == DATASET_ENCODING_U8_UNIT && same_dataset(&ph, &ld),
          "v2: dense data round trip through the rANS coder (version %u)", info.format_version);
    if (ok) spingalett_dataset_free(&ld);
    long size = 0;
    unsigned char *good = read_file(path, &size);
    if (good) {
        /* a flipped bit in a coded stream is caught (by the checksum, and the coder's final state) */
        unsigned char *copy = malloc((size_t)size);
        memcpy(copy, good, (size_t)size);
        copy[size - 1000] ^= 0x04;
        CHECK(!spingalett_load_dataset_from_memory(copy, (size_t)size, &ld) && spingalett_last_error_code() == SPINGALETT_ERR_INVALID,
              "v2: corrupt rANS chunk not detected");
        /* streaming readers report it when they reach the chunk, from any decoding thread */
        write_file("spingalett_test_bad.slettd", copy, size);
        ComputeMode mode = spingalett_get_compute_mode();
        for (int v = 0; v < 3; v++) {
            spingalett_set_compute_mode(v == 2 ? COMPUTE_OPENMP : COMPUTE_SINGLE_THREADED);
            DatasetReaderOptions o = {.shuffle = true, .no_prefetch = v == 1};
            SpingalettDatasetReader *r = spingalett_dataset_open_ex("spingalett_test_bad.slettd", &o);
            float *x = malloc((size_t)100 * 3072 * sizeof(float)), *y = malloc((size_t)100 * 10 * sizeof(float));
            uint32_t total = 0, got;
            while (r && (got = spingalett_dataset_read(r, x, y, 100)) > 0) total += got;
            CHECK(r && total < ph.count && spingalett_last_error_code() == SPINGALETT_ERR_INVALID &&
                  spingalett_dataset_read(r, x, y, 100) == 0, "v2: reader %d: corrupt chunk not reported", v);
            spingalett_dataset_close(r);
            free(x);
            free(y);
        }
        spingalett_set_compute_mode(mode);
        remove("spingalett_test_bad.slettd");
        free(copy);
        free(good);
    }

    /* shape, class names, a second set of targets with its own names */
    static const char *const names[10] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
    static const char *const parity[2] = {"even", "odd"};
    ph.height = ph.width = 32;
    ph.channels = 3;
    CHECK(spingalett_dataset_set_class_names(&ph, names, 10), "v2: set class names");
    float *odd = calloc((size_t)ph.count * 2, sizeof(float));
    for (uint32_t i = 0; i < ph.count; i++) odd[(size_t)i * 2 + (i % 10) % 2] = 1.0f;
    SpingalettTargetSet extra = {.name = "parity", .size = 2, .targets = odd, .class_names = parity};
    DatasetSaveOptions opt = {.target_name = "digit", .extra_targets = &extra, .extra_target_count = 1};
    CHECK(spingalett_save_dataset(&ph, path, &opt), "v2: save with metadata and two sets of targets");
    ok = spingalett_load_dataset(path, &ld);
    CHECK(ok && same_dataset(&ph, &ld) && ld.height == 32 && ld.width == 32 && ld.channels == 3 && ld.class_names &&
          !strcmp(ld.class_names[7], "seven") && ld.class_names[10] == NULL, "v2: shape and class names come back");
    if (ok) spingalett_dataset_free(&ld);
    ok = spingalett_load_dataset_targets(path, 1, &ld);
    bool parity_ok = ok && ld.target_size == 2 && ld.count == ph.count && ld.class_names && !strcmp(ld.class_names[1], "odd") &&
                     !memcmp(ld.targets, odd, (size_t)ph.count * 2 * sizeof(float)) &&
                     !memcmp(ld.inputs, ph.inputs, (size_t)ph.count * 3072 * sizeof(float));
    CHECK(parity_ok, "v2: the second set of targets loads with the inputs");
    if (ok) spingalett_dataset_free(&ld);
    CHECK(!spingalett_load_dataset_targets(path, 2, &ld) && spingalett_last_error_code() == SPINGALETT_ERR_INVALID,
          "v2: a set of targets that is not there");
    SpingalettDatasetReader *r = spingalett_dataset_open(path, false);
    info = spingalett_dataset_info(r);
    CHECK(r && info.target_set_count == 2 && info.height == 32 && info.channels == 3 &&
          !strcmp(spingalett_dataset_target_set_name(r, 0), "digit") && !strcmp(spingalett_dataset_target_set_name(r, 1), "parity") &&
          spingalett_dataset_target_set_size(r, 1) == 2 && !strcmp(spingalett_dataset_class_name(r, 1, 0), "even") &&
          spingalett_dataset_class_name(r, 1, 2) == NULL && spingalett_dataset_class_name(r, 0, 3) &&
          !strcmp(spingalett_dataset_class_name(r, 0, 3), "three"), "v2: reader metadata");
    spingalett_dataset_close(r);
    /* the shape and names also survive a split */
    SpingalettDataset copy = {0}, tail = {0};
    ok = spingalett_load_dataset(path, &copy) && spingalett_dataset_split(&copy, 100, &tail);
    CHECK(ok && tail.height == 32 && tail.class_names && !strcmp(tail.class_names[9], "nine"), "v2: split keeps shape and names");
    spingalett_dataset_free(&copy);
    spingalett_dataset_free(&tail);
    dataset_crafted(path, &ph);

    /* readers: shuffled passes are the same with a decoding thread (single-threaded mode leaves
       processors free), one chunk at a time on the caller, and chunks decoded together by the
       OpenMP threads (which take every processor); in memory every pass holds each sample once;
       a u8 reader serves what it was given */
    uint32_t *o1 = malloc(ph.count * sizeof(uint32_t)), *o2 = malloc(ph.count * sizeof(uint32_t));
    bool same = true, exact = true;
    ComputeMode mode = spingalett_get_compute_mode();
    static const ComputeMode open_modes[3] = {COMPUTE_SINGLE_THREADED, COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP};
    for (int pass = 0; pass < 3; pass++) {
        SpingalettDatasetReader *rs[3];
        for (int v = 0; v < 3; v++) {
            DatasetReaderOptions o = {.shuffle = true, .no_prefetch = v == 1};
            spingalett_set_compute_mode(open_modes[v]);
            spingalett_seed(5);
            rs[v] = spingalett_dataset_open_ex(path, &o);
            exact = exact && rs[v];
        }
        spingalett_set_compute_mode(COMPUTE_OPENMP);
        for (int k = 0; k <= pass && exact; k++) {
            exact = read_pass(rs[0], &ph, o1, 29 + 7 * k);
            for (int v = 1; v < 3 && exact; v++) {
                exact = read_pass(rs[v], &ph, o2, v == 1 ? 64 : 17);
                same = same && !memcmp(o1, o2, ph.count * sizeof(uint32_t));
            }
        }
        for (int v = 0; v < 3; v++) spingalett_dataset_close(rs[v]);
    }
    spingalett_set_compute_mode(mode);
    CHECK(exact && same, "v2: streaming readers (exact %d, same order %d)", exact, same);
    DatasetReaderOptions m = {.shuffle = true, .in_memory = true};
    r = spingalett_dataset_open_ex(path, &m);
    bool spread = false;                /* in memory the passes shuffle across chunks */
    exact = r != NULL;
    for (int pass = 0; pass < 2 && r; pass++) {
        exact = exact && read_pass(r, &ph, o1, 50);
        for (uint32_t k = 0; k + 1 < 20 && exact; k++) {
            uint32_t a1 = o1[k], b1 = o1[k + 1];
            if ((a1 > b1 ? a1 - b1 : b1 - a1) > 600) spread = true;
        }
    }
    CHECK(exact && spread, "v2: in-memory reader (exact %d, shuffled across chunks %d)", exact, spread);
    spingalett_dataset_close(r);
    uint8_t *bytes = malloc((size_t)ph.count * 3072);
    for (size_t i = 0; i < (size_t)ph.count * 3072; i++) bytes[i] = (uint8_t)(ph.inputs[i] * 255.0f + 0.5f);
    r = spingalett_dataset_open_u8(bytes, ph.targets, ph.count, 3072, 10, false);
    exact = r && read_pass(r, &ph, o1, 33);
    for (uint32_t k = 0; exact && k < ph.count; k++) exact = o1[k] == k;
    CHECK(exact, "v2: u8 reader");
    /* training through the u8 reader equals training on the arrays (in order) */
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    NeuralNetwork *na = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY), *nb = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    for (int n = 0; n < 2; n++) {
        NeuralNetwork *net = n ? nb : na;
        layer(net, 3072);
        layer(net, 16, ACT_RELU, WEIGHT_INITIALIZATION_HE);
        layer(net, 10, ACT_SOFTMAX, WEIGHT_INITIALIZATION_XAVIER);
    }
    memcpy(nb->weights, na->weights, na->total_weights * sizeof(float));
    train(.net = na, .inputs = ph.inputs, .targets = ph.targets, .sample_count = ph.count, .epochs = 2,
          .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 32, .do_not_shuffle = true, .optimizer_type = OPTIMIZER_ADAM);
    train(.net = nb, .training_mode = MODE_GENERATOR_FUNCTION, .generator = spingalett_dataset_generator, .generator_data = r,
          .epochs = 2, .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 32, .optimizer_type = OPTIMIZER_ADAM);
    CHECK(max_abs_diff(na->weights, nb->weights, na->total_weights) == 0.0f && na->time_step == nb->time_step,
          "v2: training from the u8 reader");
    free_network(na);
    free_network(nb);
    spingalett_dataset_close(r);
    free(bytes);
    free(o1);
    free(o2);
    free(odd);
    remove(path);
    spingalett_dataset_free(&ph);
    (void)back;
}


/* ---------------------------------------------------------------- deployment models (format 3) */

/* Fields of a .slett format 3 image, read as docs/ModelFormat.md describes them. */
static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t rd32(const uint8_t *p) { return p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static uint64_t rd64(const uint8_t *p) { return rd32(p) | (uint64_t)rd32(p + 4) << 32; }
static float rdf(const uint8_t *p) { uint32_t u = rd32(p); float f; memcpy(&f, &u, 4); return f; }
static void wr32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }

/* Rewrites both checksums of a `len`-byte image after a deliberate change. */
static void reseal(uint8_t *img, size_t len) {
    uint64_t size = rd64(img + 24);
    if (size < 64 || size > len) size = len;
    wr32(img + 56, test_crc32(img + 64, (size_t)size - 64));
    wr32(img + 60, test_crc32(img, 60));
}

typedef struct { uint32_t in, out; int act, prec; uint64_t w, s, b, o; } Entry;
static Entry image_layer(const uint8_t *img, uint32_t i) {
    const uint8_t *e = img + 64 + 48 * (size_t)i;
    Entry r = {rd32(e), rd32(e + 4), e[8], e[9], rd64(e + 16), rd64(e + 24), rd64(e + 32), rd64(e + 40)};
    return r;
}

static double half_value(uint16_t h) {
    int e = (h >> 10) & 31, m = h & 1023;
    double v = e == 0 ? ldexp(m, -24) : ldexp(1024 + m, e - 25);
    return h & 0x8000 ? -v : v;
}

static int stored_code(const uint8_t *img, Entry e, uint32_t j, uint32_t k) {
    if (e.prec == PRECISION_INT8) return (int8_t)img[e.w + (uint64_t)j * e.in + k];
    if (e.prec == PRECISION_INT4) {
        uint8_t b = img[e.w + (uint64_t)j * ((e.in + 1) / 2) + k / 2];
        return (int)(((k & 1) ? b >> 4 : b & 15) ^ 8) - 8;
    }
    uint8_t b = img[e.w + (uint64_t)j * ((e.in + 3) / 4) + k / 4];
    return (int)(((b >> (2 * (k % 4))) & 3) ^ 2) - 2;
}

static double stored_weight(const uint8_t *img, Entry e, uint32_t j, uint32_t k) {
    uint64_t i = (uint64_t)j * e.in + k;
    if (e.prec == PRECISION_FLOAT32) return rdf(img + e.w + 4 * i);
    if (e.prec == PRECISION_FP16) return half_value((uint16_t)(img[e.w + 2 * i] | img[e.w + 2 * i + 1] << 8));
    if (e.prec == PRECISION_BFLOAT16) { uint32_t u = (uint32_t)(img[e.w + 2 * i] | img[e.w + 2 * i + 1] << 8) << 16; float f; memcpy(&f, &u, 4); return f; }
    return stored_code(img, e, j, k) * (double)rdf(img + e.s + 4 * (uint64_t)j);
}

/* What the engine computes for one weight layer without activation, from the stored image:
   integer layers quantize x like the engine (scale = max |x| / 127, nearest even) and accumulate
   exactly, which must match bit for bit; float layers accumulate in double, so `bound` receives a
   bound on the rounding error. */
static void reference_layer(const uint8_t *img, Entry e, const float *x, float *y, double *bound) {
    if (e.prec >= PRECISION_INT8) {
        float amax = 0;
        for (uint32_t k = 0; k < e.in; k++) if (fabsf(x[k]) > amax) amax = fabsf(x[k]);
        float a = amax / 127.0f, inv = amax > 0 ? 127.0f / amax : 0;
        int8_t *q = malloc(e.in);
        for (uint32_t k = 0; k < e.in; k++) q[k] = (int8_t)lrintf(x[k] * inv);
        for (uint32_t j = 0; j < e.out; j++) {
            int64_t acc = 0;
            for (uint32_t k = 0; k < e.in; k++) acc += (int64_t)stored_code(img, e, j, k) * q[k];
            float rs = rdf(img + e.s + 4 * (uint64_t)j), bias = rdf(img + e.b + 4 * (uint64_t)j);
            y[j] = bias + (rs * a) * (float)acc;
            bound[j] = 0;
        }
        free(q);
    } else {
        for (uint32_t j = 0; j < e.out; j++) {
            double acc = rdf(img + e.b + 4 * (uint64_t)j), mag = fabs(acc);
            for (uint32_t k = 0; k < e.in; k++) { double t = stored_weight(img, e, j, k) * x[k]; acc += t; mag += fabs(t); }
            y[j] = (float)acc;
            bound[j] = 1e-6 * mag + 1e-30;
        }
    }
}

/* One layer (no activation) per size: the engine against the image, with rows of very different
   magnitudes (per-row scales), a zero row, and an all-zero sample. Covers every vector tail. */
static void model_kernels(PrecisionMode p) {
    static const uint32_t sizes[] = {1, 3, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257, 300, 513, 784, 1031};
    uint32_t N = 6, worst_exact = 0; double worst = 0; lcg_state = 99 + p;
    for (size_t t = 0; t < sizeof sizes / sizeof *sizes; t++) {
        uint32_t K = sizes[t];
        NeuralNetwork *net = new_spingalett(.loss_func = LOSS_MSE);
        layer(net, K); layer(net, N, ACT_NONE);
        for (uint32_t j = 0; j < N; j++)
            for (uint32_t k = 0; k < K; k++)
                net->weights[(size_t)j * K + k] = j == 2 ? 0.0f : (frand() * 2 - 1) * powf(10.0f, (float)j - 2.0f);
        for (uint32_t j = 0; j < N; j++) net->biases[j] = frand() - 0.5f;
        SpingalettModel *m = spingalett_model_from_network(net, p);
        size_t size = 0;
        uint8_t *img = spingalett_save_to_memory(net, p, false, &size);
        CHECK(m && img && size == m->image_size && !memcmp(img, m->image, size), "kernels p=%d K=%u: model image", p, K);
        if (!m || !img) { free_network(net); spingalett_model_free(m); spingalett_free(img); continue; }
        void *ws = malloc(m->workspace_size);           /* exact size: ASan catches overruns */
        float *x = malloc(K * sizeof(float)), y[6], r[6]; double bound[6];
        for (int s = 0; s < 3; s++) {
            for (uint32_t k = 0; k < K; k++) x[k] = s == 2 ? 0.0f : (frand() * 2 - 1) * (s ? 0.01f : 3.0f);
            int rc = spingalett_model_run(m, x, y, ws);
            reference_layer(img, image_layer(img, 0), x, r, bound);
            for (uint32_t j = 0; j < N; j++) {
                double d = fabs((double)y[j] - r[j]);
                if (bound[j] == 0) { if (d != 0) worst_exact++; }
                else if (d / bound[j] > worst) worst = d / bound[j];
            }
            CHECK(rc == SPINGALETT_OK, "kernels p=%d K=%u: run failed", p, K);
        }
        free(ws); free(x); spingalett_free(img); spingalett_model_free(m); free_network(net);
    }
    printf("  kernels precision=%d: %s\n", p, p >= PRECISION_INT8 ? (worst_exact ? "MISMATCH" : "bit-exact") : (worst <= 1 ? "within rounding" : "OFF"));
    CHECK(worst_exact == 0 && worst <= 1.0, "kernels precision %d: %u inexact integer outputs, float error %.2f x bound", p, worst_exact, worst);
}

static NeuralNetwork *deploy_net(void) {
    L ls[] = {{64, ACT_NONE}, {48, ACT_RELU}, {37, ACT_TANH}, {29, ACT_LEAKY_RELU}, {10, ACT_SOFTMAX}};
    lcg_state = 2024;
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    for (int i = 0; i < 5; i++) layer(.net = net, .neurons_amount = ls[i].n, .act_func = ls[i].act);
    for (uint64_t i = 0; i < net->total_weights; i++) net->weights[i] = (frand() * 2 - 1) * 0.35f;
    for (uint64_t i = 0; i < net->total_biases; i++) net->biases[i] = frand() * 0.2f - 0.1f;
    return net;
}

/* Quantized models against the float network, and batched predict against single runs. */
static void model_inference(PrecisionMode p, float tol) {
    NeuralNetwork *net = deploy_net();
    uint32_t N = 300, in = 64, out = 10;                 /* two predict chunks */
    float *x = malloc((size_t)N * in * sizeof(float)), *ref = malloc((size_t)N * out * sizeof(float));
    float *one = malloc((size_t)N * out * sizeof(float)), *batch = malloc((size_t)N * out * sizeof(float));
    for (size_t i = 0; i < (size_t)N * in; i++) x[i] = frand() * 2 - 1;
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    predict(.net = net, .inputs = x, .sample_count = N, .outputs = ref);

    SpingalettModel *m = spingalett_model_from_network(net, p);
    CHECK(m && m->input_size == in && m->output_size == out && m->layer_count == 4 && m->loss == LOSS_CROSS_ENTROPY,
          "inference p=%d: model fields", p);
    if (!m) { free(x); free(ref); free(one); free(batch); free_network(net); return; }
    SpingalettLayerInfo li;
    CHECK(spingalett_model_layer(m, 1, &li) && li.inputs == 48 && li.outputs == 37 && li.activation == ACT_TANH &&
          li.precision == p && !spingalett_model_layer(m, 4, &li), "inference p=%d: layer info", p);

    void *ws = malloc(m->workspace_size);
    for (uint32_t s = 0; s < N; s++) spingalett_model_run(m, x + (size_t)s * in, one + (size_t)s * out, ws);
    free(ws);
    float qerr = max_abs_diff(ref, one, (size_t)N * out);
    uint32_t agree = 0;
    for (uint32_t s = 0; s < N; s++) {
        uint32_t a = 0, b = 0;
        for (uint32_t k = 1; k < out; k++) {
            if (ref[s * out + k] > ref[s * out + a]) a = k;
            if (one[s * out + k] > one[s * out + b]) b = k;
        }
        agree += a == b;
    }

    ComputeMode cm[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS};
    float worst = 0;
    for (int c = 0; c < 3; c++) {
        spingalett_set_compute_mode(cm[c]);
        memset(batch, 0, (size_t)N * out * sizeof(float));
        CHECK(spingalett_model_predict(m, x, N, batch), "inference p=%d: predict failed", p);
        float d = max_abs_diff(one, batch, (size_t)N * out);
        if (d > worst) worst = d;
        /* a few samples at a time: one, fewer than four, fewer than a tile of the batched kernels,
           one past a tile */
        static const uint32_t few[] = {1, 3, 5, 11, 13};
        for (size_t f = 0; f < sizeof few / sizeof *few; f++) {
            CHECK(spingalett_model_predict(m, x + (size_t)7 * in, few[f], batch), "inference p=%d: predict failed", p);
            d = max_abs_diff(one + (size_t)7 * out, batch, (size_t)few[f] * out);
            if (d > worst) worst = d;
        }
    }
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    bool integer = p >= PRECISION_INT8;
    printf("  inference precision=%d: max|q - fp32| = %.2e, argmax agrees %u/%u, |predict - run| = %.1e\n", p, qerr, agree, N, worst);
    /* random weights give near ties, so argmax agreement is only a sanity check here; Examples/MNIST.c
       measures accuracy on real data */
    uint32_t min_agree = p == PRECISION_INT2 ? 30u : (p == PRECISION_INT4 ? 80u : 97u);
    CHECK(qerr <= tol && agree * 100 >= N * min_agree, "inference p=%d: error %.3e, agreement %u/%u", p, qerr, agree, N);
    CHECK(integer ? worst == 0.0f : worst < 1e-5f, "inference p=%d: predict differs from run by %.3e", p, worst);

    EvalMetrics e1 = evaluate(.net = net, .inputs = x, .targets = ref, .sample_count = N);
    EvalMetrics e2 = spingalett_model_evaluate(m, x, ref, N);
    if (p == PRECISION_FLOAT32)
        CHECK(fabsf(e1.loss - e2.loss) < 1e-5f && e1.accuracy == e2.accuracy, "inference: model_evaluate %.6f/%.3f vs %.6f/%.3f",
              e2.loss, e2.accuracy, e1.loss, e1.accuracy);
    else
        CHECK(isfinite(e2.loss) && fabsf(e2.accuracy - (float)agree / N) < 1e-6f, "inference p=%d: model_evaluate", p);

    spingalett_model_free(m);
    free(x); free(ref); free(one); free(batch); free_network(net);
}

/* The validator must reject every damaged image without reading out of bounds. */
#if !defined(_WIN32)
#include <pthread.h>

typedef struct {
    const SpingalettModel *model;
    const float *x;
    float *const *expect;               /* outputs for each count in counts */
    const uint32_t *counts;
    uint32_t sizes, seed;
    bool same;
} PredictJob;

static void *predict_job(void *arg) {
    PredictJob *j = (PredictJob *)arg;
    float *y = malloc((size_t)40 * j->model->output_size * sizeof(float));
    j->same = y != NULL;
    for (uint32_t r = 0; r < 40 && j->same; r++) {
        uint32_t k = (r * 5 + j->seed) % j->sizes, n = j->counts[k];
        j->same = spingalett_model_predict(j->model, j->x, n, y) &&
                  !memcmp(y, j->expect[k], (size_t)n * j->model->output_size * sizeof(float));
    }
    free(y);
    return NULL;
}
#endif

/* Models prepare their weights once and keep a workspace between calls: results are the same for
   repeated calls of any size, for a model that owns nothing (spingalett_model_init over the same
   image, which prepares on every call), and for calls from several threads at once. */
static void model_shared(void) {
    spingalett_seed(41);
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(.net = net, .height = 8, .width = 8, .channels = 3);
    conv2d(.net = net, .filters = 8, .kernel = 3, .padding = 1, .act_func = ACT_RELU, .weight_initialization = WEIGHT_INITIALIZATION_HE);
    conv2d(.net = net, .filters = 8, .kernel = 3, .padding = 1, .groups = 8, .act_func = ACT_RELU, .weight_initialization = WEIGHT_INITIALIZATION_HE);
    max_pool2d(.net = net, .kernel = 2);
    layer(net, 24, ACT_RELU, WEIGHT_INITIALIZATION_HE);
    layer(net, 5, ACT_SOFTMAX, WEIGHT_INITIALIZATION_XAVIER);
    static const uint32_t counts[] = {1, 2, 5, 12, 13, 40};
    enum { SIZES = sizeof counts / sizeof *counts };
    float *x = malloc((size_t)40 * 192 * sizeof(float));
    lcg_state = 9;
    for (size_t i = 0; i < (size_t)40 * 192; i++) x[i] = frand();
    ComputeMode mode = spingalett_get_compute_mode();
    spingalett_set_compute_mode(COMPUTE_OPENMP);
    for (int p = 0; p < PRECISION_COUNT; p++) {
        SpingalettModel *m = spingalett_model_from_network(net, (PrecisionMode)p);
        float *expect[SIZES], *y = malloc((size_t)40 * 5 * sizeof(float));
        bool same = m && y;
        for (int k = 0; k < SIZES; k++) {
            expect[k] = malloc((size_t)counts[k] * 5 * sizeof(float));
            same = same && expect[k] && spingalett_model_predict(m, x, counts[k], expect[k]);
        }
        SpingalettModel bare;
        same = same && spingalett_model_init(&bare, m->image, m->image_size) == SPINGALETT_OK;
        for (int k = SIZES; k-- > 0 && same;) {      /* in the other order, on both models */
            same = spingalett_model_predict(m, x, counts[k], y) && !memcmp(y, expect[k], (size_t)counts[k] * 5 * sizeof(float));
            same = same && spingalett_model_predict(&bare, x, counts[k], y) &&
                   !memcmp(y, expect[k], (size_t)counts[k] * 5 * sizeof(float));
        }
        CHECK(same, "model p=%d: repeated predictions or a model without an owner differ", p);
#if !defined(_WIN32)
        PredictJob jobs[4];
        pthread_t threads[4];
        int started = 0;
        for (int t = 0; t < 4 && same; t++) {
            jobs[t] = (PredictJob){m, x, expect, counts, SIZES, (uint32_t)t * 3u, false};
            if (pthread_create(&threads[t], NULL, predict_job, &jobs[t]) == 0) started++;
        }
        bool threaded = started == 4;
        for (int t = 0; t < started; t++) {
            pthread_join(threads[t], NULL);
            threaded = threaded && jobs[t].same;
        }
        CHECK(!same || threaded, "model p=%d: predictions from several threads differ", p);
#endif
        for (int k = 0; k < SIZES; k++) free(expect[k]);
        free(y);
        spingalett_model_free(m);
    }
    spingalett_set_compute_mode(mode);
    free(x);
    free_network(net);
    printf("  shared models: repeated, unowned and concurrent predictions agree\n");
}

static void model_validation(void) {
    NeuralNetwork *net = deploy_net();
    size_t size = 0;
    uint8_t *img = spingalett_save_to_memory(net, PRECISION_INT8, true, &size);
    uint8_t *buf = malloc(size + 8);
    SpingalettModel m;
    int failures_before = failures;

    CHECK(spingalett_model_init(&m, img, size) == SPINGALETT_OK, "validation: intact image rejected");
    CHECK(spingalett_model_init(&m, img, size + 4096) == SPINGALETT_OK && m.image_size == size, "validation: larger region");
    for (size_t i = 0; i < size; i++) {                         /* every truncation */
        memcpy(buf, img, i);
        int rc = spingalett_model_init(&m, buf, i);
        if (rc == SPINGALETT_OK) { CHECK(0, "validation: image truncated to %zu bytes accepted", i); break; }
    }
    for (size_t i = 0; i < size; i += (i < 64 + 4 * 48 ? 1 : 97)) {   /* flipped bits */
        memcpy(buf, img, size);
        buf[i] ^= (uint8_t)(1u << (i % 8));
        if (spingalett_model_init(&m, buf, size) == SPINGALETT_OK) { CHECK(0, "validation: flipped byte %zu accepted", i); break; }
    }
    memcpy(buf + 1, img, size);
    CHECK(spingalett_model_init(&m, buf + 1, size) == SPINGALETT_ERR_INVALID, "validation: misaligned image accepted");

    /* inconsistent contents with valid checksums */
    struct { size_t at; uint32_t value; const char *what; } bad[] = {
        {64 + 48 + 0, 47, "inputs differ from the previous layer's outputs"},
        {64 + 4, 0, "zero outputs"},
        {64 + 16, 3, "misaligned weights"},
        {64 + 32, 0xFFFFFF00u, "biases beyond the image"},
        {64 + 24, 0, "integer layer without scales"},
        {8, 1, "single layer"},
        {8, 7, "more layers than entries"},
        {64 + 8, ACT_COUNT, "unknown activation"},
        {64 + 9, PRECISION_COUNT, "unknown precision"},
        {64 + 12, 0x3F800000u, "dropout rate 1"},
        {24, 63, "file size below the header"},
    };
    for (size_t t = 0; t < sizeof bad / sizeof *bad; t++) {
        memcpy(buf, img, size);
        if (bad[t].at == 64 + 8 || bad[t].at == 64 + 9) buf[bad[t].at] = (uint8_t)bad[t].value;
        else wr32(buf + bad[t].at, bad[t].value);
        if (bad[t].at == 24) wr32(buf + 28, 0);
        reseal(buf, size);
        int rc = spingalett_model_init(&m, buf, size);
        CHECK(rc != SPINGALETT_OK, "validation: %s accepted", bad[t].what);
    }

    memcpy(buf, img, size);
    buf[6] = 7;                                                  /* a future format version */
    reseal(buf, size);
    CHECK(spingalett_model_init(&m, buf, size) == SPINGALETT_ERR_FORMAT_VERSION, "validation: future version");
    CHECK(spingalett_model_init(NULL, img, size) == SPINGALETT_ERR_INVALID && spingalett_model_init(&m, NULL, size) == SPINGALETT_ERR_INVALID,
          "validation: NULL arguments");
    CHECK(load_spingalett_from_memory(buf, size) == NULL && spingalett_last_error_code() == SPINGALETT_ERR_FORMAT_VERSION,
          "validation: load of a future version");

    /* the full library's loaders accept unaligned copies */
    memcpy(buf + 1, img, size);
    NeuralNetwork *b = load_spingalett_from_memory(buf + 1, size);
    SpingalettModel *owned = spingalett_model_from_memory(buf + 1, size);
    CHECK(b && owned, "validation: unaligned copy not loaded");
    spingalett_model_free(owned);
    if (b) free_network(b);

    CHECK(spingalett_save_to_memory(NULL, PRECISION_INT8, false, &size) == NULL &&
          spingalett_save_to_memory(net, PRECISION_COUNT, false, &size) == NULL && size == 0, "validation: save arguments");
    NeuralNetwork *wide = new_spingalett(.loss_func = LOSS_MSE);
    layer(wide, 131073); layer(wide, 1, ACT_NONE);
    CHECK(spingalett_model_from_network(wide, PRECISION_INT8) == NULL && spingalett_last_error_code() == SPINGALETT_ERR_INVALID,
          "validation: integer layer with 131073 inputs");
    SpingalettModel *f32 = spingalett_model_from_network(wide, PRECISION_FLOAT32);
    CHECK(f32 != NULL, "validation: float layer with 131073 inputs");
    spingalett_model_free(f32);
    free_network(wide);

    printf("  model validation %s\n", failures == failures_before ? "checked" : "FAILED");
    free(buf); spingalett_free(img); free_network(net);
}

/* Optimizer state, legacy files, NaN inputs and C header export. */
static void model_files(void) {
    float x[16 * 6], y[16 * 3];
    lcg_state = 31;
    for (int i = 0; i < 16 * 6; i++) x[i] = frand() * 2 - 1;
    for (int i = 0; i < 16 * 3; i++) y[i] = frand();
    L ls[] = {{6, ACT_NONE}, {9, ACT_TANH}, {3, ACT_SIGMOID}};
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    NeuralNetwork *a = build(LOSS_MSE, ls, 3, NULL, NULL);
    train(.net = a, .inputs = x, .targets = y, .sample_count = 16, .epochs = 5, .optimizer_type = OPTIMIZER_ADAM, .learning_rate = 0.01f, .training_strategy = STRATEGY_FULL_BATCH);
    size_t size = 0;
    void *img = spingalett_save_to_memory(a, PRECISION_FLOAT32, true, &size);
    NeuralNetwork *b = load_spingalett_from_memory(img, size);
    CHECK(b && b->time_step == a->time_step, "files: optimizer state not restored");
    if (b) {
        train(.net = a, .inputs = x, .targets = y, .sample_count = 16, .epochs = 5, .optimizer_type = OPTIMIZER_ADAM, .learning_rate = 0.01f, .training_strategy = STRATEGY_FULL_BATCH);
        train(.net = b, .inputs = x, .targets = y, .sample_count = 16, .epochs = 5, .optimizer_type = OPTIMIZER_ADAM, .learning_rate = 0.01f, .training_strategy = STRATEGY_FULL_BATCH);
        float d = max_abs_diff(a->weights, b->weights, a->total_weights);
        printf("  files: training resumed from memory differs by %.1e\n", d);
        CHECK(d == 0.0f, "files: resumed training differs by %.3e", d);
        free_network(b);
    }
    spingalett_free(img);

    /* a version 1 file runs as a model in its own precision */
    NeuralNetwork *v1 = load_spingalett(SPINGALETT_TEST_DATA_DIR "/xor_v1.nn");
    SpingalettModel *mv1 = spingalett_model_load(SPINGALETT_TEST_DATA_DIR "/xor_v1.nn");
    CHECK(v1 && mv1, "files: xor_v1.nn as a model");
    if (v1 && mv1) {
        float in[2] = {1, 0}, out[1];
        float *f = forward(.net = v1, .input = in);
        CHECK(spingalett_model_run(mv1, in, out, NULL) == SPINGALETT_OK && fabsf(out[0] - f[0]) < 1e-6f,
              "files: xor_v1 model output %.6f vs %.6f", out[0], f[0]);
    }
    spingalett_model_free(mv1);
    if (v1) free_network(v1);

    /* NaN in, NaN out, in every precision */
    for (int p = 0; p < PRECISION_COUNT; p++) {
        SpingalettModel *m = spingalett_model_from_network(a, (PrecisionMode)p);
        float in[6] = {0.1f, NAN, 0.3f, 0, 0, 0}, out[3];
        CHECK(m && spingalett_model_run(m, in, out, NULL) == SPINGALETT_OK && isnan(out[0]), "files: NaN input, precision %d", p);
        spingalett_model_free(m);
    }

    /* C header: the array holds the image, the macros the sizes */
    const char *path = "spingalett_test_model.h";
    CHECK(spingalett_export_c_header(a, path, "test_model", PRECISION_INT4), "files: export failed");
    CHECK(!spingalett_export_c_header(a, path, "1model", PRECISION_INT4) && !spingalett_export_c_header(a, path, "a-b", PRECISION_INT4) &&
          !spingalett_export_c_header(a, path, NULL, PRECISION_INT4), "files: invalid names accepted");
    SpingalettModel *m = spingalett_model_from_network(a, PRECISION_INT4);
    FILE *f = fopen(path, "r");
    char line[512];
    unsigned long hsize = 0, hin = 0, hout = 0, hws = 0;
    size_t count = 0;
    bool data = false, same = m != NULL;
    while (f && fgets(line, sizeof line, f)) {
        sscanf(line, "#define TEST_MODEL_SIZE %luu", &hsize);
        sscanf(line, "#define TEST_MODEL_INPUTS %luu", &hin);
        sscanf(line, "#define TEST_MODEL_OUTPUTS %luu", &hout);
        sscanf(line, "#define TEST_MODEL_WORKSPACE %luu", &hws);
        if (strstr(line, "static const uint8_t test_model[TEST_MODEL_SIZE] = {")) { data = true; continue; }
        if (!data) continue;
        if (line[0] == '}') break;
        for (char *c = line; (c = strstr(c, "0x")) != NULL; c += 4) {
            unsigned v;
            sscanf(c, "0x%2x", &v);
            if (!m || count >= m->image_size || ((const uint8_t *)m->image)[count] != v) same = false;
            count++;
        }
    }
    if (f) fclose(f);
    remove(path);
    CHECK(m && same && count == m->image_size && hsize == m->image_size && hin == 6 && hout == 3 && hws == m->workspace_size,
          "files: header contents (%zu bytes, size %lu, in %lu, out %lu, workspace %lu)", count, hsize, hin, hout, hws);
    printf("  files: optimizer state, version 1 model, NaN and C header checked\n");
    spingalett_model_free(m);
    free_network(a);
}


static NeuralNetwork *conv_deploy_net(void);

/* SpingalettTests export-headers DIR: the deploy_net model as C headers in three precisions, the
   conv_deploy_net model in two, a network with batch normalization in INT8, and the outputs the
   library computes for a few inputs, for Spingalett.EngineTests.c. */
static int export_test_headers(const char *dir) {
    NeuralNetwork *net = deploy_net();
    const PrecisionMode precisions[] = {PRECISION_INT8, PRECISION_INT4, PRECISION_FP16};
    const char *names[] = {"test_model_int8", "test_model_int4", "test_model_fp16"}, *suffix[] = {"int8", "int4", "fp16"};
    enum { SAMPLES = 8 };
    float x[SAMPLES * 64];
    lcg_state = 77;
    for (int i = 0; i < SAMPLES * 64; i++) x[i] = frand() * 4 - 2;
    char path[1024];
    snprintf(path, sizeof path, "%s/test_model_expected.h", dir);
    FILE *f = fopen(path, "w");
    if (!f) return 1;
    fprintf(f, "/* Generated by SpingalettTests export-headers. */\n#define TEST_SAMPLES %d\nstatic const float test_inputs[%d][64] = {\n", SAMPLES, SAMPLES);
    for (int i = 0; i < SAMPLES * 64; i++) fprintf(f, "%s%.9g,%s", i % 64 ? "" : "    {", (double)x[i], i % 64 == 63 ? "},\n" : " ");
    fprintf(f, "};\n");
    int rc = 0;
    for (int p = 0; p < 3; p++) {
        snprintf(path, sizeof path, "%s/%s.h", dir, names[p]);
        if (!spingalett_export_c_header(net, path, names[p], precisions[p])) rc = 1;
        SpingalettModel *m = spingalett_model_from_network(net, precisions[p]);
        float y[SAMPLES * 10];
        for (int s = 0; s < SAMPLES && m; s++) spingalett_model_run(m, x + s * 64, y + s * 10, NULL);
        fprintf(f, "static const float expected_%s[%d] = {\n", suffix[p], SAMPLES * 10);
        for (int i = 0; i < SAMPLES * 10; i++) fprintf(f, "%s%.9g,%s", i % 10 ? "" : "    ", (double)y[i], i % 10 == 9 ? "\n" : " ");
        fprintf(f, "};\n");
        if (!m) rc = 1;
        spingalett_model_free(m);
    }
    /* a convolution network, as INT8 and FLOAT32 */
    NeuralNetwork *conv = conv_deploy_net();
    const uint32_t cin = conv->topology[0], cout = conv->topology[conv->layers - 1];
    float *cx = malloc((size_t)SAMPLES * cin * sizeof(float)), *cy = malloc((size_t)SAMPLES * cout * sizeof(float));
    for (uint32_t i = 0; i < SAMPLES * cin; i++) cx[i] = frand() * 2 - 1;
    fprintf(f, "#define TEST_CONV_INPUTS %u\nstatic const float test_conv_inputs[%d][%u] = {\n", cin, SAMPLES, cin);
    for (uint32_t i = 0; i < SAMPLES * cin; i++)
        fprintf(f, "%s%.9g,%s", i % cin ? "" : "    {", (double)cx[i], i % cin == cin - 1 ? "},\n" : " ");
    fprintf(f, "};\n");
    const PrecisionMode conv_precisions[] = {PRECISION_INT8, PRECISION_FLOAT32};
    const char *conv_names[] = {"test_model_conv_int8", "test_model_conv_f32"}, *conv_suffix[] = {"conv_int8", "conv_f32"};
    for (int p = 0; p < 2; p++) {
        snprintf(path, sizeof path, "%s/%s.h", dir, conv_names[p]);
        if (!spingalett_export_c_header(conv, path, conv_names[p], conv_precisions[p])) rc = 1;
        SpingalettModel *m = spingalett_model_from_network(conv, conv_precisions[p]);
        for (int s = 0; s < SAMPLES && m; s++) spingalett_model_run(m, cx + (size_t)s * cin, cy + (size_t)s * cout, NULL);
        fprintf(f, "static const float expected_%s[%u] = {\n", conv_suffix[p], SAMPLES * cout);
        for (uint32_t i = 0; i < SAMPLES * cout; i++)
            fprintf(f, "%s%.9g,%s", i % cout ? "" : "    ", (double)cy[i], i % cout == cout - 1 ? "\n" : " ");
        fprintf(f, "};\n");
        if (!m) rc = 1;
        spingalett_model_free(m);
    }
    /* batch normalization: of the input and after pooling (kept, format 5), after a conv (folded) */
    NeuralNetwork *norm = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(.net = norm, .height = 9, .width = 7, .channels = 3);
    batch_norm(.net = norm);
    conv2d(.net = norm, .filters = 6, .kernel = 3, .padding = 1, .act_func = ACT_NONE);
    batch_norm(.net = norm, .act_func = ACT_RELU);
    max_pool2d(.net = norm, .kernel = 2);
    batch_norm(.net = norm, .act_func = ACT_TANH, .epsilon = 1e-3f);
    layer(.net = norm, .neurons_amount = 4, .act_func = ACT_SOFTMAX);
    for (uint64_t i = 0; i < norm->total_weights; i++) norm->weights[i] = (frand() * 2 - 1) * 0.5f + 0.5f;
    for (uint64_t i = 0; i < norm->total_biases; i++) {
        norm->biases[i] = frand() * 0.2f - 0.1f;
        norm->running_mean[i] = frand() * 0.4f - 0.2f;
        norm->running_var[i] = 0.5f + frand();
    }
    const uint32_t nout = norm->topology[norm->layers - 1];
    snprintf(path, sizeof path, "%s/test_model_norm_int8.h", dir);
    if (!spingalett_export_c_header(norm, path, "test_model_norm_int8", PRECISION_INT8)) rc = 1;
    SpingalettModel *nm = spingalett_model_from_network(norm, PRECISION_INT8);
    for (int s = 0; s < SAMPLES && nm; s++) spingalett_model_run(nm, cx + (size_t)s * cin, cy + (size_t)s * nout, NULL);
    fprintf(f, "static const float expected_norm_int8[%u] = {\n", SAMPLES * nout);
    for (uint32_t i = 0; i < SAMPLES * nout; i++)
        fprintf(f, "%s%.9g,%s", i % nout ? "" : "    ", (double)cy[i], i % nout == nout - 1 ? "\n" : " ");
    fprintf(f, "};\n");
    if (!nm || rd16((const uint8_t *)nm->image + 6) != 5) rc = 1;
    spingalett_model_free(nm);
    free_network(norm);

    free(cx); free(cy);
    free_network(conv);
    fclose(f);
    free_network(net);
    return rc;
}

/* ---------------------------------------------------------------- convolution models (format 4) */


/* A version 4 or 5 layer table entry with its input shape (header or previous entry); groups is 1
   for version 4 convolutions. */
typedef struct { uint32_t in, out; int act, prec, type; uint64_t w, s, b, o;
                 uint32_t ih, iw, ic, oh, ow, oc, kh, kw, sh, sw, ph, pw, groups; } Entry4;
static Entry4 image_layer4(const uint8_t *img, uint32_t i) {
    const size_t size = rd16(img + 6) >= 5 ? 80 : 64;
    const uint8_t *e = img + 64 + size * i;
    Entry4 r = {rd32(e), rd32(e + 4), e[8], e[9], e[10], rd64(e + 16), rd64(e + 24), rd64(e + 32), rd64(e + 40),
                0, 0, 0, rd16(e + 48), rd16(e + 50), 0, rd16(e + 52), rd16(e + 54), rd16(e + 56), rd16(e + 58),
                rd16(e + 60), rd16(e + 62), size == 80 ? rd32(e + 64) : e[10] == LAYER_CONV2D};
    r.ih = i ? rd16(e - size + 48) : rd32(img + 32);
    r.iw = i ? rd16(e - size + 50) : rd32(img + 36);
    r.ic = r.in / (r.ih * r.iw);
    r.oc = r.out / (r.oh * r.ow);
    return r;
}

/* Weight k of filter j of a stored convolution, decoded as docs/ModelFormat.md describes. */
static double conv_weight(const uint8_t *img, Entry4 e, uint32_t j, uint32_t k) {
    Entry as_dense = {e.kh * e.kw * (e.ic / e.groups), e.oc, e.act, e.prec, e.w, e.s, e.b, e.o};
    return stored_weight(img, as_dense, j, k);
}

/* What the engine computes for a convolution without activation, from the stored image (as
   reference_layer: integer layers bit for bit, float ones within `bound`). */
static void reference_conv(const uint8_t *img, Entry4 e, const float *x, float *y, double *bound) {
    bool integer = e.prec >= PRECISION_INT8;
    int8_t *q = malloc(e.in);
    float a = 0;
    if (integer) {
        float amax = 0;
        for (uint32_t k = 0; k < e.in; k++) if (fabsf(x[k]) > amax) amax = fabsf(x[k]);
        a = amax / 127.0f;
        float inv = amax > 0 ? 127.0f / amax : 0;
        for (uint32_t k = 0; k < e.in; k++) q[k] = (int8_t)lrintf(x[k] * inv);
    }
    const uint32_t cg = e.ic / e.groups, og = e.oc / e.groups;      /* channels and filters per group */
    Entry as_dense = {e.kh * e.kw * cg, e.oc, e.act, e.prec, e.w, e.s, e.b, e.o};
    for (uint32_t oh = 0; oh < e.oh; oh++)
        for (uint32_t ow = 0; ow < e.ow; ow++)
            for (uint32_t j = 0; j < e.oc; j++) {
                int64_t acc = 0;
                double facc = rdf(img + e.b + 4 * (uint64_t)j), mag = fabs(facc);
                for (uint32_t kh = 0; kh < e.kh; kh++)
                    for (uint32_t kw = 0; kw < e.kw; kw++) {
                        int64_t ih = (int64_t)oh * e.sh - e.ph + kh, iw = (int64_t)ow * e.sw - e.pw + kw;
                        if (ih < 0 || iw < 0 || ih >= e.ih || iw >= e.iw) continue;
                        for (uint32_t c = 0; c < cg; c++) {
                            uint32_t k = (kh * e.kw + kw) * cg + c;
                            size_t xi = ((size_t)ih * e.iw + (size_t)iw) * e.ic + (j / og) * cg + c;
                            if (integer) acc += (int64_t)stored_code(img, as_dense, j, k) * q[xi];
                            else { double t = conv_weight(img, e, j, k) * x[xi]; facc += t; mag += fabs(t); }
                        }
                    }
                size_t o = ((size_t)oh * e.ow + ow) * e.oc + j;
                if (integer) {
                    y[o] = rdf(img + e.b + 4 * (uint64_t)j) + (rdf(img + e.s + 4 * (uint64_t)j) * a) * (float)acc;
                    bound[o] = 0;
                } else {
                    y[o] = (float)facc;
                    bound[o] = 1e-6 * mag + 1e-30;
                }
            }
    free(q);
}

/* One convolution per window geometry and precision: the engine against the stored image (filters
   not a multiple of four, padding, strides, rectangular and pointwise windows, windows shorter and
   longer than SLETT_COLUMN_WINDOW, zero inputs). */
static void model_conv_kernels(PrecisionMode p) {
    static const struct { uint32_t h, w, c, f, kh, kw, sh, sw, ph, pw, groups; } g[] = {
        {6, 5, 4, 9, 3, 3, 2, 2, 1, 1, 1}, {7, 7, 1, 13, 3, 3, 1, 1, 1, 1, 1}, {5, 8, 3, 4, 2, 3, 1, 2, 0, 1, 1},
        {4, 4, 33, 6, 1, 1, 1, 1, 0, 0, 1}, {9, 3, 17, 5, 4, 2, 3, 1, 2, 1, 1}, {3, 3, 70, 3, 3, 3, 1, 1, 0, 0, 1},
        {10, 10, 2, 40, 2, 2, 2, 2, 0, 0, 1}, {6, 6, 3, 16, 3, 3, 1, 1, 1, 1, 1}, {5, 5, 1, 64, 5, 5, 1, 1, 2, 2, 1},
        /* depthwise (multipliers 1 and 2), grouped, grouped pointwise */
        {6, 5, 20, 20, 3, 3, 1, 1, 1, 1, 20}, {7, 6, 3, 6, 3, 3, 2, 2, 1, 1, 3}, {9, 9, 16, 16, 5, 5, 2, 2, 2, 2, 16},
        {5, 8, 4, 6, 2, 3, 1, 2, 0, 1, 2}, {4, 4, 32, 64, 1, 1, 1, 1, 0, 0, 8}, {6, 6, 48, 24, 3, 3, 1, 1, 1, 1, 4},
    };
    uint32_t inexact = 0; double worst = 0; lcg_state = 515 + p;
    for (size_t t = 0; t < sizeof g / sizeof *g; t++) {
        NeuralNetwork *net = new_spingalett(.loss_func = LOSS_MSE);
        layer(.net = net, .height = g[t].h, .width = g[t].w, .channels = g[t].c);
        conv2d(.net = net, .filters = g[t].f, .kernel_h = g[t].kh, .kernel_w = g[t].kw, .stride_h = g[t].sh,
               .stride_w = g[t].sw, .padding_h = g[t].ph, .padding_w = g[t].pw, .groups = g[t].groups,
               .act_func = ACT_NONE);
        for (uint64_t i = 0; i < net->total_weights; i++) net->weights[i] = (frand() * 2 - 1) * (i % 7 == 0 ? 2.0f : 0.3f);
        for (uint64_t i = 0; i < net->total_biases; i++) net->biases[i] = frand() - 0.5f;
        SpingalettModel *m = spingalett_model_from_network(net, p);
        CHECK(m != NULL, "conv kernels p=%d geometry %zu: no model", p, t);
        if (!m) { free_network(net); continue; }
        const uint8_t *img = m->image;
        Entry4 e = image_layer4(img, 0);
        uint32_t in = net->topology[0], out = net->topology[1];
        void *ws = malloc(m->workspace_size);
        float *x = malloc(in * sizeof(float)), *y = malloc(out * sizeof(float)), *r = malloc(out * sizeof(float));
        double *bound = malloc(out * sizeof(double));
        for (int s = 0; s < 3; s++) {
            /* wide values, small ReLU-like values (half zero: skipped by short windows), all zero */
            for (uint32_t k = 0; k < in; k++) x[k] = s == 2 ? 0.0f : s ? fmaxf(0.0f, frand() * 2 - 1) * 0.05f : (frand() * 2 - 1) * 2.0f;
            CHECK(spingalett_model_run(m, x, y, ws) == SPINGALETT_OK, "conv kernels p=%d: run failed", p);
            reference_conv(img, e, x, r, bound);
            for (uint32_t j = 0; j < out; j++) {
                double d = fabs((double)y[j] - r[j]);
                if (bound[j] == 0) { if (d != 0) inexact++; }
                else if (d / bound[j] > worst) worst = d / bound[j];
            }
            /* batched prediction (two copies of the sample) computes what a run computes */
            float *xx = malloc(2 * (size_t)in * sizeof(float)), *yy = malloc(2 * (size_t)out * sizeof(float));
            memcpy(xx, x, in * sizeof(float)); memcpy(xx + in, x, in * sizeof(float));
            CHECK(spingalett_model_predict(m, xx, 2, yy), "conv kernels p=%d: predict failed", p);
            float pd = fmaxf(max_abs_diff(yy, y, out), max_abs_diff(yy + out, y, out));
            CHECK(p >= PRECISION_INT8 ? pd == 0.0f : pd < 1e-4f, "conv kernels p=%d geometry %zu: predict differs from run by %g",
                  p, t, pd);
            free(xx); free(yy);
        }
        free(ws); free(x); free(y); free(r); free(bound);
        spingalett_model_free(m); free_network(net);
    }
    printf("  conv kernels precision=%d: %s\n", p, p >= PRECISION_INT8 ? (inexact ? "MISMATCH" : "bit-exact") : (worst <= 1 ? "within rounding" : "OFF"));
    CHECK(inexact == 0 && worst <= 1.0, "conv kernels precision %d: %u inexact integer outputs, float error %.2f x bound", p, inexact, worst);
}

/* Every layer kind: same and strided convolutions, rectangular windows, overlapping padded average
   pooling, a pointwise convolution and a dense head. */
static NeuralNetwork *conv_deploy_net(void) {
    lcg_state = 4242;
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(.net = net, .height = 9, .width = 7, .channels = 3);
    conv2d(.net = net, .filters = 6, .kernel = 3, .padding = 1, .act_func = ACT_RELU);
    max_pool2d(.net = net, .kernel = 2);
    conv2d(.net = net, .filters = 5, .kernel_h = 2, .kernel_w = 3, .stride_w = 2, .padding_w = 1, .act_func = ACT_TANH,
           .dropout_rate = 0.25f);
    avg_pool2d(.net = net, .kernel = 2, .stride = 1, .padding = 1);
    conv2d(.net = net, .filters = 7, .kernel = 1, .act_func = ACT_LEAKY_RELU);
    layer(.net = net, .neurons_amount = 4, .act_func = ACT_SOFTMAX);
    for (uint64_t i = 0; i < net->total_weights; i++) net->weights[i] = (frand() * 2 - 1) * 0.5f;
    for (uint64_t i = 0; i < net->total_biases; i++) net->biases[i] = frand() * 0.2f - 0.1f;
    return net;
}

/* Saving and loading convolution networks; models in every precision against the network, batched
   prediction against single runs; damaged version 4 images. */
static void model_conv(void) {
    NeuralNetwork *net = conv_deploy_net();
    uint32_t N = 40, in = net->topology[0], out = 4;

    /* the layer table, and the version: 4 here, 3 for dense networks */
    size_t size = 0, dense_size = 0;
    for (uint64_t i = 0; i < net->total_weights; i++) { net->opt_m_weights[i] = frand(); net->opt_v_weights[i] = frand(); }
    net->time_step = 17;
    uint8_t *img = spingalett_save_to_memory(net, PRECISION_FLOAT32, true, &size);
    NeuralNetwork *dense = deploy_net();
    uint8_t *dimg = spingalett_save_to_memory(dense, PRECISION_FLOAT32, false, &dense_size);
    CHECK(img && dimg && rd16(img + 6) == 4 && rd16(dimg + 6) == 3, "conv model: format versions");
    if (!img || !dimg) { free_network(net); free_network(dense); spingalett_free(img); spingalett_free(dimg); return; }
    Entry4 e1 = image_layer4(img, 0), e2 = image_layer4(img, 1), e3 = image_layer4(img, 2), e6 = image_layer4(img, 5);
    CHECK(rd32(img + 32) == 9 && rd32(img + 36) == 7 && e1.type == LAYER_CONV2D && e1.oh == 9 && e1.ow == 7 && e1.oc == 6 &&
          e1.kh == 3 && e1.ph == 1 && e1.sh == 1 && e2.type == LAYER_MAX_POOL2D && e2.oh == 4 && e2.ow == 3 && e2.w == 0 &&
          e2.b == 0 && e2.act == ACT_NONE && e3.kh == 2 && e3.kw == 3 && e3.sw == 2 && e3.pw == 1 && e3.oh == 3 && e3.ow == 2 &&
          e6.type == LAYER_DENSE && e6.oh == 1 && e6.kh == 0, "conv model: layer table");

    /* round trip: shapes, parameters, optimizer state */
    NeuralNetwork *back = load_spingalett_from_memory(img, size);
    bool same = back && back->layers == net->layers && back->total_weights == net->total_weights && back->time_step == 17;
    for (uint32_t l = 0; same && l < net->layers; l++) {
        SpingalettNetworkLayer a, b;
        same = spingalett_network_layer(net, l, &a) && spingalett_network_layer(back, l, &b) &&
               a.type == b.type && a.height == b.height && a.width == b.width && a.channels == b.channels &&
               a.outputs == b.outputs && a.activation == b.activation && a.dropout_rate == b.dropout_rate &&
               a.kernel_h == b.kernel_h && a.kernel_w == b.kernel_w && a.stride_h == b.stride_h &&
               a.stride_w == b.stride_w && a.padding_h == b.padding_h && a.padding_w == b.padding_w &&
               a.weight_count == b.weight_count && a.bias_count == b.bias_count;
    }
    if (same)
        same = !memcmp(back->weights, net->weights, net->total_weights * 4) && !memcmp(back->biases, net->biases, net->total_biases * 4) &&
               !memcmp(back->opt_m_weights, net->opt_m_weights, net->total_weights * 4) &&
               !memcmp(back->opt_v_weights, net->opt_v_weights, net->total_weights * 4);
    CHECK(same, "conv model: save and load round trip");
    if (back) free_network(back);

    /* every precision: the engine against the network (dequantized as stored), batched prediction
       against single runs */
    float *x = malloc((size_t)N * in * sizeof(float)), *ref = malloc((size_t)N * out * sizeof(float));
    float *one = malloc((size_t)N * out * sizeof(float)), *batch = malloc((size_t)N * out * sizeof(float));
    for (size_t i = 0; i < (size_t)N * in; i++) x[i] = frand() * 2 - 1;
    for (int p = 0; p < PRECISION_COUNT; p++) {
        SpingalettModel *m = spingalett_model_from_network(net, (PrecisionMode)p);
        CHECK(m && m->input_size == in && m->output_size == out && m->layer_count == 6, "conv model p=%d: fields", p);
        if (!m) continue;
        SpingalettLayerInfo li;
        CHECK(spingalett_model_layer(m, 2, &li) && li.type == LAYER_CONV2D && li.height == 3 && li.width == 2 &&
              li.channels == 5 && li.kernel_h == 2 && li.kernel_w == 3 && li.stride_w == 2 && li.padding_w == 1 &&
              li.activation == ACT_TANH && spingalett_model_layer(m, 1, &li) && li.type == LAYER_MAX_POOL2D,
              "conv model p=%d: layer info", p);
        size_t isize = 0;
        uint8_t *pimg = spingalett_save_to_memory(net, (PrecisionMode)p, false, &isize);
        NeuralNetwork *stored = load_spingalett_from_memory(pimg, isize);    /* the weights as stored */
        spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
        predict(.net = stored, .inputs = x, .sample_count = N, .outputs = ref);
        void *ws = malloc(m->workspace_size);
        for (uint32_t s = 0; s < N; s++) spingalett_model_run(m, x + (size_t)s * in, one + (size_t)s * out, ws);
        free(ws);
        float err = max_abs_diff(ref, one, (size_t)N * out), worst = 0;
        ComputeMode cm[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS};
        for (int c = 0; c < 3; c++) {
            spingalett_set_compute_mode(cm[c]);
            memset(batch, 0, (size_t)N * out * sizeof(float));
            CHECK(spingalett_model_predict(m, x, N, batch), "conv model p=%d: predict failed", p);
            float d = max_abs_diff(one, batch, (size_t)N * out);
            if (d > worst) worst = d;
        }
        spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
        bool integer = p >= PRECISION_INT8;
        printf("  conv model precision=%d: |run - network| = %.1e, |predict - run| = %.1e\n", p, err, worst);
        /* float layers: the stored weights in float; integer layers also quantize activations */
        float tol = integer ? (p == PRECISION_INT8 ? 0.05f : 0.5f) : 1e-5f;
        CHECK(err <= tol, "conv model p=%d: engine differs from the network by %.3e", p, err);
        CHECK(integer ? worst == 0.0f : worst < 1e-5f, "conv model p=%d: predict differs from run by %.3e", p, worst);
        spingalett_free(pimg);
        if (stored) free_network(stored);
        spingalett_model_free(m);
    }

    /* damaged version 4 tables with valid checksums */
    uint8_t *buf = malloc(size);
    SpingalettModel m;
    struct { size_t at; int bytes; uint32_t value; const char *what; } bad[] = {
        {64 + 10, 1, LAYER_TYPE_COUNT, "unknown layer type"},
        {64 + 48, 2, 8, "output height that the window does not give"},
        {64 + 52, 2, 0, "kernel height 0"},
        {64 + 56, 2, 0, "stride 0"},
        {64 + 60, 2, 3, "padding as large as the kernel"},
        {64 + 64 + 8, 1, ACT_RELU, "pooling with an activation"},
        {64 + 64 + 16, 4, 64, "pooling with weights"},
        {64 + 64 + 10, 1, LAYER_CONV2D, "pooling read as a convolution"},
        {64 + 5 * 64 + 48, 2, 2, "dense layer with a height"},
        {64 + 5 * 64 + 52, 2, 1, "dense layer with a kernel"},
        {32, 4, 3, "input height that does not divide the inputs"},
        {36, 4, 0, "input width 0"},
    };
    for (size_t t = 0; t < sizeof bad / sizeof *bad; t++) {
        memcpy(buf, img, size);
        for (int k = 0; k < bad[t].bytes; k++) buf[bad[t].at + k] = (uint8_t)(bad[t].value >> (8 * k));
        reseal(buf, size);
        CHECK(spingalett_model_init(&m, buf, size) != SPINGALETT_OK && load_spingalett_from_memory(buf, size) == NULL,
              "conv model validation: %s accepted", bad[t].what);
    }
    for (size_t i = 0; i < size; i++) {                         /* every truncation */
        if (spingalett_model_init(&m, img, i) == SPINGALETT_OK) { CHECK(0, "conv model: truncation to %zu accepted", i); break; }
    }

    /* a header for compiling the model in */
    CHECK(spingalett_export_c_header(net, "spingalett_test_conv.h", "conv_model", PRECISION_INT8), "conv model: export");
    FILE *fp = fopen("spingalett_test_conv.h", "r");
    char line[256]; bool v4 = false;
    while (fp && fgets(line, sizeof line, fp)) if (strstr(line, "format version 4")) v4 = true;
    if (fp) fclose(fp);
    CHECK(v4, "conv model: exported header names format version 4");
    remove("spingalett_test_conv.h");

    free(buf); free(x); free(ref); free(one); free(batch);
    spingalett_free(img); spingalett_free(dimg);
    free_network(net); free_network(dense);
}

/* ---- batch normalization ---- */

/* Networks with batch normalization for the gradient checks: after convolutions and dense layers,
   after the input, after pooling, as the output layer, with every activation. */
static NeuralNetwork *norm_net(int which) {
    NeuralNetwork *net;
    switch (which) {
        case 0:     /* conv (no activation) -> BN + ReLU -> max pool -> dense softmax */
            net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
            layer(.net = net, .height = 6, .width = 6, .channels = 2);
            conv2d(.net = net, .filters = 3, .kernel = 3, .padding = 1, .act_func = ACT_NONE);
            batch_norm(.net = net, .act_func = ACT_RELU);
            max_pool2d(.net = net, .kernel = 2);
            layer(.net = net, .neurons_amount = 4, .act_func = ACT_SOFTMAX);
            return net;
        case 1:     /* BN of the input, conv, average pool, dense, BN + sigmoid as the output layer */
            net = new_spingalett(.loss_func = LOSS_MSE);
            layer(.net = net, .height = 5, .width = 4, .channels = 3);
            batch_norm(.net = net, .act_func = ACT_TANH, .epsilon = 1e-3f);
            conv2d(.net = net, .filters = 4, .kernel = 2, .act_func = ACT_TANH);
            avg_pool2d(.net = net, .kernel = 2, .stride = 1);
            layer(.net = net, .neurons_amount = 3, .act_func = ACT_NONE);
            batch_norm(.net = net, .act_func = ACT_SIGMOID);
            return net;
        case 2:     /* dense layers normalized per unit */
            net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
            layer(.net = net, .neurons_amount = 5);
            layer(.net = net, .neurons_amount = 7, .act_func = ACT_NONE);
            batch_norm(.net = net, .act_func = ACT_LEAKY_RELU);
            layer(.net = net, .neurons_amount = 6, .act_func = ACT_NONE);
            batch_norm(.net = net, .act_func = ACT_FOO52);
            layer(.net = net, .neurons_amount = 3, .act_func = ACT_SOFTMAX);
            return net;
        default:    /* BN after an activated, strided conv and after pooling, into a sigmoid output */
            net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
            layer(.net = net, .height = 7, .width = 7, .channels = 2);
            conv2d(.net = net, .filters = 5, .kernel = 3, .stride = 2, .act_func = ACT_TANH);
            batch_norm(.net = net, .act_func = ACT_NONE);
            max_pool2d(.net = net, .kernel = 2, .stride = 1);
            batch_norm(.net = net, .act_func = ACT_TANH);
            layer(.net = net, .neurons_amount = 2, .act_func = ACT_SIGMOID);
            return net;
    }
}

/* The loss with batch statistics (training mode): what train() differentiates. */
static double norm_train_loss(NeuralNetwork *net, const float *x, const float *y, uint32_t N) {
    SpingalettTrainer *tr = spingalett_trainer_new(net, N);
    const float *o = spingalett_trainer_forward(tr, x, N);
    uint32_t out = net->topology[net->layers - 1];
    ActivationFunction oa = net->act_func[net->layers - 2];
    double s = 0;
    for (uint32_t i = 0; i < N; i++) s += sample_loss(net->loss_func, oa, o + (size_t)i * out, y + (size_t)i * out, out);
    spingalett_trainer_free(tr);
    return s / N;
}

static void norm_gradcheck(int which, ComputeMode mode, TrainingStrategy strat) {
    lcg_state = 2000 + which;
    NeuralNetwork *net = norm_net(which);
    const uint32_t N = 6, in = net->topology[0], out = net->topology[net->layers - 1];
    float *x, *y;
    make_data(N, in, out, net->loss_func, net->act_func[net->layers - 2], &x, &y);
    spingalett_set_compute_mode(mode);
    for (uint32_t l = 1; l < net->layers; l++) {
        uint64_t w = net->weight_offsets[l - 1], b = net->bias_offsets[l - 1];
        uint64_t rows = net->shapes[l].type == LAYER_BATCH_NORM ? net->shapes[l].channels : 0;
        for (uint64_t i = 0; i < rows; i++) {           /* gamma around 1, beta around 0 */
            net->weights[w + i] = 0.6f + frand() * 0.8f;
            net->biases[b + i] = frand() * 0.4f - 0.2f;
        }
    }
    for (uint32_t l = 1; l < net->layers; l++) {
        if (net->shapes[l].type == LAYER_BATCH_NORM) continue;
        uint64_t w = net->weight_offsets[l - 1], b = net->bias_offsets[l - 1];
        uint64_t nw = (l + 1 < net->layers ? net->weight_offsets[l] : net->total_weights) - w;
        uint64_t nb = (l + 1 < net->layers ? net->bias_offsets[l] : net->total_biases) - b;
        for (uint64_t i = 0; i < nw; i++) net->weights[w + i] = frand() * 1.2f - 0.6f;
        for (uint64_t i = 0; i < nb; i++) net->biases[b + i] = frand() * 0.2f - 0.1f;
    }
    uint64_t nw = net->total_weights, nb = net->total_biases;
    float *w0 = malloc(nw * 4), *b0 = malloc(nb * 4);
    memcpy(w0, net->weights, nw * 4); memcpy(b0, net->biases, nb * 4);
    train(.net = net, .inputs = x, .targets = y, .sample_count = N, .epochs = 1, .learning_rate = 1.0f,
          .optimizer_type = OPTIMIZER_SGD, .training_strategy = strat, .batch_size = N);
    float *ga = malloc((nw + nb) * 4);
    for (uint64_t i = 0; i < nw; i++) ga[i] = w0[i] - net->weights[i];
    for (uint64_t i = 0; i < nb; i++) ga[nw + i] = b0[i] - net->biases[i];
    memcpy(net->weights, w0, nw * 4); memcpy(net->biases, b0, nb * 4);

    int bad = 0, kinks = 0; double maxrel = 0;
    for (uint64_t i = 0; i < nw + nb; i++) {
        float *p = i < nw ? &net->weights[i] : &net->biases[i - nw];
        float orig = *p, h = 1e-3f, h2 = 2.5e-4f;
        *p = orig + h; double lp = norm_train_loss(net, x, y, N);
        *p = orig - h; double lm = norm_train_loss(net, x, y, N);
        *p = orig + h2; double lp2 = norm_train_loss(net, x, y, N);
        *p = orig - h2; double lm2 = norm_train_loss(net, x, y, N);
        *p = orig;
        double gn = (lp - lm) / (2.0 * h), gn2 = (lp2 - lm2) / (2.0 * h2);
        if (fabs(gn - gn2) > 0.1 * (fabs(gn) + fabs(gn2)) + 3e-4) { kinks++; continue; }
        double rel = fmin(fabs(gn - ga[i]) / fmax(1e-2, fabs(gn) + fabs(ga[i])),
                          fabs(gn2 - ga[i]) / fmax(1e-2, fabs(gn2) + fabs(ga[i])));
        if (rel > maxrel) maxrel = rel;
        if (rel > 3e-2) bad++;
    }
    printf("  norm gradcheck net %d mode=%d strat=%d  maxrel=%.2e  outliers=%d/%llu kinks=%d\n", which, mode, strat,
           maxrel, bad, (unsigned long long)(nw + nb), kinks);
    CHECK(bad == 0 && kinks * 10 < (int)(nw + nb), "norm gradcheck net %d mode %d strat %d: %d outliers", which, mode,
          strat, bad);
    free(ga); free(w0); free(b0); free(x); free(y);
    free_network(net);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
}

/* Training normalizes with batch statistics and moves the running ones; inference uses the running
   ones, in predict(), forward(), models (folded or not) and after a round trip through a file. */
static void norm_inference(ComputeMode mode) {
    lcg_state = 99;
    spingalett_seed(3);
    spingalett_set_compute_mode(mode);
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(.net = net, .height = 8, .width = 8, .channels = 2);
    batch_norm(.net = net, .momentum = 0.5f);                              /* after the input: kept */
    conv2d(.net = net, .filters = 6, .kernel = 3, .padding = 1, .act_func = ACT_NONE,
           .weight_initialization = WEIGHT_INITIALIZATION_HE);
    batch_norm(.net = net, .act_func = ACT_RELU);                          /* folds into the conv */
    max_pool2d(.net = net, .kernel = 2);
    batch_norm(.net = net, .act_func = ACT_TANH, .dropout_rate = 0.2f);    /* after pooling: kept */
    layer(.net = net, .neurons_amount = 10, .act_func = ACT_NONE, .weight_initialization = WEIGHT_INITIALIZATION_HE);
    batch_norm(.net = net, .act_func = ACT_RELU, .epsilon = 1e-3f);        /* folds into the dense layer */
    layer(.net = net, .neurons_amount = 3, .act_func = ACT_SOFTMAX, .weight_initialization = WEIGHT_INITIALIZATION_XAVIER);
    const uint32_t N = 64, in = net->topology[0];
    float *x = malloc((size_t)N * in * 4), *y = calloc((size_t)N * 3, 4);
    for (size_t i = 0; i < (size_t)N * in; i++) x[i] = frand() * 3.0f + 2.0f;     /* far from zero mean */
    for (uint32_t s = 0; s < N; s++) y[s * 3 + s % 3] = 1.0f;

    SpingalettNetworkLayer info;
    CHECK(spingalett_network_layer(net, 1, &info) && info.type == LAYER_BATCH_NORM && info.channels == 2 &&
          info.weight_count == 2 && info.bias_count == 2 && fabsf(info.momentum - 0.5f) < 1e-7f &&
          fabsf(info.epsilon - 1e-5f) < 1e-12f && net->weights[net->weight_offsets[0]] == 1.0f,
          "norm: layer description and initialization");
    float mean[2], var[2];
    CHECK(spingalett_get_parameters(net, 1, PARAM_RUNNING_MEAN, mean, 2) &&
          spingalett_get_parameters(net, 1, PARAM_RUNNING_VARIANCE, var, 2) && mean[0] == 0.0f && var[1] == 1.0f &&
          !spingalett_get_parameters(net, 2, PARAM_RUNNING_MEAN, mean, 6), "norm: running statistics accessors");

    TrainReport r = train(.net = net, .inputs = x, .targets = y, .sample_count = N, .epochs = 3, .learning_rate = 1e-2f,
                          .optimizer_type = OPTIMIZER_ADAM, .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 16);
    CHECK(r.status == TRAIN_COMPLETED, "norm: training");
    /* the input's running mean approaches the data's (3.5 here) */
    spingalett_get_parameters(net, 1, PARAM_RUNNING_MEAN, mean, 2);
    spingalett_get_parameters(net, 1, PARAM_RUNNING_VARIANCE, var, 2);
    CHECK(fabsf(mean[0] - 3.5f) < 0.4f && fabsf(var[0] - 0.75f) < 0.3f, "norm: running statistics follow the data (%g, %g)",
          mean[0], var[0]);

    /* predict() against forward() one sample at a time, and against a hand-computed first layer */
    float *p = malloc((size_t)N * 3 * 4), *q = malloc((size_t)N * 3 * 4);
    predict(.net = net, .inputs = x, .outputs = p, .sample_count = N);
    float worst = 0;
    for (uint32_t s = 0; s < N; s++) {
        float *o = forward(.net = net, .input = x + (size_t)s * in);
        worst = fmaxf(worst, max_abs_diff(o, p + (size_t)s * 3, 3));
    }
    CHECK(worst < 1e-6f, "norm: forward() and predict() differ by %g", worst);

    /* deployment models: two normalizations fold, two remain (format 5) */
    SpingalettModel *m = spingalett_model_from_network(net, PRECISION_FLOAT32);
    CHECK(m && m->layer_count == net->layers - 3 && rd16((const uint8_t *)m->image + 6) == 5,
          "norm: folded model (%u layers, version %u)", m ? m->layer_count : 0, m ? rd16((const uint8_t *)m->image + 6) : 0);
    if (m) {
        spingalett_model_predict(m, x, N, q);
        float d = max_abs_diff(p, q, (size_t)N * 3);
        CHECK(d < 1e-4f, "norm: folded model differs from the network by %g", d);
        float one[3];
        worst = 0;
        for (uint32_t s = 0; s < N; s++) {
            spingalett_model_run(m, x + (size_t)s * in, one, NULL);
            worst = fmaxf(worst, max_abs_diff(one, q + (size_t)s * 3, 3));
        }
        CHECK(worst < 1e-5f, "norm: model run and batched prediction differ by %g", worst);
        SpingalettLayerInfo li;
        CHECK(spingalett_model_layer(m, 0, &li) && li.type == LAYER_BATCH_NORM && li.precision == PRECISION_FLOAT32 &&
              spingalett_model_layer(m, 1, &li) && li.type == LAYER_CONV2D && li.activation == ACT_RELU && li.groups == 1,
              "norm: model layers");
        spingalett_model_free(m);
    }
    for (int prec = PRECISION_FP16; prec < PRECISION_COUNT; prec++) {
        SpingalettModel *qm = spingalett_model_from_network(net, (PrecisionMode)prec);
        CHECK(qm != NULL, "norm: model in precision %d", prec);
        if (!qm) continue;
        spingalett_model_predict(qm, x, N, q);
        float one[3];
        worst = 0;
        for (uint32_t s = 0; s < N; s++) {
            spingalett_model_run(qm, x + (size_t)s * in, one, NULL);
            worst = fmaxf(worst, max_abs_diff(one, q + (size_t)s * 3, 3));
        }
        CHECK(prec >= PRECISION_INT8 ? worst == 0.0f : worst < 1e-5f,
              "norm: precision %d: model run and batched prediction differ by %g", prec, worst);
        float d = max_abs_diff(p, q, (size_t)N * 3);
        CHECK(d < (prec <= PRECISION_BFLOAT16 ? 0.05f : prec == PRECISION_INT8 ? 0.15f : 1.0f),
              "norm: precision %d model differs from the network by %g", prec, d);
        spingalett_model_free(qm);
    }

    /* round trip: FLOAT32 keeps every normalization, parameters, statistics and optimizer state */
    size_t size = 0;
    uint8_t *img = spingalett_save_to_memory(net, PRECISION_FLOAT32, true, &size);
    NeuralNetwork *back = img ? load_spingalett_from_memory(img, size) : NULL;
    bool same = back && back->layers == net->layers && back->total_biases == net->total_biases &&
                !memcmp(back->weights, net->weights, net->total_weights * 4) &&
                !memcmp(back->biases, net->biases, net->total_biases * 4) &&
                !memcmp(back->running_mean, net->running_mean, net->total_biases * 4) &&
                !memcmp(back->running_var, net->running_var, net->total_biases * 4) &&
                !memcmp(back->opt_v_weights, net->opt_v_weights, net->total_weights * 4) &&
                back->shapes[1].momentum == 0.5f && back->shapes[7].eps == 1e-3f && back->dropout_rates[5] == 0.2f;
    CHECK(img && rd16(img + 6) == 5 && same, "norm: FLOAT32 round trip");
    if (back) {
        predict(.net = back, .inputs = x, .outputs = q, .sample_count = N);
        CHECK(!memcmp(p, q, (size_t)N * 3 * 4), "norm: loaded network predicts differently");
        free_network(back);
    }
    spingalett_free(img);
    /* INT8 without optimizer state folds; loaded back, the network has two normalizations fewer */
    img = spingalett_save_to_memory(net, PRECISION_INT8, false, &size);
    back = img ? load_spingalett_from_memory(img, size) : NULL;
    CHECK(back && back->layers == net->layers - 2, "norm: INT8 files fold normalizations");
    if (back) free_network(back);
    spingalett_free(img);

    /* restore_best_weights restores the running statistics with the weights */
    float *ref_mean = malloc(net->total_biases * 4);
    r = train(.net = net, .inputs = x, .targets = y, .sample_count = N, .epochs = 4, .learning_rate = 0.5f,
              .optimizer_type = OPTIMIZER_SGD, .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 16,
              .val_inputs = x, .val_targets = y, .val_count = N, .restore_best_weights = true,
              .callback = NULL);
    if (r.restored_best) {
        EvalMetrics e = evaluate(.net = net, .inputs = x, .targets = y, .sample_count = N);
        CHECK(fabsf(e.loss - r.best_value) < 1e-5f, "norm: restored weights give the best validation loss (%g vs %g)",
              e.loss, r.best_value);
    }
    free(ref_mean);
    free(p); free(q); free(x); free(y);
    free_network(net);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
}

/* Weight decay leaves gamma alone; a dense layer normalized with batches of one sample is refused. */
static void norm_misc(void) {
    lcg_state = 5;
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_MSE);
    layer(.net = net, .neurons_amount = 4);
    layer(.net = net, .neurons_amount = 5, .act_func = ACT_NONE);
    batch_norm(.net = net, .act_func = ACT_TANH);
    layer(.net = net, .neurons_amount = 2, .act_func = ACT_NONE);
    float x[16], y[8];
    for (int i = 0; i < 16; i++) x[i] = frand();
    for (int i = 0; i < 8; i++) y[i] = frand();
    SpingalettTrainer *tr = spingalett_trainer_new(net, 4);
    spingalett_trainer_forward(tr, x, 4);
    spingalett_trainer_backward(tr, y);
    float g0[5], gw[5], g1[5], w0[20], ww[20], w1[20];
    spingalett_get_parameters(net, 2, PARAM_WEIGHTS, g0, 5);
    spingalett_get_parameters(net, 1, PARAM_WEIGHTS, w0, 20);
    spingalett_get_parameters(net, 2, PARAM_WEIGHT_GRADIENTS, gw, 5);
    spingalett_get_parameters(net, 1, PARAM_WEIGHT_GRADIENTS, ww, 20);
    OptimizerArgs o = {.type = OPTIMIZER_SGD, .learning_rate = 0.1f, .weight_decay = 0.5f};
    spingalett_trainer_step(tr, &o);
    spingalett_get_parameters(net, 2, PARAM_WEIGHTS, g1, 5);
    spingalett_get_parameters(net, 1, PARAM_WEIGHTS, w1, 20);
    float dg = 0, dw = 0;
    for (int i = 0; i < 5; i++) dg = fmaxf(dg, fabsf(g1[i] - (g0[i] - 0.1f * gw[i] / 4)));
    for (int i = 0; i < 20; i++) dw = fmaxf(dw, fabsf(w1[i] - (w0[i] - 0.1f * (ww[i] / 4 + 0.5f * w0[i]))));
    CHECK(dg < 1e-6f && dw < 1e-6f, "norm: weight decay applies to weights (%g) but not to gamma (%g)", dw, dg);
    spingalett_trainer_free(tr);

    TrainReport r = train(.net = net, .inputs = x, .targets = y, .sample_count = 4, .epochs = 1,
                          .training_strategy = STRATEGY_SAMPLE);
    CHECK(r.status == TRAIN_FAILED, "norm: dense normalization with batches of one sample must be refused");
    r = train(.net = net, .inputs = x, .targets = y, .sample_count = 4, .epochs = 1,
              .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 3);
    CHECK(r.status == TRAIN_COMPLETED, "norm: a last batch of one sample trains");
    free_network(net);

    /* invalid arguments */
    net = new_spingalett(.loss_func = LOSS_MSE);
    batch_norm(.net = net);
    CHECK(net->layers == 0, "norm: a normalization cannot be the input layer");
    layer(.net = net, .neurons_amount = 3);
    batch_norm(.net = net, .epsilon = -1.0f);
    batch_norm(.net = net, .momentum = 1.5f);
    CHECK(net->layers == 1, "norm: epsilon and momentum are checked");
    free_network(net);
}

/* Version 5 tables that disagree with their layer kinds are refused. */
static void norm_validation(void) {
    lcg_state = 11;
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(.net = net, .height = 4, .width = 4, .channels = 2);
    batch_norm(.net = net);
    conv2d(.net = net, .filters = 3, .kernel = 3, .padding = 1, .act_func = ACT_NONE);
    batch_norm(.net = net, .act_func = ACT_RELU);
    layer(.net = net, .neurons_amount = 2, .act_func = ACT_SOFTMAX);
    size_t size = 0;
    uint8_t *img = spingalett_save_to_memory(net, PRECISION_FLOAT32, false, &size);
    SpingalettModel m;
    CHECK(img && rd16(img + 6) == 5 && spingalett_model_init(&m, img, size) == SPINGALETT_OK, "norm validation: valid image");
    if (!img) { free_network(net); return; }
    uint8_t *buf = malloc(size);
    const size_t e0 = 64, e1 = 64 + 80;            /* the input's normalization, the convolution */
    struct { size_t at; uint32_t value; int bytes; const char *what; } bad[] = {
        {e0 + 9, PRECISION_INT8, 1, "integer normalization"},
        {e0 + 68, 0, 4, "epsilon 0"},
        {e0 + 72, 0x40000000u, 4, "momentum 2"},
        {e0 + 64, 1, 4, "groups on a normalization"},
        {e0 + 52, 1, 2, "a window on a normalization"},
        {e0 + 24, 0, 8, "no running statistics"},
        {e1 + 64, 0, 4, "convolution without groups"},
        {e1 + 64, 4, 4, "groups not dividing the channels"},
        {e1 + 68, 0x3f800000u, 4, "epsilon on a convolution"},
        {6, 4, 2, "normalization in a version 4 image"},
    };
    for (size_t t = 0; t < sizeof bad / sizeof *bad; t++) {
        memcpy(buf, img, size);
        for (int k = 0; k < bad[t].bytes; k++) buf[bad[t].at + k] = (uint8_t)(k < 4 ? bad[t].value >> (8 * k) : 0);
        reseal(buf, size);
        CHECK(spingalett_model_init(&m, buf, size) != SPINGALETT_OK, "norm validation: %s accepted", bad[t].what);
    }
    free(buf);
    spingalett_free(img);
    free_network(net);
}

static void norm_determinism(void) {
    static const ComputeMode modes[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENMP, COMPUTE_OPENMP};
    static const unsigned threads[] = {1, 1, 3, 4};
    float *ref = NULL, *ref_stats = NULL;
    bool same = true;
    for (int v = 0; v < 4; v++) {
        spingalett_set_num_threads(threads[v]);
        spingalett_set_compute_mode(modes[v]);
        lcg_state = 78;
        spingalett_seed(6);
        NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
        layer(.net = net, .height = 12, .width = 11, .channels = 3);
        conv2d(.net = net, .filters = 70, .kernel = 3, .padding = 1);
        batch_norm(.net = net, .act_func = ACT_TANH);
        max_pool2d(.net = net, .kernel = 2);
        layer(.net = net, .neurons_amount = 150);
        batch_norm(.net = net, .act_func = ACT_SIGMOID);
        layer(.net = net, .neurons_amount = 3, .act_func = ACT_SOFTMAX);
        const uint32_t n = 96, in = net->topology[0];
        float *x = malloc((size_t)n * in * sizeof(float)), *y = calloc((size_t)n * 3, sizeof(float));
        for (size_t i = 0; i < (size_t)n * in; i++) x[i] = frand() * 2 - 1;
        for (uint32_t s = 0; s < n; s++) y[s * 3 + s % 3] = 1.0f;
        train(.net = net, .inputs = x, .targets = y, .sample_count = n, .epochs = 2, .learning_rate = 0.01f,
              .optimizer_type = OPTIMIZER_ADAM, .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 48,
              .do_not_shuffle = true);
        if (!ref) {
            ref = malloc(net->total_weights * sizeof(float));
            ref_stats = malloc(net->total_biases * sizeof(float));
            memcpy(ref, net->weights, net->total_weights * sizeof(float));
            memcpy(ref_stats, net->running_var, net->total_biases * sizeof(float));
        } else if (memcmp(ref, net->weights, net->total_weights * sizeof(float)) ||
                   memcmp(ref_stats, net->running_var, net->total_biases * sizeof(float))) {
            same = false;
        }
        free(x); free(y); free_network(net);
    }
    CHECK(same, "normalized training differs between thread counts or compute modes");
    printf("  normalized training: identical on 1, 3 and 4 threads and single-threaded%s\n", same ? "" : " -- NO");
    free(ref); free(ref_stats);
    spingalett_set_num_threads(4);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
}

/* A normalized conv net learns the bars task with a large learning rate. */
static void norm_learns(ComputeMode mode) {
    lcg_state = 31338;
    spingalett_seed(9);
    spingalett_set_compute_mode(mode);
    const uint32_t N = 400;
    float *x = malloc((size_t)N * 64 * 4), *y = calloc((size_t)N * 2, 4);
    for (uint32_t s = 0; s < N; s++) {
        bool vertical = s % 2;
        uint32_t pos = (uint32_t)(frand() * 6) + 1;
        for (uint32_t i = 0; i < 64; i++) {
            uint32_t r = i / 8, c = i % 8;
            x[s * 64 + i] = (frand() - 0.5f) * 0.6f + ((vertical ? c : r) == pos ? 1.0f : 0.0f) + 5.0f;
        }
        y[s * 2 + vertical] = 1.0f;
    }
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(.net = net, .height = 8, .width = 8, .channels = 1);
    conv2d(.net = net, .filters = 4, .kernel = 3, .padding = 1, .act_func = ACT_NONE,
           .weight_initialization = WEIGHT_INITIALIZATION_HE);
    batch_norm(.net = net, .act_func = ACT_RELU);
    max_pool2d(.net = net, .kernel = 2);
    layer(.net = net, .neurons_amount = 2, .act_func = ACT_SOFTMAX, .weight_initialization = WEIGHT_INITIALIZATION_XAVIER);
    train(.net = net, .inputs = x, .targets = y, .sample_count = N, .epochs = 15, .learning_rate = 0.05f,
          .optimizer_type = OPTIMIZER_ADAM, .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 32);
    EvalMetrics m = evaluate(.net = net, .inputs = x, .targets = y, .sample_count = N);
    printf("  normalized conv learns bars mode=%d: accuracy %.1f%%, loss %.4f\n", mode, 100.0 * m.accuracy, m.loss);
    CHECK(m.accuracy > 0.95f, "normalized conv net must learn the bars (mode %d): %.3f", mode, m.accuracy);
    free(x); free(y);
    free_network(net);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
}

/* Reduce on plateau: a loss that cannot improve halves the learning rate after every epoch without
   improvement, down to the floor; with a schedule, the schedule's rate is scaled. */
static float plateau_lr[8];
static bool record_lr(NeuralNetwork *n, const TrainProgress *p, void *u) {
    (void)n; (void)u;
    if (p->epoch <= 8) plateau_lr[p->epoch - 1] = p->learning_rate;
    return false;
}

static void plateau(void) {
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_MSE);
    layer(.net = net, .neurons_amount = 2);
    layer(.net = net, .neurons_amount = 1, .act_func = ACT_SIGMOID, .weight_initialization = WEIGHT_INITIALIZATION_NONE);
    float x[4] = {0, 1, 1, 0}, y[2] = {0.5f, 0.5f};    /* zero weights already give 0.5: no improvement */
    TrainReport r = train(.net = net, .inputs = x, .targets = y, .sample_count = 2, .epochs = 8, .learning_rate = 0.4f,
                          .training_strategy = STRATEGY_FULL_BATCH, .lr_plateau_factor = 0.5f, .lr_plateau_patience = 1,
                          .lr_plateau_min_lr = 0.04f, .callback = record_lr);
    static const float expected[8] = {0.4f, 0.4f, 0.2f, 0.1f, 0.05f, 0.04f, 0.04f, 0.04f};
    bool ok = r.status == TRAIN_COMPLETED;
    for (int e = 0; e < 8; e++) ok = ok && fabsf(plateau_lr[e] - expected[e]) < 1e-6f;
    CHECK(ok, "reduce on plateau: rates %g %g %g %g %g %g", plateau_lr[0], plateau_lr[1], plateau_lr[2], plateau_lr[3],
          plateau_lr[4], plateau_lr[5]);
    printf("  reduce on plateau: %g %g %g %g %g %g %g\n", plateau_lr[0], plateau_lr[1], plateau_lr[2], plateau_lr[3],
           plateau_lr[4], plateau_lr[5], plateau_lr[6]);
    r = train(.net = net, .inputs = x, .targets = y, .sample_count = 2, .epochs = 4, .learning_rate = 0.4f,
              .lr_plateau_factor = 0.25f, .lr_plateau_patience = 2, .lr_scheduler = spingalett_lr_linear_warmup,
              .callback = record_lr, .training_strategy = STRATEGY_FULL_BATCH);
    CHECK(r.status == TRAIN_COMPLETED && fabsf(plateau_lr[3] - 0.1f) < 1e-6f, "reduce on plateau with a schedule: %g",
          plateau_lr[3]);
    CHECK(train(.net = net, .inputs = x, .targets = y, .sample_count = 2, .epochs = 1, .lr_plateau_factor = 1.5f,
                .lr_plateau_patience = 1).status == TRAIN_FAILED, "a plateau factor above 1 accepted");
    free_network(net);
}

/* Label smoothing trains as the smoothed targets do, in train() and in the step API. */
static void label_smoothing(void) {
    const uint32_t N = 12;
    float x[12 * 4], t[12 * 3], ts[12 * 3];
    lcg_state = 321;
    for (int i = 0; i < 48; i++) x[i] = frand() * 2 - 1;
    for (uint32_t s = 0; s < N; s++)
        for (uint32_t k = 0; k < 3; k++) t[s * 3 + k] = k == s % 3 ? 1.0f : 0.0f;
    for (int i = 0; i < 36; i++) ts[i] = 0.85f * t[i] + 0.15f / 3.0f;
    float w[2][64];
    for (int v = 0; v < 2; v++) {
        spingalett_seed(5);
        NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
        layer(.net = net, .neurons_amount = 4);
        layer(.net = net, .neurons_amount = 6, .act_func = ACT_TANH, .weight_initialization = WEIGHT_INITIALIZATION_XAVIER);
        layer(.net = net, .neurons_amount = 3, .act_func = ACT_SOFTMAX, .weight_initialization = WEIGHT_INITIALIZATION_XAVIER);
        TrainReport r = train(.net = net, .inputs = x, .targets = v ? ts : t, .sample_count = N, .epochs = 3,
                              .learning_rate = 0.1f, .optimizer_type = OPTIMIZER_ADAM, .training_strategy = STRATEGY_SMALL_BATCH,
                              .batch_size = 4, .do_not_shuffle = true, .label_smoothing = v ? 0.0f : 0.15f);
        CHECK(r.status == TRAIN_COMPLETED, "label smoothing: training failed");
        memcpy(w[v], net->weights, net->total_weights * sizeof(float));
        if (v == 0) {
            CHECK(train(.net = net, .inputs = x, .targets = t, .sample_count = N, .epochs = 1,
                        .label_smoothing = 1.0f).status == TRAIN_FAILED, "label smoothing of 1 accepted");
            /* the step API, against the smoothed targets */
            SpingalettTrainer *tr = spingalett_trainer_new(net, N);
            OptimizerArgs sgd = {.type = OPTIMIZER_SGD, .learning_rate = 0.5f};
            float before[64], before_b[16];
            memcpy(before, net->weights, net->total_weights * sizeof(float));
            memcpy(before_b, net->biases, net->total_biases * sizeof(float));
            CHECK(spingalett_trainer_set_label_smoothing(tr, 0.15f) && !spingalett_trainer_set_label_smoothing(tr, -0.1f),
                  "trainer label smoothing setter");
            float a = spingalett_train_on_batch(tr, x, t, N, &sgd);
            float after[64];
            memcpy(after, net->weights, net->total_weights * sizeof(float));
            memcpy(net->weights, before, net->total_weights * sizeof(float));
            memcpy(net->biases, before_b, net->total_biases * sizeof(float));
            spingalett_trainer_set_label_smoothing(tr, 0.0f);
            float b = spingalett_train_on_batch(tr, x, ts, N, &sgd);
            float diff = 0.0f;
            for (uint64_t i = 0; i < net->total_weights; i++) diff = fmaxf(diff, fabsf(after[i] - net->weights[i]));
            CHECK(fabsf(a - b) < 1e-5f && diff < 1e-5f, "label smoothing in the step API: loss %g vs %g, weights %g", a, b, diff);
            spingalett_trainer_free(tr);
        }
        free_network(net);
    }
    float diff = 0.0f;
    for (int i = 0; i < 6 * 4 + 3 * 6; i++) diff = fmaxf(diff, fabsf(w[0][i] - w[1][i]));
    CHECK(diff < 1e-5f, "label smoothing differs from smoothed targets by %g", diff);
    printf("  label smoothing: as the smoothed targets (largest difference %.1e)\n", diff);
}

/* ------------------------------------------------------------------------- graphs */

/* Networks whose layers read other layers than the one before them. */
static NeuralNetwork *graph_net(int which) {
    NeuralNetwork *net;
    uint32_t a, b, c, d;
    switch (which) {
        case 0:     /* a residual block of two convolutions, global pooling, softmax */
            net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
            layer(.net = net, .height = 6, .width = 6, .channels = 3);
            a = conv2d(.net = net, .filters = 4, .kernel = 3, .padding = 1, .act_func = ACT_TANH);
            conv2d(.net = net, .filters = 4, .kernel = 3, .padding = 1, .act_func = ACT_NONE);
            add_layers(.net = net, .inputs = {a, a + 1}, .act_func = ACT_TANH);
            global_avg_pool2d(.net = net);
            layer(.net = net, .neurons_amount = 3, .act_func = ACT_SOFTMAX);
            return net;
        case 1:     /* normalized residual block with a strided 1 x 1 projection on the shortcut */
            net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
            layer(.net = net, .height = 6, .width = 6, .channels = 2);
            conv2d(.net = net, .filters = 4, .kernel = 3, .padding = 1, .act_func = ACT_NONE);
            a = batch_norm(.net = net, .act_func = ACT_TANH);
            conv2d(.net = net, .filters = 4, .kernel = 3, .padding = 1, .stride = 2, .act_func = ACT_NONE);
            b = batch_norm(.net = net, .act_func = ACT_NONE);
            conv2d(.net = net, .inputs = {a}, .filters = 4, .kernel = 1, .stride = 2, .act_func = ACT_NONE);
            c = batch_norm(.net = net, .act_func = ACT_NONE);
            add_layers(.net = net, .inputs = {b, c}, .act_func = ACT_SIGMOID);
            global_avg_pool2d(.net = net);
            layer(.net = net, .neurons_amount = 3, .act_func = ACT_SOFTMAX);
            return net;
        case 2:     /* branches concatenated: 1 x 1 conv, 3 x 3 conv and pooling of the input */
            net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
            layer(.net = net, .height = 5, .width = 5, .channels = 3);
            a = conv2d(.net = net, .filters = 2, .kernel = 1, .act_func = ACT_TANH);
            b = conv2d(.net = net, .inputs = {0}, .input_count = 1, .filters = 3, .kernel = 3, .padding = 1,
                       .act_func = ACT_FOO52);
            c = max_pool2d(.net = net, .inputs = {0}, .input_count = 1, .kernel = 3, .stride = 1, .padding = 1);
            concat_layers(.net = net, .inputs = {a, b, c}, .act_func = ACT_NONE);
            conv2d(.net = net, .filters = 2, .kernel = 3, .act_func = ACT_TANH);
            layer(.net = net, .neurons_amount = 2, .act_func = ACT_SIGMOID);
            return net;
        case 3:     /* dense layers read several times, one input named twice, a concatenation */
            net = new_spingalett(.loss_func = LOSS_MSE);
            layer(.net = net, .neurons_amount = 5);
            a = layer(.net = net, .neurons_amount = 6, .act_func = ACT_TANH);
            b = layer(.net = net, .neurons_amount = 6, .act_func = ACT_SIGMOID);
            c = add_layers(.net = net, .inputs = {a, b, a}, .act_func = ACT_LEAKY_RELU);
            d = layer(.net = net, .inputs = {a}, .neurons_amount = 4, .act_func = ACT_TANH);
            concat_layers(.net = net, .inputs = {c, d, 0}, .input_count = 3, .act_func = ACT_TANH);
            layer(.net = net, .neurons_amount = 3, .act_func = ACT_SOFTMAX);
            return net;
        default:    /* a normalization and a pooling of the same maps, an identity layer, pooled maps pooled */
            net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
            layer(.net = net, .height = 4, .width = 6, .channels = 2);
            a = conv2d(.net = net, .filters = 3, .kernel = 3, .padding = 1, .act_func = ACT_SIGMOID);
            b = batch_norm(.net = net, .act_func = ACT_TANH);
            b = add_layers(.net = net, .inputs = {b}, .act_func = ACT_NONE);
            c = avg_pool2d(.net = net, .inputs = {a}, .kernel = 2);
            d = global_avg_pool2d(.net = net, .inputs = {c});
            global_avg_pool2d(.net = net, .inputs = {b});
            concat_layers(.net = net, .inputs = {d, d + 1}, .act_func = ACT_NONE);
            layer(.net = net, .neurons_amount = 2, .act_func = ACT_SOFTMAX);
            return net;
    }
}

/* Uniform weights, gamma around 1 and beta around 0. */
static void graph_init(NeuralNetwork *net) {
    for (uint32_t l = 1; l < net->layers; l++) {
        uint64_t w = net->weight_offsets[l - 1], b = net->bias_offsets[l - 1];
        uint64_t nw = (l + 1 < net->layers ? net->weight_offsets[l] : net->total_weights) - w;
        uint64_t nb = (l + 1 < net->layers ? net->bias_offsets[l] : net->total_biases) - b;
        bool norm = net->shapes[l].type == LAYER_BATCH_NORM;
        for (uint64_t i = 0; i < nw; i++) net->weights[w + i] = norm ? 0.6f + frand() * 0.8f : frand() * 1.2f - 0.6f;
        for (uint64_t i = 0; i < nb; i++) net->biases[b + i] = norm ? frand() * 0.4f - 0.2f : frand() * 0.2f - 0.1f;
    }
}

static void graph_gradcheck(int which, ComputeMode mode, TrainingStrategy strat) {
    lcg_state = 3000 + which;
    NeuralNetwork *net = graph_net(which);
    const uint32_t N = 6, in = net->topology[0], out = net->topology[net->layers - 1];
    float *x, *y;
    make_data(N, in, out, net->loss_func, net->act_func[net->layers - 2], &x, &y);
    spingalett_set_compute_mode(mode);
    graph_init(net);
    uint64_t nw = net->total_weights, nb = net->total_biases;
    float *w0 = malloc(nw * 4 + 4), *b0 = malloc(nb * 4 + 4);
    memcpy(w0, net->weights, nw * 4); memcpy(b0, net->biases, nb * 4);
    train(.net = net, .inputs = x, .targets = y, .sample_count = N, .epochs = 1, .learning_rate = 1.0f,
          .optimizer_type = OPTIMIZER_SGD, .training_strategy = strat, .batch_size = N);
    float *ga = malloc((nw + nb) * 4);
    for (uint64_t i = 0; i < nw; i++) ga[i] = w0[i] - net->weights[i];
    for (uint64_t i = 0; i < nb; i++) ga[nw + i] = b0[i] - net->biases[i];
    memcpy(net->weights, w0, nw * 4); memcpy(net->biases, b0, nb * 4);

    int bad = 0, kinks = 0; double maxrel = 0;
    for (uint64_t i = 0; i < nw + nb; i++) {
        float *p = i < nw ? &net->weights[i] : &net->biases[i - nw];
        float orig = *p, h = 1e-3f, h2 = 2.5e-4f;
        *p = orig + h; double lp = norm_train_loss(net, x, y, N);
        *p = orig - h; double lm = norm_train_loss(net, x, y, N);
        *p = orig + h2; double lp2 = norm_train_loss(net, x, y, N);
        *p = orig - h2; double lm2 = norm_train_loss(net, x, y, N);
        *p = orig;
        double gn = (lp - lm) / (2.0 * h), gn2 = (lp2 - lm2) / (2.0 * h2);
        if (fabs(gn - gn2) > 0.1 * (fabs(gn) + fabs(gn2)) + 3e-4) { kinks++; continue; }
        double rel = fmin(fabs(gn - ga[i]) / fmax(1e-2, fabs(gn) + fabs(ga[i])),
                          fabs(gn2 - ga[i]) / fmax(1e-2, fabs(gn2) + fabs(ga[i])));
        if (rel > maxrel) maxrel = rel;
        if (rel > 3e-2) bad++;
    }
    printf("  graph gradcheck net %d mode=%d strat=%d  maxrel=%.2e  outliers=%d/%llu kinks=%d\n", which, mode, strat,
           maxrel, bad, (unsigned long long)(nw + nb), kinks);
    CHECK(bad == 0 && kinks * 10 < (int)(nw + nb), "graph gradcheck net %d mode %d strat %d: %d outliers", which, mode,
          strat, bad);
    free(ga); free(w0); free(b0); free(x); free(y);
    free_network(net);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
}

/* A residual network of `blocks` blocks of two normalized 3 x 3 convolutions on 8 x 8 x c maps. */
static NeuralNetwork *resnet(uint32_t blocks, uint32_t channels, float dropout) {
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(.net = net, .height = 8, .width = 8, .channels = 3);
    conv2d(.net = net, .filters = channels, .kernel = 3, .padding = 1, .act_func = ACT_NONE);
    uint32_t x = batch_norm(.net = net, .act_func = ACT_RELU);
    for (uint32_t k = 0; k < blocks; k++) {
        conv2d(.net = net, .filters = channels, .kernel = 3, .padding = 1, .act_func = ACT_NONE);
        batch_norm(.net = net, .act_func = ACT_RELU, .dropout_rate = dropout);
        conv2d(.net = net, .filters = channels, .kernel = 3, .padding = 1, .act_func = ACT_NONE);
        uint32_t y = batch_norm(.net = net, .act_func = ACT_NONE);
        x = add_layers(.net = net, .inputs = {x, y}, .act_func = ACT_RELU, .dropout_rate = dropout);
    }
    global_avg_pool2d(.net = net);
    layer(.net = net, .neurons_amount = 4, .act_func = ACT_SOFTMAX);
    return net;
}

/* Training a residual network (with dropout on layers read twice) gives the same bits on any
   number of threads; a chain whose layers name their inputs trains exactly as the plain chain. */
static void graph_determinism(void) {
    static const ComputeMode modes[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENMP, COMPUTE_OPENMP};
    static const unsigned threads[] = {1, 1, 3, 4};
    float *ref = NULL;
    bool same = true;
    for (int v = 0; v < 4; v++) {
        spingalett_set_num_threads(threads[v]);
        spingalett_set_compute_mode(modes[v]);
        lcg_state = 79;
        spingalett_seed(7);
        NeuralNetwork *net = resnet(2, 24, 0.2f);
        const uint32_t n = 64, in = net->topology[0];
        float *x = malloc((size_t)n * in * sizeof(float)), *y = calloc((size_t)n * 4, sizeof(float));
        for (size_t i = 0; i < (size_t)n * in; i++) x[i] = frand() * 2 - 1;
        for (uint32_t s = 0; s < n; s++) y[s * 4 + s % 4] = 1.0f;
        train(.net = net, .inputs = x, .targets = y, .sample_count = n, .epochs = 2, .learning_rate = 0.01f,
              .optimizer_type = OPTIMIZER_ADAMW, .weight_decay = 1e-3f, .training_strategy = STRATEGY_SMALL_BATCH,
              .batch_size = 32, .augment_shift = 1, .augment_flip = true);
        if (!ref) {
            ref = malloc(net->total_weights * sizeof(float));
            memcpy(ref, net->weights, net->total_weights * sizeof(float));
        } else if (memcmp(ref, net->weights, net->total_weights * sizeof(float))) {
            same = false;
        }
        free(x); free(y); free_network(net);
    }
    CHECK(same, "residual training differs between thread counts or compute modes");
    printf("  residual training: identical on 1, 3 and 4 threads and single-threaded%s\n", same ? "" : " -- NO");
    free(ref);
    spingalett_set_num_threads(4);

    /* the same chain, implicitly and with every input named */
    float *w[2] = {NULL, NULL};
    for (int named = 0; named < 2; named++) {
        lcg_state = 80;
        spingalett_seed(8);
        NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
        layer(.net = net, .height = 6, .width = 6, .channels = 2);
        uint32_t a = conv2d(.net = net, .filters = 5, .kernel = 3, .act_func = ACT_RELU);
        if (named) {
            uint32_t b = max_pool2d(.net = net, .inputs = {a}, .kernel = 2);
            layer(.net = net, .inputs = {b}, .neurons_amount = 3, .act_func = ACT_SOFTMAX);
        } else {
            max_pool2d(.net = net, .kernel = 2);
            layer(.net = net, .neurons_amount = 3, .act_func = ACT_SOFTMAX);
        }
        CHECK(!net->graph, "a chain whose layers name their inputs is still a chain");
        const uint32_t n = 40;
        float *x = malloc((size_t)n * 72 * sizeof(float)), *y = calloc((size_t)n * 3, sizeof(float));
        for (size_t i = 0; i < (size_t)n * 72; i++) x[i] = frand() * 2 - 1;
        for (uint32_t s = 0; s < n; s++) y[s * 3 + s % 3] = 1.0f;
        train(.net = net, .inputs = x, .targets = y, .sample_count = n, .epochs = 3, .learning_rate = 0.01f,
              .optimizer_type = OPTIMIZER_ADAM, .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 8);
        w[named] = malloc(net->total_weights * sizeof(float));
        memcpy(w[named], net->weights, net->total_weights * sizeof(float));
        size_t size = 0;
        void *img = spingalett_save_to_memory(net, PRECISION_FLOAT32, false, &size);
        CHECK(img && rd16((const uint8_t *)img + 6) == 4, "a chain is written in format 4, not %u",
              img ? rd16((const uint8_t *)img + 6) : 0);
        spingalett_free(img);
        free(x); free(y); free_network(net);
    }
    CHECK(!memcmp(w[0], w[1], (5 * 18 + 3 * 20) * sizeof(float)), "named inputs change the training of a chain");
    free(w[0]); free(w[1]);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
}

/* predict(), forward(), models of every precision and files agree on graphs; batched model
   prediction equals single runs bit for bit; outputs share memory. */
static void graph_inference(ComputeMode mode) {
    spingalett_set_compute_mode(mode);
    for (int which = 0; which < 6; which++) {
        lcg_state = 4000 + which;
        spingalett_seed(10 + which);
        NeuralNetwork *net = which < 5 ? graph_net(which) : resnet(3, 16, 0.0f);
        graph_init(net);
        const uint32_t n = 37, in = net->topology[0], out = net->topology[net->layers - 1];
        float *x, *y;
        make_data(n, in, out, net->loss_func, net->act_func[net->layers - 2], &x, &y);
        /* a few steps, so that the running statistics are not the initial ones */
        train(.net = net, .inputs = x, .targets = y, .sample_count = n, .epochs = 2, .learning_rate = 0.01f,
              .optimizer_type = OPTIMIZER_ADAM, .training_strategy = STRATEGY_SMALL_BATCH, .batch_size = 8);
        float *p = malloc((size_t)n * out * sizeof(float)), *q = malloc((size_t)n * out * sizeof(float));
        CHECK(predict(.net = net, .inputs = x, .sample_count = n, .outputs = p), "graph %d: predict failed", which);
        float worst = 0.0f;
        for (uint32_t s = 0; s < n; s++) {
            const float *o = forward(.net = net, .input = x + (size_t)s * in);
            for (uint32_t k = 0; k < out; k++) worst = fmaxf(worst, fabsf(o[k] - p[(size_t)s * out + k]));
        }
        CHECK(worst < 1e-5f, "graph %d mode %d: forward differs from predict by %g", which, mode, worst);

        /* through a file: format 6, the same network and the same outputs */
        size_t size = 0;
        void *img = spingalett_save_to_memory(net, PRECISION_FLOAT32, true, &size);
        CHECK(img && rd16((const uint8_t *)img + 6) == 6, "graph %d: written in format %u", which,
              img ? rd16((const uint8_t *)img + 6) : 0);
        NeuralNetwork *back = img ? load_spingalett_from_memory(img, size) : NULL;
        CHECK(back && back->layers == net->layers && back->total_weights == net->total_weights,
              "graph %d: does not load back", which);
        if (back) {
            for (uint32_t l = 1; l < net->layers; l++) {
                SpingalettNetworkLayer u, v;
                spingalett_network_layer(net, l, &u);
                spingalett_network_layer(back, l, &v);
                CHECK(!memcmp(&u, &v, sizeof u), "graph %d: layer %u differs after loading", which, l);
            }
            CHECK(predict(.net = back, .inputs = x, .sample_count = n, .outputs = q) &&
                  !memcmp(p, q, (size_t)n * out * sizeof(float)), "graph %d: loaded network predicts otherwise", which);
            free_network(back);
        }
        spingalett_free(img);

        for (int pr = 0; pr < PRECISION_COUNT; pr++) {
            SpingalettModel *m = spingalett_model_from_network(net, (PrecisionMode)pr);
            CHECK(m != NULL, "graph %d: no model in precision %d", which, pr);
            if (!m) continue;
            CHECK(spingalett_model_predict(m, x, n, q), "graph %d: model predict failed", which);
            float *r = malloc(out * sizeof(float)), diff = 0.0f;
            bool exact = true;
            void *ws = malloc(m->workspace_size);
            for (uint32_t s = 0; s < n; s++) {
                CHECK(spingalett_model_run(m, x + (size_t)s * in, r, ws) == SPINGALETT_OK, "graph %d: run failed", which);
                exact = exact && !memcmp(r, q + (size_t)s * out, out * sizeof(float));
                for (uint32_t k = 0; k < out; k++) diff = fmaxf(diff, fabsf(q[(size_t)s * out + k] - p[(size_t)s * out + k]));
            }
            /* integer models compute exactly what single runs compute */
            CHECK(exact || pr < PRECISION_INT8, "graph %d mode %d precision %d: batched prediction differs from single runs",
                  which, mode, pr);
            static const float tolerance[PRECISION_COUNT] = {1e-4f, 5e-3f, 3e-2f, 6e-2f, 0.6f, 1.0f};
            CHECK(diff < tolerance[pr], "graph %d precision %d: the model differs from the network by %g", which, pr,
                  diff);
            free(ws); free(r);
            spingalett_model_free(m);
        }
        free(p); free(q); free(x); free(y);
        free_network(net);
    }

    /* a deep residual network's outputs share memory: a few of the widest, not all of them */
    NeuralNetwork *deep = resnet(16, 32, 0.0f);
    SpingalettModel *m = spingalett_model_from_network(deep, PRECISION_FLOAT32);
    uint64_t all = 0;
    for (uint32_t l = 1; l < deep->layers; l++) all += deep->topology[l] * 4u;
    CHECK(m && m->activations_ <= 4u * 8 * 8 * 32 * 4 && m->activations_ * 10 < all,
          "residual model: %zu bytes of activations for %llu of outputs", m ? m->activations_ : 0, (unsigned long long)all);
    if (mode == COMPUTE_SINGLE_THREADED)
        printf("  residual network of 16 blocks: %zu bytes of activations in the engine's workspace (outputs: %llu)\n",
               m ? m->activations_ : 0, (unsigned long long)all);
    spingalett_model_free(m);
    free_network(deep);
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
}

/* Normalizations fold into products that feed nothing else; graphs keep the others. */
static void graph_folding(void) {
    lcg_state = 5000;
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_CROSS_ENTROPY);
    layer(.net = net, .height = 5, .width = 5, .channels = 2);
    uint32_t a = conv2d(.net = net, .filters = 3, .kernel = 3, .padding = 1, .act_func = ACT_NONE);
    uint32_t b = batch_norm(.net = net, .act_func = ACT_RELU);          /* folds: a feeds only b */
    uint32_t c = conv2d(.net = net, .filters = 3, .kernel = 3, .padding = 1, .act_func = ACT_NONE);
    uint32_t d = batch_norm(.net = net, .act_func = ACT_NONE);          /* stays: c also feeds e */
    uint32_t e = add_layers(.net = net, .inputs = {b, c, d}, .act_func = ACT_RELU);
    (void)a; (void)e;
    global_avg_pool2d(.net = net);
    layer(.net = net, .neurons_amount = 2, .act_func = ACT_SOFTMAX);
    graph_init(net);
    for (uint64_t i = 0; i < net->total_biases; i++) {
        net->running_mean[i] = frand() * 0.4f - 0.2f;
        net->running_var[i] = 0.5f + frand();
    }
    SpingalettModel *m = spingalett_model_from_network(net, PRECISION_FLOAT32);
    CHECK(m && m->layer_count == net->layers - 2, "graph folding: %u layers, expected %u", m ? m->layer_count : 0,
          net->layers - 2);
    float x[50], p[2], q[2];
    for (int i = 0; i < 50; i++) x[i] = frand() * 2 - 1;
    predict(.net = net, .inputs = x, .sample_count = 1, .outputs = p);
    if (m) {
        spingalett_model_predict(m, x, 1, q);
        CHECK(fabsf(p[0] - q[0]) < 1e-5f && fabsf(p[1] - q[1]) < 1e-5f, "graph folding: outputs %g %g vs %g %g",
              q[0], q[1], p[0], p[1]);
        SpingalettLayerInfo info;
        spingalett_model_layer(m, 3, &info);    /* the addition, after conv, conv, normalization */
        CHECK(info.type == LAYER_ADD && info.input_count == 3 && info.input_layers[0] == 1 && info.input_layers[1] == 2 &&
              info.input_layers[2] == 3, "graph folding: the addition reads %u layers", info.input_count);
    }
    spingalett_model_free(m);
    free_network(net);
}

/* Builders reject inputs that do not fit; unused layers fail training and prediction; corrupt
   version 6 images are rejected. */
static void graph_validation(void) {
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_MSE);
    CHECK(layer(.net = net, .neurons_amount = 4, .inputs = {1}) == SPINGALETT_NO_LAYER, "the input layer reads nothing");
    CHECK(layer(.net = net, .height = 4, .width = 4, .channels = 2) == 0, "input layer index");
    uint32_t a = conv2d(.net = net, .filters = 3, .kernel = 3, .padding = 1);
    uint32_t b = conv2d(.net = net, .filters = 2, .kernel = 3, .padding = 1);
    uint32_t c = max_pool2d(.net = net, .inputs = {a}, .kernel = 2);
    CHECK(a == 1 && b == 2 && c == 3, "layer indices %u %u %u", a, b, c);
    CHECK(add_layers(.net = net, .inputs = {a, b}) == SPINGALETT_NO_LAYER, "adding layers of other shapes");
    CHECK(concat_layers(.net = net, .inputs = {a, c}) == SPINGALETT_NO_LAYER, "concatenating other sizes");
    CHECK(conv2d(.net = net, .inputs = {a, b}, .filters = 2, .kernel = 1) == SPINGALETT_NO_LAYER, "conv of two layers");
    CHECK(layer(.net = net, .inputs = {9}, .neurons_amount = 2) == SPINGALETT_NO_LAYER, "a later layer as input");
    CHECK(layer(.net = net, .inputs = {1, 2}, .input_count = 17, .neurons_amount = 2) == SPINGALETT_NO_LAYER,
          "more than SPINGALETT_MAX_INPUTS inputs");
    CHECK(net->layers == 4, "rejected layers left the network with %u layers", net->layers);
    uint32_t d = concat_layers(.net = net, .inputs = {a, b, a}, .act_func = ACT_RELU);
    SpingalettNetworkLayer info;
    CHECK(d == 4 && spingalett_network_layer(net, d, &info) && info.channels == 8 && info.input_count == 3 &&
          info.inputs[2] == a, "concatenation of 3 + 2 + 3 channels");
    layer(.net = net, .neurons_amount = 2, .act_func = ACT_SIGMOID);      /* reads d; c is left unused */
    float x[32] = {0}, y[2] = {0}, o[2];
    spingalett_clear_error();
    TrainReport r = train(.net = net, .inputs = x, .targets = y, .sample_count = 1, .epochs = 1);
    CHECK(r.status == TRAIN_FAILED && spingalett_last_error_code() == SPINGALETT_ERR_INVALID &&
          strstr(spingalett_last_error_message(), "layer 3"), "training with an unused layer: %s",
          spingalett_last_error_message());
    CHECK(!predict(.net = net, .inputs = x, .sample_count = 1, .outputs = o), "prediction with an unused layer");
    CHECK(!forward(.net = net, .input = x), "forward with an unused layer");
    free_network(net);

    /* corrupt images: an input after the layer, an output over an input, channels that do not add up */
    net = graph_net(2);
    graph_init(net);
    size_t size = 0;
    uint8_t *img = spingalett_save_to_memory(net, PRECISION_FLOAT32, false, &size);
    uint8_t *buf = malloc(size);
    SpingalettModel m;
    CHECK(spingalett_model_init(&m, img, size) == SPINGALETT_OK, "graph image does not load");
    const size_t entry = 112, concat = 64 + 3 * entry;       /* layer 4, the concatenation */
    struct { size_t at; uint32_t value; const char *what; } bad[] = {
        {64 + 2 * entry + 84, 3, "an entry reading itself"},
        {64 + 1 * entry + 84, 5, "an entry reading a later layer"},
        {concat + 80, 17, "17 inputs"},
        {concat + 80, 2, "a concatenation of fewer channels"},
        {64 + 4 * entry + 96, (uint32_t)rd64(img + 64 + 3 * entry + 96), "an output over its input"},
        {64 + 4 * entry + 96, 3, "a misaligned output"},
        {40, 16, "too few bytes of activations"},
        {64 + 4 * entry + 10, 5, "a convolution marked as an addition"},
    };
    for (size_t t = 0; t < sizeof bad / sizeof bad[0]; t++) {
        memcpy(buf, img, size);
        wr32(buf + bad[t].at, bad[t].value);
        if (bad[t].at == 64 + 4 * entry + 10) buf[bad[t].at + 1] = 0, buf[bad[t].at + 2] = 0, buf[bad[t].at + 3] = 0;
        reseal(buf, size);
        CHECK(spingalett_model_init(&m, buf, size) != SPINGALETT_OK, "graph validation: %s accepted", bad[t].what);
    }
    free(buf);
    spingalett_free(img);
    free_network(net);
}

/* ------------------------------------------------------------------------- ONNX import */

/* Models exported from PyTorch (Tests/make_onnx_tests.py) compute what PyTorch computed. */
static void onnx_models(void) {
    static const struct { const char *name; uint32_t layers; LossFunction loss; } cases[] = {
        {"onnx_mlp", 4, LOSS_CROSS_ENTROPY}, {"onnx_cnn", 6, LOSS_MSE}, {"onnx_resnet", 15, LOSS_CROSS_ENTROPY},
        {"onnx_resnet_dynamo", 0, LOSS_CROSS_ENTROPY}, {"onnx_matmul", 2, LOSS_CROSS_ENTROPY},
    };
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        char path[512];
        snprintf(path, sizeof path, "%s/%s.bin", SPINGALETT_TEST_DATA_DIR, cases[c].name);
        long size = 0;
        unsigned char *data = read_file(path, &size);
        CHECK(data && size >= 16 && !memcmp(data, "SPGT", 4), "%s: no expected data", path);
        if (!data) continue;
        uint32_t n = rd32(data + 4), in = rd32(data + 8), out = rd32(data + 12);
        const float *x = (const float *)(const void *)(data + 16), *expected = x + (size_t)n * in;
        snprintf(path, sizeof path, "%s/%s.onnx", SPINGALETT_TEST_DATA_DIR, cases[c].name);
        NeuralNetwork *net = spingalett_import_onnx(path);
        CHECK(net != NULL, "%s: import failed: %s", cases[c].name, spingalett_last_error_message());
        if (!net) { free(data); continue; }
        CHECK(spingalett_input_size(net) == in && spingalett_output_size(net) == out && net->loss_func == cases[c].loss &&
              (!cases[c].layers || net->layers == cases[c].layers), "%s: %u layers, %u -> %u, loss %d", cases[c].name,
              net->layers, spingalett_input_size(net), spingalett_output_size(net), net->loss_func);
        float *y = malloc((size_t)n * out * sizeof(float));
        float worst = 0.0f;
        predict(.net = net, .inputs = x, .sample_count = n, .outputs = y);
        for (size_t i = 0; i < (size_t)n * out; i++) worst = fmaxf(worst, fabsf(y[i] - expected[i]));
        CHECK(worst < 1e-5f, "%s: outputs differ from PyTorch's by %g", cases[c].name, worst);
        /* and as a deployment model, through a file */
        SpingalettModel *m = spingalett_model_from_network(net, PRECISION_FLOAT32);
        float mworst = INFINITY;
        if (m) {
            spingalett_model_predict(m, x, n, y);
            mworst = 0.0f;
            for (size_t i = 0; i < (size_t)n * out; i++) mworst = fmaxf(mworst, fabsf(y[i] - expected[i]));
            spingalett_model_free(m);
        }
        CHECK(mworst < 1e-5f, "%s: the model differs from PyTorch's outputs by %g", cases[c].name, mworst);
        printf("  %-20s %3u layers, %llu parameters, |outputs - PyTorch| %.1e (model %.1e)\n", cases[c].name, net->layers,
               (unsigned long long)spingalett_parameter_count(net), worst, mworst);
        free(y);
        free(data);
        free_network(net);
    }

    /* an operator it does not have is named; bytes that are no model fail */
    char path[512];
    snprintf(path, sizeof path, "%s/onnx_unsupported.onnx", SPINGALETT_TEST_DATA_DIR);
    spingalett_clear_error();
    CHECK(!spingalett_import_onnx(path) && strstr(spingalett_last_error_message(), "Resize"),
          "an unsupported operator: %s", spingalett_last_error_message());
    static const unsigned char junk[] = {0x0a, 0xff, 0xff, 0xff, 0xff, 0x0f, 0x12};
    CHECK(!spingalett_import_onnx_from_memory(junk, sizeof junk) && !spingalett_import_onnx_from_memory(junk, 0) &&
          !spingalett_import_onnx_from_memory(NULL, 4), "junk accepted as an ONNX model");
    /* truncations of a real model never crash and never succeed */
    snprintf(path, sizeof path, "%s/onnx_resnet.onnx", SPINGALETT_TEST_DATA_DIR);
    long size = 0;
    unsigned char *model = read_file(path, &size);
    int accepted = 0;
    for (long cut = 0; model && cut < size; cut += 97) {
        NeuralNetwork *net = spingalett_import_onnx_from_memory(model, (size_t)cut);
        accepted += net != NULL;
        free_network(net);
    }
    CHECK(accepted == 0, "%d truncated models imported", accepted);
    free(model);
}

/* PyTorch state dicts (torch.save and safetensors, Tests/make_test_models.py) in a network built to
   match compute what PyTorch computed; a network of other layers is refused and left unchanged. */
static NeuralNetwork *torch_cnn(void) {
    NeuralNetwork *net = new_spingalett(.loss_func = LOSS_MSE);
    layer(.net = net, .height = 8, .width = 8, .channels = 3);
    conv2d(.net = net, .filters = 8, .kernel = 3, .padding = 1, .act_func = ACT_NONE);
    batch_norm(.net = net, .act_func = ACT_RELU);
    max_pool2d(.net = net, .kernel = 2);
    conv2d(.net = net, .filters = 8, .kernel = 3, .padding = 1, .groups = 4, .act_func = ACT_NONE);
    batch_norm(.net = net, .act_func = ACT_RELU);
    layer(.net = net, .neurons_amount = 16, .act_func = ACT_TANH);
    layer(.net = net, .neurons_amount = 4, .act_func = ACT_NONE);
    return net;
}

static void pytorch_weights(void) {
    char path[512];
    snprintf(path, sizeof path, "%s/torch_cnn.bin", SPINGALETT_TEST_DATA_DIR);
    long size = 0;
    unsigned char *data = read_file(path, &size);
    CHECK(data && size >= 16 && !memcmp(data, "SPGT", 4), "%s: no expected data", path);
    if (!data) return;
    uint32_t n = rd32(data + 4), in = rd32(data + 8), out = rd32(data + 12);
    const float *x = (const float *)(const void *)(data + 16), *expected = x + (size_t)n * in;
    static const char *const files[] = {"torch_cnn.pt", "torch_cnn.safetensors"};
    for (int f = 0; f < 2; f++) {
        NeuralNetwork *net = torch_cnn();
        snprintf(path, sizeof path, "%s/%s", SPINGALETT_TEST_DATA_DIR, files[f]);
        CHECK(spingalett_load_pytorch(net, path, NULL, 0), "%s: %s", files[f], spingalett_last_error_message());
        float y[64], worst = 0.0f;
        predict(.net = net, .inputs = x, .sample_count = n, .outputs = y);
        for (size_t i = 0; i < (size_t)n * out; i++) worst = fmaxf(worst, fabsf(y[i] - expected[i]));
        CHECK(worst < 1e-5f && in == spingalett_input_size(net), "%s: outputs differ from PyTorch's by %g", files[f], worst);
        printf("  %-22s |outputs - PyTorch| %.1e\n", files[f], worst);
        free_network(net);
    }
    /* explicit module names, in another order: refused, as is a network of other layers */
    NeuralNetwork *net = torch_cnn();
    const float before = net->weights[0];
    static const char *const wrong[] = {"body.13", "body.0", "body.1", "body.4", "body.5", "body.15"};
    snprintf(path, sizeof path, "%s/torch_cnn.pt", SPINGALETT_TEST_DATA_DIR);
    CHECK(!spingalett_load_pytorch(net, path, wrong, 6) && net->weights[0] == before, "modules in the wrong order accepted");
    static const char *const right[] = {"body.0", "body.1", "body.4", "body.5", "body.13", "body.15"};
    CHECK(spingalett_load_pytorch(net, path, right, 6), "modules named in order: %s", spingalett_last_error_message());
    layer(.net = net, .neurons_amount = 2);
    CHECK(!spingalett_load_pytorch(net, path, NULL, 0) && strstr(spingalett_last_error_message(), "layer 8"),
          "a network with another layer: %s", spingalett_last_error_message());
    free_network(net);
    /* truncated files never crash and never load */
    unsigned char *file = read_file(path, &size);
    int accepted = 0;
    for (long cut = 0; file && cut < size; cut += 61) {
        NeuralNetwork *t = torch_cnn();
        accepted += spingalett_load_pytorch_from_memory(t, file, (size_t)cut, NULL, 0);
        free_network(t);
    }
    CHECK(accepted == 0, "%d truncated state dicts loaded", accepted);
    free(file);
    free(data);
}

int main(int argc, char **argv) {
    if (argc > 2 && !strcmp(argv[1], "export-headers")) {
        spingalett_set_verbose(false);
        return export_test_headers(argv[2]);
    }
    const char *only = argc > 1 ? argv[1] : "";
    spingalett_set_verbose(false);
    spingalett_set_num_threads(4);
    spingalett_seed(1);

    if (!*only || !strcmp(only, "grad")) {
        printf("[gradient checks]\n");
        L a[] = {{4, ACT_NONE}, {6, ACT_SIGMOID}, {5, ACT_TANH}, {3, ACT_NONE}};
        L b[] = {{4, ACT_NONE}, {6, ACT_TANH}, {3, ACT_SOFTMAX}};
        L c[] = {{4, ACT_NONE}, {7, ACT_TANH}, {3, ACT_SIGMOID}};
        L d[] = {{4, ACT_NONE}, {8, ACT_RELU}, {6, ACT_LEAKY_RELU}, {5, ACT_FOO52}, {2, ACT_SIGMOID}};
        L e[] = {{3, ACT_NONE}, {40, ACT_TANH}, {33, ACT_SIGMOID}, {3, ACT_SOFTMAX}};
        ComputeMode modes[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS};
        TrainingStrategy strats[] = {STRATEGY_FULL_BATCH, STRATEGY_SMALL_BATCH, STRATEGY_SAMPLE};
        for (int m = 0; m < 3; m++) for (int s = 0; s < 3; s++) {
            gradcheck("mse sig/tanh/none", LOSS_MSE, a, 4, modes[m], strats[s]);
            gradcheck("mse tanh/softmax", LOSS_MSE, b, 3, modes[m], strats[s]);
            gradcheck("ce tanh/softmax", LOSS_CROSS_ENTROPY, b, 3, modes[m], strats[s]);
            gradcheck("ce tanh/sigmoid", LOSS_CROSS_ENTROPY, c, 3, modes[m], strats[s]);
            gradcheck("mse relu/leaky/foo52", LOSS_MSE, d, 5, modes[m], strats[s]);
            gradcheck("ce wide (AVX tails)", LOSS_CROSS_ENTROPY, e, 4, modes[m], strats[s]);
        }
        static const char *conv_names[] = {"conv same/maxpool/softmax", "conv stride/avgpool/1x1",
                                           "conv rect/maxpool overlap", "conv deep relu/foo52/gap",
                                           "conv narrow window/relu/maxpool", "depthwise x2/pointwise",
                                           "groups 2/rect/grouped 1x1", "depthwise strided/maxpool",
                                           "16 channels a group, strided", "16 channels, padded 3x3"};
        for (int m = 0; m < 3; m++) for (int s = 0; s < 3; s++)
            for (int c = 0; c < 10; c++) {
                lcg_state = 1000 + c;
                gradcheck_fan_in_limit = c >= 8 ? 16u : 0u;
                gradcheck_net(conv_names[c], conv_net(c), modes[m], strats[s]);
            }
    }
    if (!*only || !strcmp(only, "conv")) {
        printf("[convolution and pooling]\n");
        conv_api();
        ComputeMode cm[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS};
        for (int m = 0; m < 3; m++) conv_chunks(cm[m]);
        conv_weight_gradient();
        for (int m = 0; m < 2; m++) conv_learns(cm[m]);
        deterministic_threads();
        augmentation();
    }
    if (!*only || !strcmp(only, "norm")) {
        printf("[batch normalization]\n");
        ComputeMode cm[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS};
        TrainingStrategy strats[] = {STRATEGY_FULL_BATCH, STRATEGY_SMALL_BATCH};
        for (int m = 0; m < 3; m++) for (int s = 0; s < 2; s++)
            for (int k = 0; k < 4; k++) norm_gradcheck(k, cm[m], strats[s]);
        for (int m = 0; m < 3; m++) norm_inference(cm[m]);
        norm_misc();
        norm_validation();
        norm_determinism();
        for (int m = 0; m < 2; m++) norm_learns(cm[m]);
    }
    if (!*only || !strcmp(only, "graph")) {
        printf("[graphs]\n");
        ComputeMode cm[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS};
        TrainingStrategy strats[] = {STRATEGY_FULL_BATCH, STRATEGY_SMALL_BATCH};
        for (int m = 0; m < 3; m++) for (int s = 0; s < 2; s++)
            for (int k = 0; k < 5; k++) graph_gradcheck(k, cm[m], strats[s]);
        graph_determinism();
        for (int m = 0; m < 3; m++) graph_inference(cm[m]);
        graph_folding();
        graph_validation();
    }
    if (!*only || !strcmp(only, "onnx")) {
        printf("[ONNX import, PyTorch weights]\n");
        onnx_models();
        pytorch_weights();
    }
    if (!*only || !strcmp(only, "equiv")) {
        printf("[backend equivalence]\n");
        OptimizerType opts[] = {OPTIMIZER_SGD, OPTIMIZER_MOMENTUM, OPTIMIZER_RMSPROP, OPTIMIZER_ADAM, OPTIMIZER_ADAMW};
        for (int o = 0; o < 5; o++) {
            equivalence(opts[o], STRATEGY_FULL_BATCH, 0.0f, 0.0f);
            equivalence(opts[o], STRATEGY_FULL_BATCH, 0.01f, 0.0f);
            equivalence(opts[o], STRATEGY_SAMPLE, 0.0f, 0.0f);
            equivalence(opts[o], STRATEGY_SAMPLE, 0.01f, 0.0f);
        }
        equivalence(OPTIMIZER_ADAM, STRATEGY_FULL_BATCH, 0.0f, 0.05f);
        equivalence(OPTIMIZER_ADAM, STRATEGY_SAMPLE, 0.0f, 0.05f);
    }
    if (!*only || !strcmp(only, "cont")) {
        printf("[continued training]\n");
        continuation(OPTIMIZER_ADAM);
        continuation(OPTIMIZER_ADAMW);
        continuation(OPTIMIZER_RMSPROP);
        OptimizerType opts[] = {OPTIMIZER_SGD, OPTIMIZER_MOMENTUM, OPTIMIZER_RMSPROP, OPTIMIZER_ADAM, OPTIMIZER_ADAMW};
        ComputeMode cm[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS};
        for (int m = 0; m < 3; m++) { clip_norm(STRATEGY_SAMPLE, cm[m]); clip_norm(STRATEGY_FULL_BATCH, cm[m]); }
        for (int m = 0; m < 3; m++) chunked_batches(cm[m]);
        for (int o = 0; o < 5; o++) { strategy_consistency(opts[o], 0.05f, 0.0f); strategy_consistency(opts[o], 0.0f, 0.02f); }
    }
    if (!*only || !strcmp(only, "optim")) {
        printf("[initialization, optimizer formulas, shuffling]\n");
        initialization();
        optimizer_first_step(OPTIMIZER_ADAM);
        optimizer_first_step(OPTIMIZER_ADAMW);
        optimizer_first_step(OPTIMIZER_RMSPROP);
        shuffling();
    }
    if (!*only || !strcmp(only, "sched")) {
        printf("[lr schedulers]\n");
        schedulers();
    }
    if (!*only || !strcmp(only, "dropout")) {
        printf("[dropout]\n");
        ComputeMode cm[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS};
        for (int m = 0; m < 3; m++) {
            dropout_gradcheck(cm[m], STRATEGY_FULL_BATCH);
            dropout_gradcheck(cm[m], STRATEGY_SMALL_BATCH);
            dropout_gradcheck(cm[m], STRATEGY_SAMPLE);
        }
        dropout_mask_stats();
        dropout_equivalence(STRATEGY_FULL_BATCH, 0);
        dropout_equivalence(STRATEGY_SMALL_BATCH, 8);
        dropout_equivalence(STRATEGY_SAMPLE, 0);
        dropout_misc();
        dropout_xor();
    }
    if (!*only || !strcmp(only, "gen")) {
        printf("[generator mode]\n");
        ComputeMode cm[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS};
        for (int m = 0; m < 3; m++) generator_mode(cm[m]);
    }
    if (!*only || !strcmp(only, "predict")) {
        printf("[batched inference]\n");
        ComputeMode cm[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS};
        for (int m = 0; m < 3; m++) predict_matches_forward(cm[m]);
    }
    if (!*only || !strcmp(only, "valid")) {
        printf("[evaluation, validation, early stopping]\n");
        evaluate_matches_manual();
        ComputeMode cm[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS};
        for (int m = 0; m < 3; m++) { early_stopping(cm[m], STRATEGY_SMALL_BATCH); early_stopping(cm[m], STRATEGY_SAMPLE); }
        early_stopping(COMPUTE_SINGLE_THREADED, STRATEGY_FULL_BATCH);
        train_report_misc();
    }
    if (!*only || !strcmp(only, "step")) {
        printf("[low-level training API]\n");
        L a[] = {{4, ACT_NONE}, {6, ACT_SIGMOID}, {5, ACT_TANH}, {3, ACT_NONE}};
        L b[] = {{4, ACT_NONE}, {6, ACT_TANH}, {3, ACT_SOFTMAX}};
        L c[] = {{4, ACT_NONE}, {8, ACT_LEAKY_RELU}, {3, ACT_SIGMOID}};
        ComputeMode cm[] = {COMPUTE_SINGLE_THREADED, COMPUTE_OPENMP, COMPUTE_OPENBLAS};
        for (int m = 0; m < 3; m++) {
            trainer_gradcheck("mse sig/tanh/none", LOSS_MSE, a, 4, cm[m]);
            trainer_gradcheck("ce tanh/softmax", LOSS_CROSS_ENTROPY, b, 3, cm[m]);
            trainer_gradcheck("ce leaky/sigmoid", LOSS_CROSS_ENTROPY, c, 3, cm[m]);
            trainer_matches_train(cm[m]);
        }
        trainer_custom_grads(LOSS_MSE, ACT_NONE);
        trainer_custom_grads(LOSS_MSE, ACT_SOFTMAX);
        trainer_custom_grads(LOSS_CROSS_ENTROPY, ACT_SOFTMAX);
        trainer_custom_grads(LOSS_CROSS_ENTROPY, ACT_SIGMOID);
        label_smoothing();
        plateau();
    }
    if (!*only || !strcmp(only, "data")) {
        printf("[data sets]\n");
        datasets();
        cifar_files();
        dataset_files();
        dataset_files_v2();
    }
    if (!*only || !strcmp(only, "io")) {
        printf("[save/load]\n");
        roundtrip(PRECISION_FLOAT32, 0.0f);
        roundtrip(PRECISION_FP16, 1e-3f);
        roundtrip(PRECISION_BFLOAT16, 8e-3f);
        roundtrip(PRECISION_INT8, 1.5e-2f);
        load_robustness();
        error_codes();
    }
    if (!*only || !strcmp(only, "model")) {
        printf("[deployment models]\n");
        for (int p = 0; p < PRECISION_COUNT; p++) model_kernels((PrecisionMode)p);
        model_inference(PRECISION_FLOAT32, 1e-5f);
        model_inference(PRECISION_FP16, 2e-3f);
        model_inference(PRECISION_BFLOAT16, 2e-2f);
        model_inference(PRECISION_INT8, 3e-2f);
        model_inference(PRECISION_INT4, 0.1f);
        model_inference(PRECISION_INT2, 0.5f);
        model_shared();
        model_validation();
        model_files();
        for (int p = 0; p < PRECISION_COUNT; p++) model_conv_kernels((PrecisionMode)p);
        model_conv();
    }
    if (!*only || !strcmp(only, "xor")) {
        printf("[xor convergence]\n");
        OptimizerType opts[] = {OPTIMIZER_SGD, OPTIMIZER_MOMENTUM, OPTIMIZER_RMSPROP, OPTIMIZER_ADAM, OPTIMIZER_ADAMW};
        for (int o = 0; o < 5; o++) {
            xor_converges(opts[o], STRATEGY_SAMPLE, COMPUTE_SINGLE_THREADED);
            xor_converges(opts[o], STRATEGY_FULL_BATCH, COMPUTE_OPENBLAS);
            xor_converges(opts[o], STRATEGY_SMALL_BATCH, COMPUTE_OPENMP);
        }
    }
    printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures != 0;
}
