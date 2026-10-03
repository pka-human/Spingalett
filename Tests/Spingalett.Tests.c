/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Spingalett test suite. Uses only the public API, so the same checks run against every build
 * configuration; backends that are not compiled in fall back to single-threaded and the
 * cross-backend comparisons then pass trivially.
 *
 *   Spingalett.Tests [group]     groups: grad equiv cont optim sched dropout gen io xor (default: all)
 *
 * Numerical gradients come from central differences of an independently computed loss; analytic
 * gradients from a single SGD step with lr = 1 (W_before - W_after). Everything is seeded, so
 * results are deterministic.
 */

#include <Spingalett/Spingalett.h>
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

// One full-batch SGD step with lr=1 => analytic grad = W_before - W_after.
static void gradcheck(const char *name, LossFunction loss, const L *ls, int nl, ComputeMode mode, TrainingStrategy strat) {
    uint32_t N = strat == STRATEGY_SAMPLE ? 1 : 6; lcg_state = 777;
    float *x, *y;
    make_data(N, ls[0].n, ls[nl - 1].n, loss, ls[nl - 1].act, &x, &y);
    spingalett_set_compute_mode(mode);
    NeuralNetwork *net = build(loss, ls, nl, NULL, NULL);
    for (uint64_t i = 0; i < net->total_weights; i++) net->weights[i] = frand() * 1.2f - 0.6f;
    for (uint64_t i = 0; i < net->total_biases; i++) net->biases[i] = frand() * 0.2f - 0.1f;
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
        double rel = fabs(gn - ga[i]) / fmax(1e-2, fabs(gn) + fabs(ga[i]));
        if (rel > maxrel) maxrel = rel;
        if (rel > 3e-2) bad++;
    }
    printf("  gradcheck %-28s mode=%d strat=%d  maxrel=%.2e  outliers=%d/%llu kinks=%d\n", name, mode, strat, maxrel, bad, (unsigned long long)(nw + nb), kinks);
    CHECK(bad == 0 && kinks * 10 < (int)(nw + nb), "gradcheck %s mode %d strat %d: %d outliers", name, mode, strat, bad);
    free(ga); free(w0); free(b0); free(x); free(y);
    free_network(net);
}

static float max_abs_diff(const float *a, const float *b, uint64_t n) {
    float m = 0; for (uint64_t i = 0; i < n; i++) { float d = fabsf(a[i] - b[i]); if (d > m) m = d; } return m;
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
    save_spingalett(.net = a, .filename = "spingalett_test_roundtrip.nn", .precision = p);
    NeuralNetwork *b = load_spingalett("spingalett_test_roundtrip.nn");
    remove("spingalett_test_roundtrip.nn");
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

static void load_robustness(void) {
    L ls[] = {{6, ACT_NONE}, {10, ACT_RELU}, {3, ACT_SIGMOID}};
    NeuralNetwork *a = build(LOSS_MSE, ls, 3, NULL, NULL);
    save_spingalett(.net = a, .filename = "spingalett_test_robust.nn");
    layer(.net = NULL, .neurons_amount = 3);              /* leaves a sticky error behind */
    NeuralNetwork *b = load_spingalett("spingalett_test_robust.nn");
    CHECK(b != NULL, "load failed after an unrelated earlier error");
    if (b) free_network(b);
    CHECK(truncate_file("spingalett_test_robust.nn", 7), "could not truncate test file");
    spingalett_clear_error();
    NeuralNetwork *c = load_spingalett("spingalett_test_robust.nn");
    CHECK(c == NULL && spingalett_last_error_code() != SPINGALETT_OK, "truncated file must fail to load");
    if (c) free_network(c);
    remove("spingalett_test_robust.nn");
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
static bool capture_loss(NeuralNetwork *n, size_t e, float err) { (void)n; (void)e; cb_loss = err; return false; }

/* Same seed + same time_step => same dropout masks, so train() with a negligible lr evaluates the
   masked loss, and train() with lr = 1 and SGD yields the masked gradient. */
static void dropout_run(NeuralNetwork *net, const float *x, const float *y, uint32_t N, float lr, TrainingStrategy st) {
    net->time_step = 0;
    spingalett_seed(1234);
    train(.net = net, .inputs = x, .targets = y, .sample_count = N, .epochs = 1, .learning_rate = lr,
          .optimizer_type = OPTIMIZER_SGD, .training_strategy = st, .batch_size = N,
          .callback = capture_loss, .callback_interval = 1);
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
    save_spingalett(.net = a, .filename = "spingalett_test_dropout.nn");
    NeuralNetwork *c = load_spingalett("spingalett_test_dropout.nn");
    remove("spingalett_test_dropout.nn");
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
        CHECK(a->time_step == 0 && empty.calls == 1, "empty generator must stop training");
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

int main(int argc, char **argv) {
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
    if (!*only || !strcmp(only, "io")) {
        printf("[save/load]\n");
        roundtrip(PRECISION_FLOAT32, 0.0f);
        roundtrip(PRECISION_FP16, 1e-3f);
        roundtrip(PRECISION_BFLOAT16, 8e-3f);
        roundtrip(PRECISION_INT8, 1.5e-2f);
        load_robustness();
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
