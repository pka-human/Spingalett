/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Inspects, quantizes and exports .slett models.
 *
 *   ModelTool info <model.slett>
 *   ModelTool convert <in.slett> <out.slett> [--precision P] [--no-optimizer]
 *                                      rewrite (any format version) as format 3, keeping the
 *                                      optimizer state when the input has some
 *   ModelTool header <model.slett> <out.h> <name> [--precision P]
 *                                      C header with the model image, for firmware
 *   ModelTool eval <model.slett> <data.slettd | images labels> [--precision P]
 *                                      accuracy and loss in every precision (or in P)
 *   ModelTool bench <model.slett> [--precision P]
 *                                      latency of one sample and batched throughput
 *
 * P is fp32, fp16, bf16, int8, int4 or int2; by default the precision the model is stored in.
 */

#include <Spingalett/Spingalett.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if !defined(TIME_UTC) && defined(_WIN32)
#include <windows.h>
#endif

static const char *precision_name(PrecisionMode p) {
    static const char *names[] = {"fp32", "fp16", "bf16", "int8", "int4", "int2"};
    return (unsigned)p < PRECISION_COUNT ? names[p] : "?";
}

static const char *activation_name(ActivationFunction a) {
    static const char *names[] = {"sigmoid", "relu", "tanh", "leaky relu", "foo52", "softmax", "none"};
    return (unsigned)a < ACT_COUNT ? names[a] : "?";
}

static bool parse_precision(const char *s, PrecisionMode *p) {
    static const char *names[][2] = {{"fp32", "float32"}, {"fp16", "half"}, {"bf16", "bfloat16"}, {"int8", "i8"}, {"int4", "i4"}, {"int2", "ternary"}};
    for (int i = 0; i < PRECISION_COUNT; i++)
        if (!strcmp(s, names[i][0]) || !strcmp(s, names[i][1])) { *p = (PrecisionMode)i; return true; }
    return false;
}

static double now(void) {
#if defined(TIME_UTC)
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#elif defined(_WIN32)
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

static int fail(const char *what) {
    fprintf(stderr, "%s: %s\n", what, spingalett_last_error_message());
    return 1;
}

/* The model as stored, its network (float) and the precision of its first layer. */
static bool open_model(const char *path, SpingalettModel **model, NeuralNetwork **net, PrecisionMode *stored) {
    *model = spingalett_model_load(path);
    *net = *model ? load_spingalett(path) : NULL;
    if (!*model || !*net) {
        spingalett_model_free(*model);
        return false;
    }
    SpingalettLayerInfo first;
    spingalett_model_layer(*model, 0, &first);
    *stored = first.precision;
    return true;
}

static int info(const char *path) {
    SpingalettModel *m;
    NeuralNetwork *net;
    PrecisionMode stored;
    if (!open_model(path, &m, &net, &stored)) return fail(path);
    printf("%s\n  %u inputs, %u outputs, %u weight layers, %llu parameters\n  image %zu bytes, workspace %zu bytes\n",
           path, m->input_size, m->output_size, m->layer_count,
           (unsigned long long)(net->total_weights + net->total_biases), m->image_size, m->workspace_size);
    for (uint32_t i = 0; i < m->layer_count; i++) {
        SpingalettLayerInfo l;
        spingalett_model_layer(m, i, &l);
        printf("  layer %u: %5u -> %-5u %-10s %s\n", i + 1, l.inputs, l.outputs, activation_name(l.activation), precision_name(l.precision));
    }
    spingalett_model_free(m);
    free_network(net);
    return 0;
}

static int convert(const char *in, const char *out, bool have_precision, PrecisionMode precision, bool optimizer) {
    SpingalettModel *m;
    NeuralNetwork *net;
    PrecisionMode stored;
    if (!open_model(in, &m, &net, &stored)) return fail(in);
    spingalett_model_free(m);
    /* files saved without optimizer state record no optimizer steps */
    save_spingalett(.net = net, .filename = out, .precision = have_precision ? precision : stored,
                    .do_not_save_optimizer = !optimizer || net->time_step == 0);
    free_network(net);
    if (spingalett_last_error_code() != SPINGALETT_OK) return fail(out);
    printf("wrote %s (%s)\n", out, precision_name(have_precision ? precision : stored));
    return 0;
}

static int header(const char *in, const char *out, const char *name, bool have_precision, PrecisionMode precision) {
    SpingalettModel *m;
    NeuralNetwork *net;
    PrecisionMode stored;
    if (!open_model(in, &m, &net, &stored)) return fail(in);
    spingalett_model_free(m);
    PrecisionMode p = have_precision ? precision : stored;
    bool ok = spingalett_export_c_header(net, out, name, p);
    SpingalettModel *q = ok ? spingalett_model_from_network(net, p) : NULL;
    free_network(net);
    if (!ok || !q) return fail(out);
    printf("wrote %s: %s, %zu bytes of model, %zu bytes of workspace\n", out, precision_name(p), q->image_size, q->workspace_size);
    spingalett_model_free(q);
    return 0;
}

static int eval(const char *path, const char *data, const char *labels, bool have_precision, PrecisionMode precision) {
    SpingalettModel *m;
    NeuralNetwork *net;
    PrecisionMode stored;
    if (!open_model(path, &m, &net, &stored)) return fail(path);
    spingalett_model_free(m);
    SpingalettDataset d;
    bool loaded = labels ? spingalett_load_idx(data, labels, net->topology[net->layers - 1], &d) : spingalett_load_dataset(data, &d);
    if (!loaded) {
        free_network(net);
        return fail(data);
    }
    if (d.input_size != net->topology[0] || d.target_size != net->topology[net->layers - 1]) {
        fprintf(stderr, "%s: %u inputs and %u targets per sample; the model has %u and %u\n", data, d.input_size,
                d.target_size, net->topology[0], net->topology[net->layers - 1]);
        spingalett_dataset_free(&d);
        free_network(net);
        return 1;
    }
    printf("%u samples\n  precision     bytes    accuracy      loss\n", d.count);
    int rc = 0;
    for (int p = 0; p < PRECISION_COUNT; p++) {
        if (have_precision && p != (int)precision) continue;
        SpingalettModel *q = spingalett_model_from_network(net, (PrecisionMode)p);
        if (!q) { rc = fail(precision_name((PrecisionMode)p)); continue; }
        EvalMetrics e = spingalett_model_evaluate(q, d.inputs, d.targets, d.count);
        printf("  %-6s %13zu    %7.2f%%  %8.4f%s\n", precision_name((PrecisionMode)p), q->image_size,
               100.0 * (double)e.accuracy, (double)e.loss, (PrecisionMode)p == stored ? "   (stored)" : "");
        spingalett_model_free(q);
    }
    spingalett_dataset_free(&d);
    free_network(net);
    return rc;
}

static int bench(const char *path, bool have_precision, PrecisionMode precision) {
    SpingalettModel *m;
    NeuralNetwork *net;
    PrecisionMode stored;
    if (!open_model(path, &m, &net, &stored)) return fail(path);
    spingalett_model_free(m);
    uint32_t n = 4096, in = net->topology[0], out = net->topology[net->layers - 1];
    float *x = malloc((size_t)n * in * sizeof(float)), *y = malloc((size_t)n * out * sizeof(float));
    for (size_t i = 0; i < (size_t)n * in; i++) x[i] = (float)((i * 2654435761u) % 1000) / 1000.0f;
    printf("  precision     bytes   one sample     batched\n");
    for (int p = 0; p < PRECISION_COUNT; p++) {
        if (have_precision && p != (int)precision) continue;
        SpingalettModel *q = spingalett_model_from_network(net, (PrecisionMode)p);
        if (!q) continue;
        void *ws = malloc(q->workspace_size);
        uint32_t runs = 0;
        double t0 = now(), t;
        do {
            spingalett_model_run(q, x + (size_t)(runs % n) * in, y, ws);
            runs++;
        } while ((t = now() - t0) < 0.3);
        double single = t / runs;
        t0 = now();
        spingalett_model_predict(q, x, n, y);
        double batched = (now() - t0) / n;
        printf("  %-6s %13zu  %8.2f us  %8.2f us/sample\n", precision_name((PrecisionMode)p), q->image_size, single * 1e6, batched * 1e6);
        free(ws);
        spingalett_model_free(q);
    }
    free(x);
    free(y);
    free_network(net);
    return 0;
}

static int usage(void) {
    fprintf(stderr,
        "usage: ModelTool info <model.slett>\n"
        "       ModelTool convert <in.slett> <out.slett> [--precision P] [--no-optimizer]\n"
        "       ModelTool header <model.slett> <out.h> <name> [--precision P]\n"
        "       ModelTool eval <model.slett> <data.slettd | images labels> [--precision P]\n"
        "       ModelTool bench <model.slett> [--precision P]\n"
        "P: fp32, fp16, bf16, int8, int4, int2 (default: as stored)\n");
    return 2;
}

int main(int argc, char **argv) {
    spingalett_set_verbose(false);
#if defined(SPINGALETT_HAS_OPENMP)
    spingalett_set_compute_mode(COMPUTE_OPENMP);
#endif
    const char *args[4] = {0};
    int nargs = 0;
    bool have_precision = false, optimizer = true;
    PrecisionMode precision = PRECISION_FLOAT32;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--precision") && i + 1 < argc) {
            if (!parse_precision(argv[++i], &precision)) return usage();
            have_precision = true;
        } else if (!strcmp(argv[i], "--no-optimizer")) {
            optimizer = false;
        } else if (nargs < 4) {
            args[nargs++] = argv[i];
        } else {
            return usage();
        }
    }
    const char *cmd = argc > 1 ? argv[1] : "";
    if (!strcmp(cmd, "info") && nargs == 1) return info(args[0]);
    if (!strcmp(cmd, "convert") && nargs == 2) return convert(args[0], args[1], have_precision, precision, optimizer);
    if (!strcmp(cmd, "header") && nargs == 3) return header(args[0], args[1], args[2], have_precision, precision);
    if (!strcmp(cmd, "eval") && (nargs == 2 || nargs == 3)) return eval(args[0], args[1], nargs == 3 ? args[2] : NULL, have_precision, precision);
    if (!strcmp(cmd, "bench") && nargs == 1) return bench(args[0], have_precision, precision);
    return usage();
}
