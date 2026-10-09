/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Runs a .slett model with the runtime alone (Spingalett.Runtime.h, libspingalett-runtime).
 *
 *   RunModel <model.slett>                          the model's layers, then the time of one sample
 *                                                   and of batches, on random inputs
 *   RunModel <model.slett> <inputs.f32> [outputs.f32]
 *                                                   predicts the samples of inputs.f32 (raw floats,
 *                                                   the model's input size per sample) and writes
 *                                                   their outputs, or prints each one's class
 *   --threads N                                     at most N threads (all cores by default)
 *
 * A program written for the runtime's header also builds against the full library, and computes
 * the same there.
 */

#define SPINGALETT_SHORT_NAMES          /* ComputeMode, COMPUTE_OPENMP, LAYER_DENSE, ACT_RELU, ... */
#include <Spingalett/Spingalett.Runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if !defined(TIME_UTC) && defined(_WIN32)
#include <windows.h>
#endif

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

static void describe(const SpingalettModel *m) {
    static const char *kinds[] = {"dense", "convolution", "max pooling", "average pooling", "batch normalization",
                                  "addition", "concatenation", "global average pooling", "transposed convolution",
                                  "upsampling", "layer normalization"};
    static const char *precisions[] = {"fp32", "fp16", "bf16", "int8", "int4", "int2"};
    static const char *activations[] = {"", "sigmoid", "relu", "tanh", "leaky relu", "foo52", "softmax"};
    printf("%u inputs -> %u outputs, %u layers, %zu bytes, workspace %zu bytes\n", m->input_size, m->output_size,
           m->layer_count, m->image_size, m->workspace_size);
    for (uint32_t i = 0; i < m->layer_count; i++) {
        SpingalettLayerInfo l;
        if (!spingalett_model_layer(m, i, &l)) break;
        printf("  %3u  %-22s %ux%ux%u", i + 1, (unsigned)l.type < LAYER_TYPE_COUNT ? kinds[l.type] : "?", l.height,
               l.width, l.channels);
        if (l.type == LAYER_DENSE || l.type == LAYER_CONV2D || l.type == LAYER_CONV_TRANSPOSE2D)
            printf("  %s", (unsigned)l.precision < PRECISION_COUNT ? precisions[l.precision] : "?");
        if ((unsigned)l.activation < ACT_COUNT && l.activation != ACT_NONE) printf("  %s", activations[l.activation]);
        printf("\n");
    }
}

static int compare_doubles(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* The median time of `repeats` predictions of `count` samples (after one that prepares the weights). */
static double time_predict(const SpingalettModel *m, const float *x, uint32_t count, float *y, int repeats) {
    double times[64];
    spingalett_model_predict(m, x, count, y);
    for (int r = 0; r < repeats; r++) {
        double t0 = now();
        spingalett_model_predict(m, x, count, y);
        times[r] = now() - t0;
    }
    qsort(times, (size_t)repeats, sizeof *times, compare_doubles);
    return times[repeats / 2];
}

static int bench(const SpingalettModel *m) {
    const uint32_t batch = 256, in = m->input_size, out = m->output_size;
    float *x = (float *)malloc((size_t)batch * in * sizeof(float)), *y = (float *)malloc((size_t)batch * out * sizeof(float));
    if (!x || !y) return 1;
    for (size_t i = 0; i < (size_t)batch * in; i++) x[i] = (float)((i * 2654435761u) % 2000) / 1000.0f - 1.0f;
    spingalett_set_compute_mode(COMPUTE_SINGLE_THREADED);
    double one = time_predict(m, x, 1, y, 63);
    spingalett_set_compute_mode(COMPUTE_OPENMP);
    double many = time_predict(m, x, batch, y, 15);
    printf("one sample (one thread): %.1f us; batches of %u: %.2f us a sample, %.0f samples/s (kernels %s)\n",
           one * 1e6, batch, many / batch * 1e6, batch / many, spingalett_cpu_kernels());
    free(x);
    free(y);
    return 0;
}

static int run(const SpingalettModel *m, const char *inputs, const char *outputs) {
    FILE *f = fopen(inputs, "rb");
    if (!f) { perror(inputs); return 1; }
    long bytes = -1;
    if (fseek(f, 0, SEEK_END) == 0) bytes = ftell(f);
    const size_t sample = (size_t)m->input_size * sizeof(float);
    if (bytes <= 0 || bytes % (long)sample || fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "%s: not a whole number of samples of %u floats\n", inputs, m->input_size);
        fclose(f);
        return 1;
    }
    const uint32_t count = (uint32_t)((size_t)bytes / sample);
    float *x = (float *)malloc((size_t)bytes), *y = (float *)malloc((size_t)count * m->output_size * sizeof(float));
    bool ok = x && y && fread(x, 1, (size_t)bytes, f) == (size_t)bytes;
    fclose(f);
    double t0 = now();
    if (ok && !spingalett_model_predict(m, x, count, y)) {
        free(x); free(y);
        return fail("predict");
    }
    double t = now() - t0;
    if (ok && outputs) {
        FILE *o = fopen(outputs, "wb");
        ok = o && fwrite(y, sizeof(float), (size_t)count * m->output_size, o) == (size_t)count * m->output_size;
        if (o && fclose(o) != 0) ok = false;
        if (ok) printf("%u samples in %.3f s: outputs in %s\n", count, t, outputs);
    } else if (ok) {
        for (uint32_t s = 0; s < count; s++) {
            const float *ys = y + (size_t)s * m->output_size;
            uint32_t best = 0;
            for (uint32_t k = 1; k < m->output_size; k++) if (ys[k] > ys[best]) best = k;
            if (m->output_size > 1) printf("%u: class %u (%.4f)\n", s, best, (double)ys[best]);
            else printf("%u: %.6f\n", s, (double)ys[0]);
        }
    }
    free(x);
    free(y);
    if (!ok) fprintf(stderr, "%s: cannot read the inputs or write the outputs\n", inputs);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    const char *files[3] = {NULL, NULL, NULL};
    int nfiles = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) spingalett_set_num_threads((unsigned)atoi(argv[++i]));
        else if (nfiles < 3 && argv[i][0] != '-') files[nfiles++] = argv[i];
        else nfiles = 4;
    }
    if (nfiles < 1 || nfiles > 3) {
        fprintf(stderr, "usage: RunModel <model.slett> [<inputs.f32> [outputs.f32]] [--threads N]\n");
        return 2;
    }
    spingalett_set_verbose(false);
    spingalett_set_compute_mode(COMPUTE_OPENMP);
    SpingalettModel *m = spingalett_model_load(files[0]);
    if (!m) return fail(files[0]);
    int rc;
    if (nfiles == 1) {
        printf("Spingalett %s, %s: ", spingalett_version(), files[0]);
        describe(m);
        rc = bench(m);
    } else {
        rc = run(m, files[1], files[2]);
    }
    spingalett_model_free(m);
    return rc;
}
