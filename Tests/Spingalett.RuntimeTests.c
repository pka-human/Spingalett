/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The runtime library on its own: a C99 program of Spingalett.Runtime.h, linked with nothing else.
 *
 *   SpingalettRuntimeTests DATA_DIR [EXPORT_DIR]
 *
 * Checks the runtime's API (versions, errors, settings, logging) and the models of DATA_DIR/runtime,
 * written by earlier releases, which every 1.x runtime must run: integer models give the engine's
 * single runs bit for bit, float ones agree with them within rounding. With EXPORT_DIR, written by
 * "SpingalettTests export-models", every model there must give the full library's results bit for
 * bit: batched on one thread and on four, sample by sample, and evaluated.
 */

#include <Spingalett/Spingalett.Runtime.h>
#include "Spingalett.RuntimeData.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    void *data = NULL;
    long n = -1;
    if (fseek(f, 0, SEEK_END) == 0) n = ftell(f);
    if (n >= 0 && fseek(f, 0, SEEK_SET) == 0 && (data = malloc((size_t)n + 1)) != NULL &&
        fread(data, 1, (size_t)n, f) != (size_t)n) {
        free(data);
        data = NULL;
    }
    fclose(f);
    if (data) *size = (size_t)n;
    return data;
}

/* dir/name, false when it does not fit */
static bool join(char *path, size_t size, const char *dir, const char *name) {
    int n = snprintf(path, size, "%s/%s", dir, name);
    return n > 0 && (size_t)n < size;
}

static int warnings = 0;
static void count_warnings(SpingalettLogLevel level, const char *message) {
    if (level == SPINGALETT_LOG_WARNING && strstr(message, "OpenBLAS")) warnings++;
}

static void api(const char *data_dir) {
    char path[1024];
    printf("[api]\n");
    CHECK(!strcmp(spingalett_version(), SPINGALETT_VERSION_STRING), "version %s, headers %s", spingalett_version(),
          SPINGALETT_VERSION_STRING);
    CHECK(spingalett_cpu_kernels() && spingalett_cpu_kernels()[0], "cpu kernels");
    printf("  version %s, kernels %s\n", spingalett_version(), spingalett_cpu_kernels());

    spingalett_clear_error();
    CHECK(!spingalett_set_compute_mode(SPINGALETT_COMPUTE_COUNT) && spingalett_last_error_code() == SPINGALETT_ERR_INVALID,
          "an unknown compute mode is refused");
    CHECK(spingalett_set_compute_mode(SPINGALETT_COMPUTE_OPENMP) &&
          spingalett_get_compute_mode() == SPINGALETT_COMPUTE_OPENMP, "compute mode");
    spingalett_set_num_threads(3);
    CHECK(spingalett_get_num_threads() == 3, "threads");
    spingalett_set_verbose(false);
    CHECK(!spingalett_get_verbose(), "verbose");

    spingalett_clear_error();
    CHECK(spingalett_last_error_code() == SPINGALETT_OK, "cleared error");
    CHECK(!spingalett_model_load("no/such/model.slett") && spingalett_last_error_code() == SPINGALETT_ERR_FILE_IO,
          "a missing file: %d", spingalett_last_error_code());
    join(path, sizeof path, data_dir, "xor_v1.nn");
    spingalett_clear_error();
    CHECK(!spingalett_model_load(path) && spingalett_last_error_code() == SPINGALETT_ERR_FORMAT_VERSION &&
          strstr(spingalett_last_error_message(), "ModelTool convert"),
          "a file of format 1 is refused with a hint: %d %s", spingalett_last_error_code(),
          spingalett_last_error_message());
    static const char junk[16] = "not a model";
    spingalett_clear_error();
    CHECK(!spingalett_model_from_memory(junk, sizeof junk) && spingalett_last_error_code() == SPINGALETT_ERR_FORMAT_VERSION,
          "junk is refused: %d", spingalett_last_error_code());
    spingalett_clear_error();
    CHECK(!spingalett_model_from_memory(NULL, 16) && spingalett_last_error_code() == SPINGALETT_ERR_INVALID, "NULL data");
    float x = 0, y = 0;
    spingalett_clear_error();
    CHECK(!spingalett_model_predict(NULL, &x, 1, &y) && spingalett_last_error_code() == SPINGALETT_ERR_INVALID,
          "predict without a model");
    spingalett_model_free(NULL);

    /* the runtime has no OpenBLAS: the mode runs single-threaded, and says so once */
    join(path, sizeof path, data_dir, "runtime/mlp_int8.slett");
    SpingalettModel *m = spingalett_model_load(path);
    CHECK(m != NULL, "%s: %s", path, spingalett_last_error_message());
    if (m) {
        float *in = (float *)calloc(m->input_size * 2, sizeof(float)), *out = (float *)malloc(m->output_size * 2 * sizeof(float));
        spingalett_set_log_callback(count_warnings);
        spingalett_set_compute_mode(SPINGALETT_COMPUTE_OPENBLAS);
        CHECK(spingalett_model_predict(m, in, 2, out) && spingalett_model_predict(m, in, 2, out) && warnings == 1,
              "OpenBLAS falls back with one warning (%d)", warnings);
        spingalett_set_log_callback(NULL);
        free(in);
        free(out);
        spingalett_model_free(m);
    }
    spingalett_set_compute_mode(SPINGALETT_COMPUTE_SINGLE_THREADED);
    spingalett_set_num_threads(4);
}

typedef struct {
    char name[96];
    unsigned count;
} Entry;

static int read_list(const char *dir, Entry *entries, int max) {
    char path[1024];
    FILE *f = join(path, sizeof path, dir, "models.txt") ? fopen(path, "r") : NULL;
    if (!f) return -1;
    int n = 0;
    while (n < max && fscanf(f, "%95s %u", entries[n].name, &entries[n].count) == 2) n++;
    fclose(f);
    return n;
}

static SpingalettModel *load(const char *dir, const char *name) {
    char path[1024], file[160];
    snprintf(file, sizeof file, "%s.slett", name);
    SpingalettModel *m = join(path, sizeof path, dir, file) ? spingalett_model_load(path) : NULL;
    CHECK(m != NULL, "%s: %s", path, spingalett_last_error_message());
    return m;
}

static bool is_integer(const SpingalettModel *m) {
    SpingalettLayerInfo info;
    return spingalett_model_layer(m, 0, &info) && info.precision >= SPINGALETT_PRECISION_INT8;
}

/* The engine's outputs for each sample (spingalett_model_run with a workspace of the caller's). */
static void run_samples(const SpingalettModel *m, const float *x, unsigned count, float *y) {
    void *workspace = malloc(m->workspace_size ? m->workspace_size : 4);
    int rc = SPINGALETT_OK;
    for (unsigned s = 0; s < count && rc == SPINGALETT_OK; s++)
        rc = spingalett_model_run(m, x + (size_t)s * m->input_size, y + (size_t)s * m->output_size, workspace);
    CHECK(rc == SPINGALETT_OK, "model run: %d", rc);
    free(workspace);
}

/* Models of earlier releases: batched prediction against the engine, on one thread and four, in one
   call and in pieces of 1, 7 and the rest. */
static void release_models(const char *data_dir) {
    char dir[1024];
    Entry entries[64];
    int n = join(dir, sizeof dir, data_dir, "runtime") ? read_list(dir, entries, 64) : -1;
    printf("[models of earlier releases]\n");
    CHECK(n > 0, "no models listed in %s", dir);
    for (int i = 0; i < n; i++) {
        SpingalettModel *m = load(dir, entries[i].name);
        if (!m) continue;
        const unsigned count = entries[i].count, in = m->input_size, out = m->output_size;
        float *x = (float *)malloc((size_t)count * in * sizeof(float));
        float *t = (float *)malloc((size_t)count * out * sizeof(float));
        float *ref = (float *)malloc((size_t)count * out * sizeof(float));
        float *y = (float *)malloc((size_t)count * out * sizeof(float));
        runtime_inputs(x, (size_t)count * in);
        runtime_targets(t, count, out);
        run_samples(m, x, count, ref);
        const bool exact = is_integer(m);
        double worst = 0;
        for (int v = 0; v < 3; v++) {
            spingalett_set_compute_mode(v == 1 ? SPINGALETT_COMPUTE_OPENMP : SPINGALETT_COMPUTE_SINGLE_THREADED);
            if (v < 2) {
                CHECK(spingalett_model_predict(m, x, count, y), "%s: predict", entries[i].name);
            } else {
                const unsigned pieces[3] = {1, 7, count - 8};
                for (unsigned p = 0, start = 0; p < 3; start += pieces[p], p++)
                    CHECK(spingalett_model_predict(m, x + (size_t)start * in, pieces[p], y + (size_t)start * out),
                          "%s: predict in pieces", entries[i].name);
            }
            for (size_t k = 0; k < (size_t)count * out; k++) {
                double d = fabs((double)y[k] - ref[k]) / (1e-3 + fabs((double)ref[k]));
                if (d > worst || d != d) worst = d != d ? INFINITY : d;
            }
        }
        SpingalettEvalMetrics e = spingalett_model_evaluate(m, x, t, count);
        CHECK(e.loss == e.loss && e.accuracy >= 0 && e.accuracy <= 1, "%s: evaluate", entries[i].name);
        CHECK(exact ? worst == 0 : worst < 1e-4, "%s: batched against single runs, relative difference %.2e",
              entries[i].name, worst);
        printf("  %-16s %u -> %u  %s\n", entries[i].name, in, out,
               exact ? (worst == 0 ? "bit-exact" : "MISMATCH") : worst < 1e-4 ? "within rounding" : "OFF");
        free(x); free(t); free(ref); free(y);
        spingalett_model_free(m);
    }
    spingalett_set_compute_mode(SPINGALETT_COMPUTE_SINGLE_THREADED);
}

static bool same_as(const char *dir, const char *name, const char *suffix, const float *y, size_t n) {
    char path[1024], file[160];
    size_t size = 0;
    snprintf(file, sizeof file, "%s.%s", name, suffix);
    float *expected = join(path, sizeof path, dir, file) ? (float *)read_file(path, &size) : NULL;
    bool same = expected && size == n * sizeof(float) && memcmp(expected, y, size) == 0;
    free(expected);
    return same;
}

/* Models written by the full library, with its results: the runtime's must be the same bits. */
static void library_models(const char *dir) {
    Entry *entries = (Entry *)malloc(512 * sizeof(Entry));
    int n = read_list(dir, entries, 512), same = 0;
    printf("[the full library's models]\n");
    CHECK(n > 0, "no models listed in %s", dir);
    for (int i = 0; i < n; i++) {
        SpingalettModel *m = load(dir, entries[i].name);
        if (!m) continue;
        const unsigned count = entries[i].count, in = m->input_size, out = m->output_size;
        const size_t outputs = (size_t)count * out;
        float *x = (float *)malloc((size_t)count * in * sizeof(float)), *t = (float *)malloc(outputs * sizeof(float));
        float *y = (float *)malloc(outputs * sizeof(float));
        runtime_inputs(x, (size_t)count * in);
        runtime_targets(t, count, out);
        spingalett_set_compute_mode(SPINGALETT_COMPUTE_SINGLE_THREADED);
        bool st = spingalett_model_predict(m, x, count, y) && same_as(dir, entries[i].name, "st", y, outputs);
        spingalett_set_compute_mode(SPINGALETT_COMPUTE_OPENMP);
        bool omp = spingalett_model_predict(m, x, count, y) && same_as(dir, entries[i].name, "omp", y, outputs);
        run_samples(m, x, count, y);
        bool run = same_as(dir, entries[i].name, "run", y, outputs);
        spingalett_set_compute_mode(SPINGALETT_COMPUTE_SINGLE_THREADED);
        SpingalettEvalMetrics e = spingalett_model_evaluate(m, x, t, count);
        float metrics[2] = {e.loss, e.accuracy};
        bool eval = same_as(dir, entries[i].name, "eval", metrics, 2);
        CHECK(st && omp && run && eval, "%s: the full library's results (one thread %d, four %d, single runs %d, "
              "evaluation %d)", entries[i].name, st, omp, run, eval);
        same += st && omp && run && eval;
        free(x); free(t); free(y);
        spingalett_model_free(m);
    }
    printf("  %d of %d models: the full library's bits\n", same, n);
    free(entries);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: SpingalettRuntimeTests DATA_DIR [EXPORT_DIR]\n");
        return 2;
    }
    api(argv[1]);
    release_models(argv[1]);
    if (argc > 2) library_models(argv[2]);
    printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
