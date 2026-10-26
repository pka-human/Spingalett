/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The runtime library on its own: a C99 program of Spingalett.Runtime.h, linked with nothing else.
 *
 *   SpingalettRuntimeTests DATA_DIR [EXPORT_DIR]
 *
 * Checks the runtime's API (versions, errors, settings, logging), its tokenizers (the expected ids and texts
 * of DATA_DIR's tokenizer files, JSON and GGUF, as the full library's tests check them) and the models of DATA_DIR/runtime,
 * written by earlier releases, which every 1.x runtime must run: integer models give the engine's
 * single runs bit for bit, float ones agree with them within rounding. With EXPORT_DIR, written by
 * "SpingalettTests export-models", every model there must give the full library's results bit for
 * bit: batched on one thread and on four, sample by sample, evaluated, and for language models in a
 * session.
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
        spingalett_clear_error();
        CHECK(!spingalett_session_new(m, NULL) && spingalett_last_error_code() == SPINGALETT_ERR_INVALID,
              "a session of a model that is no language model");
        free(in);
        free(out);
        spingalett_model_free(m);
    }
    spingalett_set_compute_mode(SPINGALETT_COMPUTE_SINGLE_THREADED);
    spingalett_set_num_threads(4);
}

/* The bytes of a line of hexadecimal digits into out; their number. */
static size_t unhex(const char *hex, char *out) {
    size_t n = 0;
    for (; hex[0] && hex[1] && hex[0] != '\n' && hex[0] != '-'; hex += 2) {
        int hi = hex[0] <= '9' ? hex[0] - '0' : (hex[0] | 32) - 'a' + 10, lo = hex[1] <= '9' ? hex[1] - '0' : (hex[1] | 32) - 'a' + 10;
        out[n++] = (char)(hi * 16 + lo);
    }
    return n;
}

/* Every case of an expectations file for a tokenizer file of the data directory: the ids, and the text they
   decode to ("*": not checked). */
static void tokenizer_cases(const char *data_dir, const char *file, const char *expectations, int least) {
    enum { LINE = 1 << 16 };
    char path[1024];
    join(path, sizeof path, data_dir, file);
    SpingalettTokenizer *tok = spingalett_tokenizer_load(path);
    CHECK(tok != NULL, "%s: %s", path, spingalett_last_error_message());
    join(path, sizeof path, data_dir, expectations);
    FILE *f = fopen(path, "r");
    CHECK(f != NULL, "%s: no expected ids", path);
    char *line = (char *)malloc(LINE), *text = (char *)malloc(LINE), *got = (char *)malloc(LINE), *want = (char *)malloc(LINE);
    uint32_t *ids = (uint32_t *)malloc(LINE * sizeof(uint32_t)), *expect = (uint32_t *)malloc(LINE * sizeof(uint32_t));
    int cases = 0, wrong = 0;
    while (tok && f && fgets(line, LINE, f)) {
        int special = 0, add = 0, used = 0;
        unsigned count = 0, k;
        char *p;
        size_t n;
        int64_t m;
        if (line[0] == '#') continue;
        if (sscanf(line, "%d %d %n", &special, &add, &used) != 2) break;
        n = unhex(line + used, text);
        text[n] = '\0';
        if (!fgets(line, LINE, f) || sscanf(line, "%u%n", &count, &used) != 1 || count >= LINE) break;
        for (p = line + used, k = 0; k < count; k++) expect[k] = (uint32_t)strtoul(p, &p, 10);
        cases++;
        m = spingalett_tokenize(.tokenizer = tok, .text = n ? text : "", .length = n, .tokens = ids, .capacity = LINE,
                                .special = special != 0, .add_special = add != 0);
        if (m != (int64_t)count || (count && memcmp(ids, expect, count * sizeof(uint32_t)))) wrong++;
        if (!fgets(line, LINE, f)) break;
        if (line[0] != '*') {
            size_t wn = unhex(line, want);
            int64_t len = spingalett_detokenize(.tokenizer = tok, .tokens = expect, .count = count, .text = got,
                                                .capacity = LINE);
            if (len != (int64_t)wn || memcmp(got, want, wn)) wrong++;
        }
        if (!fgets(line, LINE, f)) break;
    }
    CHECK(cases >= least && !wrong, "%s: %d of %d texts tokenized or decoded otherwise than expected", file, wrong, cases);
    printf("  %-24s %d texts\n", file, cases);
    if (f) fclose(f);
    free(line); free(text); free(got); free(want); free(ids); free(expect);
    spingalett_tokenizer_free(tok);
}

static void tokenizers(const char *data_dir) {
    printf("[tokenizers]\n");
    tokenizer_cases(data_dir, "tokenizer_llama3.json", "tokenizer_llama3.txt", 100);
    tokenizer_cases(data_dir, "tokenizer_metaspace.json", "tokenizer_metaspace.txt", 50);
    tokenizer_cases(data_dir, "gguf_qwen2.gguf", "tokenizer_qwen2.txt", 100);
    tokenizer_cases(data_dir, "gguf_spm.gguf", "gguf_spm.txt", 60);
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

/* The samples of a model: token ids for a language model (its first layer an embedding). */
static void samples(const SpingalettModel *m, float *x, size_t n) {
    SpingalettLayerInfo info;
    if (spingalett_model_layer(m, 0, &info) && info.type == SPINGALETT_LAYER_EMBEDDING)
        runtime_tokens(x, n, info.vocabulary);
    else
        runtime_inputs(x, n);
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
        samples(m, x, (size_t)count * in);
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
    int n = read_list(dir, entries, 512), same = 0, sessions = 0;
    printf("[the full library's models]\n");
    CHECK(n > 0, "no models listed in %s", dir);
    for (int i = 0; i < n; i++) {
        SpingalettModel *m = load(dir, entries[i].name);
        if (!m) continue;
        const unsigned count = entries[i].count, in = m->input_size, out = m->output_size;
        const size_t outputs = (size_t)count * out;
        float *x = (float *)malloc((size_t)count * in * sizeof(float)), *t = (float *)malloc(outputs * sizeof(float));
        float *y = (float *)malloc(outputs * sizeof(float));
        samples(m, x, (size_t)count * in);
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
        /* a language model's session (the others have none: the error set, no file to compare) */
        bool session = true;
        SpingalettSession *probe = spingalett_session_new(m, NULL);
        if (probe) {
            const size_t floats = runtime_session_floats(in, out / in);
            float *logits = (float *)malloc(floats * sizeof(float));
            session = logits && runtime_session(m, x, logits) && same_as(dir, entries[i].name, "sess", logits, floats);
            free(logits);
            sessions++;
        }
        spingalett_session_free(probe);
        CHECK(st && omp && run && eval && session, "%s: the full library's results (one thread %d, four %d, single "
              "runs %d, evaluation %d, session %d)", entries[i].name, st, omp, run, eval, session);
        same += st && omp && run && eval && session;
        free(x); free(t); free(y);
        spingalett_model_free(m);
    }
    printf("  %d of %d models: the full library's bits (%d language models in sessions too)\n", same, n, sessions);
    CHECK(sessions > 0, "no language models among the full library's");
    free(entries);
}

/* Graphs of operators: Tests/Data/graph_small.onnx (Tests/make_graph_tests.py) run on its inputs against the
   reference's outputs, its declared inputs and outputs, a .slett file of format 11 giving the same bits, and the
   errors of a wrong input. */
static void graphs(const char *data_dir) {
    char path[1024], saved[1024];
    printf("[graphs]\n");
    size_t size = 0;
    unsigned char *bin = NULL;
    if (join(path, sizeof path, data_dir, "graph_small.bin")) bin = (unsigned char *)read_file(path, &size);
    CHECK(bin && size >= 4, "%s: no expectations", path);
    if (!bin) return;
    uint32_t rows;
    memcpy(&rows, bin, 4);
    CHECK(size == 4 + rows * (4 * 4 + 3 * 4 + 8), "graph_small.bin of %u bytes", (unsigned)size);
    const float *x = (const float *)(bin + 4), *want = x + rows * 4;
    int64_t want_k[16];
    memcpy(want_k, want + rows * 3, rows * 8u);
    join(path, sizeof path, data_dir, "graph_small.onnx");
    SpingalettGraph *g = spingalett_graph_open(.path = path);
    CHECK(g, "graph_small.onnx: %s", spingalett_last_error_message());
    if (!g) { free(bin); return; }
    SpingalettTensorInfo in = spingalett_graph_input(g, 0), out = spingalett_graph_output(g, 1);
    CHECK(spingalett_graph_input_count(g) == 1 && spingalett_graph_output_count(g) == 2, "inputs and outputs");
    CHECK(!strcmp(in.name, "x") && in.rank == 2 && in.shape[0] == -1 && in.symbols[0] && !strcmp(in.symbols[0], "batch") &&
          in.shape[1] == 4 && in.dtype == SPINGALETT_DTYPE_FLOAT32, "the input as declared");
    CHECK(!strcmp(out.name, "k") && out.dtype == SPINGALETT_DTYPE_INT64 && out.rank == 1, "the second output as declared");
    /* the written-out GELU fused into Linear: MatMul, Add, Div, Erf, Add, Mul, Mul become one node */
    uint32_t linear = 0;
    for (uint32_t k = 0; k < spingalett_graph_node_count(g); k++) linear += !strcmp(spingalett_graph_node_op(g, k), "Linear");
    CHECK(linear == 1 && spingalett_graph_node_count(g) == 3, "%u nodes, %u Linear", spingalett_graph_node_count(g), linear);
    SpingalettTensor input = {0}, outputs[2];
    input.name = "x";
    input.dtype = SPINGALETT_DTYPE_FLOAT32;
    input.rank = 2;
    input.shape[0] = rows;
    input.shape[1] = 4;
    input.data = (void *)x;
    memset(outputs, 0, sizeof outputs);
    bool ok = spingalett_graph_run(.graph = g, .inputs = &input, .input_count = 1, .outputs = outputs, .output_count = 2);
    CHECK(ok, "run: %s", spingalett_last_error_message());
    float first[64];
    if (ok) {
        CHECK(outputs[0].dtype == SPINGALETT_DTYPE_FLOAT32 && outputs[0].rank == 2 && outputs[0].shape[0] == rows &&
              outputs[0].shape[1] == 3 && !strcmp(outputs[0].name ? outputs[0].name : "y", "y"), "the softmax's shape");
        float worst = 0.0f;
        for (uint32_t i = 0; i < rows * 3; i++) {
            const float d = fabsf(((const float *)outputs[0].data)[i] - want[i]);
            if (d > worst) worst = d;
        }
        CHECK(worst < 1e-5f, "the softmax %g from the reference's", worst);
        CHECK(!memcmp(outputs[1].data, want_k, rows * 8u), "the argmax");
        memcpy(first, outputs[0].data, rows * 3 * sizeof(float));
    }
    /* on the GPUs the runtime has (every node its kernels take there with SPINGALETT_GRAPH_GPU_MIN=0, as ctest
       runs this): the CPU's results up to rounding, the same bits run after run */
    for (int backend = 0; backend < 2 && ok; backend++) {
        const SpingalettComputeMode mode = backend ? SPINGALETT_COMPUTE_CUDA : SPINGALETT_COMPUTE_VULKAN;
        const char *device = backend ? spingalett_cuda_device() : spingalett_gpu_device();
        if (!device) continue;
        const SpingalettComputeMode before = spingalett_get_compute_mode();
        spingalett_set_compute_mode(mode);
        SpingalettTensor y[2], z[2];
        memset(y, 0, sizeof y);
        memset(z, 0, sizeof z);
        bool ran = spingalett_graph_run(.graph = g, .inputs = &input, .input_count = 1, .outputs = y, .output_count = 2);
        float on_gpu[64];
        if (ran) memcpy(on_gpu, y[0].data, rows * 3 * sizeof(float));
        ran = ran && spingalett_graph_run(.graph = g, .inputs = &input, .input_count = 1, .outputs = z, .output_count = 2);
        spingalett_set_compute_mode(before);
        CHECK(ran, "run on %s: %s", device, spingalett_last_error_message());
        if (!ran) continue;
        float worst = 0.0f;
        for (uint32_t i = 0; i < rows * 3; i++) {
            const float d = fabsf(on_gpu[i] - first[i]);
            if (d > worst) worst = d;
        }
        CHECK(worst < 1e-5f && !memcmp(y[1].data, want_k, rows * 8u), "on %s: the softmax %g from the CPU's, the argmax",
              device, worst);
        CHECK(!memcmp(z[0].data, on_gpu, rows * 3 * sizeof(float)), "on %s: other bits when run again", device);
        printf("  on %s\n", device);
    }
    /* through a file of format 11 */
    snprintf(saved, sizeof saved, "%s", "graph_small_saved.slett");
    CHECK(spingalett_graph_save(g, saved), "save: %s", spingalett_last_error_message());
    SpingalettGraph *again = spingalett_graph_open(.path = saved);
    CHECK(again, "%s: %s", saved, spingalett_last_error_message());
    if (again && ok) {
        SpingalettTensor y = {0};
        y.name = "y";
        CHECK(spingalett_graph_run(.graph = again, .inputs = &input, .input_count = 1, .outputs = &y, .output_count = 1) &&
              !memcmp(y.data, first, rows * 3 * sizeof(float)), "the saved graph's bits");
    }
    spingalett_graph_free(again);
    remove(saved);
    /* a wrong input: refused, named */
    input.shape[1] = 5;
    CHECK(!spingalett_graph_run(.graph = g, .inputs = &input, .input_count = 1, .outputs = outputs, .output_count = 2) &&
          strstr(spingalett_last_error_message(), "'x'"), "a wrong shape refused: %s", spingalett_last_error_message());
    input.shape[1] = 4;
    input.name = "nothing";
    CHECK(!spingalett_graph_run(.graph = g, .inputs = &input, .input_count = 1, .outputs = outputs, .output_count = 2),
          "an unknown input refused");
    spingalett_graph_free(g);
    const char junk[] = "not a model at all";
    CHECK(!spingalett_graph_open(.data = junk, .size = sizeof junk - 1), "bytes of no model refused");
    free(bin);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: SpingalettRuntimeTests DATA_DIR [EXPORT_DIR]\n");
        return 2;
    }
    api(argv[1]);
    tokenizers(argv[1]);
    graphs(argv[1]);
    release_models(argv[1]);
    if (argc > 2) library_models(argv[2]);
    printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
