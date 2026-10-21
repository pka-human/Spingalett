/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Runs a graph of operators (an ONNX file, or a .slett file of format 11) with the runtime alone.
 *
 *   RunGraph <model.onnx|model.slett>               the graph's inputs, outputs and operators, then the time of
 *                                                   its runs on random inputs
 *   --dim name=size                                 the size of a symbol of the inputs' shapes (1 when not given)
 *   --runs N                                        runs to time (5 by default; the median is printed)
 *   --threads N                                     at most N threads (all cores by default)
 *   --save model.slett                              writes the graph as a .slett file of format 11
 *
 * Inputs are filled with values from a fixed generator: floats in [-1, 1), integers in [0, 100) (token ids, positions
 * and masks of 1 for inputs whose names say mask), booleans true.
 */

#define SPINGALETT_SHORT_NAMES
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

static const char *type_name(SpingalettDType t) {
    switch (t) {
        case SPINGALETT_DTYPE_FLOAT32: return "float32";
        case SPINGALETT_DTYPE_FLOAT16: return "float16";
        case SPINGALETT_DTYPE_BFLOAT16: return "bfloat16";
        case SPINGALETT_DTYPE_INT64: return "int64";
        case SPINGALETT_DTYPE_INT32: return "int32";
        case SPINGALETT_DTYPE_INT8: return "int8";
        case SPINGALETT_DTYPE_UINT8: return "uint8";
        case SPINGALETT_DTYPE_BOOL: return "bool";
        default: return "?";
    }
}

static void print_info(const SpingalettTensorInfo *info) {
    printf("  %s: %s [", info->name, type_name(info->dtype));
    for (uint32_t d = 0; d < info->rank; d++) {
        if (info->shape[d] >= 0) printf("%s%lld", d ? ", " : "", (long long)info->shape[d]);
        else printf("%s%s", d ? ", " : "", info->symbols[d] ? info->symbols[d] : "?");
    }
    printf("]\n");
}

static int compare_doubles(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    const char *path = NULL, *save = NULL;
    const char *dim_names[16];
    long long dim_sizes[16];
    int dims = 0, runs = 5;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dim") && i + 1 < argc && dims < 16) {
            char *eq = strchr(argv[++i], '=');
            if (!eq) { fprintf(stderr, "--dim takes name=size\n"); return 2; }
            *eq = '\0';
            dim_names[dims] = argv[i];
            dim_sizes[dims++] = atoll(eq + 1);
        } else if (!strcmp(argv[i], "--runs") && i + 1 < argc) {
            runs = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            spingalett_set_num_threads((unsigned)atoi(argv[++i]));
        } else if (!strcmp(argv[i], "--save") && i + 1 < argc) {
            save = argv[++i];
        } else if (!path) {
            path = argv[i];
        } else {
            fprintf(stderr, "unexpected argument %s\n", argv[i]);
            return 2;
        }
    }
    if (!path || runs < 1) {
        fprintf(stderr, "usage: RunGraph <model.onnx|model.slett> [--dim name=size] [--runs N] [--threads N] "
                        "[--save model.slett]\n");
        return 2;
    }
    spingalett_set_compute_mode(COMPUTE_OPENMP);
    double t = now();
    SpingalettGraph *g = spingalett_graph_open(.path = path);
    if (!g) return fail(path);
    printf("%s: opened in %.1f ms, %u nodes\n", path, (now() - t) * 1e3, spingalett_graph_node_count(g));
    /* the operators and how many nodes of each */
    const uint32_t nodes = spingalett_graph_node_count(g);
    const char **ops = calloc(nodes + 1, sizeof(char *));
    unsigned *counts = calloc(nodes + 1, sizeof(unsigned));
    uint32_t distinct = 0;
    for (uint32_t k = 0; k < nodes; k++) {
        const char *op = spingalett_graph_node_op(g, k);
        uint32_t j = 0;
        while (j < distinct && strcmp(ops[j], op)) j++;
        if (j == distinct) ops[distinct++] = op;
        counts[j]++;
    }
    printf("operators:");
    for (uint32_t j = 0; j < distinct; j++) printf(" %s %u%s", ops[j], counts[j], j + 1 < distinct ? "," : "\n");
    if (!distinct) printf(" none\n");
    free(ops);
    free(counts);
    const uint32_t ni = spingalett_graph_input_count(g), no = spingalett_graph_output_count(g);
    SpingalettTensor *inputs = calloc(ni + 1, sizeof(SpingalettTensor));
    SpingalettTensor *outputs = calloc(no + 1, sizeof(SpingalettTensor));
    printf("inputs:\n");
    uint64_t state = 0x9E3779B97F4A7C15ull;
    for (uint32_t i = 0; i < ni; i++) {
        SpingalettTensorInfo info = spingalett_graph_input(g, i);
        print_info(&info);
        SpingalettTensor *x = &inputs[i];
        x->name = info.name;
        x->dtype = info.dtype == SPINGALETT_DTYPE_FLOAT16 || info.dtype == SPINGALETT_DTYPE_BFLOAT16
                 ? SPINGALETT_DTYPE_FLOAT32 : info.dtype;
        x->rank = info.rank;
        size_t count = 1;
        for (uint32_t d = 0; d < info.rank; d++) {
            long long size = info.shape[d] >= 0 ? info.shape[d] : 1;
            for (int k = 0; k < dims; k++)
                if (info.symbols[d] && !strcmp(info.symbols[d], dim_names[k])) size = dim_sizes[k];
            x->shape[d] = size;
            count *= (size_t)size;
        }
        const size_t width = x->dtype == SPINGALETT_DTYPE_INT64 ? 8 : x->dtype == SPINGALETT_DTYPE_FLOAT32 ||
                             x->dtype == SPINGALETT_DTYPE_INT32 ? 4 : 1;
        x->data = malloc(count * width + 1);
        const int mask = strstr(info.name, "mask") != NULL;
        for (size_t k = 0; k < count; k++) {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            const uint32_t bits = (uint32_t)(state >> 33);
            switch (x->dtype) {
                case SPINGALETT_DTYPE_FLOAT32: ((float *)x->data)[k] = (float)bits / 1073741824.0f - 1.0f; break;
                case SPINGALETT_DTYPE_INT64: ((int64_t *)x->data)[k] = mask ? 1 : bits % 100; break;
                case SPINGALETT_DTYPE_INT32: ((int32_t *)x->data)[k] = mask ? 1 : (int32_t)(bits % 100); break;
                case SPINGALETT_DTYPE_BOOL: ((uint8_t *)x->data)[k] = 1; break;
                default: ((uint8_t *)x->data)[k] = (uint8_t)(bits % 100); break;
            }
        }
    }
    /* a first run, then the timed ones */
    if (!spingalett_graph_run(.graph = g, .inputs = inputs, .input_count = ni, .outputs = outputs, .output_count = no))
        return fail("run");
    printf("outputs:\n");
    for (uint32_t o = 0; o < no; o++) {
        SpingalettTensorInfo info = {0};
        info.name = outputs[o].name ? outputs[o].name : spingalett_graph_output(g, o).name;
        info.dtype = outputs[o].dtype;
        info.rank = outputs[o].rank;
        for (uint32_t d = 0; d < info.rank; d++) info.shape[d] = outputs[o].shape[d];
        print_info(&info);
    }
    double *times = malloc(sizeof(double) * (size_t)runs);
    for (int r = 0; r < runs; r++) {
        t = now();
        if (!spingalett_graph_run(.graph = g, .inputs = inputs, .input_count = ni, .outputs = outputs, .output_count = no))
            return fail("run");
        times[r] = now() - t;
    }
    qsort(times, (size_t)runs, sizeof(double), compare_doubles);
    printf("run: %.2f ms (median of %d, %u threads)\n", times[runs / 2] * 1e3, runs, spingalett_get_num_threads());
    if (save) {
        if (!spingalett_graph_save(g, save)) return fail(save);
        printf("saved %s\n", save);
    }
    for (uint32_t i = 0; i < ni; i++) free(inputs[i].data);
    free(inputs);
    free(outputs);
    free(times);
    spingalett_graph_free(g);
    return 0;
}
