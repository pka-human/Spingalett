/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * The native engine for the JavaScript package's tests: Src/Spingalett.Inference.c compiled for the host
 * (test/native.mjs builds it), run on the samples test/native.mjs makes in JavaScript, so that the
 * WebAssembly module's results can be compared with it.
 *
 *   native run MODEL COUNT OUT          the outputs of COUNT samples, as floats, to the file OUT
 *   native generate MODEL COUNT TEMPERATURE TOP_K TOP_P SEED TOKEN...
 *                                       the tokens spingalett_generate() chooses after the prompt TOKEN...,
 *                                       over the engine's logits, printed
 *
 * The samples: for a language model (an embedding first) a window of 1 to input_size tokens below its
 * vocabulary, -1 after them; for other models inputs in [-1, 1), as Tests/Spingalett.RuntimeData.h
 * makes them; from one sequence of 32-bit numbers for all samples.
 */

#include <Spingalett/Spingalett.Inference.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t state = 2026u;

static uint32_t next(void) {
    state = state * 1664525u + 1013904223u;
    return state >> 8;
}

static void make_sample(uint32_t inputs, uint32_t vocabulary, float *x) {
    if (vocabulary) {
        const uint32_t n = 1u + next() % inputs;
        for (uint32_t i = 0; i < inputs; i++) x[i] = i < n ? (float)(next() % vocabulary) : -1.0f;
    } else {
        for (uint32_t i = 0; i < inputs; i++) x[i] = (float)next() / 8388608.0f - 1.0f;
    }
}

static void *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    void *data = NULL;
    long n = -1;
    if (fseek(f, 0, SEEK_END) == 0) n = ftell(f);
    if (n >= 0 && fseek(f, 0, SEEK_SET) == 0 && (data = malloc((size_t)n + 1u)) != NULL &&
        fread(data, 1, (size_t)n, f) != (size_t)n) {
        free(data);
        data = NULL;
    }
    fclose(f);
    if (data) *size = (size_t)n;
    return data;
}

/* The next token from a row of V logits, as Src/Spingalett.Generate.c chooses it. */
typedef struct {
    float p;
    uint32_t token;
} Candidate;

static int by_probability(const void *a, const void *b) {
    const Candidate *x = (const Candidate *)a, *y = (const Candidate *)b;
    if (x->p != y->p) return x->p > y->p ? -1 : 1;
    return x->token < y->token ? -1 : x->token > y->token;
}

static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static uint32_t next_token(const float *logits, uint32_t V, float temperature, uint32_t top_k, float top_p,
                           Candidate *c, uint64_t *s) {
    uint32_t best = 0;
    for (uint32_t v = 1; v < V; v++)
        if (logits[v] > logits[best]) best = v;
    if (!(temperature > 0.0f)) return best;
    const double inv = 1.0 / temperature, top = logits[best];
    for (uint32_t v = 0; v < V; v++) {
        c[v].p = (float)exp(((double)logits[v] - top) * inv);
        c[v].token = v;
    }
    qsort(c, V, sizeof *c, by_probability);
    uint32_t kept = top_k && top_k < V ? top_k : V;
    double total = 0.0;
    for (uint32_t k = 0; k < kept; k++) total += c[k].p;
    if (top_p > 0.0f && top_p < 1.0f) {
        double sum = 0.0;
        for (uint32_t k = 0; k < kept; k++) {
            sum += c[k].p;
            if (sum >= top_p * total) {
                kept = k + 1u;
                total = sum;
                break;
            }
        }
    }
    const double u = (double)(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0) * total;
    double sum = 0.0;
    for (uint32_t k = 0; k < kept; k++) {
        sum += c[k].p;
        if (u < sum) return c[k].token;
    }
    return c[kept - 1u].token;
}

int main(int argc, char **argv) {
    if (argc < 4 || (strcmp(argv[1], "run") != 0 && strcmp(argv[1], "generate") != 0) ||
        (!strcmp(argv[1], "run") && argc != 5) || (!strcmp(argv[1], "generate") && argc < 9)) {
        fprintf(stderr, "usage: native run MODEL COUNT OUT\n"
                        "       native generate MODEL COUNT TEMPERATURE TOP_K TOP_P SEED TOKEN...\n");
        return 2;
    }
    size_t size = 0;
    void *image = read_file(argv[2], &size);
    SpingalettModel model;
    int rc = image ? spingalett_model_init(&model, image, size) : SPINGALETT_ERR_FILE_IO;
    if (rc != SPINGALETT_OK) {
        fprintf(stderr, "%s: error %d\n", argv[2], rc);
        return 1;
    }
    SpingalettLayerInfo first;
    spingalett_model_layer(&model, 0, &first);
    const uint32_t vocabulary = first.type == SPINGALETT_LAYER_EMBEDDING ? first.vocabulary : 0u;
    const uint32_t in = model.input_size, out = model.output_size, count = (uint32_t)strtoul(argv[3], NULL, 10);
    float *x = (float *)malloc(in * sizeof(float)), *y = (float *)malloc(out * sizeof(float));
    void *workspace = malloc(model.workspace_size);

    if (!strcmp(argv[1], "run")) {
        FILE *f = fopen(argv[4], "wb");
        for (uint32_t s = 0; s < count && f && rc == SPINGALETT_OK; s++) {
            make_sample(in, vocabulary, x);
            rc = spingalett_model_run(&model, x, y, workspace);
            if (fwrite(y, sizeof(float), out, f) != out) rc = SPINGALETT_ERR_FILE_IO;
        }
        if (!f || fclose(f) != 0) rc = SPINGALETT_ERR_FILE_IO;
    } else {
        /* the window: the last input_size tokens from its first position on, -1 after them */
        const float temperature = strtof(argv[4], NULL), top_p = strtof(argv[6], NULL);
        const uint32_t top_k = (uint32_t)strtoul(argv[5], NULL, 10), V = out / in, prompt = (uint32_t)(argc - 8);
        uint64_t seed = strtoull(argv[7], NULL, 10);
        uint32_t *seq = (uint32_t *)malloc((prompt + count) * sizeof(uint32_t)), length = prompt;
        Candidate *c = (Candidate *)malloc(V * sizeof(Candidate));
        for (uint32_t i = 0; i < prompt; i++) seq[i] = (uint32_t)strtoul(argv[8 + i], NULL, 10);
        for (uint32_t made = 0; made < count && rc == SPINGALETT_OK; made++) {
            const uint32_t n = length < in ? length : in, start = length - n;
            for (uint32_t i = 0; i < in; i++) x[i] = i < n ? (float)seq[start + i] : -1.0f;
            rc = spingalett_model_run(&model, x, y, workspace);
            seq[length] = next_token(y + (size_t)(n - 1u) * V, V, temperature, top_k, top_p, c, &seed);
            printf("%s%u", made ? " " : "", seq[length++]);
        }
        printf("\n");
        free(seq);
        free(c);
    }
    free(x);
    free(y);
    free(workspace);
    free(image);
    if (rc != SPINGALETT_OK) fprintf(stderr, "%s: error %d\n", argv[2], rc);
    return rc == SPINGALETT_OK ? 0 : 1;
}
