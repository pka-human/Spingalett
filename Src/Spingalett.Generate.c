/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Text generation (spingalett_generate()): a language model's prediction on a window of tokens, a token
 * at a time. The window holds the last context tokens from its first position on; the positions after
 * them hold no token (-1, which an embedding reads as zeros), which a causal model's earlier positions
 * never see. The next token is the most likely one, or a draw from the softmax of the logits divided by
 * the temperature, among the top_k most likely and the most likely whose probabilities add up to
 * top_p. Draws come from a splitmix64 generator seeded once, in an order fixed by the logits: the same
 * tokens for the same logits.
 */

#include "Spingalett.Private.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float p;
    uint32_t token;
} Candidate;

/* by probability, the larger first; equals by token */
static int by_probability(const void *a, const void *b) {
    const Candidate *x = (const Candidate *)a, *y = (const Candidate *)b;
    if (x->p != y->p) return x->p > y->p ? -1 : 1;
    return x->token < y->token ? -1 : x->token > y->token;
}

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* The next token from a row of V logits. */
static uint32_t next_token(const float *logits, uint32_t V, const SpingalettGenerateArgs *a, Candidate *c,
                           uint64_t *state) {
    uint32_t best = 0;
    for (uint32_t v = 1; v < V; v++)
        if (logits[v] > logits[best]) best = v;
    if (!(a->temperature > 0.0f)) return best;
    const double inv = 1.0 / a->temperature, top = logits[best];
    for (uint32_t v = 0; v < V; v++) c[v] = (Candidate){(float)exp(((double)logits[v] - top) * inv), v};
    qsort(c, V, sizeof *c, by_probability);
    uint32_t kept = a->top_k && a->top_k < V ? a->top_k : V;
    double total = 0.0;
    for (uint32_t k = 0; k < kept; k++) total += c[k].p;
    if (a->top_p > 0.0f && a->top_p < 1.0f) {
        double sum = 0.0;
        for (uint32_t k = 0; k < kept; k++) {
            sum += c[k].p;
            if (sum >= a->top_p * total) {
                kept = k + 1u;
                total = sum;
                break;
            }
        }
    }
    const double u = (double)(splitmix64(state) >> 11) * (1.0 / 9007199254740992.0) * total;
    double sum = 0.0;
    for (uint32_t k = 0; k < kept; k++) {
        sum += c[k].p;
        if (u < sum) return c[k].token;
    }
    return c[kept - 1u].token;
}

uint32_t spingalett_generate_args(SpingalettGenerateArgs args) {
    NeuralNetwork *net = args.net;
    if (!net || !args.prompt || args.prompt_length == 0 || (!args.tokens && args.count) || net->layers < 2 ||
        (args.stop_count && !args.stop_tokens)) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_generate: no network, no prompt or no room for the tokens");
        return 0;
    }
    const uint32_t T = net->topology[0], out = net->topology[net->layers - 1], V = T ? out / T : 0;
    if (T == 0 || V == 0 || out % T != 0) {
        set_error(SPINGALETT_ERR_INVALID, "spingalett_generate: the network's outputs are no logits for each of its "
                                          "input tokens");
        return 0;
    }
    const uint32_t total = args.prompt_length + args.count;
    float *window = (float *)malloc((size_t)T * sizeof(float)), *logits = (float *)malloc((size_t)out * sizeof(float));
    uint32_t *seq = (uint32_t *)malloc((size_t)total * sizeof(uint32_t));
    Candidate *c = (Candidate *)malloc((size_t)V * sizeof(Candidate));
    if (!window || !logits || !seq || !c) {
        free(window); free(logits); free(seq); free(c);
        set_error(SPINGALETT_ERR_ALLOC, "spingalett_generate: out of memory");
        return 0;
    }
    memcpy(seq, args.prompt, (size_t)args.prompt_length * sizeof(uint32_t));
    uint64_t state = args.seed ? args.seed : rng_next64();
    uint32_t length = args.prompt_length, made = 0;
    bool ok = true;
    while (made < args.count) {
        const uint32_t n = length < T ? length : T, first = length - n;
        for (uint32_t i = 0; i < T; i++) window[i] = i < n ? (float)seq[first + i] : -1.0f;
        if (!(ok = spingalett_predict_args((SpingalettPredictArgs){.net = net, .inputs = window, .sample_count = 1,
                                                                   .outputs = logits})))
            break;
        const uint32_t token = next_token(logits + (size_t)(n - 1u) * V, V, &args, c, &state);
        seq[length++] = token;
        args.tokens[made++] = token;
        bool stop = false;
        for (uint32_t k = 0; k < args.stop_count && !stop; k++) stop = args.stop_tokens[k] == token;
        if (stop) break;
    }
    free(window); free(logits); free(seq); free(c);
    return ok ? made : 0;
}
