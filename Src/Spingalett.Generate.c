/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Text generation (spingalett_generate()): a language model's prediction on a window of tokens, a token
 * at a time. The window holds the last context tokens from its first position on; the positions after
 * them hold no token (-1, which an embedding reads as zeros), which a causal model's earlier positions
 * never see. The next token is the most likely one, or a draw from the softmax of the logits divided by
 * the temperature, among the top_k most likely, the most likely whose probabilities add up to top_p and those
 * min_p times as likely as the most likely, after llama.cpp's penalties of repeated tokens (SpingalettSampling,
 * spingalett_next_token()). Draws come from a splitmix64 generator seeded once, in an order fixed
 * by the logits: the same tokens for the same logits.
 *
 * A causal model runs as its deployment model of single precision, which the network keeps until its
 * parameters change, with a session (spingalett_session_feed(); on the GPU in its compute modes): the prompt
 * fed at once, then each token fed once, attending over the keys and values kept of the tokens before it.
 * Once the tokens fill the window, every step feeds the window of the last context tokens again from its
 * first position on, as the prediction over the window does. The logits are those of the deployment model,
 * which equal the network's prediction to rounding. Other networks predict on the whole window for every
 * token.
 */

#include "Spingalett.Private.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* The network as a deployment model, and the parameters it was made from (the network's host_version); the
   session of the last call, kept when it ran on a GPU (whose making takes the model to the device) for the next
   call of the same compute mode, precision and window. */
struct SpingalettGenerateModel {
    SpingalettModel *model;
    uint64_t version;
    SpingalettSession *session;
    ComputeMode mode;
    PrecisionMode precision;
    uint32_t window;
};

static void model_free(struct SpingalettGenerateModel *g) {
    if (!g) return;
    spingalett_session_free(g->session);
    spingalett_model_free(g->model);
    free(g);
}

void spingalett_generate_release(NeuralNetwork *net) {
    model_free(atomic_exchange(&net->generate_model, NULL));
}

static bool is_stop(const SpingalettGenerateArgs *a, uint32_t token, uint32_t made) {
    for (uint32_t k = 0; k < a->stop_count; k++)
        if (a->stop_tokens[k] == token) return true;
    return spingalett_stop_string(a->sampling, a->tokens, made);
}

/* The next token of a row of logits of the sequence seq[0 .. length): the penalties of its last tokens, then the
   draw. scratch holds the penalties' window. */
static uint32_t next_token(const SpingalettGenerateArgs *a, float *logits, uint32_t V, const uint32_t *seq,
                           uint32_t length, SpingalettCandidate *c, uint32_t *scratch, uint64_t *state) {
    spingalett_penalize(logits, V, a->sampling, seq, length, NULL, 0, scratch);
    const SpgDraw draw = {a->temperature, a->top_k, a->top_p, a->sampling ? a->sampling->min_p : 0.0f};
    return spingalett_next_token(logits, V, &draw, c, state);
}

/* With a session of the network's deployment model: false (*made untouched) when the model is no causal
   language model, which the prediction over the window then runs. seq: [prompt_length + count]. */
static bool generate_cached(NeuralNetwork *net, SpingalettGenerateArgs *a, uint32_t T, uint32_t V, uint32_t *seq,
                            SpingalettCandidate *c, uint32_t *scratch, uint64_t *state, bool *ok, uint32_t *made) {
    /* the model the network keeps when it was made from these parameters (the call takes it, so that
       concurrent calls make their own), else a new one */
    spingalett_network_sync(net);
    const uint64_t version = atomic_load(&net->host_version);
    struct SpingalettGenerateModel *g = atomic_exchange(&net->generate_model, NULL);
    if (g && g->version != version) {
        spingalett_model_free(g->model);
        free(g);
        g = NULL;
    }
    if (!g) {
        g = (struct SpingalettGenerateModel *)calloc(1, sizeof *g);
        SpingalettModel *model = g ? spingalett_model_from_network(net, PRECISION_FLOAT32) : NULL;
        if (!model) {
            free(g);
            return false;           /* (the prediction over the window reports what fails) */
        }
        g->model = model;
        g->version = version;
    }
    /* the kept session where it fits this call, else a new one (a model no session runs leaves the error as it
       was) */
    const ComputeMode mode = resolve_compute_mode();
    const PrecisionMode precision = spingalett_get_gpu_precision();
    SpingalettSession *session = g->session;
    g->session = NULL;
    if (session && (g->mode != mode || g->precision != precision || g->window != T)) {
        spingalett_session_free(session);
        session = NULL;
    }
    if (!session) {
        SpingalettSessionOptions options = {.max_length = T};
        session = spingalett_session_runs(g->model) ? spingalett_session_new(g->model, &options) : NULL;
    }
    if (!session) {
        model_free(atomic_exchange(&net->generate_model, g));
        return false;
    }
    float *logits = (float *)malloc((size_t)V * sizeof(float));
    *ok = logits != NULL;
    if (!*ok) set_error(SPINGALETT_ERR_ALLOC, "spingalett_generate: out of memory");
    uint32_t length = a->prompt_length, fed = 0;
    while (*ok && *made < a->count) {
        /* the window of the last T tokens: those after the ones fed, or all of it again once it is full */
        const uint32_t n = length < T ? length : T, first = length - n;
        if (first > 0) fed = 0;
        if (!spingalett_session_rewind(session, 0, fed) ||
            !spingalett_session_feed(session, &(SpingalettSessionFeed){.tokens = seq + first + fed, .count = n - fed,
                                                                       .logits = logits}, 1)) {
            *ok = false;
            break;
        }
        fed = n;
        const uint32_t token = next_token(a, logits, V, seq, length, c, scratch, state);
        seq[length++] = token;
        a->tokens[(*made)++] = token;
        if (is_stop(a, token, *made)) break;
    }
    free(logits);
    /* (kept on a GPU, whose session costs its model's way to the device; on the CPU made again) */
    if (spingalett_use_gpu() && *ok && spingalett_session_rewind(session, 0, 0)) {
        g->session = session;
        g->mode = mode;
        g->precision = precision;
        g->window = T;
    } else {
        spingalett_session_free(session);
    }
    model_free(atomic_exchange(&net->generate_model, g));
    return true;
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
    const uint32_t window = spingalett_penalty_window(args.sampling);
    uint32_t *seq = (uint32_t *)malloc((size_t)total * sizeof(uint32_t));
    uint32_t *scratch = (uint32_t *)malloc(((size_t)(window < total ? window : total) + 1u) * sizeof(uint32_t));
    SpingalettCandidate *c = (SpingalettCandidate *)malloc((size_t)V * sizeof(SpingalettCandidate));
    if (!seq || !c || !scratch) {
        free(seq); free(c); free(scratch);
        set_error(SPINGALETT_ERR_ALLOC, "spingalett_generate: out of memory");
        return 0;
    }
    memcpy(seq, args.prompt, (size_t)args.prompt_length * sizeof(uint32_t));
    uint64_t state = args.seed ? args.seed : rng_next64();
    uint32_t made = 0;
    bool ok = true;
    /* through a session (on the GPU in its compute modes); when no session runs the network, over the window */
    if (!generate_cached(net, &args, T, V, seq, c, scratch, &state, &ok, &made)) {
        float *window = (float *)malloc((size_t)T * sizeof(float)), *logits = (float *)malloc((size_t)out * sizeof(float));
        ok = window && logits;
        if (!ok) set_error(SPINGALETT_ERR_ALLOC, "spingalett_generate: out of memory");
        uint32_t length = args.prompt_length;
        while (ok && made < args.count) {
            const uint32_t n = length < T ? length : T, first = length - n;
            for (uint32_t i = 0; i < T; i++) window[i] = i < n ? (float)seq[first + i] : -1.0f;
            if (!(ok = spingalett_predict_args((SpingalettPredictArgs){.net = net, .inputs = window, .sample_count = 1,
                                                                       .outputs = logits})))
                break;
            const uint32_t token = next_token(&args, logits + (size_t)(n - 1u) * V, V, seq, length, c, scratch, &state);
            seq[length++] = token;
            args.tokens[made++] = token;
            if (is_stop(&args, token, made)) break;
        }
        free(window); free(logits);
    }
    free(seq); free(c); free(scratch);
    return ok ? made : 0;
}
