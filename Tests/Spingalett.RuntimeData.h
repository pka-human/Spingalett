/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The samples of the runtime's tests (Spingalett.RuntimeTests.c), for which SpingalettTests
   export-models also records the full library's outputs: inputs in [-1, 1) from a fixed sequence (token
   ids for language models), one-hot targets (0 or 1 for a single output), and what a session gives. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void runtime_inputs(float *x, size_t n) {
    uint32_t state = 2026u;
    for (size_t i = 0; i < n; i++) {
        state = state * 1664525u + 1013904223u;
        x[i] = (float)(state >> 8) / 8388608.0f - 1.0f;
    }
}

static void runtime_targets(float *y, uint32_t count, uint32_t outputs) {
    for (uint32_t s = 0; s < count; s++)
        for (uint32_t k = 0; k < outputs; k++)
            y[(size_t)s * outputs + k] = outputs == 1 ? (float)(s % 2u) : (float)(k == s % outputs);
}

/* Token ids for a language model's samples (a model whose first layer is an embedding): the values of
   runtime_inputs() spread over the vocabulary. */
static void runtime_tokens(float *x, size_t n, uint32_t vocabulary) {
    runtime_inputs(x, n);
    for (size_t i = 0; i < n; i++) x[i] = (float)((uint32_t)((x[i] + 1.0f) * 1000.0f) % vocabulary);
}

#if defined(SPINGALETT_RUNTIME_H)
/* Floats of runtime_session()'s logits for a window of T tokens and V logits each. */
static size_t runtime_session_floats(uint32_t T, uint32_t V) {
    return (size_t)(3u * T - 1u) * V;
}

/* A session of a language model over the first two samples of its tokens x (as floats): sample 0 fed to
   sequence 0 as a prompt of half the window, then a token at a time; three tokens of sample 1 to sequence
   1; then the rest of sample 1 together with sample 0 again from its second token (sequence 0 rewound to
   one token). The logits of every token are written in that order. false when a call fails. */
static int runtime_session(const SpingalettModel *m, const float *x, float *logits) {
    const uint32_t T = m->input_size, V = m->output_size / T, h = T / 2u;
    uint32_t *tokens = (uint32_t *)malloc((size_t)2 * T * sizeof(uint32_t));
    SpingalettSessionOptions options = {0};
    options.sequences = 2;
    SpingalettSession *s = tokens ? spingalett_session_new(m, &options) : NULL;
    int ok = s != NULL;
    uint32_t t;
    for (t = 0; ok && t < 2u * T; t++) tokens[t] = (uint32_t)x[t];
    if (ok) {
        SpingalettSessionFeed f[2];
        memset(f, 0, sizeof f);
        f[0].tokens = tokens, f[0].count = h, f[0].logits = logits, f[0].all = true;
        ok = spingalett_session_feed(s, f, 1);
        for (t = h; ok && t < T; t++) {
            f[0].tokens = tokens + t, f[0].count = 1, f[0].logits = logits + (size_t)t * V;
            ok = spingalett_session_feed(s, f, 1);
        }
        logits += (size_t)T * V;
        f[0].sequence = 1, f[0].tokens = tokens + T, f[0].count = 3, f[0].logits = logits;
        ok = ok && spingalett_session_feed(s, f, 1);
        logits += (size_t)3 * V;
        f[0].tokens = tokens + T + 3, f[0].count = T - 3, f[0].logits = logits;
        f[1].sequence = 0, f[1].tokens = tokens + 1, f[1].count = T - 1, f[1].logits = logits + (size_t)(T - 3) * V;
        f[1].all = true;
        ok = ok && spingalett_session_rewind(s, 0, 1) && spingalett_session_feed(s, f, 2);
    }
    spingalett_session_free(s);
    free(tokens);
    return ok;
}
#endif
