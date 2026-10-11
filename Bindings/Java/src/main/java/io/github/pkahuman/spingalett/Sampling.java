// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

/** How Network.generate picks each token (the defaults: the most likely one); setters chain. */
public final class Sampling {
    float temperature, topP;
    int topK;
    long seed;
    int[] stop = new int[0];

    /** The logits divided by it before the softmax; 0: the most likely token. */
    public Sampling temperature(float v) { temperature = v; return this; }
    public Sampling topK(int v) { topK = v; return this; }
    public Sampling topP(float v) { topP = v; return this; }
    /** The draws' generator (0: one draw of the library's). */
    public Sampling seed(long v) { seed = v; return this; }
    /** Tokens that end the generation (returned as the last one). */
    public Sampling stop(int... tokens) { stop = tokens.clone(); return this; }
}
