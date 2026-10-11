// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

/** Activation of a layer's outputs (GELU with erf, GELU_TANH GPT-2's approximation, SILU x sigmoid(x)). */
public enum Activation {
    NONE(0), SIGMOID(1), RELU(2), TANH(3), LEAKY_RELU(4), FOO52(5), SOFTMAX(6), GELU(7), GELU_TANH(8), SILU(9);

    final int value;

    Activation(int value) { this.value = value; }
}
