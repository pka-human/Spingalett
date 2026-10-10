// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

/** Optimizer of training. */
public enum Optimizer {
    SGD(0), MOMENTUM(1), RMSPROP(2), ADAM(3), ADAMW(4);

    final int value;

    Optimizer(int value) { this.value = value; }
}
