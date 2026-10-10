// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

/** Loss a network trains with (SPARSE_CROSS_ENTROPY: a class index a cell, a language model's next tokens). */
public enum Loss {
    MSE(0), CROSS_ENTROPY(1), SPARSE_CROSS_ENTROPY(2);

    final int value;

    Loss(int value) { this.value = value; }
}
