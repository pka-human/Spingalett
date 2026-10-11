// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

/** Initialization of a layer's weights. */
public enum Init {
    RANDOM(0), XAVIER(1), HE(2), ZEROS(3), LECUN(4);

    final int value;

    Init(int value) { this.value = value; }
}
