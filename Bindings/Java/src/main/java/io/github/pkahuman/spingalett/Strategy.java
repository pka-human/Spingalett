// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

/** How training goes through the samples. */
public enum Strategy {
    SAMPLE(0), FULL_BATCH(1), SMALL_BATCH(2);

    final int value;

    Strategy(int value) { this.value = value; }
}
