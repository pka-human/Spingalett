// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

/** Precision of deployment models, and of the GPU's products (FLOAT32 or BFLOAT16). */
public enum Precision {
    FLOAT32(0), FP16(1), BFLOAT16(2), INT8(3), INT4(4), INT2(5);

    final int value;

    Precision(int value) { this.value = value; }
}
