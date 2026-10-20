// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

/**
 * Precision of deployment models, and of the GPU's products (FLOAT32 or BFLOAT16); from Q8_0 on GGUF's block
 * types (language models' layers).
 */
public enum Precision {
    FLOAT32(0), FP16(1), BFLOAT16(2), INT8(3), INT4(4), INT2(5),
    Q8_0(6), Q4_0(7), Q4_1(8), Q5_0(9), Q5_1(10), Q2_K(11), Q3_K(12), Q4_K(13), Q5_K(14), Q6_K(15), IQ4_NL(16), IQ4_XS(17);

    final int value;

    Precision(int value) { this.value = value; }
}
