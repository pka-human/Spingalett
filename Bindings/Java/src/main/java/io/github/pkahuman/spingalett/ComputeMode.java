// SPDX-License-Identifier: MIT
// Copyright (c) 2026 pka_human (pka_human@proton.me)

package io.github.pkahuman.spingalett;

/** Where the library computes. */
public enum ComputeMode {
    SINGLE_THREADED(0), OPENMP(1), OPENBLAS(2), CUDA(3), VULKAN(4);

    final int value;

    ComputeMode(int value) { this.value = value; }
}
