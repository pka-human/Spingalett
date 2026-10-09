/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The samples of the runtime's tests (Spingalett.RuntimeTests.c), for which SpingalettTests
   export-models also records the full library's outputs: inputs in [-1, 1) from a fixed sequence,
   and one-hot targets (0 or 1 for a single output). */

#include <stddef.h>
#include <stdint.h>

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
