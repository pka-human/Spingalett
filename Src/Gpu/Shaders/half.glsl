/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Activations kept as bfloat16 (spingalett_gpu kept_half()): the kernel declares the specialization
 * constant HALF before including this file, whose bit w says that the buffer at push constant word w
 * (its byte offset over eight) holds bfloat16 values. Kernels built without SPG_HALF read and write
 * floats only.
 */

float ld(F32 x, uint i, uint w) {
#ifdef SPG_HALF
    if (((HALF >> w) & 1u) != 0u) return from_bf16(uint(BF16(x).v[i]));
#endif
    return x.v[i];
}

void st(F32 x, uint i, uint w, float v) {
#ifdef SPG_HALF
    if (((HALF >> w) & 1u) != 0u) {
        BF16(x).v[i] = uint16_t(to_bf16(v));
        return;
    }
#endif
    x.v[i] = v;
}
