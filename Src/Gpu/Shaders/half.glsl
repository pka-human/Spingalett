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

/* Four consecutive values from index i on (a multiple of four, at 16 bytes): eight bytes of bfloat16, or
   sixteen of floats. */
layout(buffer_reference, std430, buffer_reference_align = 8) buffer U32x2 { uvec2 v[]; };

vec4 ld4(F32 x, uint i, uint w) {
#ifdef SPG_HALF
    if (((HALF >> w) & 1u) != 0u) {
        const uvec2 u = U32x2(x).v[i >> 2];
        return vec4(uintBitsToFloat(u.x << 16), uintBitsToFloat(u.x & 0xFFFF0000u), uintBitsToFloat(u.y << 16),
                    uintBitsToFloat(u.y & 0xFFFF0000u));
    }
#endif
    return F32x4(x).v[i >> 2];
}

void st4(F32 x, uint i, uint w, vec4 v) {
#ifdef SPG_HALF
    if (((HALF >> w) & 1u) != 0u) {
        U32x2(x).v[i >> 2] = uvec2(to_bf16(v.x) | to_bf16(v.y) << 16, to_bf16(v.z) | to_bf16(v.w) << 16);
        return;
    }
#endif
    F32x4(x).v[i >> 2] = v;
}
