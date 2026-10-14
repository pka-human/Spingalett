/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Shared by the kernels of integer models (iquant.comp, ipass.comp), which compute every value as the CPU does, to
   the bit: IEEE division, and the activations integer models run with on the GPU. Arithmetic that must round as the
   CPU's is `precise` (no multiply-add contracted). */

/* a / b to the nearest, ties to even, for finite a and b above 0 (Vulkan's division is exact to 2.5 ulp only): 25
   bits of the quotient of the significands by long division (24 and a guard bit, the rest sticky), the result's
   exponent from theirs, subnormal or infinite where it falls */
float div_rn(float a, float b) {
    uint ia = floatBitsToUint(a), ib = floatBitsToUint(b);
    int ea = int(ia >> 23), eb = int(ib >> 23);
    uint ma = ia & 0x7FFFFFu, mb = ib & 0x7FFFFFu;
    if (ea == 0) {                          /* subnormal: normalized, the exponent below 1 */
        const int s = findMSB(ma);
        ma <<= uint(23 - s);
        ea = s - 22;
    } else {
        ma |= 0x800000u;
    }
    if (eb == 0) {
        const int s = findMSB(mb);
        mb <<= uint(23 - s);
        eb = s - 22;
    } else {
        mb |= 0x800000u;
    }
    int e = ea - eb + 127;                  /* the quotient's biased exponent, its significand in [1, 2) */
    if (ma < mb) {
        ma <<= 1;
        e--;
    }
    uint q = 0u, r = ma;
    for (int i = 0; i < 25; i++) {
        q <<= 1;
        if (r >= mb) {
            r -= mb;
            q |= 1u;
        }
        r <<= 1;
    }
    /* the bits of q below the result's last: the guard bit, more for a subnormal result */
    const uint drop = e > 0 ? 1u : min(uint(1 - e) + 1u, 26u);
    uint mant = q >> drop;
    const uint rest = q & ((1u << drop) - 1u), mid = 1u << (drop - 1u);
    if (rest > mid || (rest == mid && (r != 0u || (mant & 1u) != 0u))) mant++;
    const uint bits = e > 0 ? (uint(e - 1) << 23) + mant : mant;
    return uintBitsToFloat(min(bits, 0x7F800000u));
}

/* The activations of integer models' layers on the GPU, as the CPU's (Spingalett.SIMD.c): none, ReLU (0 for x <= 0
   and NaN), leaky ReLU. */
float int_act(float x, uint act) {
    if (act == ACT_RELU) return x > 0.0 ? x : 0.0;
    if (act == ACT_LEAKY_RELU) {
        precise float s = 0.01 * x;
        return x > 0.0 ? x : s;
    }
    return x;
}
