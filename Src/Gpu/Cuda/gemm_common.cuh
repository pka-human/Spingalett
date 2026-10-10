/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * What the matrix products (gemm.cu, gemm_mma.cu) share: gemm.comp's operand modes, epilogues and
 * convolution geometry (gemm_common.glsl), and asynchronous copies to shared memory.
 */

#pragma once

#include "common.cuh"

#define A_ROW   0u
#define A_COL   1u
#define A_CONV  2u
#define A_CONV4 3u
#define B_ROW   0u
#define B_COL   1u
#define B_CONV  2u
#define B_CONV4 3u
#define EPI_STORE       0u
#define EPI_BIAS_ACT    1u
#define EPI_SCALE_ACT   2u
#define EPI_PARTIAL     3u
#define EPI_DERIV       4u
#define FLAG_BIAS       1u
#define FLAG_PRE        2u

#define GEO_RH 0u
#define GEO_RW 1u
#define GEO_GH 2u
#define GEO_GW 3u
#define GEO_GC 4u
#define GEO_SH 5u
#define GEO_SW 6u
#define GEO_PH 7u
#define GEO_PW 8u
#define GEO_CH 11u
#define GEO_CW 12u
#define GEO_CY 13u
#define GEO_CX 14u
#define GEO_CS 15u
#define GEO_TAPS 16u

/* asynchronous copies to shared memory: `bytes` of 4 (8, 16) from src, the rest of them zeros */
DEVICE uint32_t shared_address(const void *p) {
    uint64_t r;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(r) : "l"(p));
    return (uint32_t)r;
}
DEVICE void copy4(uint32_t dst, const void *src, uint32_t bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
DEVICE void copy8(uint32_t dst, const void *src, uint32_t bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
DEVICE void copy16(uint32_t dst, const void *src, uint32_t bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
DEVICE void copies_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <uint32_t N>
DEVICE void copies_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
