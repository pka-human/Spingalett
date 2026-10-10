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

/* gemm_mma.cu's tiles of rows and columns (tm, tn) of block (x, y) of a product's dispatch: bands of SWIZZLE tiles
   of rows, a band's blocks over every tile of columns with its rows fastest. Blocks start roughly in the order of x + y
   times the blocks in x, so that a band's rows of A stay in L2 while its blocks take every tile of columns,
   rather than all of A read again from memory for each tile of columns. Which block computes a tile changes
   nothing in it. */
#define SWIZZLE 8u
DEVICE void swizzled(uint32_t &tm, uint32_t &tn) {
    const uint32_t nx = blocks_x(), ny = blocks_y(), pid = block_y() * nx + block_x();
    const uint32_t band = SWIZZLE * ny, first = pid / band * SWIZZLE, size = umin(nx - first, SWIZZLE);
    tm = first + pid % band % size;
    tn = pid % band / size;
}

/* gemm.cu's tiles: bands where the backend launches them so (Spingalett.Cuda.c's bands_pay()), a band's 2^b tiles of
   rows (b in the flags: SPG_GEMM_BAND_SHIFT) over every tile of columns in x, the bands in y; with b = 0 block (x, y)
   computes tile (x, y). Shifts and masks of the block's indices, which the compiler computes again where it needs
   them: swizzled()'s divisions made the 128 x 128 tiles of single precision spill registers, which cost the MLP's
   inference 4%. */
DEVICE void band_tile(uint32_t flags, uint32_t &tm, uint32_t &tn) {
    const uint32_t b = (flags >> SPG_GEMM_BAND_SHIFT) & 15u;
    tm = block_x();
    tn = block_y();
    if (b) tm = (block_y() << b) + (block_x() & ((1u << b) - 1u)), tn = block_x() >> b;
}

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
