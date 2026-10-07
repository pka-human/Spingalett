/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The GEMM kernels compiled for AVX-512 (-mavx512f), chosen at run time on processors that have
   it; see Spingalett.GEMM.c. */

#define SPINGALETT_GEMM_VARIANT spingalett_gemm_avx512
#include "../Spingalett.GEMM.c"
