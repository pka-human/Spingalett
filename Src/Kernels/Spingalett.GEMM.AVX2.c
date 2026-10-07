/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The GEMM kernels compiled for AVX2 and FMA (-mavx2 -mfma), chosen at run time on processors
   that have them; see Spingalett.GEMM.c. */

#define SPINGALETT_GEMM_VARIANT spingalett_gemm_avx2
#include "../Spingalett.GEMM.c"
