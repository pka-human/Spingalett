/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The indirect convolution kernels compiled for AVX-512 (-mavx512f), chosen at run time on
   processors that have it; see Spingalett.ConvGEMM.c. */

#define SPINGALETT_CONV_VARIANT avx512
#include "../Spingalett.ConvGEMM.c"
