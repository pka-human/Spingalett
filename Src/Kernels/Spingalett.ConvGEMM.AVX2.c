/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The indirect convolution kernels compiled for AVX2 and FMA (-mavx2 -mfma), chosen at run time on
   processors that have them; see Spingalett.ConvGEMM.c. */

#define SPINGALETT_CONV_VARIANT avx2
#include "../Spingalett.ConvGEMM.c"
