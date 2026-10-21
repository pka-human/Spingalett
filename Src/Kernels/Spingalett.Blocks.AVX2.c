/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The kernels of the block precisions compiled for AVX2, FMA and F16C (maddubs and madd), run on processors that
   have them; see Spingalett.Blocks.c. */

#define SPINGALETT_BLOCK_VARIANT spingalett_block_layer_avx2
#include "../Spingalett.Blocks.c"
