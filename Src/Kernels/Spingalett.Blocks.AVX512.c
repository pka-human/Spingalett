/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The kernels of the block precisions compiled for AVX-512 VNNI (dpbusd on 256-bit vectors), run on processors that
   have them; see Spingalett.Blocks.c. */

#define SPINGALETT_BLOCK_VARIANT spingalett_block_layer_avx512
#include "../Spingalett.Blocks.c"
