/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The kernels of the block precisions compiled for AVX-VNNI (256-bit dpbusd without AVX-512), run on processors that
   have them; see Spingalett.Blocks.c. */

#define SPINGALETT_BLOCK_VARIANT spingalett_block_layer_avxvnni
#include "../Spingalett.Blocks.c"
