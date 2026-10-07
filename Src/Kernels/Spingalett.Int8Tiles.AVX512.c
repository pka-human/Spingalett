/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The INT8 tile kernel compiled for AVX-512 VNNI (-mavx512vnni), run on processors that have it;
   see Spingalett.Int8Tiles.c. */

#define SPINGALETT_I8_TILE_VARIANT spingalett_i8_tile_avx512
#include "../Spingalett.Int8Tiles.c"
