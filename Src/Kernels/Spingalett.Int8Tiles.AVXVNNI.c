/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The INT8 tile kernel compiled for AVX-VNNI (-mavxvnni: 256-bit dpbusd without AVX-512), run on
   processors that have it and not AVX-512 VNNI; see Spingalett.Int8Tiles.c. */

#define SPINGALETT_I8_TILE_VARIANT spingalett_i8_tile_avxvnni
#include "../Spingalett.Int8Tiles.c"
