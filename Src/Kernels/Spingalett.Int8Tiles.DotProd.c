/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* The INT8 tile kernel compiled for the ARM dot product instructions (-march=armv8.2-a+dotprod),
   run on processors that have them; see Spingalett.Int8Tiles.c. */

#define SPINGALETT_I8_TILE_VARIANT spingalett_i8_tile_dotprod
#include "../Spingalett.Int8Tiles.c"
