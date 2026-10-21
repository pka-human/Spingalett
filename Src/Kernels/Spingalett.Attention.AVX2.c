/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* Sessions' attention compiled for AVX2, run on processors that have it; see Spingalett.Attention.c. */

#define SPINGALETT_ATTEND_VARIANT spingalett_attend_blocks_avx2
#include "../Spingalett.Attention.c"
