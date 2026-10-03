/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/*
 * Force-included (-include) into the static SDL2 built for the AppImage. With _GNU_SOURCE,
 * glibc 2.38+ headers redirect strtol(), sscanf() and friends to new __isoc23_* symbols, which
 * would make the image require glibc 2.38. SDL defines _GNU_SOURCE in every file anyway; doing
 * it here first allows switching the redirection off (it only changes how "0b" prefixes parse).
 */
#define _GNU_SOURCE
#include <features.h>
#undef __GLIBC_USE_C2X_STRTOL
#define __GLIBC_USE_C2X_STRTOL 0
