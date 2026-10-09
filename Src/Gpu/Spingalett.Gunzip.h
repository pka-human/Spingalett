/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The contents of a gzip file of `size` bytes, followed by a zero (text); its length in *length. NULL if it
   is not such a file, its data are damaged (the CRC disagrees) or memory runs out. free() it. */
char *spg_gunzip(const uint8_t *gz, size_t size, size_t *length);
