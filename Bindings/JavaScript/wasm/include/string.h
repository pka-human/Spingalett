/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

/* What the engine takes from <string.h>, for its WebAssembly build without a C library: copies and
   fills as the compiler's own (memory.copy and memory.fill, or loads and stores for small sizes). */

#ifndef SPINGALETT_WASM_STRING_H
#define SPINGALETT_WASM_STRING_H

#include <stddef.h>

static inline void *memcpy(void *restrict d, const void *restrict s, size_t n) { return __builtin_memcpy(d, s, n); }
static inline void *memset(void *d, int c, size_t n) { return __builtin_memset(d, c, n); }

static inline int memcmp(const void *a, const void *b, size_t n) {
    const unsigned char *p = (const unsigned char *)a, *q = (const unsigned char *)b;
    for (size_t i = 0; i < n; i++)
        if (p[i] != q[i]) return p[i] < q[i] ? -1 : 1;
    return 0;
}

#endif
