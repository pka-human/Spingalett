/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#include "Spingalett.Private.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#if defined(_WIN32)
#include <malloc.h>
#endif

void *spingalett_aligned_alloc(size_t size) {
    if (size == 0) size = SPINGALETT_ALIGNMENT;
    size = (size + SPINGALETT_ALIGNMENT - 1) & ~((size_t)SPINGALETT_ALIGNMENT - 1);
#if defined(_WIN32)        /* MSVC and MinGW: the Windows C runtime has no aligned_alloc */
    return _aligned_malloc(size, SPINGALETT_ALIGNMENT);
#elif defined(__APPLE__) || defined(__ANDROID__)
    void *ptr = NULL;
    if (posix_memalign(&ptr, SPINGALETT_ALIGNMENT, size) != 0) return NULL;
    return ptr;
#else
    return aligned_alloc(SPINGALETT_ALIGNMENT, size);
#endif
}

void *spingalett_aligned_calloc(size_t count, size_t elem_size) {
    if (elem_size != 0 && count > SIZE_MAX / elem_size) return NULL;
    size_t total = count * elem_size;
    void *p = spingalett_aligned_alloc(total);
    if (p) memset(p, 0, total);
    return p;
}

void spingalett_aligned_free(void *ptr) {
    if (!ptr) return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void *spingalett_read_file(const char *path, size_t *size) {
    *size = 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        set_error(SPINGALETT_ERR_FILE_IO, "load: cannot open file for reading");
        return NULL;
    }
    long length = -1;
    if (fseek(fp, 0, SEEK_END) == 0) length = ftell(fp);
    if (length < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        set_error(SPINGALETT_ERR_FILE_IO, "load: cannot determine the file size");
        return NULL;
    }
    void *data = spingalett_aligned_alloc((size_t)length);
    if (!data) {
        fclose(fp);
        set_error(SPINGALETT_ERR_ALLOC, "load: file buffer allocation failed");
        return NULL;
    }
    bool ok = fread(data, 1, (size_t)length, fp) == (size_t)length;
    fclose(fp);
    if (!ok) {
        spingalett_aligned_free(data);
        set_error(SPINGALETT_ERR_FILE_IO, "load: read error");
        return NULL;
    }
    *size = (size_t)length;
    return data;
}
