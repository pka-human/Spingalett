/*
* SPDX-License-Identifier: MIT
* Copyright (c) 2026 pka_human (pka_human@proton.me)
*/

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L         /* O_CLOEXEC, mmap */
#endif
#include "Spingalett.Private.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#if defined(_WIN32)
#  include <malloc.h>
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  define SPG_MMAP 1
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

/* ------------------------------------------------------------------------- files */

/* Maps the file read-only; false when the platform cannot (the caller reads it then). */
static bool map_file(SpgFileView *f, const char *path) {
#if defined(_WIN32)
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (file == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER size;
    bool ok = GetFileSizeEx(file, &size) && size.QuadPart > 0 && (uint64_t)size.QuadPart <= (uint64_t)SIZE_MAX;
    HANDLE mapping = ok ? CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL) : NULL;
    CloseHandle(file);
    if (!mapping) return false;
    void *view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);                   /* the view keeps the mapping */
    if (!view) return false;
    f->data = (const uint8_t *)view;
    f->size = (size_t)size.QuadPart;
    f->handle = view;
    f->mapped = true;
    return true;
#elif defined(SPG_MMAP)
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st;
    bool ok = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0 && (uint64_t)st.st_size <= (uint64_t)SIZE_MAX;
    void *p = ok ? mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
    close(fd);
    if (p == MAP_FAILED) return false;
    f->data = (const uint8_t *)p;
    f->size = (size_t)st.st_size;
    f->handle = p;
    f->mapped = true;
    return true;
#else
    (void)f; (void)path;
    return false;
#endif
}

bool spingalett_file_open(SpgFileView *f, const char *path) {
    memset(f, 0, sizeof *f);
    if (map_file(f, path)) return true;
    /* empty files, pipes and platforms without mappings: read into memory */
    size_t size = 0;
    void *data = spingalett_read_file(path, &size);
    if (!data) return false;
    f->data = (const uint8_t *)data;
    f->size = size;
    f->handle = data;
    return true;
}

void spingalett_file_close(SpgFileView *f) {
    if (!f->handle) return;
    if (f->mapped) {
#if defined(_WIN32)
        UnmapViewOfFile(f->handle);
#elif defined(SPG_MMAP)
        munmap(f->handle, f->size);
#endif
    } else {
        spingalett_aligned_free(f->handle);
    }
    memset(f, 0, sizeof *f);
}
