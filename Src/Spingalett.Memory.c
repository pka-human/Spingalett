#include "Spingalett.Private.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

void *spingalett_aligned_alloc(size_t size) {
    if (size == 0) size = SPINGALETT_ALIGNMENT;
    size = (size + SPINGALETT_ALIGNMENT - 1) & ~((size_t)SPINGALETT_ALIGNMENT - 1);
#if defined(_MSC_VER)
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
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
