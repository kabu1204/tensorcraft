#pragma once

#include <cstdlib>
#include <sys/mman.h>
#include <unistd.h>
#include <cstddef>

static inline void *alloc_pinned(size_t len, size_t alignment)
{
#ifdef __linux__
    long ps = sysconf(_SC_PAGESIZE);
    void *p = mmap(NULL, (len + ps - 1) & ~(ps - 1),
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                   
    if (p == MAP_FAILED) {
        return nullptr;
    }
    
    if (mlock(p, len)) {
        munmap(p, len);
        return nullptr;
    }

    return p;
#else
    return std::aligned_alloc(alignment, len);
#endif
}
