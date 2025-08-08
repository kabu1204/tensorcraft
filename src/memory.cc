#include "memory.h"

// cacheline align
static AlignedHostAllocator default_host_allocator(64);

TensorAllocator& get_default_host_allocator() {
    return default_host_allocator;
}