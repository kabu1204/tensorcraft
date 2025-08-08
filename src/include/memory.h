#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <new>
#include <stdexcept>
#include "arch.h"

static inline bool is_pow2(size_t x) {
    return (x > 0) && (__builtin_popcount(x) == 1);
}

class TensorAllocator {
public:
    virtual void* allocate(size_t n_bytes, bool pin_memory = false) = 0;
    virtual void free(void* ptr) = 0;
    virtual void copy(void* dst, const void* src, size_t n_bytes) = 0;
    virtual ~TensorAllocator() = default;
};

class AlignedHostAllocator : public TensorAllocator {
private:
    const size_t alignment_;

public:
    AlignedHostAllocator(size_t alignment) : alignment_(alignment) {
        if (!is_pow2(alignment)) {
            throw std::invalid_argument("alignment must be a power of 2");
        }
    }

    AlignedHostAllocator(AlignedHostAllocator&& other) = delete;
    AlignedHostAllocator(const AlignedHostAllocator& other) = delete;

    void* allocate(size_t n_bytes, bool pin_memory = false) override {
        if (n_bytes == 0) {
            return nullptr;
        }

        if (pin_memory) {
            return alloc_pinned(n_bytes, alignment_);
        } else {
            void* ptr = nullptr;
            int result = posix_memalign(&ptr, alignment_, n_bytes);
            if (result != 0) {
                throw std::bad_alloc{};
            }
            return ptr;
        }
    }

    void free(void* ptr) override {
        std::free(ptr);
    }

    void copy(void* dst, const void* src, size_t n_bytes) override {
        std::memcpy(dst, src, n_bytes);
    }

    ~AlignedHostAllocator() = default;
};

TensorAllocator& get_default_host_allocator();

// TODO
class CachedHostAllocator;
class CUDAAllocator;
class CachedCUDAAllocator;