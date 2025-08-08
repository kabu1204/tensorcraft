#pragma once

#include "memory.h"
#include "common.h"
#include "ops.h"

#include <cstddef>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <type_traits>
#include <utility>
#include <algorithm>
#include <initializer_list>
#include <limits>

// for ease of lifecycle management
class TensorStorage {
private:
    void* data_;
    size_t size_;
    TensorAllocator& allocator_;
    bool pin_memory_;
    // TODO: device

public:
    TensorStorage() = delete;

    explicit TensorStorage(size_t size, bool pin_memory, TensorAllocator& allocator)
        : size_(size), allocator_(allocator), pin_memory_(pin_memory) {
        data_ = allocator_.allocate(size, pin_memory);
        if (!data_) {
            throw std::bad_alloc{};
        }
    }

    // Disallow implicit copy
    TensorStorage(const TensorStorage& other) = delete;
    TensorStorage& operator=(const TensorStorage& other) = delete;

    // Move
    TensorStorage(TensorStorage&& other) noexcept
        : data_(other.data_), size_(other.size_), allocator_(other.allocator_), pin_memory_(other.pin_memory_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    TensorStorage& operator=(TensorStorage&& other) {
        if (this != &other) {
            // TODO: check device and allocator
            if (data_) {
                allocator_.free(data_);
            }
            data_ = other.data_;
            size_ = other.size_;
            pin_memory_ = other.pin_memory_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void copy(const TensorStorage& from) {
        if (!from.data_ || !data_) {
            throw std::runtime_error("Cannot copy: invalid storage");
        }
        if (from.size_ > size_) {
            throw std::runtime_error("Source storage larger than destination");
        }
        allocator_.copy(data_, from.data_, from.size_);
    }

    TensorStorage clone() const {
        TensorStorage new_storage{size_, pin_memory_, allocator_};
        allocator_.copy(new_storage.data(), data_, size_);

        // RVO by compiler
        return new_storage;
    }

    ~TensorStorage() {
        if (data_) {
            allocator_.free(data_);
        }
    }

    void* data() const { return data_; }
    size_t size() const { return size_; }
    bool is_empty() const { return size_ == 0; }

    bool is_pinned() const {
        return pin_memory_;
    }
};

class Tensor {
private:
    std::vector<uint64_t> shape_;
    std::vector<uint64_t> strides_;
    size_t n_elements_;
    Dtype dtype_;
    size_t offset_;  // Byte offset from start of storage

    std::shared_ptr<TensorStorage> storage_;

public:
    Tensor(const std::vector<uint64_t>& shape, Dtype dtype = Dtype::Float32, bool pin_memory = false)
    : shape_{shape}, dtype_{dtype}, offset_{0} {
        if (dtype >= Dtype::Count) {
            throw std::runtime_error{"Invalid dtype"};
        }
        if (shape.empty()) {
            throw std::runtime_error{"Shape cannot be empty"};
        }
        n_elements_ = compute_total_elements(shape_);

        // TODO(feat): get allocator according to different backends (devices)
        TensorAllocator& allocator = get_default_host_allocator();
        storage_ = std::make_shared<TensorStorage>(n_elements_ * dtype_size(dtype), pin_memory, allocator);
        compute_strides();
    }

    // Move
    Tensor(Tensor&& other) noexcept
        : shape_{std::exchange(other.shape_, {})},
        strides_{std::exchange(other.strides_, {})},
        n_elements_{std::exchange(other.n_elements_, 0)},
        dtype_{std::exchange(other.dtype_, Dtype::Float32)},
        offset_{std::exchange(other.offset_, 0)},
        storage_{std::move(other.storage_)} 
        { }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            shape_ = std::exchange(other.shape_, {});
            strides_ = std::exchange(other.strides_, {});
            n_elements_ = std::exchange(other.n_elements_, 0);
            dtype_ = std::exchange(other.dtype_, Dtype::Float32);
            offset_ = std::exchange(other.offset_, 0);
            storage_ = std::move(other.storage_);
        }
        return *this;
    }

    // NOTE: Implicit copy is SHALLOW
    //     with shared_ptr, the copy is shallow
    //     should handled correctly by the compiler
    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;

    // managed by shared_ptr
    ~Tensor() = default;

    Tensor() = delete;

    void compute_strides() {
        strides_.resize(shape_.size());
        if (shape_.empty()) return;
        
        strides_[shape_.size() - 1] = 1;
        for (int i = shape_.size() - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    bool is_contiguous() const {
        if (shape_.size() != strides_.size()) return false;
        
        std::vector<uint64_t> expected_strides(shape_.size());
        if (shape_.empty()) return true;
        
        expected_strides[shape_.size() - 1] = 1;
        for (int i = shape_.size() - 2; i >= 0; --i) {
            expected_strides[i] = expected_strides[i + 1] * shape_[i + 1];
        }
        return strides_ == expected_strides;
    }

    size_t compute_total_elements(const std::vector<uint64_t>& shape) const {
        size_t total = 1;
        for (auto dim : shape) {
            total *= dim;
        }
        return total;
    }

public:
    const std::vector<uint64_t>& shape() const { return shape_; }

    const std::vector<uint64_t>& strides() const { return strides_; }
    uint64_t stride(uint64_t dim) const { 
        if (dim >= strides_.size()) {
            throw std::runtime_error{"Dimension out of bounds"};
        }
        return strides_[dim]; 
    }

    uint64_t n_elements() const { return compute_total_elements(shape_); }
    size_t ndim() const { return shape_.size(); }
    size_t nbytes() const { return n_elements() * dtype_size(dtype_); }
    
    Dtype dtype() const { return dtype_; }
    size_t element_size() const { return dtype_size(dtype_); }
    
    std::shared_ptr<TensorStorage> storage() const { return storage_; }
    
    size_t offset() const { return offset_; }

    // NOTE: 
    //     caller of data() and typed_data() 
    //     should NOT take ownership of the data
    void* data() const { 
        return static_cast<char*>(storage_->data()) + offset_; 
    }
    template<typename T>
    T* typed_data() const {
        static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type");
        if constexpr (std::is_same_v<T, float>) {
            if (dtype_ != Dtype::Float32) {
                throw std::runtime_error("Type mismatch: tensor dtype is not float32");
            }
        } else if constexpr (std::is_same_v<T, int8_t>) {
            if (dtype_ != Dtype::Int8) {
                throw std::runtime_error("Type mismatch: tensor dtype is not int8");
            }
        } else if constexpr (std::is_same_v<T, int16_t>) {
            if (dtype_ != Dtype::Int16) {
                throw std::runtime_error("Type mismatch: tensor dtype is not int16");
            }
        } else if constexpr (std::is_same_v<T, int32_t>) {
            if (dtype_ != Dtype::Int32) {
                throw std::runtime_error("Type mismatch: tensor dtype is not int32");
            }
        } else if constexpr (std::is_same_v<T, int64_t>) {
            if (dtype_ != Dtype::Int64) {
                throw std::runtime_error("Type mismatch: tensor dtype is not int64");
            }
        } else {
            throw std::runtime_error("Unsupported type for typed_data()");
        }
        return static_cast<T*>(data());
    }

public:
    static Tensor empty(const std::vector<uint64_t>& shape, Dtype dtype = Dtype::Float32, bool pin_memory = false) {
        return Tensor(shape, dtype, pin_memory);
    }

    static Tensor zeros(const std::vector<uint64_t>& shape, Dtype dtype = Dtype::Float32, bool pin_memory = false) {
        Tensor tensor(shape, dtype, pin_memory);
        return ops::fill(tensor, 0.0f);
    }

    static Tensor ones(const std::vector<uint64_t>& shape, Dtype dtype = Dtype::Float32, bool pin_memory = false) {
        Tensor tensor(shape, dtype, pin_memory);
        return ops::fill(tensor, 1.0f);
    }

    static Tensor arange(size_t start, size_t end, size_t step = 1, Dtype dtype = Dtype::Int32, bool pin_memory = false) {
        if (step == 0) {
            throw std::runtime_error{"step should > 0"};
        }
        size_t len = (end - start) / step;
        Tensor tensor({len}, dtype, pin_memory);
        return ops::arange(tensor, start, step);
    }

    static Tensor arange(int64_t end, Dtype dtype = Dtype::Int32, bool pin_memory = false) {
        return arange(0, end, 1, dtype, pin_memory);
    }

    Tensor view(
        const std::vector<uint64_t>& shape,
        const std::vector<uint64_t>& strides,
        const size_t offset
    ) const {
        return Tensor(shape, strides, dtype_, offset, storage_);
    }

    // Print tensor information and contents
    void print(const std::string& name = "Tensor", uint64_t max_elements = 10) const {
        CHECK_THROW(data());
        
        std::printf("%s [", name.c_str());
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::printf("%llu", shape_[i]);
            if (i < shape_.size() - 1) std::printf("x");
        }
        std::printf(", %s]\n", dtype_name(dtype_));
        
        if (!is_contiguous()) {
            std::printf("  Non-contiguous tensor, strides: [");
            for (size_t i = 0; i < strides_.size(); ++i) {
                std::printf("%llu", strides_[i]);
                if (i < strides_.size() - 1) std::printf(", ");
            }
            std::printf("]\n");
        }
        
        uint64_t total_elements = n_elements();
        uint64_t elements_to_show = std::min(max_elements, total_elements);
        
        if (total_elements == 0) {
            std::printf("  Empty tensor\n");
            return;
        }
        
        std::printf("  Elements: ");
        print_elements(elements_to_show);
        
        if (total_elements > max_elements) {
            std::printf(" ... (+%llu more)", total_elements - max_elements);
        }
        std::printf("\n");
    }

private:
    void print_elements(uint64_t count) const {
        switch (dtype_) {
            case Dtype::Float32: {
                auto* typed_data = static_cast<const float*>(data());
                for (uint64_t i = 0; i < count; ++i) {
                    std::printf("%.6f", typed_data[i]);
                    if (i < count - 1) std::printf(", ");
                }
                break;
            }
            case Dtype::Float16: {
                // TODO: support float16
                auto* typed_data = static_cast<const uint16_t*>(data());
                for (uint64_t i = 0; i < count; ++i) {
                    std::printf("%.6hu", typed_data[i]);
                    if (i < count - 1) std::printf(", ");
                }
                break;
            }
            case Dtype::Int8: {
                auto* typed_data = static_cast<const int8_t*>(data());
                for (uint64_t i = 0; i < count; ++i) {
                    std::printf("%d", static_cast<int>(typed_data[i]));
                    if (i < count - 1) std::printf(", ");
                }
                break;
            }
            case Dtype::Int16: {
                auto* typed_data = static_cast<const int16_t*>(data());
                for (uint64_t i = 0; i < count; ++i) {
                    std::printf("%d", static_cast<int>(typed_data[i]));
                    if (i < count - 1) std::printf(", ");
                }
                break;
            }
            case Dtype::Int32: {
                auto* typed_data = static_cast<const int32_t*>(data());
                for (uint64_t i = 0; i < count; ++i) {
                    std::printf("%d", typed_data[i]);
                    if (i < count - 1) std::printf(", ");
                }
                break;
            }
            default:
                std::printf("[Unsupported dtype for printing]");
        }
    }

public:
    
    // Print detailed tensor structure
    void print_detailed(const std::string& name = "Tensor") const {
        CHECK_THROW(data());
        
        std::printf("=== %s ===\n", name.c_str());
        std::printf("Shape: [");
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::printf("%llu", shape_[i]);
            if (i < shape_.size() - 1) std::printf(", ");
        }
        std::printf("]\n");
        std::printf("Data type: %s\n", dtype_name(dtype_));
        std::printf("Element size: %zu bytes\n", dtype_size(dtype_));
        std::printf("Strides: [");
        for (size_t i = 0; i < strides_.size(); ++i) {
            std::printf("%llu", strides_[i]);
            if (i < strides_.size() - 1) std::printf(", ");
        }
        std::printf("]\n");
        std::printf("Contiguous: %s\n", is_contiguous() ? "Yes" : "No");
        std::printf("Total elements: %llu\n", n_elements());
        std::printf("Total size: %zu bytes\n", nbytes());
        std::printf("Memory: %p\n", data());
        std::printf("===================\n");
    }

private:
    // Private constructor for creating views (shares storage)
    Tensor(
        const std::vector<uint64_t>& shape,
        const std::vector<uint64_t>& strides,
        Dtype dtype,
        size_t offset,
        std::shared_ptr<TensorStorage> storage
    ) : shape_{shape}, strides_{strides}, dtype_{dtype}, offset_{offset}, storage_{storage} {
        n_elements_ = compute_total_elements(shape_);
    }
};

// Implementation of standalone tensor printing functions
inline void print_tensor(const Tensor& tensor, const std::string& name, uint32_t max_elements) {
    tensor.print(name, max_elements);
}

inline void print_tensor_detailed(const Tensor& tensor, const std::string& name) {
    tensor.print_detailed(name);
}