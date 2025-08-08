#pragma once

#include "memory.h"
#include "common.h"
#include "ops.h"
#include "slice.h"

#include <cassert>
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
#include <array>


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
        // TODO: check device and allocator
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

    TensorAllocator& allocator() { return allocator_; }
    const TensorAllocator& allocator() const { return allocator_; }
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
    //     should be handled correctly by the compiler
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

    // shape/stride view transforms (return a new view on underlying storage)
    // permute dimensions according to dims (e.g. {1,0,2}) without copy
    Tensor permute(const std::vector<size_t>& dims) const;
    // swap two dimensions without copy
    Tensor transpose(size_t dim0, size_t dim1) const;
    // 2D convenience transpose
    Tensor T() const;
    // reshape view
    Tensor reshape(const std::vector<uint64_t>& new_shape) const;
    Tensor reshape(std::initializer_list<uint64_t> new_shape) const { return reshape(std::vector<uint64_t>(new_shape)); }

    // materialize a contiguous tensor with row-major strides
    Tensor contiguous() const;

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

    template<typename T, typename... Indices>
    T& at(Indices... indices) {
        auto* base_ptr = typed_data<T>();
        const size_t flat_index = compute_flat_index_from_indices(indices...);
        return base_ptr[flat_index];
    }

    template<typename T, typename... Indices>
    const T& at(Indices... indices) const {
        auto* base_ptr = typed_data<T>();
        const size_t flat_index = compute_flat_index_from_indices(indices...);
        return base_ptr[flat_index];
    }

    // indexing and slicing views
    Tensor select(size_t dim, int64_t index) const;
    Tensor narrow(size_t dim, int64_t start, int64_t length) const;
    Tensor slice(size_t dim, int64_t start, int64_t end, int64_t step = 1) const;
    Tensor slice(const std::vector<Slice>& slices) const;

    // operator[] interface indexing and slicing
    Tensor operator[](int64_t index) const;
    Tensor operator[](Slice s) const;

    // brace-init multi-dimensional slicing
    Tensor operator[](std::initializer_list<Index> indices) const;
    Tensor operator[](std::initializer_list<IndexArg> indices) const;
    Tensor index(const std::vector<Index>& indices, bool keep_dim = false) const;
    Tensor operator()(const std::vector<Index>& indices) const { return index(indices); }

    // numpy-style format
    std::string format_array_recursive(size_t dim, int indent, std::vector<uint64_t>& idx) const;
    std::string format_shape() const;
    std::string format_strides() const;
    std::string info(uint64_t max_elements = 32) const;
    std::string info_full() const;

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

    template<typename... Indices>
    size_t compute_flat_index_from_indices(Indices... indices) const {
        static_assert((std::is_integral_v<Indices> && ...), "Indices must be integral types");
        constexpr size_t kNumIdx = sizeof...(Indices);
        if (kNumIdx != shape_.size()) {
            throw std::runtime_error{"Number of indices must match tensor dimensions"};
        }
        std::array<int64_t, kNumIdx> idx_array{static_cast<int64_t>(indices)...};
        size_t flat_index = 0;
        for (size_t dim = 0; dim < kNumIdx; ++dim) {
            int64_t dim_size = static_cast<int64_t>(shape_[dim]);
            int64_t idx = idx_array[dim];
            if (idx < 0) idx += dim_size;  // negative index support
            if (idx < 0 || idx >= dim_size) {
                throw std::runtime_error{"Index out of bounds"};
            }
            flat_index += static_cast<size_t>(idx) * static_cast<size_t>(strides_[dim]);
        }
        return flat_index;
    }
};
