#include "tensor.h"
#include "slice.h"

#include <algorithm>
#include <vector>
#include <initializer_list>
#include <variant>

static inline void normalize_index(int64_t& idx, int64_t dim_size) {
    if (idx < 0) idx += dim_size;
    if (idx < 0 || idx >= dim_size) {
        throw std::runtime_error("Index out of bounds");
    }
}

static inline void normalize_slice_bounds(int64_t& start, int64_t& end, int64_t step, int64_t dim_size) {
    if (step == 0) throw std::runtime_error("slice step must not be 0");
    if (step < 0) throw std::runtime_error("negative step not supported yet");

    if (start == Slice::kNegInf) start = 0;
    if (end == Slice::kPosInf) end = dim_size;

    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;

    start = std::clamp<int64_t>(start, 0, dim_size);
    end = std::clamp<int64_t>(end, 0, dim_size);

    if (end < start) end = start; // empty
}

Tensor Tensor::select(size_t dim, int64_t index) const {
    if (dim >= shape_.size()) throw std::runtime_error("select dim out of range");
    int64_t dim_size = static_cast<int64_t>(shape_[dim]);
    normalize_index(index, dim_size);

    // new shape removes the selected dimension
    std::vector<uint64_t> new_shape;
    new_shape.reserve(shape_.size() - 1);
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i == dim) continue;
        new_shape.push_back(shape_[i]);
    }

    // strides keep all except removed dim
    std::vector<uint64_t> new_strides;
    new_strides.reserve(strides_.size() - 1);
    for (size_t i = 0; i < strides_.size(); ++i) {
        if (i == dim) continue;
        new_strides.push_back(strides_[i]);
    }

    size_t new_offset = offset_ + static_cast<size_t>(index) * strides_[dim] * dtype_size(dtype_);
    return Tensor(new_shape, new_strides, dtype_, new_offset, storage_);
}

Tensor Tensor::narrow(size_t dim, int64_t start, int64_t length) const {
    if (dim >= shape_.size()) throw std::runtime_error("narrow dim out of range");
    int64_t dim_size = static_cast<int64_t>(shape_[dim]);
    if (length < 0) throw std::runtime_error("narrow length must be >= 0");
    normalize_index(start, dim_size);
    int64_t end = std::min<int64_t>(dim_size, start + length);

    std::vector<uint64_t> new_shape = shape_;
    new_shape[dim] = static_cast<uint64_t>(std::max<int64_t>(0, end - start));

    std::vector<uint64_t> new_strides = strides_;
    size_t new_offset = offset_ + static_cast<size_t>(start) * strides_[dim] * dtype_size(dtype_);
    return Tensor(new_shape, new_strides, dtype_, new_offset, storage_);
}

Tensor Tensor::slice(size_t dim, int64_t start, int64_t end, int64_t step) const {
    if (dim >= shape_.size()) throw std::runtime_error("slice dim out of range");
    int64_t dim_size = static_cast<int64_t>(shape_[dim]);
    normalize_slice_bounds(start, end, step, dim_size);

    int64_t range = std::max<int64_t>(0, end - start);
    uint64_t size_dim = static_cast<uint64_t>((range + step - 1) / step);

    std::vector<uint64_t> new_shape = shape_;
    new_shape[dim] = size_dim;

    // important
    std::vector<uint64_t> new_strides = strides_;
    new_strides[dim] *= static_cast<uint64_t>(step);

    size_t new_offset = offset_ + static_cast<size_t>(start) * strides_[dim] * dtype_size(dtype_);
    return Tensor(new_shape, new_strides, dtype_, new_offset, storage_);
}

Tensor Tensor::slice(const std::vector<Slice>& slices) const {
    if (slices.size() != shape_.size()) {
        throw std::runtime_error("Number of slices must match tensor dimensions");
    }
    std::vector<uint64_t> new_shape = shape_;
    std::vector<uint64_t> new_strides = strides_;
    size_t new_offset = offset_;

    for (size_t dim = 0; dim < slices.size(); ++dim) {
        int64_t s = slices[dim].start;
        int64_t e = slices[dim].end;
        int64_t st = slices[dim].step;
        int64_t dim_size = static_cast<int64_t>(shape_[dim]);
        normalize_slice_bounds(s, e, st, dim_size);

        int64_t range = std::max<int64_t>(0, e - s);
        uint64_t size_dim = static_cast<uint64_t>((range + st - 1) / st);
        new_shape[dim] = size_dim;
        new_offset += static_cast<size_t>(s) * strides_[dim] * dtype_size(dtype_);
        new_strides[dim] *= static_cast<uint64_t>(st);
    }

    return Tensor(new_shape, new_strides, dtype_, new_offset, storage_);
}

Tensor Tensor::operator[](int64_t index) const {
    // Index along the first dimension but keep the dim (size 1)
    return slice(0, index, index + 1, 1);
}

Tensor Tensor::operator[](Slice s) const {
    // Slice along the first dimension of the current view
    return slice(0, s.start, s.end, s.step);
}

Tensor Tensor::operator[](std::initializer_list<Index> indices) const {
    std::vector<Index> idxv(indices.begin(), indices.end());
    return index(idxv);
}

Tensor Tensor::operator[](std::initializer_list<IndexArg> indices) const {
    std::vector<Index> idxv;
    idxv.reserve(indices.size());
    for (const auto& ia : indices) {
        idxv.push_back(static_cast<Index>(ia));
    }
    return index(idxv);
}

Tensor Tensor::index(const std::vector<Index>& indices, bool keep_dim) const {
    if (indices.size() > shape_.size()) {
        throw std::runtime_error("too many indices");
    }

    // We'll progressively apply select/slice on a running view
    Tensor view = *this;
    size_t dim = 0;
    for (const auto& idx : indices) {
        if (std::holds_alternative<int64_t>(idx)) {
            int64_t v = std::get<int64_t>(idx);
            // keep dimension as size-1 slice
            if (keep_dim) {
                view = view.slice(dim, v, v + 1, 1);
                ++dim;
            } else {
                view = view.select(dim, std::get<int64_t>(idx));
            }
        } else {
            const Slice& s = std::get<Slice>(idx);
            view = view.slice(dim, s.start, s.end, s.step);
            ++dim; // slicing keeps the dimension
        }
    }
    return view;
}


