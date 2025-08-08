#include "tensor.h"
#include "common.h"
#include <iostream>


static inline void validate_permutation(const std::vector<size_t>& dims, size_t ndim) {
    if (dims.size() != ndim) {
        throw std::runtime_error("permute: dims size must == tensor ndim");
    }
    std::vector<int> seen(ndim, 0);
    for (size_t d : dims) {
        if (d >= ndim) throw std::runtime_error("permute: dim out of range");
        if (seen[d]) throw std::runtime_error("permute: duplicate dim");
        seen[d] = 1;
    }
}

Tensor Tensor::permute(const std::vector<size_t>& dims) const {
    validate_permutation(dims, shape_.size());
    std::vector<uint64_t> new_shape(shape_.size());
    std::vector<uint64_t> new_strides(strides_.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        new_shape[i] = shape_[dims[i]];
        new_strides[i] = strides_[dims[i]];
    }
    return Tensor(new_shape, new_strides, dtype_, offset_, storage_);
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    const size_t n = shape_.size();
    if (dim0 >= n || dim1 >= n) {
        throw std::runtime_error("transpose: dim out of range");
    }
    if (dim0 == dim1) return *this;
    std::vector<size_t> dims(n);
    for (size_t i = 0; i < n; ++i) dims[i] = i;
    std::swap(dims[dim0], dims[dim1]);
    return permute(dims);
}

Tensor Tensor::T() const {
    if (shape_.size() != 2) {
        throw std::runtime_error("T(): only valid for 2D tensors");
    }
    return transpose(0, 1);
}

Tensor Tensor::reshape(const std::vector<uint64_t>& new_shape) const {
    // Only zero-copy reshape if storage is contiguous with current view
    if (!is_contiguous()) {
        throw std::runtime_error("reshape: tensor is not contiguous");
    }
    if (new_shape.empty()) {
        throw std::runtime_error("reshape: new_shape cannot be empty");
    }
    size_t new_elems = 1;
    for (auto d : new_shape) new_elems *= d;
    if (new_elems != n_elements()) {
        throw std::runtime_error("reshape: number of elements must not change");
    }
    std::vector<uint64_t> new_strides(new_shape.size());
    if (!new_shape.empty()) {
        new_strides[new_shape.size() - 1] = 1;
        for (int i = static_cast<int>(new_shape.size()) - 2; i >= 0; --i) {
            new_strides[static_cast<size_t>(i)] = new_strides[static_cast<size_t>(i + 1)] * new_shape[static_cast<size_t>(i + 1)];
        }
    }
    return Tensor(new_shape, new_strides, dtype_, offset_, storage_);
}

Tensor Tensor::contiguous() const {
    // If already contiguous and offset is 0, we can return a shallow copy (same storage)
    if (is_contiguous() && offset_ == 0) {
        return Tensor(shape_, strides_, dtype_, offset_, storage_);
    }

    CHECK_THROW(storage_);
    bool pinned = storage_->is_pinned();
    TensorAllocator& src_allocator = storage_->allocator();

    // build contiguous strides for the result
    std::vector<uint64_t> contig_strides(shape_.size());
    if (!shape_.empty()) {
        contig_strides[shape_.size() - 1] = 1;
        for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
            contig_strides[static_cast<size_t>(i)] = contig_strides[static_cast<size_t>(i + 1)] * shape_[static_cast<size_t>(i + 1)];
        }
    }

    // create fresh storage with same allocator/pin
    const size_t elem_size = dtype_size(dtype_);
    auto new_storage = std::make_shared<TensorStorage>(n_elements() * elem_size, pinned, src_allocator);
    Tensor result(shape_, contig_strides, dtype_, /*offset=*/0, new_storage);

    if (is_contiguous()) {
        src_allocator.copy(result.data(), static_cast<const char*>(storage_->data()) + offset_, n_elements() * elem_size);
        return result;
    }

    // general n-d strided copy using unified iterator
    {
        const size_t esize = elem_size;
        auto it = begin();
        auto it_end = end();
        char* dst = static_cast<char*>(result.data());
        for (; it != it_end; ++it) {
            const void* src_ptr = (*it).data();
            src_allocator.copy(dst, src_ptr, esize);
            dst += esize;
        }
    }

    return result;
}

std::string Tensor::info(uint64_t max_elements) const {
    (void)max_elements;
    CHECK_THROW(data());
    std::string out;
    out.reserve(256);
    out += "tensor(";
    std::vector<uint64_t> index_buffer(shape_.size(), 0);
    out += format_array_recursive(0, 7, index_buffer); // indent aligns with 'array('
    out += ", dtype=";
    out += dtype_name(dtype_);
    out += ", shape=" + format_shape();
    out += ", strides=" + format_strides();
    out += ")\n";
    return out;
}

std::string Tensor::format_array_recursive(size_t dim, int indent, std::vector<uint64_t>& idx) const {
    std::string out;
    if (dim + 1 == shape_.size()) {
        // last dimension: print a row
        out += "[";
        uint64_t width = shape_[dim];
        switch (dtype_) {
            case Dtype::Float32: {
                auto* typed = static_cast<const float*>(data());
                char buf[64];
                for (uint64_t i = 0; i < width; ++i) {
                    idx[dim] = i;
                    // compute linear index
                    size_t li = 0;
                    for (size_t d = 0; d < shape_.size(); ++d) {
                        li += idx[d] * strides_[d];
                    }
                    int n = std::snprintf(buf, sizeof(buf), "%.6g", static_cast<double>(typed[li]));
                    if (n > 0) out.append(buf, static_cast<size_t>(std::min(n, static_cast<int>(sizeof(buf)-1))));
                    if (std::strchr(buf, '.') == nullptr && std::strchr(buf, 'e') == nullptr && std::strchr(buf, 'E') == nullptr) {
                        out += ".";
                    }
                    if (i + 1 < width) out += ", ";
                }
                break;
            }
            case Dtype::Int32: {
                auto* typed = static_cast<const int32_t*>(data());
                for (uint64_t i = 0; i < width; ++i) {
                    idx[dim] = i;
                    size_t li = 0;
                    for (size_t d = 0; d < shape_.size(); ++d) {
                        li += idx[d] * strides_[d];
                    }
                    out += std::to_string(typed[li]);
                    if (i + 1 < width) out += ", ";
                }
                break;
            }
            case Dtype::Int8: {
                auto* typed = static_cast<const int8_t*>(data());
                for (uint64_t i = 0; i < width; ++i) {
                    idx[dim] = i;
                    size_t li = 0;
                    for (size_t d = 0; d < shape_.size(); ++d) {
                        li += idx[d] * strides_[d];
                    }
                    out += std::to_string(static_cast<int>(typed[li]));
                    if (i + 1 < width) out += ", ";
                }
                break;
            }
            case Dtype::Int16: {
                auto* typed = static_cast<const int16_t*>(data());
                for (uint64_t i = 0; i < width; ++i) {
                    idx[dim] = i;
                    size_t li = 0;
                    for (size_t d = 0; d < shape_.size(); ++d) {
                        li += idx[d] * strides_[d];
                    }
                    out += std::to_string(static_cast<int>(typed[li]));
                    if (i + 1 < width) out += ", ";
                }
                break;
            }
            case Dtype::Float16: {
                auto* typed = static_cast<const uint16_t*>(data());
                for (uint64_t i = 0; i < width; ++i) {
                    idx[dim] = i;
                    size_t li = 0;
                    for (size_t d = 0; d < shape_.size(); ++d) {
                        li += idx[d] * strides_[d];
                    }
                    out += std::to_string(typed[li]);
                    if (i + 1 < width) out += ", ";
                }
                break;
            }
            default:
                out += "[unsupported dtype]";
        }
        out += "]";
        return out;
    }

    // nested dimensions
    out += "[";
    uint64_t depth = shape_[dim];
    for (uint64_t i = 0; i < depth; ++i) {
        idx[dim] = i;
        // newline and indentation for inner arrays
        if (i > 0) {
            out += "\n";
            out.append(static_cast<size_t>(indent), ' ');
        }
        out += format_array_recursive(dim + 1, indent + 2, idx);
        if (i + 1 < depth) out += ",";
    }
    out += "]";
    return out;
}

std::string Tensor::format_shape() const {
    std::string out;
    out.reserve(256);
    out += "(";
    for (size_t i = 0; i < shape_.size(); ++i) {
        out += std::to_string(shape_[i]);
        if (i < shape_.size() - 1) out += ", ";
    }
    out += ")";
    return out;
}

std::string Tensor::format_strides() const {
    std::string out;
    out.reserve(256);
    out += "(";
    for (size_t i = 0; i < strides_.size(); ++i) {
        out += std::to_string(strides_[i]);
        if (i < strides_.size() - 1) out += ", ";
    }
    out += ")";
    return out;
}

// Return formatted detailed tensor structure
std::string Tensor::info_full() const {
    CHECK_THROW(data());
    std::string out;
    out.reserve(256);
    out += "shape: " + format_shape() + "\n";
    out += "dtype: ";
    out += dtype_name(dtype_);
    out += "\n";
    out += "element_size: ";
    out += std::to_string(dtype_size(dtype_));
    out += " bytes\n";
    out += "strides: " + format_strides() + "\n";
    out += "contiguous: ";
    out += (is_contiguous() ? "True" : "False");
    out += "\n";
    out += "n_elements: ";
    out += std::to_string(n_elements());
    out += "\n";
    out += "nbytes: ";
    out += std::to_string(nbytes());
    out += " bytes\n";
    out += "data: ";
    {
        char buf[32];
        int n = std::snprintf(buf, sizeof(buf), "%p", data());
        if (n > 0) out.append(buf, static_cast<size_t>(std::min(n, static_cast<int>(sizeof(buf)-1))));
    }
    out += "\n";
    return out;
}
