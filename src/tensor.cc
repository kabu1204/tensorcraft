#include "tensor.h"
#include <iostream>


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
