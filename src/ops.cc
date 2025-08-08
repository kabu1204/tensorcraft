#include "ops.h"
#include "tensor.h"

namespace ops {

// TODO: different dtypes, different backends

Tensor fill(const Tensor& a, float value) {
    auto* data = a.typed_data<float>();
    std::fill(data, data + a.n_elements(), value);
    return a;
}

Tensor arange(const Tensor& a, int64_t start, int64_t step) {
    auto* data = a.typed_data<float>();
    for (uint64_t i = 0; i < a.n_elements(); ++i) {
        data[i] = static_cast<float>(start + i * step);
    }
    return a;
}

}