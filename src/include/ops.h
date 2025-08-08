#pragma once

#include <cstdint>

class Tensor;

namespace ops {

// NOTE: To align with libtorch's behavior, 
//       we always return a shallow copy of 
//       the tensor by value
Tensor fill(const Tensor& a, float value);
Tensor arange(const Tensor& a, int64_t start, int64_t step = 1);

// element-wise ops
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

}