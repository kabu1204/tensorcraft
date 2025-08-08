#include "tensor.h"
#include "slice.h"

int main() {
    // Create a 1D tensor with values 0..5 (float32)
    Tensor base = Tensor::arange(6, Dtype::Float32);

    // View it as 2x3 (row-major contiguous view)
    Tensor a = base.view({2, 3}, {3, 1}, 0);

    // Indexing
    float v = a.at<float>(1, 2); // 5.0f

    // Slicing
    Tensor cols = a[{{0, 2}, {1, 3}}]; // a[0:2, 1:3]

    // Reshape requires contiguity
    // Tensor r = cols.reshape({1,4}); // would throw
    Tensor cols_c = cols.contiguous(); // materialize as contiguous
    Tensor r = cols_c.reshape({1, 4}); // ok

    // Permute/transpose
    Tensor t = a.T(); // 2D transpose

    // Unified iterator (read)
    for (auto it = a.begin(); it != a.end(); ++it) {
      float x = (*it).as_float32();
      (void)x;
    }

    // Unified iterator (write)
    for (auto it = a.begin(); it != a.end(); ++it) {
      (*it).as_float32() = 1.0f;
    }
}