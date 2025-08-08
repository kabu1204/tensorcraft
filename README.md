## TensorCraft

TensorCraft is a small, educational C++17 tensor library, designed as a compact foundation for learning and experimentation rather than a fully-functional framework. It is now CPU-only.

### Features
- **Tensors with shared storage** via `TensorStorage` (views are zero‑copy)
- **Row‑major strides** with full exposure of `shape`, `strides`, and `offset`
- **Indexing and slicing**: `select`, `narrow`, `slice`, `operator[]` with integers and `Slice`
- **View transforms**: `permute`, `transpose`, `T()` (2D), `reshape` (requires contiguity)
- **Materialization**: `.contiguous()` to produce a contiguous copy (uses same allocator/pin as the source)
- **Unified iterator** for user and internal use: `for (auto it = t.begin(); it != t.end(); ++it) { (*it).as_float32(); }`
- **Convenience creators**: `empty`, `zeros`, `ones`, `arange`
- **Info utilities**: `info()` (pretty print) and `info_full()`

Supported dtypes (host): `Float32`.

### Build
Requires CMake ≥ 3.20 and a C++17 compiler.

```bash
cd tensorcraft
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

This builds the static library `libtensorcraft.a` and example/test executables.

### Running examples/tests
From the `build/` directory:

```bash
./tensor_examples
./tensor_shape
./tensor_init
./test_tensor_storage
./tensor_slice_example
./tensor_view_ops_test
```

### Quick start

```cpp
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
```

### Core concepts
- `shape[d]`: length of dimension d.
- `strides[d]` (elements): how many elements to skip when incrementing index along dim d.
- `offset` (bytes): byte offset into shared storage; views can adjust `shape/strides/offset` without copying.
- Flat index (elements): `sum(i_d * strides[d])`. Byte address: `base + (offset + flat_index * element_size)`.

### API
- Construction: `Tensor::empty/zeros/ones/arange`
- Views: `view(shape, strides, offset)`, `select`, `narrow`, `slice` (and `operator[]` using `Slice` or brace-init), `index(...)`
- Transforms: `permute`, `transpose`, `T()`, `reshape` (throws if tensor is not contiguous), `.contiguous()`
- Data access:
  - `typed_data<T>()` validates dtype and returns `T*`
  - `at<T>(i, j, ...)` validates indices and returns an lvalue reference
  - Iteration: `begin()/end()` yield `ElementRef` with `as_*()` typed accessors

### Memory/allocators
- Default host allocator: `AlignedHostAllocator` (cacheline aligned)
- `TensorStorage` owns memory, views share `std::shared_ptr<TensorStorage>`
- `.contiguous()` allocates with the same allocator and pin setting as the source storage, and uses the allocator’s `copy` to move data

### Roadmap
- Negative slice step
- Broadcasting and advanced indexing
- More dtypes
- Operators (kernels)
- CUDA backend

### Project layout
- `src/include/`: headers (`tensor.h`, `slice.h`, `ops.h`, `memory.h`, `common.h`)
- `src/`: implementations (`tensor.cc`, `slice.cc`, `ops.cc`, `memory.cc`)
- `examples/`: examples and tests
- `docs/`: documentations

