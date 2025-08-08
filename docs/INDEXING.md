# Tensor Indexing and Slicing

## Flat index

Given indices `(i0, i1, ..., i{n-1})`, the flat index (in elements) is:

```
flat_index = sum_{d=0...n-1} (i_d * strides[d])
```

with each `i_d` validated into `[0, shape[d])`.

The byte pointer for element `(i0,...,i{n-1})` is:

```
byte_ptr = base_ptr + (offset + flat_index * dtype_size)
```

## Slicing

Sslicing: `[start, end)` with a positive `step`.
- Bounds normalization:
  - If `start` is unspecified (−inf) it defaults to 0. If `end` is unspecified (+inf) it defaults to `shape[dim]`.
  - Negative bounds are translated by adding `shape[dim]`.
  - Finally, both are clamped into `[0, shape[dim]]` and empty ranges collapse (`end < start` → `end = start`).
- Size and stride update:
  - New size along the sliced dim: `size_dim = ceil_div(max(0, end - start), step)`
  - New element stride along that dim: `new_strides[dim] = old_strides[dim] * step`
  - New byte offset adds: `offset += start * old_strides[dim] * dtype_size`
- Other dims keep their original `shape` and `strides`.

This makes slicing a zero-copy view.

## Narrow

`narrow(dim, start, length)` is a convenient slice with `end = min(shape[dim], start + length)` and `step = 1`.

## Select

`select(dim, index)` indexes one element along `dim` and REMOVES that dimension:
- New shape and strides drop `dim`.
- Offset is increased by `index * strides[dim] * dtype_size`.

This mirrors PyTorch `select`, which reduces the rank by 1.

## operator[] and multi-indexing

- `t[int]` -> reduce-dim slice of size 1: `t.slice(0, i, i+1)`
- `t[Slice{s,e,st}]` -> `t.slice(0, s, e, st)`
- `t[{Index...}]` (initializer_list) to index multiple dims at once. Each `Index` is either an `int64_t` or a `Slice`.
- For brace-init slices, `{start, end}` defaults `step=1`, so you can write `t[{1, {3,5}}]`.

- For multi-indexing, integers will reduce dimensions.
- Slices always keep the dimension.
- `select(dim, ...)` is available when you explicitly want to drop the dimension.

## Example calculations

Consider a contiguous 2D tensor with `shape = [3, 4]` and `strides = [4, 1]` (elements). Dtype is float32 (`dtype_size = 4`).

- Element `(1, 2)`:
  - `flat_index = 1*4 + 2*1 = 6`
  - `byte_offset = base_offset + 6 * 4`

- Slice `[:, 1:3]`:
  - Along dim 1: `start=1, end=3, step=1`
  - New `shape = [3, 2]`
  - New `strides = [4, 1]` (unchanged because step=1)
  - New `offset = old_offset + 1 * 1 * 4`

- Slice `[:, ::2]`:
  - Along dim 1: `start=0, end=4, step=2`
  - New `shape = [3, 2]`
  - New `strides = [4, 2]`
  - New `offset = old_offset + 0`

- Index `t[1]` (keep dim):
  - Translates to `slice(0, 1, 2)` along first dimension
  - New `shape = [1, 4]`, `strides = [4, 1]`, `offset += 1 * 4 * 4`

- Multi-index `t[{1, {0,2}}]`:
  - First dim `1` -> `slice(0, 1, 2)`, keep-dim
  - Second dim `{0, 2}` -> `slice(1, 0, 2)`
  - Result `shape = [1, 2]`, `strides = [4, 1]`, `offset += (1 * 4) * 4`

