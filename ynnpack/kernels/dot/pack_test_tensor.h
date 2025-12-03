// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_DOT_PACK_TEST_TENSOR_H_
#define XNNPACK_YNNPACK_KERNELS_DOT_PACK_TEST_TENSOR_H_

#include <cassert>
#include <cstddef>
#include <cstring>
#include <vector>

#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/test/buffer.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/dot/pack.h"

namespace ynn {

// If `tile_k > 1`, we need to transpose b such that `tile_k` values of the k
// dimension are contiguous in memory.
template <typename T>
Tensor<T> pack_b(Tensor<T> b, size_t tile_k, size_t tile_n) {
  const size_t elem_count = type_element_count(type_of<T>());
  const size_t elem_size_bits = sizeof(T) * 8 / elem_count;

  assert(tile_k % elem_count == 0);

  // Get n, k dimensions from b, and remove them from the extents.
  std::vector<size_t> extents = b.extents();
  size_t n = extents.back() * elem_count;
  extents.pop_back();
  size_t k = extents.back();
  extents.pop_back();

  // Remember these extents, which are the batch dimensions for the transpose
  // operation below.
  std::vector<size_t> batch_extents = extents;

  // Add the new transposed dimensions.
  // blocks_n is always 1 (and thus `block_n = n`) in these tests.
  extents.push_back(ceil_div(k, tile_k));
  extents.push_back(align_up(n, tile_n));
  extents.push_back(tile_k / elem_count);

  // Make the result.
  Tensor<T> result(extents, Alignment({.bytes = tile_k * tile_n * sizeof(T)}));
  packer p(/*transpose=*/false, elem_size_bits, tile_k, align_up(n, tile_n));
  for (const auto& i : EnumerateIndices(batch_extents)) {
    p.pack(k, n, b.stride(b.rank() - 2) * sizeof(T), b.slice_leading(i).base(),
           result.stride(result.rank() - 3) * sizeof(T), 0,
           result.slice_leading(i).base());
  }
  return result;
}

template <typename T>
Tensor<T> transpose_a(Tensor<T> a, size_t tile_k) {
  const size_t elem_count = type_element_count(type_of<T>());
  const size_t elem_size_bits = sizeof(T) * 8 / elem_count;

  assert(tile_k % elem_count == 0);

  // Get n, k dimensions from a, and remove them from the extents.
  std::vector<size_t> extents = a.extents();
  size_t k = extents.back() * elem_count;
  extents.pop_back();
  size_t m = extents.back();
  extents.pop_back();

  // Remember these extents, which are the batch dimensions for the transpose
  // operation below.
  std::vector<size_t> batch_extents = extents;

  // Add the new transposed dimensions.
  extents.push_back(ceil_div(k, tile_k));
  extents.push_back(m * tile_k);

  // Make the result.
  Tensor<T> result(extents);
  packer p(/*transpose=*/true, elem_size_bits, tile_k, m * tile_k);
  for (const auto& i : EnumerateIndices(batch_extents)) {
    p.pack(k, m, a.stride(a.rank() - 2) * sizeof(T), a.slice_leading(i).base(),
           result.stride(result.rank() - 2) * sizeof(T), 0,
           result.slice_leading(i).base());
  }
  return result;
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_DOT_PACK_TEST_TENSOR_H_
