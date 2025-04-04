// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_TEST_SUBGRAPH_STENCIL_H_
#define THIRD_PARTY_XNNPACK_TEST_SUBGRAPH_STENCIL_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "src/xnnpack/buffer.h"
#include "src/xnnpack/math.h"

namespace xnnpack {

struct StencilParams {
  size_t size;
  size_t dilation;
  size_t stride;
  size_t padding_min;
  size_t padding_max;

  size_t padding() const { return padding_min + padding_max; }
  size_t dilated_kernel_extent() const { return (size - 1) * dilation + 1; }

  size_t output_extent(size_t input_extent) const {
    return doz(input_extent + padding(), dilated_kernel_extent()) / stride + 1;
  }

  size_t input_extent(size_t output_extent) const {
    assert(output_extent > 0);
    size_t unpadded = stride * (output_extent - 1) + dilated_kernel_extent();
    return std::max<size_t>(1, doz(unpadded, padding()));
  }

  // Many XNPACK implementations of stencil operations can't work correctly if
  // they are "empty", i.e. they don't use any non-padded input values. Many
  // randomly generated kernel parameters need the empty reduction case to work.
  // This helper detects these cases, so they can be skipped.
  bool result_is_identity(size_t input_extent, size_t output_extent) const {
    for (size_t i = 0; i < output_extent; ++i) {
      size_t count = 0;
      for (size_t j = 0; j < size; ++j) {
        size_t input_x = i * stride + j * dilation;
        if (input_x >= padding_min && input_x < input_extent + padding_min) {
          ++count;
        }
      }
      if (count == 0) return true;
    }
    return false;
  }
};

inline std::ostream& operator<<(std::ostream& os, const StencilParams& params) {
  // This is intended to be copy-pasteable into a test to make reproducing
  // reported failures easier.
  return os << "{" << params.size << ", " << params.dilation << ", "
            << params.stride << ", " << params.padding_min << ", "
            << params.padding_max << "}";
}

template <typename Rng>
StencilParams random_stencil_params(Rng& rng, int max_dilation = 2,
                                    int max_kernel_size = 7) {
  std::uniform_int_distribution<> size_dist{1, max_kernel_size};
  std::uniform_int_distribution<> dilation_dist{1, max_dilation};
  std::uniform_int_distribution<> stride_dist{1, 2};

  StencilParams result;
  result.size = size_dist(rng);
  result.dilation = dilation_dist(rng);
  result.stride =
      std::min<size_t>(stride_dist(rng),
                       std::max<size_t>(result.dilated_kernel_extent() - 1, 1));

  // TODO(b/404587443): Many XNNPACK operators produce incorrect results if
  // there is more padding than necessary. This logic ensures we don't do that.
  std::uniform_int_distribution<> padding_min_dist{
      0, static_cast<int>(result.dilated_kernel_extent()) - 1};
  result.padding_min = padding_min_dist(rng);
  std::uniform_int_distribution<> padding_max_dist{
      0, static_cast<int>(result.dilated_kernel_extent() - result.padding_min -
                          1)};
  result.padding_max = padding_max_dist(rng);
  return result;
}

// Replace the dimension `dim` in x with two dimensions: a "spatial" dimension,
// and a "kernel" dimension. These dimensions "overlap" in memory, such that
// `result(i, di) = x(i * stride + di * dilation)`. The spatial dimension will
// be reduced in extent such that no padding is required.
template <typename T>
Tensor<T> make_stencil_dim(Tensor<T> x, size_t dim,
                           const StencilParams& kernel) {
  x = x.expand_dims({dim + 1});

  std::vector<size_t> extents = x.extents();
  std::vector<size_t> strides = x.strides();
  extents[dim] = doz(extents[dim], kernel.dilated_kernel_extent() - 1);
  extents[dim + 1] = kernel.size;
  strides[dim + 1] = strides[dim] * kernel.dilation;
  strides[dim] *= kernel.stride;
  extents[dim] = divide_round_up(extents[dim], kernel.stride);
  x.set_shape(std::move(extents), std::move(strides));
  return x;
}

}  // namespace xnnpack

#endif  // THIRD_PARTY_XNNPACK_TEST_SUBGRAPH_STENCIL_H_
