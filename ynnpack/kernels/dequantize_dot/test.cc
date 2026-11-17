// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/tolerance.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/dequantize_dot/dequantize_dot.h"

namespace ynn {

struct Shape {
  size_t m;
  size_t n;
};

std::string to_string(const Shape& shape) {
  std::stringstream sstr;
  sstr << shape.m << "x" << shape.n;
  return sstr.str();
}

template <typename Output>
void Reference(size_t m, size_t n, size_t stride_dot_m, const int32_t* dot,
               size_t stride_a_offset_m, const int32_t* a_offset,
               size_t stride_b_offset_n, const int32_t* b_offset,
               size_t stride_offset_n, const float* offset,
               const float* a_scale, size_t stride_a_scale_m,
               const float* b_scale, size_t stride_b_scale_n,
               size_t stride_output_m, size_t stride_output_n, Output* output,
               const dequantize_dot_params&) {
  for (size_t i = 0; i < m; ++i) {
    const int32_t* dot_i = offset_bytes(dot, i * stride_dot_m);
    const int32_t a_offset_i = *offset_bytes(a_offset, i * stride_a_offset_m);
    const float a_scale_i = *offset_bytes(a_scale, i * stride_a_scale_m);
    Output* output_i = offset_bytes(output, i * stride_output_m);

    for (size_t j = 0; j < n; ++j) {
      int32_t dot_ij = *offset_bytes(dot_i, j * sizeof(int32_t));
      const int32_t b_offset_j = *offset_bytes(b_offset, j * stride_b_offset_n);
      const float b_scale_j = *offset_bytes(b_scale, j * stride_b_scale_n);
      const float offset_j = *offset_bytes(offset, j * stride_offset_n);

      float output_ij = static_cast<float>(dot_ij - a_offset_i * b_offset_j);
      output_ij = output_ij * a_scale_i * b_scale_j + offset_j;

      *offset_bytes(output_i, j * stride_output_n) =
          static_cast<Output>(output_ij);
    }
  }
}

template <typename Output>
void TestKernel(uint64_t arch_flags, dequantize_dot_kernel_fn kernel,
                const Shape& dot_shape, size_t a_offset_m, size_t b_offset_n,
                size_t offset_n, size_t a_scale_m, size_t b_scale_n) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP() << "Unsupported hardware";
  }

  ReplicableRandomDevice rng;

  const size_t m = std::max({dot_shape.m, a_offset_m, a_scale_m});
  const size_t n = std::max({dot_shape.n, b_offset_n, offset_n, b_scale_n});
  ASSERT_EQ(dot_shape.n, n);

  Tensor<int32_t> dot({dot_shape.m, dot_shape.n});
  Tensor<int32_t> a_offset({a_offset_m});
  Tensor<int32_t> b_offset({b_offset_n});
  Tensor<float> offset({offset_n});
  Tensor<float> a_scale({a_scale_m});
  Tensor<float> b_scale({b_scale_n});
  Tensor<Output> output({m, n});
  Tensor<Output> reference({m, n});

  const int max_abs_dot = 8000;
  const int max_abs_ab_offset = 1000;
  const float max_abs_ab_scale = 0.01f;
  const float max_abs_offset = 1.0f;

  fill_random(dot.data(), dot.size(), rng, -max_abs_dot, max_abs_dot);
  fill_random(a_offset.data(), a_offset.size(), rng, -max_abs_ab_offset,
              max_abs_ab_offset);
  fill_random(b_offset.data(), b_offset.size(), rng, -max_abs_ab_offset,
              max_abs_ab_offset);
  fill_random(offset.data(), offset.size(), rng, -max_abs_offset,
              max_abs_offset);
  fill_random(a_scale.data(), a_scale.size(), rng, 0.01f * max_abs_ab_scale,
              max_abs_ab_scale);
  fill_random(b_scale.data(), b_scale.size(), rng, 0.01f * max_abs_ab_scale,
              max_abs_ab_scale);

  broadcast_extent_1(dot);
  broadcast_extent_1(a_offset);
  broadcast_extent_1(b_offset);
  broadcast_extent_1(offset);
  broadcast_extent_1(a_scale);
  broadcast_extent_1(b_scale);
  broadcast_extent_1(output);

  dequantize_dot_params params = {};

  kernel(m, n, dot.stride_bytes(0), dot.base(), a_offset.stride_bytes(0),
         a_offset.base(), b_offset.stride_bytes(0), b_offset.base(),
         offset.stride_bytes(0), offset.base(), a_scale.stride_bytes(0),
         a_scale.base(), b_scale.stride_bytes(0), b_scale.base(),
         output.stride_bytes(0), output.base(), &params);

  Reference<Output>(m, n, dot.stride_bytes(0), dot.base(),
                    a_offset.stride_bytes(0), a_offset.base(),
                    b_offset.stride_bytes(0), b_offset.base(),
                    offset.stride_bytes(0), offset.base(), a_scale.base(),
                    a_scale.stride_bytes(0), b_scale.base(),
                    b_scale.stride_bytes(0), reference.stride_bytes(0),
                    reference.stride_bytes(1), reference.base(), params);

  tolerance_spec tol = {/*relative=*/3.0f, /*absolute=*/2.0f};
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      ASSERT_NEAR(output(i, j), reference(i, j),
                  tol.absolute_error(reference(i, j)))
          << "at (" << i << ", " << j << "), shape " << m << "x" << n;
    }
  }
}

const std::vector<Shape> all_shapes = []() {
  std::vector<Shape> shapes;
  for (size_t m : {1, 2, 5}) {
    for (size_t n : simd_sizes_up_to(256)) {
      shapes.push_back({m, n});
    }
  }
  return shapes;
}();

#define YNN_DEQUANTIZE_DOT_KERNEL(arch_flags, kernel, type)                   \
  class DequantizeDotTest_##kernel : public testing::TestWithParam<Shape> {}; \
                                                                              \
  TEST_P(DequantizeDotTest_##kernel, no_broadcast) {                          \
    const Shape& shape = GetParam();                                          \
    TestKernel<type>(arch_flags, kernel, shape, shape.m, shape.n, shape.n,    \
                     shape.m, shape.n);                                       \
  }                                                                           \
  TEST_P(DequantizeDotTest_##kernel, broadcast_dot) {                         \
    const Shape& shape = GetParam();                                          \
    TestKernel<type>(arch_flags, kernel, {1, shape.n}, shape.m, shape.n,      \
                     shape.n, shape.m, shape.n);                              \
  }                                                                           \
  TEST_P(DequantizeDotTest_##kernel, broadcast_a_offset) {                    \
    const Shape& shape = GetParam();                                          \
    TestKernel<type>(arch_flags, kernel, shape, 1, shape.n, shape.n, shape.m, \
                     shape.n);                                                \
  }                                                                           \
  TEST_P(DequantizeDotTest_##kernel, broadcast_b_offset) {                    \
    const Shape& shape = GetParam();                                          \
    TestKernel<type>(arch_flags, kernel, shape, shape.m, 1, shape.n, shape.m, \
                     shape.n);                                                \
  }                                                                           \
  TEST_P(DequantizeDotTest_##kernel, broadcast_offset) {                      \
    const Shape& shape = GetParam();                                          \
    TestKernel<type>(arch_flags, kernel, shape, shape.m, shape.n, 1, shape.m, \
                     shape.n);                                                \
  }                                                                           \
  TEST_P(DequantizeDotTest_##kernel, broadcast_a_scale) {                     \
    const Shape& shape = GetParam();                                          \
    TestKernel<type>(arch_flags, kernel, shape, shape.m, shape.n, shape.n, 1, \
                     shape.n);                                                \
  }                                                                           \
  TEST_P(DequantizeDotTest_##kernel, broadcast_b_scale) {                     \
    const Shape& shape = GetParam();                                          \
    TestKernel<type>(arch_flags, kernel, shape, shape.m, shape.n, shape.n,    \
                     shape.m, 1);                                             \
  }                                                                           \
                                                                              \
  INSTANTIATE_TEST_SUITE_P(kernel, DequantizeDotTest_##kernel,                \
                           testing::ValuesIn(all_shapes),                     \
                           [](const auto& i) { return to_string(i.param); });

#include "ynnpack/kernels/dequantize_dot/kernels.inc"
#undef YNN_DEQUANTIZE_DOT_KERNEL

}  // namespace ynn
