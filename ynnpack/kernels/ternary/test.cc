// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/ternary/reference.h"
#include "ynnpack/kernels/ternary/ternary.h"

using testing::Combine;
using testing::Values;
using testing::ValuesIn;
using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

struct Shape {
  size_t m;
  size_t n;
  size_t padding_a;
  size_t padding_b;
  size_t padding_c;
  size_t padding_x;
};

std::string to_string(const Shape& shape) {
  std::stringstream sstr;
  sstr << shape.m << "x" << shape.n;
  if (shape.padding_a != 0 || shape.padding_b != 0 || shape.padding_x != 0) {
    sstr << "_" << shape.padding_a << "_" << shape.padding_b << "_"
         << shape.padding_x;
  }
  return sstr.str();
}

struct KernelInfo {
  uint64_t arch_flags = 0;
  ternary_kernel_fn kernel;
  init_ternary_params_fn init_params;

  // Constructor for a kernel function.
  KernelInfo(uint64_t arch_flags, ternary_kernel_fn kernel,
             init_ternary_params_fn init_params)
      : arch_flags(arch_flags), kernel(kernel), init_params(init_params) {}
};

template <typename A, typename B, typename C, typename X, typename OpInfo>
void TestImpl(const KernelInfo& kernel_info, const OpInfo& op_info, size_t m,
              size_t a_n, size_t b_n, size_t c_n, const Shape& shape) {
  if (!is_arch_supported(kernel_info.arch_flags)) {
    GTEST_SKIP() << "Unsupported hardware";
  }

  ReplicableRandomDevice rng;

  ternary_kernel_fn kernel = kernel_info.kernel;
  init_ternary_params_fn init_params = kernel_info.init_params;

  size_t n = std::max(std::max(a_n, b_n), c_n);

  Tensor<A> a({m, a_n + shape.padding_a});
  Tensor<B> b({m, b_n + shape.padding_b});
  Tensor<C> c({m, c_n + shape.padding_c});
  Tensor<X> x({m, n + shape.padding_x});

  quantization_params a_quantization = random_quantization(A(), rng);
  quantization_params b_quantization = random_quantization(B(), rng);
  quantization_params c_quantization = random_quantization(C(), rng);
  quantization_params x_quantization = random_quantization(X(), rng);
  // These kernels are mostly variations of multiply and multiply-add, so
  // testing extreme values is not required, and integer overflow is an issue.
  fill_random(a.data(), a.size(), rng, -255, 255, a_quantization);
  fill_random(b.data(), b.size(), rng, -255, 255, b_quantization);
  fill_random(c.data(), c.size(), rng, -255, 255, c_quantization);

  a = a.crop_padding({0, 0}, {0, shape.padding_a});
  b = b.crop_padding({0, 0}, {0, shape.padding_b});
  c = c.crop_padding({0, 0}, {0, shape.padding_c});
  x = x.crop_padding({0, 0}, {0, shape.padding_x});

  broadcast_extent_1(a);
  broadcast_extent_1(b);
  broadcast_extent_1(c);

  ternary_params params;
  if (init_params) {
    init_params(a_quantization.scale, a_quantization.zero_point,
                b_quantization.scale, b_quantization.zero_point,
                c_quantization.scale, c_quantization.zero_point,
                x_quantization.scale, x_quantization.zero_point, params);
  }

  kernel(m, n, a.stride(0) * sizeof(A), a.stride(1) * sizeof(A), a.base(),
         b.stride(0) * sizeof(B), b.stride(1) * sizeof(B), b.base(),
         c.stride(0) * sizeof(C), c.stride(1) * sizeof(C), c.base(),
         x.stride(0) * sizeof(X), x.base(), &params);

  check_results(op_info, a, b, c, x, a_quantization, b_quantization,
                c_quantization, x_quantization);
}

template <typename A, typename B, typename C, typename X, typename OpInfo>
void TestOp(const KernelInfo& kernel_info, const OpInfo& op_info,
            const Shape& shape) {
  TestImpl<A, B, C, X, OpInfo>(kernel_info, op_info, shape.m, shape.n, shape.n,
                               shape.n, shape);
}

template <typename A, typename B, typename C, typename X, typename OpInfo>
void TestOpBroadcastA(const KernelInfo& kernel_info, const OpInfo& op_info,
                      const Shape& shape) {
  TestImpl<A, B, C, X, OpInfo>(kernel_info, op_info, shape.m, 1, shape.n,
                               shape.n, shape);
}

template <typename A, typename B, typename C, typename X, typename OpInfo>
void TestOpBroadcastB(const KernelInfo& kernel_info, const OpInfo& op_info,
                      const Shape& shape) {
  TestImpl<A, B, C, X, OpInfo>(kernel_info, op_info, shape.m, shape.n, 1,
                               shape.n, shape);
}

template <typename A, typename B, typename C, typename X, typename OpInfo>
void TestOpBroadcastC(const KernelInfo& kernel_info, const OpInfo& op_info,
                      const Shape& shape) {
  TestImpl<A, B, C, X, OpInfo>(kernel_info, op_info, shape.m, shape.n, shape.n,
                               1, shape);
}

const std::vector<Shape> all_shapes = []() {
  std::vector<Shape> shapes;

  const std::vector<size_t> all_ns = simd_sizes_up_to(256);

  const size_t all_ms[] = {5};

  const size_t padding = 16;
  for (size_t i : all_ms) {
    for (size_t j : all_ns) {
      shapes.push_back({i, j, 0, 0, 0, 0});
    }
  }
  shapes.push_back({8, 4, padding, 0, 0, 0});
  shapes.push_back({8, 4, 0, padding, 0, 0});
  shapes.push_back({8, 4, 0, 0, padding, 0});
  shapes.push_back({8, 4, 0, 0, 0, padding});
  return shapes;
}();

#define YNN_ELEMENTWISE_KERNEL(arch_flags, kernel, op, init_params_fn, type_a, \
                               type_b, type_c, type_x)                         \
  class OpsTest_##kernel : public testing::TestWithParam<Shape> {};            \
                                                                               \
  TEST_P(OpsTest_##kernel, op) {                                               \
    const KernelInfo kernel_info(arch_flags, kernel, init_params_fn);          \
    const struct op op_info;                                                   \
    TestOp<type_a, type_b, type_c, type_x>(kernel_info, op_info, GetParam());  \
  }                                                                            \
  TEST_P(OpsTest_##kernel, op_broadcast_a) {                                   \
    const KernelInfo kernel_info(arch_flags, kernel, init_params_fn);          \
    const struct op op_info;                                                   \
    TestOpBroadcastA<type_a, type_b, type_c, type_x>(kernel_info, op_info,     \
                                                     GetParam());              \
  }                                                                            \
  TEST_P(OpsTest_##kernel, op_broadcast_b) {                                   \
    const KernelInfo kernel_info(arch_flags, kernel, init_params_fn);          \
    const struct op op_info;                                                   \
    TestOpBroadcastB<type_a, type_b, type_c, type_x>(kernel_info, op_info,     \
                                                     GetParam());              \
  }                                                                            \
  TEST_P(OpsTest_##kernel, op_broadcast_c) {                                   \
    const KernelInfo kernel_info(arch_flags, kernel, init_params_fn);          \
    const struct op op_info;                                                   \
    TestOpBroadcastC<type_a, type_b, type_c, type_x>(kernel_info, op_info,     \
                                                     GetParam());              \
  }                                                                            \
                                                                               \
  INSTANTIATE_TEST_SUITE_P(kernel, OpsTest_##kernel, ValuesIn(all_shapes),     \
                           [](const auto& i) { return to_string(i.param); });
#include "ynnpack/kernels/ternary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

}  // namespace ynn