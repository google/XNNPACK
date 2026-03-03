// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
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
#include "ynnpack/kernels/binary/binary.h"
#include "ynnpack/kernels/binary/reference.h"

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
  binary_kernel_fn kernel;

  // Constructor for a reference kernel.
  KernelInfo(ynn_binary_operator op, ynn_type type) {
    kernel = get_binary_reference_kernel(op, type);
    assert(kernel);
  }

  // Constructor for a kernel function.
  KernelInfo(uint64_t arch_flags, binary_kernel_fn kernel)
      : arch_flags(arch_flags), kernel(kernel) {
    assert(kernel);
  }
};

template <typename A, typename B, typename X, typename OpInfo>
void TestImpl(const KernelInfo& kernel_info, const OpInfo& op_info, size_t m,
              size_t a_n, size_t b_n, const Shape& shape) {
  if (!is_arch_supported(kernel_info.arch_flags)) {
    GTEST_SKIP() << "Unsupported hardware";
  }

  ReplicableRandomDevice rng;

  binary_kernel_fn kernel = kernel_info.kernel;

  size_t n = std::max(a_n, b_n);

  Tensor<A> a({m, a_n + shape.padding_a});
  Tensor<B> b({m, b_n + shape.padding_b});
  Tensor<X> x({m, n + shape.padding_x});

  fill_random(a.data(), a.size(), rng);
  fill_random(b.data(), b.size(), rng);

  a = a.crop_padding({0, 0}, {0, shape.padding_a});
  b = b.crop_padding({0, 0}, {0, shape.padding_b});
  x = x.crop_padding({0, 0}, {0, shape.padding_x});

  broadcast_extent_1(a);
  broadcast_extent_1(b);

  kernel(m, n, a.stride(0) * sizeof(A), a.stride(1) * sizeof(A), a.base(),
         b.stride(0) * sizeof(B), b.stride(1) * sizeof(B), b.base(),
         x.stride(0) * sizeof(X), x.base());

  check_results(op_info, a, b, x);
}

template <typename A, typename B, typename X, typename OpInfo>
void TestOp(const KernelInfo& kernel_info, const OpInfo& op_info,
            const Shape& shape, A = {}, B = {}, X = {}) {
  TestImpl<A, B, X>(kernel_info, op_info, shape.m, shape.n, shape.n, shape);
}

template <typename A, typename B, typename X, typename OpInfo>
void TestOpBroadcastA(const KernelInfo& kernel_info, const OpInfo& op_info,
                      const Shape& shape, A = {}, B = {}, X = {}) {
  TestImpl<A, B, X>(kernel_info, op_info, shape.m, 1, shape.n, shape);
}

template <typename A, typename B, typename X, typename OpInfo>
void TestOpBroadcastB(const KernelInfo& kernel_info, const OpInfo& op_info,
                      const Shape& shape, A = {}, B = {}, X = {}) {
  TestImpl<A, B, X>(kernel_info, op_info, shape.m, shape.n, 1, shape);
}

class Reference : public testing::TestWithParam<
                      std::tuple<ynn_type, ynn_binary_operator, Shape>> {};

template <typename Body>
auto SwitchType(ynn_type type, Body body) {
  switch (type) {
    case ynn_type_int32:
      return body(int32_t{});
    case ynn_type_fp32:
      return body(float{});
    default:
      break;
  }
  YNN_UNREACHABLE;
}

TEST_P(Reference, no_broadcast) {
  ynn_type type = std::get<0>(GetParam());
  ynn_binary_operator op = std::get<1>(GetParam());
  const Shape& shape = std::get<2>(GetParam());
  KernelInfo kernel_info(op, type);
  const binary_op_info& op_info = *get_binary_op_info(op);
  SwitchType(type, [&](auto type) {
    TestOp(kernel_info, op_info, shape, type, type, type);
  });
}
TEST_P(Reference, op_broadcast_a) {
  ynn_type type = std::get<0>(GetParam());
  ynn_binary_operator op = std::get<1>(GetParam());
  const Shape& shape = std::get<2>(GetParam());
  KernelInfo kernel_info(op, type);
  const binary_op_info& op_info = *get_binary_op_info(op);
  SwitchType(type, [&](auto type) {
    TestOpBroadcastA(kernel_info, op_info, shape, type, type, type);
  });
}
TEST_P(Reference, op_broadcast_b) {
  ynn_type type = std::get<0>(GetParam());
  ynn_binary_operator op = std::get<1>(GetParam());
  const Shape& shape = std::get<2>(GetParam());
  KernelInfo kernel_info(op, type);
  const binary_op_info& op_info = *get_binary_op_info(op);
  SwitchType(type, [&](auto type) {
    TestOpBroadcastB(kernel_info, op_info, shape, type, type, type);
  });
}

// clang-format off
const ynn_binary_operator all_integer_ops[] = {
    ynn_binary_add,
    ynn_binary_copysign,
    ynn_binary_divide,
    ynn_binary_max,
    ynn_binary_min,
    ynn_binary_multiply,
    ynn_binary_subtract,
    ynn_binary_pow,
};

const ynn_binary_operator all_real_ops[] = {
    ynn_binary_add,
    ynn_binary_copysign,
    ynn_binary_divide,
    ynn_binary_max,
    ynn_binary_min,
    ynn_binary_multiply,
    ynn_binary_pow,
    ynn_binary_squared_difference,
    ynn_binary_subtract,
    ynn_binary_leaky_relu,
};
// clang-format on

// For reference kernels, we assume the implementation is simple, and just test
// one shape (with various paddings).
const size_t padding = 16;

const Shape reference_shapes[] = {
    {256, 4, 0, 0, 0},
    {256, 4, padding, 0, 0},
    {256, 4, 0, padding, 0},
    {256, 4, 0, 0, padding},
};

INSTANTIATE_TEST_SUITE_P(RealOps, Reference,
                         Combine(Values(ynn_type_fp32), ValuesIn(all_real_ops),
                                 ValuesIn(reference_shapes)),
                         test_param_to_string<Reference::ParamType>);

INSTANTIATE_TEST_SUITE_P(IntegerOps, Reference,
                         Combine(Values(ynn_type_int32),
                                 ValuesIn(all_integer_ops),
                                 ValuesIn(reference_shapes)),
                         test_param_to_string<Reference::ParamType>);

const std::vector<Shape> all_shapes = []() {
  std::vector<Shape> shapes;

  const std::vector<size_t> all_ns = simd_sizes_up_to(256);

  const size_t all_ms[] = {5};

  for (size_t i : all_ms) {
    for (size_t j : all_ns) {
      shapes.push_back({i, j, 0, 0, 0});
    }
  }
  shapes.push_back({8, 4, padding, 0, 0});
  shapes.push_back({8, 4, 0, padding, 0});
  shapes.push_back({8, 4, 0, 0, padding});
  return shapes;
}();

#define YNN_ELEMENTWISE_KERNEL(arch_flags, kernel, op, type_a, type_b, type_x) \
  class kernel##_test : public testing::TestWithParam<Shape> {};               \
  TEST_P(kernel##_test, no_broadcast) {                                        \
    KernelInfo kernel_info(arch_flags, kernel);                                \
    TestOp<type_a, type_b, type_x>(kernel_info, op{}, GetParam());             \
  }                                                                            \
  TEST_P(kernel##_test, op_broadcast_a) {                                      \
    KernelInfo kernel_info(arch_flags, kernel);                                \
    TestOpBroadcastA<type_a, type_b, type_x>(kernel_info, op{}, GetParam());   \
  }                                                                            \
  TEST_P(kernel##_test, op_broadcast_b) {                                      \
    KernelInfo kernel_info(arch_flags, kernel);                                \
    TestOpBroadcastB<type_a, type_b, type_x>(kernel_info, op{}, GetParam());   \
  }                                                                            \
  INSTANTIATE_TEST_SUITE_P(test, kernel##_test, ValuesIn(all_shapes),          \
                           [](const auto& i) { return to_string(i.param); });
#include "ynnpack/kernels/binary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

}  // namespace ynn