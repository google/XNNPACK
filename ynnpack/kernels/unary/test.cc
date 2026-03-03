// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/unary/reference.h"
#include "ynnpack/kernels/unary/unary.h"

using testing::Combine;
using testing::Values;
using testing::ValuesIn;

// These need to be in the global namespace because ynn_type is in the global
// namespace.
using ynn::to_string;  // NOLINT(misc-unused-using-decls)
std::string to_string(const std::pair<ynn_type, bool>& type) {
  std::stringstream ss;
  if (type.second) {
    ss << "q";
  }
  ss << to_string(type.first);
  return ss.str();
}

namespace ynn {

struct Shape {
  size_t m;
  size_t n;
  size_t padding_a;
  size_t padding_x;
};

std::string to_string(const Shape& shape) {
  std::stringstream sstr;
  sstr << shape.m << "x" << shape.n;
  if (shape.padding_a != 0 || shape.padding_x != 0) {
    sstr << "_" << shape.padding_a << "_" << shape.padding_x;
  }
  return sstr.str();
}

struct KernelInfo {
  uint64_t arch_flags = 0;
  unary_kernel_fn kernel;

  // Constructor for a reference kernel.
  KernelInfo(ynn_unary_operator op, ynn_type type) {
    kernel = get_unary_reference_kernel(op, type);
    assert(kernel);
  }

  // Constructor for a reference convert op.
  KernelInfo(ynn_type a_type, ynn_type x_type) {
    kernel = get_convert_reference_kernel(a_type, x_type);
    assert(kernel);
  }

  // Constructor for a kernel function.
  KernelInfo(uint64_t arch_flags, unary_kernel_fn kernel)
      : arch_flags(arch_flags), kernel(kernel) {
    assert(kernel);
  }
};

template <typename A, typename X, typename OpInfo>
void TestImpl(A, X, const KernelInfo& kernel_info, const OpInfo& op_info,
              const Shape& shape) {
  if (!is_arch_supported(kernel_info.arch_flags)) {
    GTEST_SKIP() << "Unsupported hardware";
  }
  ReplicableRandomDevice rng;

  unary_kernel_fn kernel = kernel_info.kernel;

  Tensor<A> a({shape.m, shape.n + shape.padding_a});
  Tensor<X> x({shape.m, shape.n + shape.padding_x});

  interval domain = op_info.domain(type_of<A>());
  fill_random(a.data(), a.size(), rng, domain.min, domain.max);

  a = a.crop_padding({0, 0}, {0, shape.padding_a});
  x = x.crop_padding({0, 0}, {0, shape.padding_x});

  kernel(shape.m, shape.n, a.stride(0) * sizeof(A), a.base(),
         x.stride(0) * sizeof(X), x.base());

  check_results(op_info, a, x);
}

template <typename F>
constexpr decltype(auto) SwitchType(ynn_type type, F&& f) {
  switch (type) {
    case ynn_type_int32:
      return std::forward<F>(f)(int32_t());
    case ynn_type_int8:
      return std::forward<F>(f)(int8_t());
    case ynn_type_uint8:
      return std::forward<F>(f)(uint8_t());
    case ynn_type_fp16:
      return std::forward<F>(f)(half());
    case ynn_type_bf16:
      return std::forward<F>(f)(bfloat16());
    case ynn_type_fp32:
      return std::forward<F>(f)(float());
    default:
      YNN_UNREACHABLE;
  }
}

class Reference : public testing::TestWithParam<
                      std::tuple<ynn_type, ynn_unary_operator, Shape>> {};

TEST_P(Reference, op) {
  ynn_type type = std::get<0>(GetParam());
  ynn_unary_operator op = std::get<1>(GetParam());
  const Shape& shape = std::get<2>(GetParam());
  const unary_op_info& op_info = *get_unary_op_info(op);
  KernelInfo kernel_info(op, type);
  SwitchType(type, [&](auto type) {
    TestImpl(type, type, kernel_info, op_info, shape);
  });
}

class ReferenceConvert
    : public testing::TestWithParam<std::tuple<ynn_type, ynn_type, Shape>> {};

TEST_P(ReferenceConvert, op) {
  ynn_type a = std::get<0>(GetParam());
  ynn_type x = std::get<1>(GetParam());
  KernelInfo kernel_info(a, x);
  const Shape& shape = std::get<2>(GetParam());
  SwitchType(x, [&](auto x) {
    SwitchType(a,
               [&](auto a) { TestImpl(a, x, kernel_info, convert{}, shape); });
  });
}

// clang-format off
ynn_type all_convert_types[] = {
    ynn_type_int8,
    ynn_type_uint8,
    ynn_type_int32,
    ynn_type_fp16,
    ynn_type_bf16,
    ynn_type_fp32,
};

const ynn_unary_operator all_real_ops[] = {
    ynn_unary_abs,
    ynn_unary_floor,
    ynn_unary_ceil,
    ynn_unary_round,
    ynn_unary_negate,
    ynn_unary_square,
    ynn_unary_square_root,
    ynn_unary_cube_root,
    ynn_unary_reciprocal_square_root,
    ynn_unary_log,
    ynn_unary_log1p,
    ynn_unary_exp,
    ynn_unary_expm1,
    ynn_unary_erf,
    ynn_unary_tanh,
    ynn_unary_sign,
    ynn_unary_sine,
    ynn_unary_cosine,
    ynn_unary_sigmoid,
    ynn_unary_hardswish,
};

const ynn_unary_operator all_integer_ops[] = {
    ynn_unary_abs,
    ynn_unary_negate,
    ynn_unary_square,
    ynn_unary_sign,
};
// clang-format on

// For reference kernels, we assume the implementation is simple, and just test
// one shape (with various paddings).
const size_t padding = 16;

const Shape reference_shapes[] = {
    {256, 4, 0, 0},
    {256, 4, padding, 0},
    {256, 4, 0, padding},
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

INSTANTIATE_TEST_SUITE_P(Convert, ReferenceConvert,
                         Combine(ValuesIn(all_convert_types),
                                 ValuesIn(all_convert_types),
                                 ValuesIn(reference_shapes)),
                         test_param_to_string<ReferenceConvert::ParamType>);

const std::vector<Shape> all_shapes = []() {
  std::vector<Shape> shapes;

  const std::vector<size_t> all_ns = simd_sizes_up_to(256);

  const size_t all_ms[] = {5};

  for (size_t i : all_ms) {
    for (size_t j : all_ns) {
      shapes.push_back({i, j, 0, 0});
    }
  }
  shapes.push_back({8, 4, padding, 0});
  shapes.push_back({8, 4, 0, padding});
  return shapes;
}();

#define YNN_ELEMENTWISE_KERNEL(arch_flags, kernel, op, type_a, type_x) \
  class kernel##_test : public testing::TestWithParam<Shape> {};       \
  TEST_P(kernel##_test, no_broadcast) {                                \
    KernelInfo kernel_info(arch_flags, kernel);                        \
    TestImpl(type_a{}, type_x{}, kernel_info, op{}, GetParam());       \
  }                                                                    \
  INSTANTIATE_TEST_SUITE_P(test, kernel##_test, ValuesIn(all_shapes),  \
                           [](const auto& i) { return to_string(i.param); });
#include "ynnpack/kernels/unary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

}  // namespace ynn
