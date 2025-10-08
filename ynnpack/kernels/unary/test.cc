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
  init_unary_params_fn init_params;

  // Constructor for a reference kernel.
  KernelInfo(ynn_unary_operator op, ynn_type a_type, bool a_quantized,
             ynn_type x_type, bool x_quantized) {
    const unary_kernel* kernel = get_unary_reference_kernel(
        op, a_type, a_quantized, x_type, x_quantized);
    this->kernel = kernel->op;
    init_params = kernel->init_params;
  }

  // Constructor for a kernel function.
  KernelInfo(uint64_t arch_flags, unary_kernel_fn kernel,
             init_unary_params_fn init_params)
      : arch_flags(arch_flags), kernel(kernel), init_params(init_params) {}
};

template <typename A, typename X, typename OpInfo>
void TestImpl(A, X, const KernelInfo& kernel_info, const OpInfo& op_info,
              const Shape& shape) {
  if (!is_arch_supported(kernel_info.arch_flags)) {
    GTEST_SKIP() << "Unsupported hardware";
  }
  ReplicableRandomDevice rng;

  unary_kernel_fn kernel = kernel_info.kernel;
  init_unary_params_fn init_params = kernel_info.init_params;

  Tensor<A> a({shape.m, shape.n + shape.padding_a});
  Tensor<X> x({shape.m, shape.n + shape.padding_x});
  a = a.crop_padding({0, 0}, {0, shape.padding_a});
  x = x.crop_padding({0, 0}, {0, shape.padding_x});

  quantization_params a_quantization = random_quantization(A(), rng);
  quantization_params x_quantization = random_quantization(X(), rng);
  interval domain = op_info.domain(type_of<A>());
  TypeGenerator<A> a_gen(domain.min, domain.max, a_quantization);
  a.generate([&]() { return a_gen(rng); });

  unary_params params;
  if (init_params) {
    init_params(a_quantization.scale, a_quantization.zero_point,
                x_quantization.scale, x_quantization.zero_point, params);
  }
  kernel(shape.m, shape.n, a.stride(0) * sizeof(A), a.base(),
         x.stride(0) * sizeof(X), x.base(), &params);

  check_results(op_info, a, x, a_quantization, x_quantization);
}

template <typename F>
constexpr decltype(auto) SwitchType(ynn_type type, bool is_quantized, F&& f) {
  switch (type) {
    case ynn_type_int32:
      return is_quantized ? std::forward<F>(f)(quantized<int32_t>())
                          : std::forward<F>(f)(int32_t());
    case ynn_type_int8:
      assert(is_quantized);
      return std::forward<F>(f)(quantized<int8_t>());
    case ynn_type_uint8:
      assert(is_quantized);
      return std::forward<F>(f)(quantized<uint8_t>());
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

class IntegerOps : public testing::TestWithParam<
                       std::tuple<ynn_type, ynn_unary_operator, Shape>> {};
class RealOps : public testing::TestWithParam<
                    std::tuple<ynn_type, ynn_unary_operator, Shape>> {};

TEST_P(IntegerOps, op) {
  ynn_type type = std::get<0>(GetParam());
  ynn_unary_operator op = std::get<1>(GetParam());
  const Shape& shape = std::get<2>(GetParam());
  KernelInfo kernel_info(op, type, /*a_quantized=*/false, type,
                         /*x_quantized=*/false);
  const unary_op_info& op_info = *get_unary_op_info(op);
  SwitchIntegerType(type, [&](auto type) {
    TestImpl(type, type, kernel_info, op_info, shape);
  });
}

TEST_P(RealOps, op) {
  ynn_type type = std::get<0>(GetParam());
  ynn_unary_operator op = std::get<1>(GetParam());
  const Shape& shape = std::get<2>(GetParam());
  KernelInfo kernel_info(op, type, /*a_quantized=*/true, type,
                         /*x_quantized=*/true);
  const unary_op_info& op_info = *get_unary_op_info(op);
  SwitchRealType(type, [&](auto type) {
    TestImpl(type, type, kernel_info, op_info, shape);
  });
}

class ReferenceConvert
    : public testing::TestWithParam<std::tuple<
          std::pair<ynn_type, bool>, std::pair<ynn_type, bool>, Shape>> {};

TEST_P(ReferenceConvert, op) {
  ynn_type a, x;
  bool a_is_quantized, x_is_quantized;
  std::tie(a, a_is_quantized) = std::get<0>(GetParam());
  std::tie(x, x_is_quantized) = std::get<1>(GetParam());
  KernelInfo kernel_info(ynn_unary_convert, a, a_is_quantized, x,
                         x_is_quantized);
  const Shape& shape = std::get<2>(GetParam());
  SwitchType(x, x_is_quantized, [&](auto x) {
    SwitchType(a, a_is_quantized,
               [&](auto a) { TestImpl(a, x, kernel_info, convert{}, shape); });
  });
}

// clang-format off
const ynn_type all_integer_types[] = {
    ynn_type_int32,
};

const ynn_type all_real_types[] = {
    ynn_type_int8,
    ynn_type_uint8,
    ynn_type_fp16,
    ynn_type_bf16,
    ynn_type_fp32,
};

std::pair<ynn_type, bool> all_convert_types[] = {
    {ynn_type_int8, true},
    {ynn_type_uint8, true},
    {ynn_type_int32, true},
    {ynn_type_int32, false},
    {ynn_type_fp16, false},
    {ynn_type_bf16, false},
    {ynn_type_fp32, false},
};

const ynn_unary_operator all_integer_ops[] = {
    ynn_unary_abs,
    ynn_unary_negate,
    ynn_unary_square,
    ynn_unary_sign,
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
// clang-format on

// For reference kernels, we assume the implementation is simple, and just test
// one shape (with various paddings).
const size_t padding = 16;

const Shape reference_shapes[] = {
    {256, 4, 0, 0},
    {256, 4, padding, 0},
    {256, 4, 0, padding},
};

// TODO: Dividing these into integer and real ops doesn't really work, because
// we can have a mix (e.g. converting qint8 -> int32_t). We need to have the
// `quantized<T>` decorators in the kernel metadata I think.
INSTANTIATE_TEST_SUITE_P(UnaryTest, IntegerOps,
                         Combine(ValuesIn(all_integer_types),
                                 ValuesIn(all_integer_ops),
                                 ValuesIn(reference_shapes)),
                         test_param_to_string<IntegerOps::ParamType>);

INSTANTIATE_TEST_SUITE_P(UnaryTest, RealOps,
                         Combine(ValuesIn(all_real_types),
                                 ValuesIn(all_real_ops),
                                 ValuesIn(reference_shapes)),
                         test_param_to_string<RealOps::ParamType>);

INSTANTIATE_TEST_SUITE_P(UnaryTest, ReferenceConvert,
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

#define YNN_ELEMENTWISE_KERNEL(arch_flags, kernel, op, init_params_fn, type_a, \
                               type_x)                                         \
  class kernel##_test : public testing::TestWithParam<Shape> {};               \
  TEST_P(kernel##_test, no_broadcast) {                                        \
    KernelInfo kernel_info(arch_flags, kernel, init_params_fn);                \
    TestImpl(type_a{}, type_x{}, kernel_info, op{}, GetParam());               \
  }                                                                            \
  INSTANTIATE_TEST_SUITE_P(test, kernel##_test, ValuesIn(all_shapes),          \
                           [](const auto& i) { return to_string(i.param); });
#include "ynnpack/kernels/unary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

}  // namespace ynn
