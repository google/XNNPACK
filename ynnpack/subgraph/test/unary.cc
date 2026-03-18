// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
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
#include "ynnpack/subgraph/test/subgraph_builder.h"

// This needs to be in the global namespace for argument dependent lookup to
// work.
using ::ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

using ::testing::Combine;
using ::testing::ValuesIn;

template <typename A, typename X>
void TestOp(A, X, const unary_op_info& op_info, ynn_unary_operator op) {
  ReplicableRandomDevice rng;
  std::uniform_int_distribution<size_t> rank_dist(1, YNN_MAX_TENSOR_RANK);
  for (auto _ : FuzzTest(std::chrono::milliseconds(100))) {
    size_t rank = rank_dist(rng);
    // We want the total number of elements to be reasonable, so choose max_dim
    // such that a random shape of rank `p.rank` produces this max size.
    constexpr size_t max_size = 1024;
    const size_t max_dim = static_cast<size_t>(std::ceil(
        std::pow(static_cast<double>(max_size),
                 1.0 / static_cast<double>(std::max<size_t>(1, rank)))));
    quantization_params a_quantization = random_quantization(A(), rng);
    quantization_params output_quantization = random_quantization(X(), rng);

    SubgraphBuilder subgraph(2);
    subgraph.AddInput(type_of<A>(), rank, 0, a_quantization)
        .AddOutput(type_of<X>(), rank, 1, output_quantization)
        .AddUnary(op, 0, 1);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank, 1, max_dim);

      Tensor<A> a(shape);
      Tensor<X> output(shape);

      interval domain = op_info.domain(type_of<A>());
      fill_random(a.data(), a.size(), rng, domain.min, domain.max,
                  a_quantization);

      runtime.ReshapeExternalTensor(shape, a.data(), 0).ReshapeRuntime();

      ASSERT_EQ(runtime.GetExternalTensorShape(1), shape);

      runtime.SetupExternalTensor(output.data(), 1).InvokeRuntime();

      check_results(op_info, a, output, a_quantization, output_quantization);
    }
  }
}

class IntegerOps
    : public testing::TestWithParam<std::tuple<ynn_type, ynn_unary_operator>> {
};
class RealOps
    : public testing::TestWithParam<std::tuple<ynn_type, ynn_unary_operator>> {
};

template <typename T>
void TestOp(T type, ynn_unary_operator op) {
  const unary_op_info& op_info = *get_unary_op_info(op);
  TestOp(type, type, op_info, op);
}

TEST_P(IntegerOps, op) {
  ynn_type type = std::get<0>(GetParam());
  ynn_unary_operator op = std::get<1>(GetParam());
  SwitchIntegerType(type, [&](auto type) { TestOp(type, op); });
}

TEST_P(RealOps, op) {
  ynn_type type = std::get<0>(GetParam());
  ynn_unary_operator op = std::get<1>(GetParam());
  SwitchRealType(type, [&](auto type) { TestOp(type, op); });
}

class Convert : public testing::TestWithParam<std::tuple<ynn_type, ynn_type>> {
};

template <typename A, typename X>
void TestConvert(A a, X x) {
  TestOp(a, x, convert(), ynn_unary_convert);
}

template <typename F>
constexpr decltype(auto) SwitchType(ynn_type type, F&& f) {
  switch (type) {
    case ynn_type_int8:
      return std::forward<F>(f)(quantized<int8_t>());
    case ynn_type_uint8:
      return std::forward<F>(f)(quantized<uint8_t>());
    case ynn_type_fp16:
      return std::forward<F>(f)(half());
    case ynn_type_bf16:
      return std::forward<F>(f)(bfloat16());
    case ynn_type_fp32:
      return std::forward<F>(f)(float());
    case ynn_type_int32:
      return std::forward<F>(f)(int32_t());
    default:
      YNN_UNREACHABLE;
  }
}

TEST_P(Convert, op) {
  ynn_type from = std::get<0>(GetParam());
  ynn_type to = std::get<1>(GetParam());
  SwitchType(from, [&](auto from) {
    SwitchType(to, [&](auto to) { TestConvert(from, to); });
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
    ynn_unary_erf,
    ynn_unary_square_root,
    ynn_unary_cube_root,
    ynn_unary_reciprocal_square_root,
    ynn_unary_log,
    ynn_unary_log1p,
    ynn_unary_exp,
    ynn_unary_expm1,
    ynn_unary_tanh,
    ynn_unary_convert,
    ynn_unary_sign,
};
// clang-format on

INSTANTIATE_TEST_SUITE_P(UnaryTest, IntegerOps,
                         Combine(ValuesIn(all_integer_types),
                                 ValuesIn(all_integer_ops)),
                         test_param_to_string<IntegerOps::ParamType>);

INSTANTIATE_TEST_SUITE_P(UnaryTest, RealOps,
                         Combine(ValuesIn(all_real_types),
                                 ValuesIn(all_real_ops)),
                         test_param_to_string<RealOps::ParamType>);

INSTANTIATE_TEST_SUITE_P(UnaryTest, Convert,
                         Combine(ValuesIn(all_real_types),
                                 ValuesIn(all_real_types)),
                         test_param_to_string<Convert::ParamType>);

}  // namespace ynn