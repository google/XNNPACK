// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/binary/reference.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

// This needs to be in the global namespace for argument dependent lookup to
// work.
using ::ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

using ::testing::Combine;
using ::testing::ValuesIn;

void update_shape(std::vector<size_t>& shape,
                  const std::vector<size_t>& update) {
  for (size_t d = 0; d < shape.size(); ++d) {
    if (shape[d] != 1) shape[d] = update[d];
  }
}

std::vector<size_t> reversed(std::vector<size_t> shape) {
  std::reverse(shape.begin(), shape.end());
  return shape;
}

template <typename T>
void TestOp(T, const binary_op_info& op_info, ynn_binary_operator op) {
  ReplicableRandomDevice rng;
  std::uniform_int_distribution<size_t> rank_dist(1, YNN_MAX_TENSOR_RANK);
  std::bernoulli_distribution random_bool(0.5);

  for (auto _ : FuzzTest(std::chrono::milliseconds(100))) {
    const size_t rank = rank_dist(rng);
    std::vector<size_t> a_shape(rank);
    std::vector<size_t> b_shape(rank);
    // Randomly broadcast some dims from one side at a time.
    for (size_t d = 0; d < rank; ++d) {
      if (random_bool(rng)) {
        // Don't broadcast this dimension.
      } else if (random_bool(rng)) {
        a_shape[d] = 1;
      } else {
        b_shape[d] = 1;
      }
    }
    if (random_bool(rng)) {
      a_shape.resize(std::min(rank, rank_dist(rng)));
    } else {
      b_shape.resize(std::min(rank, rank_dist(rng)));
    }
    // We want the total number of elements to be reasonable, so choose max_dim
    // such that a random shape of rank `p.rank` produces this max size.
    constexpr size_t max_size = 1024;
    const size_t max_dim = static_cast<size_t>(std::ceil(
        std::pow(static_cast<double>(max_size),
                 1.0 / static_cast<double>(std::max<size_t>(1, rank)))));
    quantization_params a_quantization = random_quantization(T(), rng);
    quantization_params b_quantization = random_quantization(T(), rng);
    quantization_params x_quantization = random_quantization(T(), rng);

    SubgraphBuilder subgraph(3);
    subgraph.AddInput(type_of<T>(), reversed(a_shape), 0, a_quantization)
        .AddInput(type_of<T>(), reversed(b_shape), 1, b_quantization)
        .AddOutput(type_of<T>(), rank, 2, x_quantization)
        .AddBinary(op, 0, 1, 2);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank, 1, max_dim);
      update_shape(a_shape, shape);
      update_shape(b_shape, shape);

      for (size_t d = 0; d < rank; ++d) {
        if (d < a_shape.size() && d < b_shape.size()) {
          shape[d] = std::max(a_shape[d], b_shape[d]);
        } else if (d < a_shape.size()) {
          shape[d] = a_shape[d];
        } else {
          shape[d] = b_shape[d];
        }
      }

      Tensor<T> a(reversed(a_shape));
      Tensor<T> b(reversed(b_shape));
      Tensor<T> x(reversed(shape));

      fill_random(a.data(), a.size(), rng, a_quantization);
      fill_random(b.data(), b.size(), rng, b_quantization);

      runtime.ReshapeExternalTensor(a.extents(), a.data(), 0)
          .ReshapeExternalTensor(b.extents(), b.data(), 1)
          .ReshapeRuntime();

      ASSERT_EQ(runtime.GetExternalTensorShape(2), x.extents());

      runtime.SetupExternalTensor(x.data(), 2).InvokeRuntime();

      broadcast_extent_1(a);
      broadcast_extent_1(b);

      check_results(op_info, a, b, x, a_quantization, b_quantization,
                    x_quantization);
    }
  }
}

class IntegerOps
    : public testing::TestWithParam<std::tuple<ynn_type, ynn_binary_operator>> {
};
class RealOps
    : public testing::TestWithParam<std::tuple<ynn_type, ynn_binary_operator>> {
};

template <typename T>
void TestOp(T type, ynn_binary_operator op) {
  const binary_op_info& op_info = *get_binary_op_info(op);
  TestOp(type, op_info, op);
}

TEST_P(IntegerOps, op) {
  ynn_type type = std::get<0>(GetParam());
  ynn_binary_operator op = std::get<1>(GetParam());
  SwitchIntegerType(type, [&](auto type) { TestOp(type, op); });
}

TEST_P(RealOps, op) {
  ynn_type type = std::get<0>(GetParam());
  ynn_binary_operator op = std::get<1>(GetParam());
  SwitchRealType(type, [&](auto type) { TestOp(type, op); });
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

INSTANTIATE_TEST_SUITE_P(BinaryTest, IntegerOps,
                         Combine(ValuesIn(all_integer_types),
                                 ValuesIn(all_integer_ops)),
                         test_param_to_string<IntegerOps::ParamType>);

INSTANTIATE_TEST_SUITE_P(BinaryTest, RealOps,
                         Combine(ValuesIn(all_real_types),
                                 ValuesIn(all_real_ops)),
                         test_param_to_string<RealOps::ParamType>);

}  // namespace ynn