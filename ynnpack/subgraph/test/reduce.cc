// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
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
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

template <typename T>
std::function<T(T, T)> GetReferenceOp(ynn_reduce_operator op) {
  switch (op) {
    case ynn_reduce_sum:
      return [](T a, T b) { return a + b; };
    case ynn_reduce_sum_squared:
      return [](T a, T b) { return a + b * b; };
    case ynn_reduce_min:
      return [](T a, T b) { return std::min(a, b); };
    case ynn_reduce_max:
      return [](T a, T b) { return std::max(a, b); };
    default:
      YNN_UNREACHABLE;
  }
}

template <typename T>
float Tolerance(ynn_reduce_operator op, size_t k, float max_abs_value) {
  switch (op) {
    case ynn_reduce_sum:
      return epsilon(type_of<T>()) * k * max_abs_value * 3.0f;
    case ynn_reduce_sum_squared:
      return epsilon(type_of<T>()) * k * max_abs_value * max_abs_value * 6.0f;
    default:
      return 0.0f;
  }
}

template <typename A, typename C>
void TestReduce(A, C, ynn_reduce_operator op) {
  ReplicableRandomDevice rng;
  std::uniform_int_distribution<size_t> rank_dist(1, YNN_MAX_TENSOR_RANK);
  std::bernoulli_distribution random_bool(0.5);

  const float max_abs_value = 10.0f;
  TypeGenerator<A> a_gen(-max_abs_value, max_abs_value, quantization_params{});
  TypeGenerator<C> c_gen(-max_abs_value, max_abs_value, quantization_params{});

  for (auto _ : FuzzTest(std::chrono::milliseconds(500))) {
    const bool keep_dims = random_bool(rng);

    const size_t input_rank = rank_dist(rng);
    const size_t num_k_dims =
        std::uniform_int_distribution<size_t>(1, input_rank)(rng);
    const size_t output_rank = input_rank - (keep_dims ? 0 : num_k_dims);

    // Select random axes to reduce.
    std::vector<int32_t> reduce_axes(input_rank);
    std::iota(reduce_axes.begin(), reduce_axes.end(), 0);
    std::shuffle(reduce_axes.begin(), reduce_axes.end(), rng);
    reduce_axes.resize(num_k_dims);

    // Build the subgraph.
    SubgraphBuilder subgraph(3);
    const uint32_t a_id = 0;
    uint32_t c_id = 1;
    const uint32_t output_id = 2;
    subgraph.AddInput(type_of<A>(), input_rank, a_id)
        .AddOutput(type_of<C>(), output_rank, output_id);

    const bool init_c = random_bool(rng);
    const C init_value = c_gen(rng);
    if (init_c) {
      subgraph.AddScalar<C>(init_value, c_id);
    } else {
      subgraph.AddInput(type_of<C>(), output_rank, c_id);
    }

    subgraph.AddReduce(op, reduce_axes, a_id, c_id, output_id,
                       keep_dims ? YNN_NODE_FLAG_KEEP_DIMS : 0);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> a_shape = random_shape(rng, input_rank);
      std::vector<size_t> c_shape = a_shape;
      size_t num_k_elements = 1;
      for (int32_t i : reduce_axes) {
        num_k_elements *= a_shape[i];
        c_shape[i] = 1;
      }

      Tensor<A> a(a_shape);
      a.generate([&]() { return a_gen(rng); });

      runtime.ReshapeExternalTensor(a_shape, a.data(), a_id);

      Tensor<C> expected(c_shape);
      if (init_c) {
        expected.fill(init_value);
      } else {
        expected.generate([&]() { return c_gen(rng); });
      }

      std::vector<size_t> expected_shape = c_shape;
      if (!keep_dims) {
        std::sort(reduce_axes.begin(), reduce_axes.end(),
                  std::greater<int32_t>());
        for (int32_t i : reduce_axes) {
          expected_shape.erase(expected_shape.begin() + i);
        }
      }

      Tensor<C> c = expected.deep_copy();
      if (!init_c) {
        std::vector<size_t> b_shape = c_shape;
        runtime.ReshapeExternalTensor(expected_shape, c.data(), c_id);
      }
      runtime.ReshapeRuntime();
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      ASSERT_EQ(runtime.GetExternalTensorShape(output_id), expected_shape);

      runtime.SetupExternalTensor(c.data(), output_id).InvokeRuntime();
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      // Compute the reference result.
      auto op_impl = GetReferenceOp<C>(op);
      broadcast_extent_1(expected);
      for (const auto& i : EnumerateIndices(a_shape)) {
        expected(i) = op_impl(expected(i), a(i));
      }

      // Verify results.
      for (const auto& i : EnumerateIndices(c_shape)) {
        if (std::is_integral<C>::value) {
          ASSERT_EQ(c(i), expected(i));
        } else {
          const float tolerance =
              Tolerance<C>(op, num_k_elements + 1, max_abs_value);
          ASSERT_NEAR(c(i), expected(i), tolerance);
        }
      }
    }
  }
}

class Reduce : public testing::TestWithParam<
                   std::tuple<ynn_reduce_operator, multi_type>> {};

TEST_P(Reduce, Test) {
  SwitchTwoTypes(std::get<1>(GetParam()), [&](auto a_type, auto c_type) {
    TestReduce(a_type, c_type, std::get<0>(GetParam()));
  });
}

INSTANTIATE_TEST_SUITE_P(
    Sum, Reduce,
    testing::Combine(testing::Values(ynn_reduce_sum),
                     testing::Values(multi_type::fp32, multi_type::fp16_fp32,
                                     multi_type::bf16_fp32,
                                     multi_type::int8_int32,
                                     multi_type::uint8_int32)),
    [](const testing::TestParamInfo<Reduce::ParamType>& info) {
      return test_param_to_string(info);
    });

INSTANTIATE_TEST_SUITE_P(
    MinMax, Reduce,
    testing::Combine(testing::Values(ynn_reduce_min, ynn_reduce_max),
                     testing::Values(multi_type::fp32, multi_type::fp16,
                                     multi_type::bf16, multi_type::int8,
                                     multi_type::uint8)),
    [](const testing::TestParamInfo<Reduce::ParamType>& info) {
      return test_param_to_string(info);
    });

template <typename T>
void TestMinMax(T) {
  ReplicableRandomDevice rng;
  std::uniform_int_distribution<size_t> rank_dist(1, YNN_MAX_TENSOR_RANK - 1);
  std::bernoulli_distribution random_bool(0.5);

  const float max_abs_value = 10.0f;
  TypeGenerator<T> t_gen(-max_abs_value, max_abs_value, quantization_params{});

  for (auto _ : FuzzTest(std::chrono::milliseconds(500))) {
    const bool keep_dims = random_bool(rng);

    const size_t input_rank = rank_dist(rng);
    const size_t num_k_dims =
        std::uniform_int_distribution<size_t>(1, input_rank)(rng);
    const size_t output_rank = input_rank - (keep_dims ? 0 : num_k_dims) + 1;

    // Select random axes to reduce.
    std::vector<int32_t> reduce_axes(input_rank);
    std::iota(reduce_axes.begin(), reduce_axes.end(), 0);
    std::shuffle(reduce_axes.begin(), reduce_axes.end(), rng);
    reduce_axes.resize(num_k_dims);

    // Build the subgraph.
    SubgraphBuilder subgraph(3);
    const uint32_t a_id = 0;
    uint32_t c_id = 1;
    const uint32_t output_id = 2;
    subgraph.AddInput(type_of<T>(), input_rank, a_id)
        .AddOutput(type_of<T>(), output_rank, output_id);

    const bool init_c = random_bool(rng);
    const T init_value = t_gen(rng);
    if (init_c) {
      subgraph.AddScalar<T>(init_value, c_id);
    } else {
      subgraph.AddInput(type_of<T>(), output_rank, c_id);
    }

    subgraph.AddReduce(ynn_reduce_min_max, reduce_axes, a_id, c_id, output_id,
                       keep_dims ? YNN_NODE_FLAG_KEEP_DIMS : 0);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> a_shape = random_shape(rng, input_rank);
      std::vector<size_t> c_shape = a_shape;
      size_t num_k_elements = 1;
      for (int32_t i : reduce_axes) {
        num_k_elements *= a_shape[i];
        c_shape[i] = 1;
      }
      c_shape.insert(c_shape.begin(), 2);

      Tensor<T> a(a_shape);
      a.generate([&]() { return t_gen(rng); });

      runtime.ReshapeExternalTensor(a_shape, a.data(), a_id);

      Tensor<T> expected(c_shape);
      if (init_c) {
        expected.fill(init_value);
      } else {
        expected.generate([&]() { return t_gen(rng); });
      }

      std::vector<size_t> expected_shape = c_shape;
      if (!keep_dims) {
        std::sort(reduce_axes.begin(), reduce_axes.end(),
                  std::greater<int32_t>());
        for (int32_t i : reduce_axes) {
          expected_shape.erase(expected_shape.begin() + i + 1);
        }
      }

      Tensor<T> c = expected.deep_copy();
      if (!init_c) {
        std::vector<size_t> b_shape = c_shape;
        runtime.ReshapeExternalTensor(expected_shape, c.data(), c_id);
      }
      runtime.ReshapeRuntime();
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      ASSERT_EQ(runtime.GetExternalTensorShape(output_id), expected_shape);

      runtime.SetupExternalTensor(c.data(), output_id).InvokeRuntime();
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      // Compute the reference result.
      Tensor<T> expected_min = expected.slice(0, 0).remove_dim(0);
      Tensor<T> expected_max = expected.slice(0, 1).remove_dim(0);
      broadcast_extent_1(expected_min);
      broadcast_extent_1(expected_max);
      for (const auto& i : EnumerateIndices(a_shape)) {
        expected_min(i) = std::min(expected_min(i), a(i));
        expected_max(i) = std::max(expected_max(i), a(i));
      }

      // Verify results.
      for (const auto& i : EnumerateIndices(c_shape)) {
        ASSERT_EQ(c(i), expected(i));
      }
    }
  }
}

template <typename F>
constexpr decltype(auto) SwitchType(ynn_type type, F&& f) {
  switch (type) {
    case ynn_type_int8:
      return std::forward<F>(f)(int8_t());
    case ynn_type_uint8:
      return std::forward<F>(f)(uint8_t());
    case ynn_type_int32:
      return std::forward<F>(f)(int32_t());
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

class MinMax : public testing::TestWithParam<ynn_type> {};

TEST_P(MinMax, Test) {
  SwitchType(GetParam(), [&](auto type) { TestMinMax(type); });
}

INSTANTIATE_TEST_SUITE_P(
    MinMax, MinMax,
    testing::Values(ynn_type_fp32, ynn_type_fp16, ynn_type_bf16, ynn_type_int8,
                    ynn_type_uint8),
    [](const testing::TestParamInfo<MinMax::ParamType>& info) {
      return to_string(info.param);
    });

}  // namespace ynn