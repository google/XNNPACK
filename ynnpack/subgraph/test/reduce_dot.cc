// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

// This test generates pipelines of the form sum(a*b) with random broadcasting
// of a and b. This is essentially a fuzz tester for rewriting sum(a*b) to dot.

template <typename AB, typename C>
void ReferenceImpl(const std::vector<size_t>& shape, Tensor<AB> a, Tensor<AB> b,
                   Tensor<C>& c) {
  if (!std::is_same<C, float>::value && !std::is_same<C, double>::value &&
      !std::is_same<C, int32_t>::value) {
    Tensor<float> c_float(c.extents());
    c_float.assign(c);
    ReferenceImpl(shape, a, b, c_float);
    c.assign(c_float);
  } else {
    broadcast_extent_1(a);
    broadcast_extent_1(b);
    broadcast_extent_1(c);

    for (const auto& i : EnumerateIndices(shape)) {
      c(i) = c(i) + static_cast<C>(a(i)) * static_cast<C>(b(i));
    }
  }
}

template <typename T>
float Tolerance(size_t k, float max_abs_value) {
  return type_info<T>::epsilon() * (k + 1) * max_abs_value * max_abs_value *
         4.0f;
}

template <typename AB, typename C>
void TestReduceDot(AB, C) {
  ReplicableRandomDevice rng;
  std::uniform_int_distribution<size_t> rank_dist(1, 4);
  std::bernoulli_distribution random_bool(0.5);

  const float max_abs_value = 1.0f;

  for (auto _ : FuzzTest(std::chrono::milliseconds(1000))) {
    const size_t rank = rank_dist(rng);
    const std::vector<size_t> common_shape = random_shape(rng, rank);

    std::vector<size_t> a_shape(rank);
    std::vector<size_t> b_shape(rank);
    for (size_t i = 0; i < rank; ++i) {
      const int choice = std::uniform_int_distribution<int>(0, 2)(rng);
      if (choice == 0) {
        a_shape[i] = common_shape[i];
        b_shape[i] = common_shape[i];
      } else if (choice == 1) {
        a_shape[i] = 1;
        b_shape[i] = common_shape[i];
      } else {
        a_shape[i] = common_shape[i];
        b_shape[i] = 1;
      }
    }

    const bool keep_dims = random_bool(rng);
    const size_t num_reduce_axes =
        std::uniform_int_distribution<size_t>(1, rank)(rng);
    std::vector<int32_t> reduce_axes(rank);
    std::iota(reduce_axes.begin(), reduce_axes.end(), 0);
    std::shuffle(reduce_axes.begin(), reduce_axes.end(), rng);
    reduce_axes.resize(num_reduce_axes);

    std::vector<size_t> c_shape = common_shape;
    size_t num_reduce_elements = 1;
    for (int32_t axis : reduce_axes) {
      num_reduce_elements *= common_shape[axis];
      c_shape[axis] = 1;
    }

    std::vector<size_t> output_shape = c_shape;
    if (!keep_dims) {
      std::vector<int32_t> sorted_axes = reduce_axes;
      std::sort(sorted_axes.begin(), sorted_axes.end(),
                std::greater<int32_t>());
      for (int32_t axis : sorted_axes) {
        output_shape.erase(output_shape.begin() + axis);
      }
    }

    SubgraphBuilder subgraph(4);
    uint32_t a_id = 0;
    uint32_t b_id = 1;
    uint32_t mul_id = 2;
    uint32_t output_id = 3;

    subgraph.AddInput(type_of<AB>(), a_shape, a_id)
        .AddInput(type_of<AB>(), b_shape, b_id)
        .AddTensor(type_of<C>(), common_shape, mul_id)
        .AddOutput(type_of<C>(), output_shape, output_id);

    if (!std::is_same<AB, C>::value) {
      uint32_t a_converted_id = YNN_INVALID_VALUE_ID;
      uint32_t b_converted_id = YNN_INVALID_VALUE_ID;
      subgraph.AddTensor(type_of<C>(), a_shape, a_converted_id)
          .AddTensor(type_of<C>(), b_shape, b_converted_id);

      subgraph.AddConvert(a_id, type_of<C>(), a_converted_id)
          .AddConvert(b_id, type_of<C>(), b_converted_id)
          .AddBinary(ynn_binary_multiply, a_converted_id, b_converted_id,
                     mul_id);
    } else {
      subgraph.AddBinary(ynn_binary_multiply, a_id, b_id, mul_id);
    }
    subgraph.AddReduce(ynn_reduce_sum, reduce_axes, mul_id,
                       YNN_INVALID_VALUE_ID, output_id,
                       keep_dims ? YNN_NODE_FLAG_KEEP_DIMS : 0);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    Tensor<AB> a(a_shape);
    Tensor<AB> b(b_shape);
    fill_random(a.data(), a.size(), rng, -max_abs_value, max_abs_value);
    fill_random(b.data(), b.size(), rng, -max_abs_value, max_abs_value);

    runtime.ReshapeExternalTensor(a_shape, a.data(), a_id);
    runtime.ReshapeExternalTensor(b_shape, b.data(), b_id);

    Tensor<C> c(output_shape);
    runtime.SetupExternalTensor(c.data(), output_id);

    runtime.ReshapeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    ASSERT_EQ(runtime.GetExternalTensorShape(output_id), output_shape);

    runtime.InvokeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    Tensor<C> expected(c_shape);
    expected.fill(0);
    ReferenceImpl(common_shape, a, b, expected);

    for (const auto& i : EnumerateIndices(output_shape)) {
      std::vector<size_t> expected_i = i;
      if (!keep_dims) {
        expected_i = std::vector<size_t>(rank, 0);
        std::vector<int32_t> sorted_axes = reduce_axes;
        std::sort(sorted_axes.begin(), sorted_axes.end());
        size_t dst_axis = 0;
        for (size_t r = 0; r < rank; ++r) {
          if (std::find(sorted_axes.begin(), sorted_axes.end(), r) !=
              sorted_axes.end()) {
            expected_i[r] = 0;
          } else {
            expected_i[r] = i[dst_axis++];
          }
        }
      }
      if (is_integral<C>::value) {
        ASSERT_EQ(c(i), expected(expected_i));
      } else {
        const float tolerance =
            Tolerance<C>(num_reduce_elements, max_abs_value);
        ASSERT_NEAR(static_cast<float>(c(i)),
                    static_cast<float>(expected(expected_i)), tolerance);
      }
    }
  }
}

class ReduceDot : public testing::TestWithParam<multi_type> {};

TEST_P(ReduceDot, Test) {
  SwitchTwoTypes(GetParam(), [&](auto ab_type, auto c_type) {
    TestReduceDot(ab_type, c_type);
  });
}

multi_type reduce_dot_types[] = {
#if defined(YNN_ARCH_X86) || defined(YNN_ARCH_ARM64)
    // TODO(b/501068911): Replace this with YNN_ENABLE_FP64
    multi_type::fp64,
#endif
    multi_type::fp32,      multi_type::fp16,      multi_type::bf16,
    multi_type::fp16_fp32, multi_type::bf16_fp32, multi_type::int8_int32,
};

INSTANTIATE_TEST_SUITE_P(
    ReduceDot, ReduceDot, testing::ValuesIn(reduce_dot_types),
    [](const testing::TestParamInfo<ReduceDot::ParamType>& info) {
      return to_string(info.param);
    });

}  // namespace ynn
