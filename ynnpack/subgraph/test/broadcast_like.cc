// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

template <typename Rng>
std::pair<std::vector<size_t>, std::vector<size_t>> random_broadcasted_inputs(
    Rng& rng, std::vector<size_t> output_shape, size_t a_rank, size_t b_rank) {
  // The logic here is simpler if the innermost dimension is first.
  std::reverse(output_shape.begin(), output_shape.end());
  std::vector<size_t> a_shape = output_shape;
  std::vector<size_t> b_shape = output_shape;
  std::bernoulli_distribution broadcast_dist(0.25);
  for (size_t i = 0; i < output_shape.size(); i++) {
    // We only want to broadcast one of the two inputs in this dimension,
    // including broadcasting due to smaller rank.
    if (broadcast_dist(rng) && i < b_rank) {
      a_shape[i] = 1;
    } else if (broadcast_dist(rng) && i < a_rank) {
      b_shape[i] = 1;
    }
  }
  a_shape.resize(a_rank);
  b_shape.resize(b_rank);
  std::reverse(a_shape.begin(), a_shape.end());
  std::reverse(b_shape.begin(), b_shape.end());
  return {a_shape, b_shape};
}

template <typename T>
void TestImpl(T, size_t rank) {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution broadcast_dist(0.25);
  std::uniform_int_distribution<size_t> dim_dist(1, 9);

  for (size_t input_rank = 0; input_rank <= rank; input_rank++) {
    for (std::pair<size_t, size_t> input_ranks :
         {std::make_pair(input_rank, rank), std::make_pair(rank, input_rank)}) {
      // Broadcast all of the dimensions of the template, including those the
      // input does not have.
      std::vector<int32_t> axes(rank);
      std::iota(axes.begin(), axes.end(), -static_cast<int32_t>(rank));

      // Define subgraph
      SubgraphBuilder subgraph(3);
      subgraph.AddInput(type_of<T>(), input_ranks.first, 0)
          .AddInput(type_of<half>(), input_ranks.second, 1)
          .AddOutput(type_of<T>(), rank, 2)
          .AddBroadcastLike(axes, 0, 1, 2);

      Runtime runtime(subgraph.GetSubgraph());
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      for (int reshape = 0; reshape < 2; ++reshape) {
        std::vector<size_t> output_shape = random_shape(rng, rank);
        std::vector<size_t> input_shape, template_shape;
        std::tie(input_shape, template_shape) = random_broadcasted_inputs(
            rng, output_shape, input_ranks.first, input_ranks.second);

        Tensor<T> input(input_shape);
        fill_random(input.data(), input.size(), rng);

        // Check reshaped shape is correct
        runtime.ReshapeExternalTensor(input_shape, input.base(), 0)
            .ReshapeExternalTensor(template_shape, nullptr, 1)
            .ReshapeRuntime();
        ASSERT_EQ(runtime.GetExternalTensorShape(2), output_shape);

        // Run subgraph
        Tensor<T> output(output_shape);
        runtime.SetupExternalTensor(output.base(), 2).InvokeRuntime();

        // Implement the broadcast so we can check the result.
        std::vector<int32_t> new_dims(output_shape.size() - input.rank());
        std::iota(new_dims.begin(), new_dims.end(), 0);
        input = input.expand_dims(new_dims);
        broadcast_extent_1(input);
        Tensor<T> expected(output_shape);
        expected.assign(input);

        // Verify results.
        ASSERT_THAT(output, testing::ElementsAreArray(expected));
      }
    }
  }
}

class BroadcastLike
    : public ::testing::TestWithParam<std::tuple<ynn_type, int>> {};

TEST_P(BroadcastLike, test) {
  ynn_type type = std::get<0>(GetParam());
  int rank = std::get<1>(GetParam());
  SwitchRealType(type, [&](auto type) { TestImpl(type, rank); });
}

INSTANTIATE_TEST_SUITE_P(
    BroadcastLike, BroadcastLike,
    testing::Combine(testing::Values(ynn_type_int8, ynn_type_uint8,
                                     ynn_type_fp16, ynn_type_bf16,
                                     ynn_type_fp32),
                     testing::Range(1, YNN_MAX_TENSOR_RANK)),
    test_param_to_string<BroadcastLike::ParamType>);

TEST(BroadcastLike, static_input_dynamic_template_dot) {
  const uint32_t a_id = 0;
  const uint32_t out_id = 1;
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  uint32_t a_broadcast_id = YNN_INVALID_VALUE_ID;
  uint32_t b_broadcast_id = YNN_INVALID_VALUE_ID;
  std::vector<float> b_data(2 * 4 * 8, 1.0f);

  SubgraphBuilder subgraph(2);
  subgraph.AddInput(ynn_type_fp32, 3, a_id)
      .AddTensor(ynn_type_fp32, {2, 4, 8}, b_id, b_data.data())
      .AddTensor(ynn_type_fp32, 3, a_broadcast_id)
      .AddTensor(ynn_type_fp32, 3, b_broadcast_id)
      .AddOutput(ynn_type_fp32, 3, out_id)
      .AddBroadcastLike({0}, a_id, b_id, a_broadcast_id)
      .AddBroadcastLike({0}, b_id, a_id, b_broadcast_id)
      .AddDot(1, a_broadcast_id, b_broadcast_id, YNN_INVALID_VALUE_ID, out_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  std::vector<float> a_data(1 * 6 * 4, 2.0f);
  std::vector<float> out_data(2 * 6 * 8, 0.0f);
  runtime.ReshapeExternalTensor({1, 6, 4}, a_data.data(), a_id)
      .ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  ASSERT_EQ(runtime.GetExternalTensorShape(out_id),
            (std::vector<size_t>{2, 6, 8}));
  runtime.SetupExternalTensor(out_data.data(), out_id).InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeExternalTensor({3, 6, 4}, a_data.data(), a_id);
  ASSERT_EQ(runtime.ReshapeRuntime().Status(), ynn_status_invalid_parameter);
}

// Dimensions of the template that are not in the requested axes are not
// broadcasted, even if the template has more dimensions than the input.
TEST(BroadcastLike, noop_higher_rank_template) {
  const uint32_t template_id = 0;
  uint32_t input_id = YNN_INVALID_VALUE_ID;
  std::vector<float> input_data(2 * 4 * 8, 1.0f);

  SubgraphBuilder subgraph(1);
  subgraph.AddInput(ynn_type_fp32, {0, 2, 1, 4}, template_id)
      .AddTensor(ynn_type_fp32, {2, 4, 8}, input_id, input_data.data());

  const int32_t axes[] = {0};
  uint32_t output_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_broadcast_like(subgraph.GetSubgraph(), 1, axes, input_id,
                                      template_id, &output_id, /*flags=*/0),
            ynn_status_success);
  EXPECT_EQ(output_id, input_id);
}

// If the output is already defined, it has the rank of the template, but only
// the requested axes are broadcasted.
TEST(BroadcastLike, defined_output_has_template_rank) {
  const uint32_t input_id = 0;
  const uint32_t template_id = 1;
  const uint32_t output_id = 2;

  SubgraphBuilder subgraph(3);
  subgraph.AddInput(ynn_type_fp32, 1, input_id)
      .AddInput(ynn_type_fp32, 3, template_id)
      .AddOutput(ynn_type_fp32, 3, output_id)
      .AddBroadcastLike({-1}, input_id, template_id, output_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  runtime.ReshapeExternalTensor({4}, input.data(), input_id)
      .ReshapeExternalTensor({2, 3, 4}, nullptr, template_id)
      .ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  ASSERT_EQ(runtime.GetExternalTensorShape(output_id),
            (std::vector<size_t>{1, 1, 4}));

  std::vector<float> output(4);
  runtime.SetupExternalTensor(output.data(), output_id).InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  EXPECT_THAT(output, testing::ElementsAreArray(input));
}

TEST(BroadcastLike, symbolic_extent_inequality) {
  const uint32_t x_id = 0;
  const uint32_t out_id = 1;
  uint32_t stenciled_id = YNN_INVALID_VALUE_ID;
  uint32_t pooled_id = YNN_INVALID_VALUE_ID;

  SubgraphBuilder subgraph(2);
  const uint32_t zero_id = subgraph.DefineScalar<float>(0.0f);
  subgraph.AddInput(ynn_type_fp32, 1, x_id)
      .AddOutput(ynn_type_fp32, 1, out_id)
      .AddTensor(ynn_type_fp32, 2, stenciled_id)
      .AddTensor(ynn_type_fp32, 1, pooled_id)
      .AddStencilCopy({0}, {1}, {2}, {1}, {1}, x_id, YNN_INVALID_VALUE_ID,
                      stenciled_id)
      .AddReduce(ynn_reduce_sum, {1}, stenciled_id, zero_id, pooled_id);

  const int32_t axes[] = {0};
  uint32_t x_broadcast_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(ynn_define_broadcast_like(subgraph.GetSubgraph(), 1, axes, x_id,
                                      pooled_id, &x_broadcast_id, /*flags=*/0),
            ynn_status_success);
  EXPECT_EQ(x_broadcast_id, x_id);

  subgraph.AddBroadcastLike({0}, pooled_id, x_id, out_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  std::vector<float> x = {1.0f, 3.0f};
  runtime.ReshapeExternalTensor({2}, x.data(), x_id).ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  ASSERT_EQ(runtime.GetExternalTensorShape(out_id), (std::vector<size_t>{2}));

  std::vector<float> out(2);
  runtime.SetupExternalTensor(out.data(), out_id).InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  EXPECT_THAT(out, testing::ElementsAre(4.0f, 4.0f));

  std::vector<float> x_invalid(3, 1.0f);
  runtime.ReshapeExternalTensor({3}, x_invalid.data(), x_id);
  EXPECT_EQ(runtime.ReshapeRuntime().Status(), ynn_status_invalid_parameter);
}

TEST(BroadcastLike, bad_static_shape) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t x_id = 2;
  SubgraphBuilder subgraph(3);
  subgraph.AddInput(ynn_type_fp32, {10, 20}, a_id)
      .AddInput(ynn_type_fp32, {5, 20}, b_id)
      .AddOutput(ynn_type_fp32, 2, x_id)
      .AddBroadcastLike({0}, a_id, b_id, x_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeExternalTensor({10, 20}, nullptr, a_id)
      .ReshapeExternalTensor({5, 20}, nullptr, b_id);
  EXPECT_EQ(runtime.ReshapeRuntime().Status(), ynn_status_invalid_parameter);
}

}  // namespace ynn
