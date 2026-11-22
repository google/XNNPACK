// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

template <typename T>
void TestImpl(size_t rank) {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution broadcast_dist(0.25);
  std::uniform_int_distribution<size_t> dim_dist(1, 9);

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (auto _ : FuzzTest(std::chrono::milliseconds(250))) {
    xnn_quantization_params quantization =
        random_quantization(xnn_datatype_of<T>(), rng);

    // The broadcast shape is a random shape, with 0s randomly added which pass
    // through the input shape.
    std::vector<size_t> broadcast_shape = random_shape(rng, rank);
    for (size_t& dim : broadcast_shape) {
      if (broadcast_dist(rng)) {
        dim = 0;
      }
    }

    // static_broadcast supports adding new dimensions, but only if the static
    // broadcast shape is not trying to pass the input shape through (the static
    // broadcast shape is not 0 in that dimension).
    size_t input_rank = rank;
    while (input_rank >= 1 && broadcast_shape[rank - input_rank] != 0 &&
           broadcast_dist(rng)) {
      input_rank--;
    }

    // Define subgraph
    SubgraphTester subgraph(2);
    subgraph.AddInputTensor(input_rank, xnn_datatype_of<T>(), quantization, 0)
        .AddOutputTensor(rank, xnn_datatype_of<T>(), quantization, 1)
        .AddBroadcast(broadcast_shape, 0, 1);
    ASSERT_EQ(subgraph.CreateRuntime(), xnn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> input_shape = broadcast_shape;
      std::vector<size_t> output_shape = broadcast_shape;
      for (size_t i = 0; i < rank; ++i) {
        if (input_shape[i] == 0) {
          input_shape[i] = dim_dist(rng);
          output_shape[i] = input_shape[i];
        } else {
          input_shape[i] = 1;
        }
      }
      input_shape.erase(input_shape.begin(),
                        input_shape.begin() + rank - input_rank);

      Tensor<T> input(input_shape, xnnpack::XnnExtraBytes);
      DatatypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      // Check reshaped shape is correct
      subgraph.ReshapeExternalTensor(input_shape, input.base(), 0)
          .ReshapeRuntime();
      ASSERT_EQ(subgraph.Status(), xnn_status_success);
      ASSERT_EQ(subgraph.GetExternalTensorShape(1), output_shape);

      // Run subgraph
      Tensor<T> output(output_shape);
      subgraph.SetupExternalTensor(output.base(), 1)
          .SetupRuntime()
          .InvokeRuntime();

      // Add the new dimensions back to the input and make it broadcastable.
      std::vector<size_t> new_axes(rank - input_rank);
      std::iota(new_axes.begin(), new_axes.end(), 0);
      input = input.expand_dims(new_axes);
      broadcast_extent_1(input);

      Tensor<T> expected(output_shape);
      expected.assign(input);

      // Verify results.
      ASSERT_THAT(output, testing::ElementsAreArray(expected));
    }
  }
}

template <typename T>
class Broadcast : public ::testing::TestWithParam<int> {};

using BroadcastQS8 = Broadcast<quantized<int8_t>>;
using BroadcastQU8 = Broadcast<quantized<uint8_t>>;
using BroadcastBF16 = Broadcast<xnn_bfloat16>;
using BroadcastF16 = Broadcast<xnn_float16>;
using BroadcastF32 = Broadcast<float>;

TEST_P(BroadcastQS8, test) { TestImpl<quantized<int8_t>>(GetParam()); }
TEST_P(BroadcastQU8, test) { TestImpl<quantized<uint8_t>>(GetParam()); }
TEST_P(BroadcastBF16, test) { TestImpl<xnn_bfloat16>(GetParam()); }
TEST_P(BroadcastF16, test) { TestImpl<xnn_float16>(GetParam()); }
TEST_P(BroadcastF32, test) { TestImpl<float>(GetParam()); }

auto rank_params = testing::Range(1, XNN_MAX_TENSOR_DIMS);
INSTANTIATE_TEST_SUITE_P(Broadcast, BroadcastQS8, rank_params);
INSTANTIATE_TEST_SUITE_P(Broadcast, BroadcastQU8, rank_params);
INSTANTIATE_TEST_SUITE_P(Broadcast, BroadcastBF16, rank_params);
INSTANTIATE_TEST_SUITE_P(Broadcast, BroadcastF16, rank_params);
INSTANTIATE_TEST_SUITE_P(Broadcast, BroadcastF32, rank_params);

}  // namespace xnnpack

#include "src/xnnpack/subgraph-validation.h"

TEST(SubgraphValidationTest, NullInputPointers) {
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_subgraph_check_input_type_dense(
                xnn_node_type_unary_elementwise, 0, nullptr));
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_subgraph_check_nth_input_type_dense(
                xnn_node_type_unary_elementwise, 0, nullptr, 1));
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_subgraph_check_output_type_dense(
                xnn_node_type_unary_elementwise, 0, nullptr));
}

TEST(SubgraphValidationTest, NullDatatypeMatches) {
  struct xnn_value val;
  memset(&val, 0, sizeof(val));
  val.type = xnn_value_type_dense_tensor;
  val.datatype = xnn_datatype_fp32;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_subgraph_check_datatype_matches(
                xnn_node_type_unary_elementwise, 0, nullptr, 1, &val));
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_subgraph_check_datatype_matches(
                xnn_node_type_unary_elementwise, 0, &val, 1, nullptr));
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_subgraph_check_datatype_matches_two_inputs(
                xnn_node_type_binary_elementwise, 0, nullptr, 1, &val, 2,
                &val));
}

TEST(SubgraphValidationTest, NullQuantizationMatches) {
  struct xnn_value val;
  memset(&val, 0, sizeof(val));
  val.type = xnn_value_type_dense_tensor;
  val.datatype = xnn_datatype_qint8;
  val.quantization.scale = 1.0f;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_subgraph_check_quantization_parameter_matches(
                xnn_node_type_unary_elementwise, 0, nullptr, 1, &val));
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_subgraph_check_quantization_parameter_matches(
                xnn_node_type_unary_elementwise, 0, &val, 1, nullptr));
}

TEST(SubgraphValidationTest, InvalidQuantizationScale) {
  struct xnn_value in;
  memset(&in, 0, sizeof(in));
  in.type = xnn_value_type_dense_tensor;
  in.datatype = xnn_datatype_qint8;
  in.quantization.scale = 0.0f;

  struct xnn_value out;
  memset(&out, 0, sizeof(out));
  out.type = xnn_value_type_dense_tensor;
  out.datatype = xnn_datatype_qint8;
  out.quantization.scale = 1.0f;

  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_subgraph_check_quantization_parameter_matches(
                xnn_node_type_unary_elementwise, 0, &in, 1, &out));
}

TEST(SubgraphValidationTest, BatchDimsMatchNullOrExcess) {
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_subgraph_check_batch_dims_match(
                xnn_node_type_batch_matrix_multiply, 0, nullptr, 1, nullptr,
                1));

  struct xnn_value val;
  memset(&val, 0, sizeof(val));
  val.type = xnn_value_type_dense_tensor;
  val.datatype = xnn_datatype_fp32;
  val.shape.num_dims = 2;
  EXPECT_EQ(xnn_status_invalid_parameter,
            xnn_subgraph_check_batch_dims_match(
                xnn_node_type_batch_matrix_multiply, 0, &val, 1, &val,
                XNN_MAX_TENSOR_DIMS + 1));
}
