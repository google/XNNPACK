// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>

#include <xnnpack.h>
#include <xnnpack/subgraph.h>

#include "subgraph-tester.h"
#include <gtest/gtest.h>


namespace xnnpack {

constexpr size_t XNN_UNKNOWN_DIM = 0;

TEST(DISABLED_SUBGRAPH_SHAPE_INFERENCE, tensor_with_dynamic_dims_set_correctly) {
  auto tester = SubgraphTester(1);
  tester
    .AddInputTensorF32({XNN_UNKNOWN_DIM, 1}, 0);
  const xnn_value* input = tester.Value(0);
  EXPECT_FALSE(xnn_tensor_shape_is_static(input));
}

TEST(DISABLED_SUBGRAPH_SHAPE_INFERENCE, quantized_tensor_with_dynamic_dims_set_correctly) {
  auto tester = SubgraphTester(1);
  tester
    .AddInputTensorQS8(/*zero_point=*/3, /*scale=*/1.1f, {XNN_UNKNOWN_DIM, 1}, 0);
  const xnn_value* input = tester.Value(0);
  EXPECT_FALSE(xnn_tensor_shape_is_static(input));
}

TEST(DISABLED_SUBGRAPH_SHAPE_INFERENCE, fully_connected_dynamic_input) {
  auto tester = SubgraphTester(6);
  const size_t input_channels = 3;
  const size_t output_channels = 5;
  const uint32_t input_id = 0;
  const uint32_t filter_id = 1;
  const uint32_t output_id = 2;
  tester
    .AddInputTensorF32({XNN_UNKNOWN_DIM, XNN_UNKNOWN_DIM}, input_id)
    .AddStaticTensorF32({output_channels, input_channels}, TensorType::kDense, filter_id)
    .AddOutputTensorF32({XNN_UNKNOWN_DIM, XNN_UNKNOWN_DIM}, output_id)
    .AddFullyConnected(input_id, filter_id, XNN_INVALID_VALUE_ID, output_id)
    .InferShape()
  ;

  const xnn_value* input = tester.Value(input_id);
  // Batch size cannot be inferred.
  ASSERT_EQ(input->shape.dim[0], XNN_UNKNOWN_DIM);
  EXPECT_EQ(input->shape.dim[1], input_channels);

  const xnn_value* output = tester.Value(output_id);
  // Batch size cannot be inferred.
  ASSERT_EQ(output->shape.dim[0], XNN_UNKNOWN_DIM);
  EXPECT_EQ(output->shape.dim[1], output_channels);

  // Values are still dynamic after inference.
  EXPECT_FALSE(xnn_tensor_shape_is_static(input));
  EXPECT_FALSE(xnn_tensor_shape_is_static(output));
}

TEST(DISABLED_SUBGRAPH_SHAPE_INFERENCE, fully_connected_infer_output_from_input) {
  auto tester = SubgraphTester(6);
  const size_t input_batch_dim = 7;
  const size_t input_channels = 3;
  const size_t output_channels = 5;
  const uint32_t input_id = 0;
  const uint32_t filter_id = 1;
  const uint32_t output_id = 2;
  tester
    .AddInputTensorF32({input_batch_dim, XNN_UNKNOWN_DIM}, input_id)
    .AddStaticTensorF32({output_channels, input_channels}, TensorType::kDense, filter_id)
    .AddOutputTensorF32({XNN_UNKNOWN_DIM, XNN_UNKNOWN_DIM}, output_id)
    .AddFullyConnected(input_id, filter_id, XNN_INVALID_VALUE_ID, output_id)
    .InferShape()
  ;

  const xnn_value* input = tester.Value(input_id);
  EXPECT_EQ(input->shape.dim[1], input_channels);

  const xnn_value* output = tester.Value(output_id);
  // Output batch siz is inferred.
  ASSERT_EQ(output->shape.dim[0], input_batch_dim);
  EXPECT_EQ(output->shape.dim[1], output_channels);
}

TEST(DISABLED_SUBGRAPH_SHAPE_INFERENCE, fully_connected_infer_input_from_output) {
  auto tester = SubgraphTester(6);
  const size_t output_batch_dim = 7;
  const size_t input_channels = 3;
  const size_t output_channels = 5;
  const uint32_t input_id = 0;
  const uint32_t filter_id = 1;
  const uint32_t output_id = 2;
  tester
    .AddInputTensorF32({XNN_UNKNOWN_DIM, XNN_UNKNOWN_DIM}, input_id)
    .AddStaticTensorF32({output_channels, input_channels}, TensorType::kDense, filter_id)
    .AddOutputTensorF32({output_batch_dim, XNN_UNKNOWN_DIM}, output_id)
    .AddFullyConnected(input_id, filter_id, XNN_INVALID_VALUE_ID, output_id)
    .InferShape()
  ;

  const xnn_value* input = tester.Value(input_id);
  // Input batch size is inferred.
  ASSERT_EQ(input->shape.dim[0], output_batch_dim);
  EXPECT_EQ(input->shape.dim[1], input_channels);

  const xnn_value* output = tester.Value(output_id);
  // ASSERT_EQ(output->shape.dim[0], output_batch_dim);
  EXPECT_EQ(output->shape.dim[1], output_channels);
}

TEST(DISABLED_SUBGRAPH_SHAPE_INFERENCE, fully_connected_cannot_infer_reshaped_input) {
  auto tester = SubgraphTester(6);
  const size_t input_batch_dim = 2 * 3;
  const size_t input_channels = 3;
  const size_t output_channels = 5;
  const uint32_t input_id = 0;
  const uint32_t filter_id = 1;
  const uint32_t output_id = 2;
  tester
    .AddInputTensorF32({input_batch_dim, input_channels}, input_id)
    .AddStaticTensorF32({output_channels, input_channels}, TensorType::kDense, filter_id)
    .AddOutputTensorF32({XNN_UNKNOWN_DIM, output_channels}, output_id)
    .AddFullyConnected(input_id, filter_id, XNN_INVALID_VALUE_ID, output_id, XNN_FLAG_TENSORFLOW_RESHAPE_2D)
    .InferShape()
  ;

  const xnn_value* output = tester.Value(output_id);
  // Output batch dim cannot be inferred due to input being reshaped to be 2D.
  EXPECT_EQ(output->shape.dim[0], XNN_UNKNOWN_DIM);
  EXPECT_FALSE(xnn_tensor_shape_is_static(output));
}

}  // namespace xnnpack
