// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/subgraph.h"
#include "test/subgraph/runtime-tester.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

TEST(AVERAGE_POOLING_2D_THEN_CLAMP, fusion) {
  RuntimeTester tester(3);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t output_id = 2;
  tester.AddInputTensorF32({1, 10, 10, 3}, input_id)
      .AddDynamicTensorF32({1, 9, 9, 3}, intermediate_id)
      .AddOutputTensorF32({1, 9, 9, 3}, output_id)
      .AddAveragePooling2D(0, 0, 0, 0, 2, 2, 1, 1, input_id, intermediate_id)
      .AddClamp(output_min, output_max, intermediate_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(CLAMP_THEN_CLAMP, fusion) {
  RuntimeTester tester(3);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t output_id = 2;
  tester.AddInputTensorF32({1, 10, 10, 3}, input_id)
      .AddDynamicTensorF32({1, 10, 10, 3}, intermediate_id)
      .AddOutputTensorF32({1, 10, 10, 3}, output_id)
      .AddClamp(-std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(), input_id,
                intermediate_id)
      .AddClamp(output_min, output_max, intermediate_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(CONVOLUTION_2D_THEN_CLAMP, fusion) {
  RuntimeTester tester(5);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = 2;
  uint32_t intermediate_id = 3;
  uint32_t output_id = 4;
  tester.AddInputTensorF32({1, 256, 256, 3}, input_id)
      .AddStaticTensorF32({32, 3, 3, 3}, TensorType::kDense, filter_id)
      .AddStaticTensorF32({32}, TensorType::kDense, bias_id)
      .AddDynamicTensorF32({1, 128, 128, 32}, intermediate_id)
      .AddOutputTensorF32({1, 128, 128, 32}, output_id)
      .AddConvolution2D(
          ConvolutionParams{
              Padding{1, 1, 1, 1},
              Kernel{3, 3},
              Subsampling{2, 2},
              Dilation{1, 1},
              /*groups=*/1,
              /*group_input_channels=*/3,
              /*group_output_channels=*/32,
          },
          input_id, filter_id, bias_id, intermediate_id)
      .AddClamp(output_min, output_max, intermediate_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(DECONVOLUTION_2D_THEN_CLAMP, fusion) {
  RuntimeTester tester(5);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = 2;
  uint32_t intermediate_id = 3;
  uint32_t output_id = 4;
  tester.AddInputTensorF32({1, 128, 128, 3}, input_id)
      .AddStaticTensorF32({32, 3, 3, 3}, TensorType::kDense, filter_id)
      .AddStaticTensorF32({32}, TensorType::kDense, bias_id)
      .AddDynamicTensorF32({1, 255, 255, 32}, intermediate_id)
      .AddOutputTensorF32({1, 255, 255, 32}, output_id)
      .AddDeconvolution2D(
          DeconvolutionParams{Padding{1, 1, 1, 1}, Adjustment{0, 0},
                              Kernel{3, 3}, Upsampling{2, 2}, Dilation{1, 1},
                              /*groups=*/1,
                              /*group_input_channels=*/3,
                              /*groups_output_channels=*/32},
          input_id, filter_id, bias_id, intermediate_id)
      .AddClamp(output_min, output_max, intermediate_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(DEPTHWISE_CONVOLUTION_2D_THEN_CLAMP, fusion) {
  RuntimeTester tester(5);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = 2;
  uint32_t intermediate_id = 3;
  uint32_t output_id = 4;
  tester.AddInputTensorF32({1, 128, 128, 4}, input_id)
      .AddStaticTensorF32({1, 3, 3, 4}, TensorType::kDense, filter_id)
      .AddStaticTensorF32({4}, TensorType::kDense, bias_id)
      .AddDynamicTensorF32({1, 128, 128, 4}, intermediate_id)
      .AddOutputTensorF32({1, 128, 128, 4}, output_id)
      .AddDepthwiseConvolution2D(
          DepthwiseConvolutionParams{Padding{1, 1, 1, 1}, Kernel{3, 3},
                                     Subsampling{1, 1}, Dilation{1, 1},
                                     /*depth_multiplier=*/1,
                                     /*input_channels=*/4},
          input_id, filter_id, bias_id, intermediate_id)
      .AddClamp(output_min, output_max, intermediate_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(FULLY_CONNECTED_2D_THEN_CLAMP, fusion) {
  RuntimeTester tester(5);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = 2;
  uint32_t intermediate_id = 3;
  uint32_t output_id = 4;
  tester.AddInputTensorF32({5, 3}, input_id)
      .AddStaticTensorF32({7, 3}, TensorType::kDense, filter_id)
      .AddStaticTensorF32({7}, TensorType::kDense, bias_id)
      .AddDynamicTensorF32({5, 7}, intermediate_id)
      .AddOutputTensorF32({5, 7}, output_id)
      .AddFullyConnected(input_id, filter_id, bias_id, intermediate_id)
      .AddClamp(output_min, output_max, intermediate_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  switch (tester.NumOperators()) {
    case 2:
      ASSERT_EQ(tester.Node(0)->type, xnn_node_type_fully_connected);
      ASSERT_EQ(tester.Node(1)->type, xnn_node_type_unary_elementwise);
      break;
    case 3:
      ASSERT_EQ(tester.Node(0)->type, xnn_node_type_pack_lh);
      ASSERT_EQ(tester.Node(1)->type, xnn_node_type_fully_connected);
      ASSERT_EQ(tester.Node(2)->type, xnn_node_type_unary_elementwise);
      break;
    default:
      GTEST_FAIL() << "Unexpected number of operators ("
                   << tester.NumOperators() << ") in subgraph.";
  }

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();

  const struct xnn_node* fc_node = nullptr;
  switch (tester.NumOperators()) {
    case 1:
      fc_node = tester.Node(0);
      break;
    case 2:
      ASSERT_EQ(tester.Node(0)->type, xnn_node_type_pack_lh);
      fc_node = tester.Node(1);
      break;
    default:
      GTEST_FAIL() << "Unexpected number of operators ("
                   << tester.NumOperators() << ") in subgraph.";
  }
  ASSERT_EQ(fc_node->activation.output_min, output_min);
  ASSERT_EQ(fc_node->activation.output_max, output_max);
  ASSERT_EQ(fc_node->outputs[0], output_id);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(FULLY_CONNECTED_2D_THEN_COPY_THEN_FULLY_CONNECTED, fusion) {
  RuntimeTester tester(11);
  uint32_t fc1_input_id = 0;
  uint32_t fc1_filter_id = 1;
  uint32_t fc1_bias_id = 2;
  uint32_t fc1_output_id = 4;
  uint32_t reshape_output_id = 5;
  uint32_t fc2_filter_id = 7;
  uint32_t fc2_bias_id = 8;
  uint32_t output_id = 9;
  tester.AddInputTensorF32({5, 3}, fc1_input_id)
      .AddStaticTensorF32({7, 3}, TensorType::kDense, fc1_filter_id)
      .AddStaticTensorF32({7}, TensorType::kDense, fc1_bias_id)
      .AddDynamicTensorF32({5, 7}, fc1_output_id)
      .AddDynamicTensorF32({5, 7}, reshape_output_id)
      .AddStaticTensorF32({9, 7}, TensorType::kDense, fc2_filter_id)
      .AddStaticTensorF32({9}, TensorType::kDense, fc2_bias_id)
      .AddOutputTensorF32({5, 9}, output_id)
      .AddFullyConnected(fc1_input_id, fc1_filter_id, fc1_bias_id,
                         fc1_output_id)
      .AddCopy(fc1_output_id, reshape_output_id)
      .AddFullyConnected(reshape_output_id, fc2_filter_id, fc2_bias_id,
                         output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  switch (tester.NumOperators()) {
    case 3:
      ASSERT_EQ(tester.Node(0)->type, xnn_node_type_fully_connected);
      ASSERT_EQ(tester.Node(1)->type, xnn_node_type_copy);
      ASSERT_EQ(tester.Node(2)->type, xnn_node_type_fully_connected);
      break;
    case 5:
      ASSERT_EQ(tester.Node(0)->type, xnn_node_type_pack_lh);
      ASSERT_EQ(tester.Node(1)->type, xnn_node_type_fully_connected);
      ASSERT_EQ(tester.Node(2)->type, xnn_node_type_copy);
      ASSERT_EQ(tester.Node(3)->type, xnn_node_type_pack_lh);
      ASSERT_EQ(tester.Node(4)->type, xnn_node_type_fully_connected);
      break;
    default:
      GTEST_FAIL() << "Unexpected number of operators ("
                   << tester.NumOperators() << ") in subgraph.";
  }

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();

  const struct xnn_node* fc_node = nullptr;
  switch (tester.NumOperators()) {
    case 2:
      fc_node = tester.Node(0);
      break;
    case 4:
      ASSERT_EQ(tester.Node(0)->type, xnn_node_type_pack_lh);
      fc_node = tester.Node(1);
      break;
    default:
      GTEST_FAIL() << "Unexpected number of operators ("
                   << tester.NumOperators() << ") in subgraph.";
  }
  // Copy is optimized away.
  ASSERT_EQ(fc_node->outputs[0], reshape_output_id);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(MAX_POOLING_THEN_CLAMP, fusion) {
  RuntimeTester tester(3);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t output_id = 2;
  tester.AddInputTensorF32({1, 10, 10, 3}, input_id)
      .AddDynamicTensorF32({1, 9, 9, 3}, intermediate_id)
      .AddOutputTensorF32({1, 9, 9, 3}, output_id)
      .AddMaxPooling2D(0, 0, 0, 0, 2, 2, 1, 1, 1, 1, input_id, intermediate_id)
      .AddClamp(output_min, output_max, intermediate_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(CONSTANT_PAD_THEN_CONVOLUTION, fusion) {
  RuntimeTester tester(5);
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t filter_id = 2;
  uint32_t bias_id = 3;
  uint32_t output_id = 4;
  size_t pre_paddings[4] = {0, 2, 4, 0};
  size_t post_paddings[4] = {0, 6, 8, 0};
  float padding_value = 0.0f;

  tester.AddInputTensorF32({1, 254, 254, 3}, input_id)
      .AddDynamicTensorF32({1, 262, 266, 3}, intermediate_id)
      .AddStaticTensorF32({32, 3, 3, 3}, TensorType::kDense, filter_id)
      .AddStaticTensorF32({32}, TensorType::kDense, bias_id)
      .AddOutputTensorF32({1, 131, 133, 32}, output_id)
      .AddConstantPad(pre_paddings, post_paddings, padding_value, input_id,
                      intermediate_id)
      .AddConvolution2D(
          ConvolutionParams{
              Padding{0, 0, 0, 0},
              Kernel{3, 3},
              Subsampling{2, 2},
              Dilation{1, 1},
              /*groups=*/1,
              /*group_input_channels=*/3,
              /*group_output_channels=*/32,
          },
          intermediate_id, filter_id, bias_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(1)->params.convolution_2d.input_padding_top, 2);
  ASSERT_EQ(tester.Node(1)->params.convolution_2d.input_padding_left, 4);
  ASSERT_EQ(tester.Node(1)->params.convolution_2d.input_padding_right, 8);
  ASSERT_EQ(tester.Node(1)->params.convolution_2d.input_padding_bottom, 6);
  ASSERT_EQ(tester.Node(1)->outputs[0], output_id);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(CONSTANT_PAD_THEN_CONVOLUTION,
     not_fused_due_to_non_zero_padding_in_n_dimension) {
  RuntimeTester tester(5);
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t filter_id = 2;
  uint32_t bias_id = 3;
  uint32_t output_id = 4;
  // Non-zero pre-padding in the N or C dimension.
  size_t pre_paddings[4] = {1, 2, 4, 0};
  size_t post_paddings[4] = {0, 6, 8, 0};
  float padding_value = 0.0f;

  tester.AddInputTensorF32({1, 254, 254, 3}, input_id)
      .AddDynamicTensorF32({2, 262, 266, 3}, intermediate_id)
      .AddStaticTensorF32({32, 3, 3, 3}, TensorType::kDense, filter_id)
      .AddStaticTensorF32({32}, TensorType::kDense, bias_id)
      .AddOutputTensorF32({2, 131, 133, 32}, output_id)
      .AddConstantPad(pre_paddings, post_paddings, padding_value, input_id,
                      intermediate_id)
      .AddConvolution2D(
          ConvolutionParams{
              Padding{0, 0, 0, 0},
              Kernel{3, 3},
              Subsampling{2, 2},
              Dilation{1, 1},
              /*groups=*/1,
              /*group_input_channels=*/3,
              /*group_output_channels=*/32,
          },
          intermediate_id, filter_id, bias_id, output_id)
      .Optimize();
  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);
}

TEST(CONSTANT_PAD_THEN_CONVOLUTION, not_fused_due_to_padding_value_not_zero) {
  RuntimeTester tester(5);
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t filter_id = 2;
  uint32_t bias_id = 3;
  uint32_t output_id = 4;
  size_t pre_paddings[4] = {1, 2, 4, 0};
  size_t post_paddings[4] = {0, 6, 8, 0};
  float padding_value = 1.0f;

  tester.AddInputTensorF32({1, 254, 254, 3}, input_id)
      .AddDynamicTensorF32({2, 262, 266, 3}, intermediate_id)
      .AddStaticTensorF32({32, 3, 3, 3}, TensorType::kDense, filter_id)
      .AddStaticTensorF32({32}, TensorType::kDense, bias_id)
      .AddOutputTensorF32({2, 131, 133, 32}, output_id)
      .AddConstantPad(pre_paddings, post_paddings, padding_value, input_id,
                      intermediate_id)
      .AddConvolution2D(
          ConvolutionParams{
              Padding{0, 0, 0, 0},
              Kernel{3, 3},
              Subsampling{2, 2},
              Dilation{1, 1},
              /*groups=*/1,
              /*group_input_channels=*/3,
              /*group_output_channels=*/32,
          },
          intermediate_id, filter_id, bias_id, output_id)
      .Optimize();
  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);
}

TEST(CONSTANT_PAD_THEN_DEPTHWISE_CONVOLUTION, fusion) {
  RuntimeTester tester(5);
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t filter_id = 2;
  uint32_t bias_id = 3;
  uint32_t output_id = 4;
  size_t pre_paddings[4] = {0, 2, 4, 0};
  size_t post_paddings[4] = {0, 6, 8, 0};
  float padding_value = 0.0f;
  tester.AddInputTensorF32({1, 128, 128, 4}, input_id)
      .AddDynamicTensorF32({1, 136, 140, 4}, intermediate_id)
      .AddStaticTensorF32({1, 3, 3, 4}, TensorType::kDense, filter_id)
      .AddStaticTensorF32({4}, TensorType::kDense, bias_id)
      .AddOutputTensorF32({1, 134, 140, 4}, output_id)
      .AddConstantPad(pre_paddings, post_paddings, padding_value, input_id,
                      intermediate_id)
      .AddDepthwiseConvolution2D(
          DepthwiseConvolutionParams{Padding{0, 0, 0, 0}, Kernel{3, 3},
                                     Subsampling{1, 1}, Dilation{1, 1},
                                     /*depth_multiplier=*/1,
                                     /*input_channels=*/4},
          intermediate_id, filter_id, bias_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(1)->params.depthwise_convolution_2d.input_padding_top,
            2);
  ASSERT_EQ(tester.Node(1)->params.depthwise_convolution_2d.input_padding_left,
            4);
  ASSERT_EQ(tester.Node(1)->params.depthwise_convolution_2d.input_padding_right,
            8);
  ASSERT_EQ(
      tester.Node(1)->params.depthwise_convolution_2d.input_padding_bottom, 6);
  ASSERT_EQ(tester.Node(1)->outputs[0], output_id);
  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(CONSTANT_PAD_THEN_DEPTHWISE_CONVOLUTION,
     not_fused_due_to_non_zero_padding_in_n_dimension) {
  RuntimeTester tester(5);
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t filter_id = 2;
  uint32_t bias_id = 3;
  uint32_t output_id = 4;
  // Non-zero pre-padding in the N or C dimension.
  size_t pre_paddings[4] = {1, 2, 4, 0};
  size_t post_paddings[4] = {0, 6, 8, 0};
  float padding_value = 0.0f;
  tester.AddInputTensorF32({1, 128, 128, 4}, input_id)
      .AddDynamicTensorF32({2, 136, 140, 4}, intermediate_id)
      .AddStaticTensorF32({1, 3, 3, 4}, TensorType::kDense, filter_id)
      .AddStaticTensorF32({4}, TensorType::kDense, bias_id)
      .AddOutputTensorF32({2, 134, 140, 4}, output_id)
      .AddConstantPad(pre_paddings, post_paddings, padding_value, input_id,
                      intermediate_id)
      .AddDepthwiseConvolution2D(
          DepthwiseConvolutionParams{Padding{0, 0, 0, 0}, Kernel{3, 3},
                                     Subsampling{1, 1}, Dilation{1, 1},
                                     /*depth_multiplier=*/1,
                                     /*input_channels=*/4},
          intermediate_id, filter_id, bias_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);
  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);
  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(CONSTANT_PAD_THEN_DEPTHWISE_CONVOLUTION,
     not_fused_due_to_padding_value_not_zero) {
  RuntimeTester tester(5);
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t filter_id = 2;
  uint32_t bias_id = 3;
  uint32_t output_id = 4;
  size_t pre_paddings[4] = {0, 2, 4, 0};
  size_t post_paddings[4] = {0, 6, 8, 0};
  float padding_value = 1.0f;
  tester.AddInputTensorF32({1, 128, 128, 4}, input_id)
      .AddDynamicTensorF32({1, 136, 140, 4}, intermediate_id)
      .AddStaticTensorF32({1, 3, 3, 4}, TensorType::kDense, filter_id)
      .AddStaticTensorF32({4}, TensorType::kDense, bias_id)
      .AddOutputTensorF32({1, 134, 140, 4}, output_id)
      .AddConstantPad(pre_paddings, post_paddings, padding_value, input_id,
                      intermediate_id)
      .AddDepthwiseConvolution2D(
          DepthwiseConvolutionParams{Padding{0, 0, 0, 0}, Kernel{3, 3},
                                     Subsampling{1, 1}, Dilation{1, 1},
                                     /*depth_multiplier=*/1,
                                     /*input_channels=*/4},
          intermediate_id, filter_id, bias_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);
  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);
  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(COPY, fused_downstream) {
  // ---input--> (Copy) ---intermediate--> (Clamp) ---output-->
  const uint32_t input_id = 0;
  const uint32_t intermediate_id = 1;
  const uint32_t output_id = 2;
  const std::vector<size_t> dims = {1, 2, 3, 4};
  RuntimeTester tester(3);
  tester.AddInputTensorF32(dims, input_id)
      .AddDynamicTensorF32(dims, intermediate_id)
      .AddOutputTensorF32(dims, output_id)
      .AddCopy(input_id, intermediate_id)
      .AddClamp(-0.5f, 0.5f, intermediate_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 1);
  EXPECT_EQ(unoptimized_output, optimized_output);
  const xnn_node* clamp_node = tester.Node(1);
  ASSERT_EQ(clamp_node->type, xnn_node_type_unary_elementwise);
  ASSERT_EQ(clamp_node->unary_operator, xnn_unary_clamp);
  EXPECT_EQ(clamp_node->inputs[0], input_id);
  EXPECT_EQ(clamp_node->outputs[0], output_id);
}

TEST(COPY, fused_downstream_node_with_multiple_inputs) {
  // ---static data---------------------------\
  // ---input--> (Copy) ---intermediate--> (Add) ---output-->
  const uint32_t input_id = 0;
  const uint32_t copy_out_id = 1;
  const uint32_t output_id = 2;
  const uint32_t static_id = 3;
  const std::vector<size_t> dims = {1, 2, 3, 4};
  RuntimeTester tester(4);
  tester.AddInputTensorF32(dims, input_id)
      .AddDynamicTensorF32(dims, copy_out_id)
      .AddOutputTensorF32(dims, output_id)
      .AddStaticTensorF32(dims, TensorType::kDense, static_id)
      .AddCopy(input_id, copy_out_id)
      .AddAddition(static_id, copy_out_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 1);
  EXPECT_EQ(unoptimized_output, optimized_output);
  const xnn_node* addition_node = tester.Node(1);
  ASSERT_EQ(addition_node->type, xnn_node_type_binary_elementwise);
  ASSERT_EQ(addition_node->binary_operator, xnn_binary_add);
  ASSERT_EQ(addition_node->num_inputs, 2);
  EXPECT_EQ(addition_node->inputs[0], static_id);
  EXPECT_EQ(addition_node->inputs[1], input_id);
}

TEST(COPY, not_fused_downstream_due_to_persistent_tensor) {
  // ---input--> (Copy) ---persistent--> (Clamp) ---output-->
  // We cannot fuse Copy downstream because we need to write to the persistent
  // tensor.
  const uint32_t input_id = 0;
  const uint32_t intermediate_id = 1;
  const uint32_t output_id = 2;
  const std::vector<size_t> dims = {1, 2, 3, 4};
  RuntimeTester tester(3);
  tester.AddInputTensorF32(dims, input_id)
      .AddInputOutputTensorF32(dims, intermediate_id)
      .AddOutputTensorF32(dims, output_id)
      .AddCopy(input_id, intermediate_id)
      .AddClamp(-0.5f, 0.5f, intermediate_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 2);
}

TEST(COPY, fused_upstream) {
  // ---input--> (Clamp) ---intermediate--> (Copy) ---output-->
  const uint32_t input_id = 0;
  const uint32_t intermediate_id = 1;
  const uint32_t output_id = 2;
  const std::vector<size_t> dims = {1, 2, 3, 4};
  RuntimeTester tester(3);
  tester.AddInputTensorF32(dims, input_id)
      .AddDynamicTensorF32(dims, intermediate_id)
      .AddOutputTensorF32(dims, output_id)
      .AddClamp(-0.5f, 0.5f, input_id, intermediate_id)
      .AddCopy(intermediate_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 1);
  EXPECT_EQ(unoptimized_output, optimized_output);

  const xnn_node* clamp_node = tester.Node(0);
  ASSERT_EQ(clamp_node->type, xnn_node_type_unary_elementwise);
  ASSERT_EQ(clamp_node->unary_operator, xnn_unary_clamp);
  EXPECT_EQ(clamp_node->inputs[0], input_id);
  EXPECT_EQ(clamp_node->outputs[0], output_id);
}

TEST(COPY, fused_upstream_with_multiple_outputs) {
  // ---input--> (Split) ---split_out1--> (Copy) ---copy_out1--> (Concat)
  // ---output-->
  //                \-------split_out2--> (Copy) ---copy_out2---/
  const uint32_t input_id = 0;
  const uint32_t split_out1 = 1;
  const uint32_t split_out2 = 2;
  const uint32_t copy_out1 = 3;
  const uint32_t copy_out2 = 4;
  const uint32_t output_id = 5;
  const std::vector<size_t> dims = {1, 2, 3, 4};
  const size_t axis = 1;
  const std::vector<size_t> split_dims = {1, 1, 3, 4};

  RuntimeTester tester(6);
  tester.AddInputTensorF32(dims, input_id)
      .AddDynamicTensorF32(split_dims, split_out1)
      .AddDynamicTensorF32(split_dims, split_out2)
      .AddDynamicTensorF32(split_dims, copy_out1)
      .AddDynamicTensorF32(split_dims, copy_out2)
      .AddOutputTensorF32(dims, output_id)
      .AddEvenSplit(axis, input_id, {split_out1, split_out2})
      .AddCopy(split_out1, copy_out1)
      .AddCopy(split_out2, copy_out2)
      .AddConcatenate(axis, {copy_out1, copy_out2}, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 4);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 2);
  EXPECT_EQ(unoptimized_output, optimized_output);

  const xnn_node* split_node = tester.Node(0);
  ASSERT_EQ(split_node->type, xnn_node_type_even_split);
  EXPECT_EQ(split_node->inputs[0], input_id);
  ASSERT_EQ(split_node->num_outputs, 2);
  EXPECT_EQ(split_node->outputs[0], copy_out1);
  EXPECT_EQ(split_node->outputs[1], copy_out2);

  const xnn_node* concat_node = tester.Node(3);
  ASSERT_EQ(concat_node->type, xnn_node_type_concatenate);
  ASSERT_EQ(concat_node->num_inputs, 2);
  EXPECT_EQ(concat_node->inputs[0], copy_out1);
  EXPECT_EQ(concat_node->inputs[1], copy_out2);
  EXPECT_EQ(concat_node->outputs[0], output_id);
}

TEST(COPY, not_fused_upstream_due_to_persistent_tensor) {
  // ---input--> (Clamp) ---persistent tensor--> (Copy) ---output-->
  // Clamp needs to write to persistent tensor, so we cannot fuse Copy upstream.
  // However, Copy can potentially be fused downstream, we verify that in
  // another test.
  const uint32_t input_id = 0;
  const uint32_t persistent_id = 1;
  const uint32_t output_id = 2;
  const std::vector<size_t> dims = {1, 2, 3, 4};
  RuntimeTester tester(3);
  tester.AddInputTensorF32(dims, input_id)
      .AddInputOutputTensorF32(dims, persistent_id)
      .AddOutputTensorF32(dims, output_id)
      .AddClamp(-0.5f, 0.5f, input_id, persistent_id)
      .AddCopy(persistent_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 2);
  EXPECT_EQ(unoptimized_output, optimized_output);

  const xnn_node* clamp_node = tester.Node(0);
  ASSERT_EQ(clamp_node->type, xnn_node_type_unary_elementwise);
  ASSERT_EQ(clamp_node->unary_operator, xnn_unary_clamp);
  EXPECT_EQ(clamp_node->outputs[0], persistent_id);
}

TEST(COPY,
     not_fused_upstream_due_to_persistent_tensor_but_can_be_fused_downstream) {
  // ---input--> (Clamp) ---persistent tensor--> (Copy) ---copy_out-->
  // (HardSwish) ---output--> We cannot fuse Copy upstream, but later on we can
  // fuse it downstream.
  const uint32_t input_id = 0;
  const uint32_t persistent_id = 1;
  const uint32_t copy_out_id = 2;
  const uint32_t output_id = 3;
  const std::vector<size_t> dims = {1, 2, 3, 4};
  RuntimeTester tester(4);
  tester.AddInputTensorF32(dims, input_id)
      .AddInputOutputTensorF32(dims, persistent_id)
      .AddDynamicTensorF32(dims, copy_out_id)
      .AddOutputTensorF32(dims, output_id)
      .AddClamp(-0.5f, 0.5f, input_id, persistent_id)
      .AddCopy(persistent_id, copy_out_id)
      .AddHardSwish(copy_out_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 3);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 2);
  EXPECT_EQ(unoptimized_output, optimized_output);
}

TEST(COPY, not_fused_upstream_or_downstream_2) {
  // input -> (Clamp) -> internal
  // persistent -> (HardSwish) -> output
  // internal -> (Copy) -> persistent
  // We cannot fuse Copy because it would write the persistent value
  // before it is read by HardSwish instead of after.
  const uint32_t input_id = 0;
  const uint32_t persistent_id = 1;
  const uint32_t internal_id = 2;
  const uint32_t output_id = 3;
  const std::vector<size_t> dims = {1, 2, 3, 4};
  RuntimeTester tester(4);
  tester.AddInputTensorF32(dims, input_id)
      .AddInputOutputTensorF32(dims, persistent_id)
      .AddDynamicTensorF32(dims, internal_id)
      .AddOutputTensorF32(dims, output_id)
      .AddClamp(-0.5f, 0.5f, input_id, internal_id)
      .AddHardSwish(persistent_id, output_id)
      .AddCopy(internal_id, persistent_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 3);

  xnnpack::Buffer<float> optimized_output = tester.RunWithoutFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 3);
}

TEST(COPY, fused_chain_of_copies) {
  // ---input--> (Copy) ---copy_out1--> (Copy) ---copy_out2--> (Copy)
  // ---output-->
  const uint32_t input_id = 0;
  const uint32_t copy_out1 = 1;
  const uint32_t copy_out2 = 2;
  const uint32_t output_id = 3;
  const std::vector<size_t> dims = {1, 2, 3, 4};
  RuntimeTester tester(4);
  tester.AddInputTensorF32(dims, input_id)
      .AddDynamicTensorF32(dims, copy_out1)
      .AddDynamicTensorF32(dims, copy_out2)
      .AddOutputTensorF32(dims, output_id)
      .AddCopy(input_id, copy_out1)
      .AddCopy(copy_out1, copy_out2)
      .AddCopy(copy_out2, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 3);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 1);
  EXPECT_EQ(unoptimized_output, optimized_output);

  const xnn_node* copy_node = tester.Node(0);
  ASSERT_EQ(copy_node->type, xnn_node_type_copy);
  EXPECT_EQ(copy_node->inputs[0], input_id);
  EXPECT_EQ(copy_node->outputs[0], output_id);
}

TEST(UNARY_QUANTIZED_TO_LUT, negate_abs) {
  // Create the subgraph -|x|, quantized.
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  const uint32_t abs_input_id = 2;
  const TensorShape dims = {2, 4, 32};  // 256 elements.
  RuntimeTester tester(3);
  using quint8 = xnnpack::quantized<uint8_t>;
  xnn_quantization_params quantization = {128, 1.0f};
  xnn_quantization_params abs_quantization = {0, 1.0f};
  xnnpack::Buffer<quint8> input_data(dims.NumElements());
  std::iota(input_data.begin(), input_data.end(), 0);
  tester
      .AddInputTensor<quint8>(dims, input_data.data(), quantization,
                              input_id)
      .AddOutputTensor<quint8>(dims, quantization,
                               output_id)
      .AddDynamicTensor<quint8>(dims, abs_input_id, abs_quantization)
      .AddUnary(xnn_unary_abs, nullptr, input_id, abs_input_id)
      .AddUnary(xnn_unary_negate, nullptr, abs_input_id, output_id);

  xnnpack::Buffer<quint8> unoptimized_output =
      tester.RunWithoutFusion<quint8>();
  EXPECT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<quint8> optimized_output = tester.RunWithFusion<quint8>();
  EXPECT_EQ(tester.NumOperators(), 1);

  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(unoptimized_output[i] - 128, -std::abs(i - 128));
    EXPECT_EQ(optimized_output[i] - 128, -std::abs(i - 128));
  }
}

TEST(UNARY_QUANTIZED_TO_LUT, cant_fuse_input) {
  // Create the subgraph -|x| with a float -> quantized conversion of the input.
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  const uint32_t quantized_input_id = 2;
  const uint32_t abs_input_id = 3;
  const TensorShape dims = {2, 4, 32};  // 256 elements.
  RuntimeTester tester(4);
  using quint8 = xnnpack::quantized<uint8_t>;
  xnn_quantization_params quantization = {128, 1.0f};
  xnn_quantization_params abs_quantization = {0, 1.0f};
  xnnpack::Buffer<float> input_data(dims.NumElements());
  std::iota(input_data.begin(), input_data.end(), -128.0f);
  tester.AddInputTensor<float>(dims, input_data.data(), input_id)
      .AddOutputTensor<quint8>(dims, quantization, output_id)
      .AddDynamicTensor<quint8>(dims, quantized_input_id, quantization)
      .AddDynamicTensor<quint8>(dims, abs_input_id, abs_quantization)
      .AddUnary(xnn_unary_convert, nullptr, input_id, quantized_input_id)
      .AddUnary(xnn_unary_abs, nullptr, quantized_input_id, abs_input_id)
      .AddUnary(xnn_unary_negate, nullptr, abs_input_id, output_id);

  xnnpack::Buffer<quint8> unoptimized_output =
      tester.RunWithoutFusion<quint8>();
  EXPECT_EQ(tester.NumOperators(), 3);

  xnnpack::Buffer<quint8> optimized_output = tester.RunWithFusion<quint8>();
  EXPECT_EQ(tester.NumOperators(), 2);

  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(unoptimized_output[i] - 128, -std::abs(i - 128));
    EXPECT_EQ(optimized_output[i] - 128, -std::abs(i - 128));
  }
}

TEST(UNARY_QUANTIZED_TO_LUT, cant_fuse_output) {
  // Create the subgraph -|x| with a quantized -> float conversion of the
  // output.
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  const uint32_t abs_input_id = 2;
  const uint32_t quantized_output_id = 3;
  const TensorShape dims = {2, 4, 32};  // 256 elements.
  RuntimeTester tester(4);
  using quint8 = xnnpack::quantized<uint8_t>;
  xnn_quantization_params quantization = {128, 1.0f};
  xnn_quantization_params abs_quantization = {0, 1.0f};
  xnnpack::Buffer<quint8> input_data(dims.NumElements());
  std::iota(input_data.begin(), input_data.end(), 0);
  tester.AddInputTensor<quint8>(dims, input_data.data(), quantization, input_id)
      .AddOutputTensor<float>(dims, output_id)
      .AddDynamicTensor<quint8>(dims, abs_input_id, abs_quantization)
      .AddDynamicTensor<quint8>(dims, quantized_output_id, quantization)
      .AddUnary(xnn_unary_abs, nullptr, input_id, abs_input_id)
      .AddUnary(xnn_unary_negate, nullptr, abs_input_id, quantized_output_id)
      .AddUnary(xnn_unary_convert, nullptr, quantized_output_id, output_id);

  xnnpack::Buffer<float> unoptimized_output = tester.RunWithoutFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 3);

  xnnpack::Buffer<float> optimized_output = tester.RunWithFusion<float>();
  EXPECT_EQ(tester.NumOperators(), 2);

  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(unoptimized_output[i], -std::abs(i - 128));
    EXPECT_EQ(optimized_output[i], -std::abs(i - 128));
  }
}

TEST(UNARY_QUANTIZED_TO_LUT, softsign) {
  // Create the subgraph x/(1 + |x|), quantized.
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  const uint32_t one_id = 2;
  const uint32_t abs_input_id = 3;
  const uint32_t abs_input_plus_one_id = 4;
  const TensorShape dims = {2, 4, 32};  // 256 elements.
  RuntimeTester tester(5);
  using quint8 = xnnpack::quantized<uint8_t>;
  quint8 one = {255};
  xnn_quantization_params input_quantization = {128, 1.0f};
  xnn_quantization_params output_quantization = {0, 1.0f / 256.0f};
  xnnpack::Buffer<quint8> input_data(dims.NumElements());
  std::iota(input_data.begin(), input_data.end(), 0);
  tester
      .AddInputTensor<quint8>(dims, input_data.data(), input_quantization,
                              input_id)
      .AddOutputTensor<quint8>(dims, output_quantization, output_id)
      .AddStaticTensor<quint8>({1}, one_id, &one, output_quantization)
      .AddDynamicTensor<quint8>(dims, abs_input_id, input_quantization)
      .AddDynamicTensor<quint8>(dims, abs_input_plus_one_id, input_quantization)
      .AddUnary(xnn_unary_abs, nullptr, input_id, abs_input_id)
      .AddBinary(xnn_binary_add, nullptr, abs_input_id, one_id,
                 abs_input_plus_one_id)
      .AddBinary(xnn_binary_divide, nullptr, input_id, abs_input_plus_one_id,
                 output_id);

  xnnpack::Buffer<quint8> unoptimized_output =
      tester.RunWithoutFusion<quint8>();
  EXPECT_EQ(tester.NumOperators(), 3);

  xnnpack::Buffer<quint8> optimized_output = tester.RunWithFusion<quint8>();
  EXPECT_EQ(tester.NumOperators(), 1);
  EXPECT_EQ(unoptimized_output, optimized_output);
}

TEST(UNARY_QUANTIZED_TO_LUT, softplus) {
  // Create the subgraph log(1 + exp(x)), where the input and output are
  // quantized, but the computation is performed in floats.
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  const uint32_t one_id = 2;
  const uint32_t input_float = 3;
  const uint32_t output_float = 4;
  const uint32_t exp_input = 5;
  const uint32_t one_plus_exp_input = 6;
  const TensorShape dims = {2, 4, 32};  // 256 elements.
  RuntimeTester tester(7);
  using quint8 = xnnpack::quantized<uint8_t>;
  float one = {1.0f};
  xnn_quantization_params input_quantization = {128, 1.0f};
  xnn_quantization_params output_quantization = {0, 1.0f / 256.0f};
  xnnpack::Buffer<quint8> input_data(dims.NumElements());
  std::iota(input_data.begin(), input_data.end(), 0);
  tester
      .AddInputTensor<quint8>(dims, input_data.data(), input_quantization,
                              input_id)
      .AddOutputTensor<quint8>(dims, output_quantization,
                               output_id)
      .AddStaticTensor<float>({1}, one_id, &one, output_quantization)
      .AddDynamicTensor<float>(dims, input_float, input_quantization)
      .AddDynamicTensor<float>(dims, output_float, output_quantization)
      .AddDynamicTensor<float>(dims, exp_input, output_quantization)
      .AddDynamicTensor<float>(dims, one_plus_exp_input, output_quantization)
      .AddUnary(xnn_unary_convert, nullptr, input_id, input_float)
      .AddUnary(xnn_unary_exp, nullptr, input_float, exp_input)
      .AddBinary(xnn_binary_add, nullptr, exp_input, one_id, one_plus_exp_input)
      .AddUnary(xnn_unary_log, nullptr, one_plus_exp_input, output_float)
      .AddUnary(xnn_unary_convert, nullptr, output_float, output_id);

  xnnpack::Buffer<quint8> unoptimized_output =
      tester.RunWithoutFusion<quint8>();
  EXPECT_EQ(tester.NumOperators(), 5);

  xnnpack::Buffer<quint8> optimized_output = tester.RunWithFusion<quint8>();
  EXPECT_EQ(tester.NumOperators(), 1);
  EXPECT_EQ(unoptimized_output, optimized_output);
}

TEST(UNARY_QUANTIZED_TO_LUT, binary_first) {
  // Create the subgraph |x - 1|, quantized.
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  const uint32_t one_id = 2;
  const uint32_t sub_id = 3;
  const TensorShape dims = {2, 4, 32};  // 256 elements.
  RuntimeTester tester(4);
  using quint8 = xnnpack::quantized<uint8_t>;
  quint8 one = {255};
  xnn_quantization_params input_quantization = {128, 1.0f};
  xnn_quantization_params output_quantization = {0, 1.0f / 256.0f};
  xnnpack::Buffer<quint8> input_data(dims.NumElements());
  std::iota(input_data.begin(), input_data.end(), 0);
  tester
      .AddInputTensor<quint8>(dims, input_data.data(), input_quantization,
                              input_id)
      .AddOutputTensor<quint8>(dims, output_quantization,
                               output_id)
      .AddStaticTensor<quint8>({1}, one_id, &one, output_quantization)
      .AddDynamicTensor<quint8>(dims, sub_id, input_quantization)
      .AddBinary(xnn_binary_add, nullptr, input_id, one_id, sub_id)
      .AddUnary(xnn_unary_abs, nullptr, sub_id, output_id);

  xnnpack::Buffer<quint8> unoptimized_output =
      tester.RunWithoutFusion<quint8>();
  EXPECT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<quint8> optimized_output = tester.RunWithFusion<quint8>();
  EXPECT_EQ(tester.NumOperators(), 1);
  EXPECT_EQ(unoptimized_output, optimized_output);
}

TEST(UNARY_QUANTIZED_TO_LUT, binary_not_unary) {
  // Create the subgraph x + max(x, y)
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  const uint32_t output_id = 2;
  const uint32_t max_id = 3;
  const TensorShape dims = {2, 4, 32};  // 256 elements.
  RuntimeTester tester(4);
  using quint8 = xnnpack::quantized<uint8_t>;
  xnn_quantization_params input_quantization = {128, 1.0f};
  xnn_quantization_params output_quantization = {128, 1.0f};
  xnnpack::Buffer<quint8> x_data(dims.NumElements());
  xnnpack::Buffer<quint8> y_data(dims.NumElements());
  std::iota(x_data.begin(), x_data.end(), 0);
  std::iota(y_data.begin(), y_data.end(), 3);
  tester.AddInputTensor<quint8>(dims, x_data.data(), input_quantization, x_id)
      .AddInputTensor<quint8>(dims, y_data.data(), input_quantization, y_id)
      .AddOutputTensor<quint8>(dims, output_quantization, output_id)
      .AddDynamicTensor<quint8>(dims, max_id, input_quantization)
      .AddBinary(xnn_binary_maximum, nullptr, x_id, y_id, max_id)
      .AddBinary(xnn_binary_add, nullptr, x_id, max_id, output_id);

  xnnpack::Buffer<quint8> unoptimized_output =
      tester.RunWithoutFusion<quint8>();
  EXPECT_EQ(tester.NumOperators(), 2);

  xnnpack::Buffer<quint8> optimized_output = tester.RunWithFusion<quint8>();
  EXPECT_EQ(tester.NumOperators(), 2);
  EXPECT_EQ(unoptimized_output, optimized_output);
}

}  // namespace xnnpack
