// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>
#include <xnnpack/subgraph.h>

#include "runtime-tester.h"
#include <gtest/gtest.h>

namespace xnnpack {

TEST(ADD_THEN_CLAMP, fusion) {
  auto tester = RuntimeTester(4);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input1_id = 0;
  uint32_t input2_id = 1;
  uint32_t intermediate_id = 2;
  uint32_t output_id = 3;
  tester
    .AddInputTensorF32({1, 2, 2, 3}, input1_id)
    .AddInputTensorF32({1, 2, 2, 3}, input2_id)
    .AddDynamicTensorF32({1, 2, 2, 3}, intermediate_id)
    .AddOutputTensorF32({1, 2, 2, 3}, output_id)
    .AddAddition(input1_id, input2_id, intermediate_id)
    .AddClamp(output_min, output_max, intermediate_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  std::vector<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);
  ASSERT_EQ(tester.Node(1)->compute_type, xnn_compute_type_invalid);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(AVERAGE_POOLING_2D_THEN_CLAMP, fusion) {
  auto tester = RuntimeTester(3);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t output_id = 2;
  tester
    .AddInputTensorF32({1, 10, 10, 3}, input_id)
    .AddDynamicTensorF32({1, 9, 9, 3}, intermediate_id)
    .AddOutputTensorF32({1, 9, 9, 3}, output_id)
    .AddAveragePooling2D(0, 0, 0, 0, 2, 2, 1, 1, input_id, intermediate_id)
    .AddClamp(output_min, output_max, intermediate_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  std::vector<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);
  ASSERT_EQ(tester.Node(1)->compute_type, xnn_compute_type_invalid);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(CLAMP_THEN_CLAMP, fusion) {
  auto tester = RuntimeTester(3);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t output_id = 2;
  tester
    .AddInputTensorF32({1, 10, 10, 3}, input_id)
    .AddDynamicTensorF32({1, 10, 10, 3}, intermediate_id)
    .AddOutputTensorF32({1, 10, 10, 3}, output_id)
    .AddClamp(
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        input_id,
        intermediate_id)
    .AddClamp(output_min, output_max, intermediate_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  std::vector<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);
  ASSERT_EQ(tester.Node(1)->compute_type, xnn_compute_type_invalid);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(CONVOLUTION_2D_THEN_CLAMP, fusion) {
  auto tester = RuntimeTester(5);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = 2;
  uint32_t intermediate_id = 3;
  uint32_t output_id = 4;
  tester
    .AddInputTensorF32({1, 256, 256, 3}, input_id)
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
          /*groups=*/ 1,
          /*group_input_channels=*/ 3,
          /*group_output_channels=*/ 32,
        }, input_id, filter_id, bias_id, intermediate_id)
    .AddClamp(output_min, output_max, intermediate_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  std::vector<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);
  ASSERT_EQ(tester.Node(1)->compute_type, xnn_compute_type_invalid);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(DIVIDE_THEN_CLAMP, fusion) {
  auto tester = RuntimeTester(4);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input1_id = 0;
  uint32_t input2_id = 1;
  uint32_t intermediate_id = 2;
  uint32_t output_id = 3;
  tester
    .AddInputTensorF32({1, 2, 2, 3}, input1_id)
    .AddInputTensorF32({1, 2, 2, 3}, input2_id)
    .AddDynamicTensorF32({1, 2, 2, 3}, intermediate_id)
    .AddOutputTensorF32({1, 2, 2, 3}, output_id)
    .AddDivide(input1_id, input2_id, intermediate_id)
    .AddClamp(output_min, output_max, intermediate_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  std::vector<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);
  ASSERT_EQ(tester.Node(1)->compute_type, xnn_compute_type_invalid);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(DECONVOLUTION_2D_THEN_CLAMP, fusion) {
  auto tester = RuntimeTester(5);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = 2;
  uint32_t intermediate_id = 3;
  uint32_t output_id = 4;
  tester
    .AddInputTensorF32({1, 128, 128, 3}, input_id)
    .AddStaticTensorF32({32, 3, 3, 3}, TensorType::kDense, filter_id)
    .AddStaticTensorF32({32}, TensorType::kDense, bias_id)
    .AddDynamicTensorF32({1, 255, 255, 32}, intermediate_id)
    .AddOutputTensorF32({1, 255, 255, 32}, output_id)
    .AddDeconvolution2D(
        DeconvolutionParams{
          Padding{1, 1, 1, 1},
          Adjustment{0, 0},
          Kernel{3, 3},
          Upsampling{2, 2},
          Dilation{1, 1},
          /*groups=*/ 1,
          /*group_input_channels=*/ 3,
          /*groups_output_channels*/ 32
          }, input_id, filter_id, bias_id, intermediate_id)
    .AddClamp(output_min, output_max, intermediate_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  std::vector<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);
  ASSERT_EQ(tester.Node(1)->compute_type, xnn_compute_type_invalid);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(DEPTHWISE_CONVOLUTION_2D_THEN_CLAMP, fusion) {
  auto tester = RuntimeTester(5);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = 2;
  uint32_t intermediate_id = 3;
  uint32_t output_id = 4;
  tester
    .AddInputTensorF32({1, 128, 128, 4}, input_id)
    .AddStaticTensorF32({1, 3, 3, 4}, TensorType::kDense, filter_id)
    .AddStaticTensorF32({4}, TensorType::kDense, bias_id)
    .AddDynamicTensorF32({1, 128, 128, 4}, intermediate_id)
    .AddOutputTensorF32({1, 128, 128, 4}, output_id)
    .AddDepthwiseConvolution2D(
        DepthwiseConvolutionParams{
          Padding{1, 1, 1, 1},
          Kernel{3, 3},
          Subsampling{1, 1},
          Dilation{1, 1},
          /*depth_multiplier=*/ 1,
          /*input_channels=*/ 4
        }, input_id, filter_id, bias_id, intermediate_id)
    .AddClamp(output_min, output_max, intermediate_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  std::vector<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);
  ASSERT_EQ(tester.Node(1)->compute_type, xnn_compute_type_invalid);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(FULLY_CONNECTED_2D_THEN_CLAMP, fusion) {
  auto tester = RuntimeTester(5);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input_id = 0;
  uint32_t filter_id = 1;
  uint32_t bias_id = 2;
  uint32_t intermediate_id = 3;
  uint32_t output_id = 4;
  tester
    .AddInputTensorF32({5, 3}, input_id)
    .AddStaticTensorF32({7, 3}, TensorType::kDense, filter_id)
    .AddStaticTensorF32({7}, TensorType::kDense, bias_id)
    .AddDynamicTensorF32({5, 7}, intermediate_id)
    .AddOutputTensorF32({5, 7}, output_id)
    .AddFullyConnected(input_id, filter_id, bias_id, intermediate_id)
    .AddClamp(output_min, output_max, intermediate_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  std::vector<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);
  ASSERT_EQ(tester.Node(1)->compute_type, xnn_compute_type_invalid);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(MULTIPLY_THEN_CLAMP, fusion) {
  auto tester = RuntimeTester(4);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input1_id = 0;
  uint32_t input2_id = 1;
  uint32_t intermediate_id = 2;
  uint32_t output_id = 3;
  tester
    .AddInputTensorF32({1, 2, 2, 3}, input1_id)
    .AddInputTensorF32({1, 2, 2, 3}, input2_id)
    .AddDynamicTensorF32({1, 2, 2, 3}, intermediate_id)
    .AddOutputTensorF32({1, 2, 2, 3}, output_id)
    .AddMultiply(input1_id, input2_id, intermediate_id)
    .AddClamp(output_min, output_max, intermediate_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  std::vector<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);
  ASSERT_EQ(tester.Node(1)->compute_type, xnn_compute_type_invalid);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(MAX_POOLING_THEN_CLAMP, fusion) {
  auto tester = RuntimeTester(3);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t output_id = 2;
  tester
    .AddInputTensorF32({1, 10, 10, 3}, input_id)
    .AddDynamicTensorF32({1, 9, 9, 3}, intermediate_id)
    .AddOutputTensorF32({1, 9, 9, 3}, output_id)
    .AddMaxPooling2D(0, 0, 0, 0, 2, 2, 1, 1, 1, 1, input_id, intermediate_id)
    .AddClamp(output_min, output_max, intermediate_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  std::vector<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);
  ASSERT_EQ(tester.Node(1)->compute_type, xnn_compute_type_invalid);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(SUBTRACT_THEN_CLAMP, fusion) {
  auto tester = RuntimeTester(4);
  float output_min = -0.5f;
  float output_max = 0.5f;
  uint32_t input1_id = 0;
  uint32_t input2_id = 1;
  uint32_t intermediate_id = 2;
  uint32_t output_id = 3;
  tester
    .AddInputTensorF32({1, 2, 2, 3}, input1_id)
    .AddInputTensorF32({1, 2, 2, 3}, input2_id)
    .AddDynamicTensorF32({1, 2, 2, 3}, intermediate_id)
    .AddOutputTensorF32({1, 2, 2, 3}, output_id)
    .AddSubtract(input1_id, input2_id, intermediate_id)
    .AddClamp(output_min, output_max, intermediate_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  std::vector<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->activation.output_min, output_min);
  ASSERT_EQ(tester.Node(0)->activation.output_max, output_max);
  ASSERT_EQ(tester.Node(0)->outputs[0], output_id);
  ASSERT_EQ(tester.Node(1)->compute_type, xnn_compute_type_invalid);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(CONSTANT_PAD_THEN_CONVOLUTION, fusion) {
  auto tester = RuntimeTester(5);
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t filter_id = 2;
  uint32_t bias_id = 3;
  uint32_t output_id = 4;
  size_t pre_paddings[4] = {0, 2, 4, 0};
  size_t post_paddings[4] = {0, 6, 8, 0};
  float padding_value = 0.0f;

  tester
    .AddInputTensorF32({1, 254, 254, 3}, input_id)
    .AddDynamicTensorF32({1, 262, 266, 3}, intermediate_id)
    .AddStaticTensorF32({32, 3, 3, 3}, TensorType::kDense, filter_id)
    .AddStaticTensorF32({32}, TensorType::kDense, bias_id)
    .AddOutputTensorF32({1, 131, 133, 32}, output_id)
    .AddConstantPad(pre_paddings, post_paddings, padding_value, input_id, intermediate_id)
    .AddConvolution2D(
        ConvolutionParams{
          Padding{0, 0, 0, 0},
          Kernel{3, 3},
          Subsampling{2, 2},
          Dilation{1, 1},
          /*groups=*/ 1,
          /*group_input_channels=*/ 3,
          /*group_output_channels=*/ 32,
        }, intermediate_id, filter_id, bias_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  std::vector<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->compute_type, xnn_compute_type_invalid);
  ASSERT_EQ(tester.Node(1)->params.convolution_2d.input_padding_top, 2);
  ASSERT_EQ(tester.Node(1)->params.convolution_2d.input_padding_left, 4);
  ASSERT_EQ(tester.Node(1)->params.convolution_2d.input_padding_right, 8);
  ASSERT_EQ(tester.Node(1)->params.convolution_2d.input_padding_bottom, 6);
  ASSERT_EQ(tester.Node(1)->outputs[0], output_id);

  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(CONSTANT_PAD_THEN_CONVOLUTION, not_fused_due_to_non_zero_padding_in_n_dimension) {
  auto tester = RuntimeTester(5);
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t filter_id = 2;
  uint32_t bias_id = 3;
  uint32_t output_id = 4;
  // Non-zero pre-padding in the N or C dimension.
  size_t pre_paddings[4] = {1, 2, 4, 0};
  size_t post_paddings[4] = {0, 6, 8, 0};
  float padding_value = 0.0f;

  tester
    .AddInputTensorF32({1, 254, 254, 3}, input_id)
    .AddDynamicTensorF32({2, 262, 266, 3}, intermediate_id)
    .AddStaticTensorF32({32, 3, 3, 3}, TensorType::kDense, filter_id)
    .AddStaticTensorF32({32}, TensorType::kDense, bias_id)
    .AddOutputTensorF32({2, 131, 133, 32}, output_id)
    .AddConstantPad(pre_paddings, post_paddings, padding_value, input_id, intermediate_id)
    .AddConvolution2D(
        ConvolutionParams{
          Padding{0, 0, 0, 0},
          Kernel{3, 3},
          Subsampling{2, 2},
          Dilation{1, 1},
          /*groups=*/ 1,
          /*group_input_channels=*/ 3,
          /*group_output_channels=*/ 32,
        }, intermediate_id, filter_id, bias_id, output_id)
    .Optimize();
  std::vector<float> optimized_output = tester.RunWithFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);
}

TEST(CONSTANT_PAD_THEN_CONVOLUTION, not_fused_due_to_padding_value_not_zero) {
  auto tester = RuntimeTester(5);
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t filter_id = 2;
  uint32_t bias_id = 3;
  uint32_t output_id = 4;
  size_t pre_paddings[4] = {0, 2, 4, 0};
  size_t post_paddings[4] = {0, 6, 8, 0};
  float padding_value = 1.0f;

  tester
    .AddInputTensorF32({1, 254, 254, 3}, input_id)
    .AddDynamicTensorF32({2, 262, 266, 3}, intermediate_id)
    .AddStaticTensorF32({32, 3, 3, 3}, TensorType::kDense, filter_id)
    .AddStaticTensorF32({32}, TensorType::kDense, bias_id)
    .AddOutputTensorF32({2, 131, 133, 32}, output_id)
    .AddConstantPad(pre_paddings, post_paddings, padding_value, input_id, intermediate_id)
    .AddConvolution2D(
        ConvolutionParams{
          Padding{0, 0, 0, 0},
          Kernel{3, 3},
          Subsampling{2, 2},
          Dilation{1, 1},
          /*groups=*/ 1,
          /*group_input_channels=*/ 3,
          /*group_output_channels=*/ 32,
        }, intermediate_id, filter_id, bias_id, output_id)
    .Optimize();
  std::vector<float> optimized_output = tester.RunWithFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);
}

TEST(CONSTANT_PAD_THEN_DEPTHWISE_CONVOLUTION, fusion) {
  auto tester = RuntimeTester(5);
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t filter_id = 2;
  uint32_t bias_id = 3;
  uint32_t output_id = 4;
  size_t pre_paddings[4] = {0, 2, 4, 0};
  size_t post_paddings[4] = {0, 6, 8, 0};
  float padding_value = 0.0f;
  tester
    .AddInputTensorF32({1, 128, 128, 4}, input_id)
    .AddDynamicTensorF32({1, 136, 140, 4}, intermediate_id)
    .AddStaticTensorF32({1, 3, 3, 4}, TensorType::kDense, filter_id)
    .AddStaticTensorF32({4}, TensorType::kDense, bias_id)
    .AddOutputTensorF32({1, 134, 140, 4}, output_id)
    .AddConstantPad(pre_paddings, post_paddings, padding_value, input_id, intermediate_id)
    .AddDepthwiseConvolution2D(
        DepthwiseConvolutionParams{
          Padding{0, 0, 0, 0},
          Kernel{3, 3},
          Subsampling{1, 1},
          Dilation{1, 1},
          /*depth_multiplier=*/ 1,
          /*input_channels=*/ 4
        }, intermediate_id, filter_id, bias_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);

  std::vector<float> optimized_output = tester.RunWithFusion<float>();

  ASSERT_EQ(tester.NumOperators(), 1);
  ASSERT_EQ(tester.Node(0)->compute_type, xnn_compute_type_invalid);
  ASSERT_EQ(tester.Node(1)->params.depthwise_convolution_2d.input_padding_top, 2);
  ASSERT_EQ(tester.Node(1)->params.depthwise_convolution_2d.input_padding_left, 4);
  ASSERT_EQ(tester.Node(1)->params.depthwise_convolution_2d.input_padding_right, 8);
  ASSERT_EQ(tester.Node(1)->params.depthwise_convolution_2d.input_padding_bottom, 6);
  ASSERT_EQ(tester.Node(1)->outputs[0], output_id);
  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(CONSTANT_PAD_THEN_DEPTHWISE_CONVOLUTION, not_fused_due_to_non_zero_padding_in_n_dimension) {
  auto tester = RuntimeTester(5);
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t filter_id = 2;
  uint32_t bias_id = 3;
  uint32_t output_id = 4;
  // Non-zero pre-padding in the N or C dimension.
  size_t pre_paddings[4] = {1, 2, 4, 0};
  size_t post_paddings[4] = {0, 6, 8, 0};
  float padding_value = 0.0f;
  tester
    .AddInputTensorF32({1, 128, 128, 4}, input_id)
    .AddDynamicTensorF32({2, 136, 140, 4}, intermediate_id)
    .AddStaticTensorF32({1, 3, 3, 4}, TensorType::kDense, filter_id)
    .AddStaticTensorF32({4}, TensorType::kDense, bias_id)
    .AddOutputTensorF32({2, 134, 140, 4}, output_id)
    .AddConstantPad(pre_paddings, post_paddings, padding_value, input_id, intermediate_id)
    .AddDepthwiseConvolution2D(
        DepthwiseConvolutionParams{
          Padding{0, 0, 0, 0},
          Kernel{3, 3},
          Subsampling{1, 1},
          Dilation{1, 1},
          /*depth_multiplier=*/ 1,
          /*input_channels=*/ 4
        }, intermediate_id, filter_id, bias_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);
  std::vector<float> optimized_output = tester.RunWithFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);
  ASSERT_EQ(unoptimized_output, optimized_output);
}

TEST(CONSTANT_PAD_THEN_DEPTHWISE_CONVOLUTION, not_fused_due_to_padding_value_not_zero) {
  auto tester = RuntimeTester(5);
  uint32_t input_id = 0;
  uint32_t intermediate_id = 1;
  uint32_t filter_id = 2;
  uint32_t bias_id = 3;
  uint32_t output_id = 4;
  size_t pre_paddings[4] = {0, 2, 4, 0};
  size_t post_paddings[4] = {0, 6, 8, 0};
  float padding_value = 1.0f;
  tester
    .AddInputTensorF32({1, 128, 128, 4}, input_id)
    .AddDynamicTensorF32({1, 136, 140, 4}, intermediate_id)
    .AddStaticTensorF32({1, 3, 3, 4}, TensorType::kDense, filter_id)
    .AddStaticTensorF32({4}, TensorType::kDense, bias_id)
    .AddOutputTensorF32({1, 134, 140, 4}, output_id)
    .AddConstantPad(pre_paddings, post_paddings, padding_value, input_id, intermediate_id)
    .AddDepthwiseConvolution2D(
        DepthwiseConvolutionParams{
          Padding{0, 0, 0, 0},
          Kernel{3, 3},
          Subsampling{1, 1},
          Dilation{1, 1},
          /*depth_multiplier=*/ 1,
          /*input_channels=*/ 4
        }, intermediate_id, filter_id, bias_id, output_id);

  std::vector<float> unoptimized_output = tester.RunWithoutFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);
  std::vector<float> optimized_output = tester.RunWithFusion<float>();
  ASSERT_EQ(tester.NumOperators(), 2);
  ASSERT_EQ(unoptimized_output, optimized_output);
}

}  // namespace xnnpack
