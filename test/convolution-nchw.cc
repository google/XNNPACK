// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "convolution-operator-tester.h"

/**************************** SPMM path ****************************/

TEST(CONVOLUTION_NCHW_F16, kernel_1x1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, kernel_1x1_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, kernel_1x1_zero_weights_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, kernel_1x1_varying_input_height_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .input_size(input_height, 37)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, kernel_1x1_varying_input_width_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .input_size(27, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, kernel_1x1_varying_input_channels_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .input_size(27, 37)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(19)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, kernel_1x1_varying_output_channels_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels < 19; output_channels *= 2) {
    ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .input_size(27, 37)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, kernel_1x1_with_qmin_with_fp32_weights) {
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, kernel_1x1_with_qmax_with_fp32_weights) {
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, kernel_1x1_without_bias_with_fp32_weights) {
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .iterations(3)
    .TestNCHWxF16();
}

// Weights cache is not supported for SPMM microkernel, add a test here, but skip the assertions.
TEST(CONVOLUTION_NCHW_F16, weights_cache_1x1_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .use_weights_cache(true)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** SPMM path, batched ****************************/

TEST(CONVOLUTION_NCHW_F16, batched_1x1_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_1x1_zero_weights_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_1x1_varying_input_height_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .batch_size(2)
      .input_size(input_height, 37)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_1x1_varying_input_width_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .batch_size(2)
      .input_size(27, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_1x1_varying_input_channels_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(19)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_1x1_varying_output_channels_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels < 19; output_channels *= 2) {
    ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_1x1_with_input_stride_with_fp32_weights) {
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .input_channel_stride(25)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_1x1_with_output_stride_with_fp32_weights) {
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .output_channel_stride(21)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_1x1_with_qmin_with_fp32_weights) {
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_1x1_with_qmax_with_fp32_weights) {
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_1x1_without_bias_with_fp32_weights) {
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DConv 3x3c3s2 HWC->CHW path ****************************/

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, kernel_3x3c3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, kernel_3x3c3s2_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, kernel_3x3c3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .input_size(input_height, 37)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(19)
      .force_nhwc_input(true)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, kernel_3x3c3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .input_size(27, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(19)
      .force_nhwc_input(true)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, kernel_3x3c3s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels < 19; output_channels *= 2) {
    ConvolutionOperatorTester()
      .input_size(27, 37)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(output_channels)
      .force_nhwc_input(true)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, kernel_3x3c3s2_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, kernel_3x3c3s2_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, kernel_3x3c3s2_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, weights_cache_3x3c3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .use_weights_cache(true)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DConv 3x3c3s2 HWC->CHW path, batched ****************************/

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, batched_3x3c3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, batched_3x3c3s2_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .batch_size(2)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, batched_3x3c3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, 37)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(19)
      .force_nhwc_input(true)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, batched_3x3c3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(19)
      .force_nhwc_input(true)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, batched_3x3c3s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels < 19; output_channels *= 2) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, 37)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(output_channels)
      .force_nhwc_input(true)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, batched_3x3c3s2_with_output_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .output_channel_stride(21)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, batched_3x3c3s2_with_qmin) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, batched_3x3c3s2_with_qmax) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, batched_3x3c3s2_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F16, weights_cache_batched_3x3c3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .use_weights_cache(true)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 3x3 path ****************************/

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .input_size(input_height, 37)
      .kernel_size(3, 3)
      .padding(1)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .input_size(27, input_width)
      .kernel_size(3, 3)
      .padding(1)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, weights_cache_depthwise_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 3x3 path, batched ****************************/

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, 37)
      .kernel_size(3, 3)
      .padding(1)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, input_width)
      .kernel_size(3, 3)
      .padding(1)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3_with_input_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .input_channel_stride(21)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3_with_output_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .output_channel_stride(23)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3_with_qmin) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3_with_qmax) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, weights_cache_batched_depthwise_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 3x3 stride-2 path ****************************/

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3s2_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3s2_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .input_size(input_height, 37)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .input_size(27, input_width)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3s2_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3s2_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_3x3s2_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 3x3 stride-2 path, batched ****************************/

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3s2_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3s2_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, 37)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, 37) //input_width)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3s2_with_input_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .input_channel_stride(21)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3s2_with_output_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .output_channel_stride(23)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3s2_with_qmin) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3s2_with_qmax) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_3x3s2_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 5x5 path ****************************/

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .input_size(input_height, 37)
      .kernel_size(5, 5)
      .padding(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .input_size(27, input_width)
      .kernel_size(5, 5)
      .padding(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 5x5 path, batched ****************************/

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, 37)
      .kernel_size(5, 5)
      .padding(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, input_width)
      .kernel_size(5, 5)
      .padding(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5_with_input_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .input_channel_stride(21)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5_with_output_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .output_channel_stride(23)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5_with_qmin) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5_with_qmax) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 5x5 stride-2 path ****************************/

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5s2_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5s2_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .input_size(input_height, 37)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .input_size(27, input_width)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5s2_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5s2_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, depthwise_5x5s2_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 5x5 stride-2 path, batched ****************************/

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5s2_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5s2_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, 37)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, input_width)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5s2_with_input_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .input_channel_stride(21)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5s2_with_output_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .output_channel_stride(23)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5s2_with_qmin) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5s2_with_qmax) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(CONVOLUTION_NCHW_F16, batched_depthwise_5x5s2_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 3x3 path ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_3x3_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .depthwise_layout(true)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_3x3_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_3x3_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, weights_cache_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 3x3 path, batched ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_3x3_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .depthwise_layout(true)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_3x3_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_3x3_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, weights_cache_batched_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 3x3 stride-2 path ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_3x3s2_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .depthwise_layout(true)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_3x3s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_3x3s2_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 3x3 stride-2 path, batched ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_3x3s2_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .depthwise_layout(true)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_3x3s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_3x3s2_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 5x5 path ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_5x5) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_5x5_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .depthwise_layout(true)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_5x5_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_5x5_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 5x5 path, batched ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_5x5) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_5x5_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .depthwise_layout(true)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_5x5_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_5x5_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 5x5 stride-2 path ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_5x5s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_5x5s2_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .depthwise_layout(true)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_5x5s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, kernel_5x5s2_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** DWCONV 5x5 stride-2 path, batched ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_5x5s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_5x5s2_with_fp32_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
    .depthwise_layout(true)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_5x5s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF16();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F16, batched_5x5s2_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF16();
}

/**************************** SPMM path ****************************/

TEST(CONVOLUTION_NCHW_F32, kernel_1x1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, kernel_1x1_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, kernel_1x1_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .input_size(input_height, 37)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, kernel_1x1_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .input_size(27, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, kernel_1x1_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    ConvolutionOperatorTester()
      .input_size(27, 37)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(19)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, kernel_1x1_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels < 19; output_channels *= 2) {
    ConvolutionOperatorTester()
      .input_size(27, 37)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, kernel_1x1_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, kernel_1x1_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, kernel_1x1_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .iterations(3)
    .TestNCHWxF32();
}

// Weights cache is not supported for SPMM microkernel, add a test here, but skip the assertions.
TEST(CONVOLUTION_NCHW_F32, weights_cache_1x1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .use_weights_cache(true)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** SPMM path, batched ****************************/

TEST(CONVOLUTION_NCHW_F32, batched_1x1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_1x1_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_1x1_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, 37)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_1x1_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, input_width)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_1x1_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(19)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_1x1_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels < 19; output_channels *= 2) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .sparsity(0.5f)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_1x1_with_input_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .input_channel_stride(25)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_1x1_with_output_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .output_channel_stride(21)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_1x1_with_qmin) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_1x1_with_qmax) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_1x1_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(0.5f)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DConv 3x3c3s2 HWC->CHW path ****************************/

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, kernel_3x3c3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, kernel_3x3c3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .input_size(input_height, 37)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(19)
      .force_nhwc_input(true)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, kernel_3x3c3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .input_size(27, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(19)
      .force_nhwc_input(true)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, kernel_3x3c3s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels < 19; output_channels *= 2) {
    ConvolutionOperatorTester()
      .input_size(27, 37)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(output_channels)
      .force_nhwc_input(true)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, kernel_3x3c3s2_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, kernel_3x3c3s2_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, kernel_3x3c3s2_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, weights_cache_3x3c3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .use_weights_cache(true)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DConv 3x3c3s2 HWC->CHW path, batched ****************************/

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, batched_3x3c3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, batched_3x3c3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, 37)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(19)
      .force_nhwc_input(true)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, batched_3x3c3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, input_width)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(19)
      .force_nhwc_input(true)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, batched_3x3c3s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t output_channels = 1; output_channels < 19; output_channels *= 2) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, 37)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(output_channels)
      .force_nhwc_input(true)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, batched_3x3c3s2_with_output_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .output_channel_stride(21)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, batched_3x3c3s2_with_qmin) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, batched_3x3c3s2_with_qmax) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, batched_3x3c3s2_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NHWC2NCHW_OP_F32, weights_cache_batched_3x3c3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .force_nhwc_input(true)
    .use_weights_cache(true)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 3x3 path ****************************/

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .input_size(input_height, 37)
      .kernel_size(3, 3)
      .padding(1)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .input_size(27, input_width)
      .kernel_size(3, 3)
      .padding(1)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, weights_cache_depthwise_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 3x3 path, batched ****************************/

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, 37)
      .kernel_size(3, 3)
      .padding(1)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, input_width)
      .kernel_size(3, 3)
      .padding(1)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3_with_input_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .input_channel_stride(21)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3_with_output_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .output_channel_stride(23)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3_with_qmin) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3_with_qmax) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, weights_cache_batched_depthwise_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 3x3 stride-2 path ****************************/

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3s2_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .input_size(input_height, 37)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .input_size(27, input_width)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3s2_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3s2_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_3x3s2_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 3x3 stride-2 path, batched ****************************/

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3s2_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, 37)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, input_width)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3s2_with_input_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .input_channel_stride(21)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3s2_with_output_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .output_channel_stride(23)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3s2_with_qmin) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3s2_with_qmax) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_3x3s2_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 5x5 path ****************************/

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .input_size(input_height, 37)
      .kernel_size(5, 5)
      .padding(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .input_size(27, input_width)
      .kernel_size(5, 5)
      .padding(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 5x5 path, batched ****************************/

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, 37)
      .kernel_size(5, 5)
      .padding(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, input_width)
      .kernel_size(5, 5)
      .padding(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5_with_input_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .input_channel_stride(21)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5_with_output_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .output_channel_stride(23)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5_with_qmin) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5_with_qmax) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 5x5 stride-2 path ****************************/

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5s2_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .input_size(input_height, 37)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .input_size(27, input_width)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5s2_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5s2_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, depthwise_5x5s2_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 5x5 stride-2 path, batched ****************************/

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5s2_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_height = 25; input_height <= 31; input_height++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(input_height, 37)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t input_width = 27; input_width <= 37; input_width++) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, input_width)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5s2_with_input_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .input_channel_stride(21)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5s2_with_output_stride) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .output_channel_stride(23)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5s2_with_qmin) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5s2_with_qmax) {
  ConvolutionOperatorTester()
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(CONVOLUTION_NCHW_F32, batched_depthwise_5x5s2_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 3x3 path ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, kernel_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, kernel_3x3_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, kernel_3x3_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, weights_cache_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 3x3 path, batched ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, batched_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, batched_3x3_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, batched_3x3_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, weights_cache_batched_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .groups(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 3x3 stride-2 path ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, kernel_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, kernel_3x3s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, kernel_3x3s2_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 3x3 stride-2 path, batched ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, batched_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, batched_3x3s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(3, 3)
      .padding(1)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, batched_3x3s2_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(3, 3)
    .padding(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 5x5 path ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, kernel_5x5) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, kernel_5x5_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, kernel_5x5_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 5x5 path, batched ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, batched_5x5) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, batched_5x5_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, batched_5x5_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 5x5 stride-2 path ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, kernel_5x5s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, kernel_5x5s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, kernel_5x5s2_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

/**************************** DWCONV 5x5 stride-2 path, batched ****************************/

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, batched_5x5s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, batched_5x5s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionOperatorTester()
      .depthwise_layout(true)
      .batch_size(2)
      .input_size(27, 37)
      .kernel_size(5, 5)
      .padding(2)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestNCHWxF32();
  }
}

TEST(DEPTHWISE_CONVOLUTION_NCHW_F32, batched_5x5s2_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 37)
    .kernel_size(5, 5)
    .padding(2)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestNCHWxF32();
}
