// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "convolution-spnchw-operator-tester.h"


/**************************** SPMM path ****************************/

TEST(CONVOLUTION_SpNHWC_OP_F32, 1x1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  ConvolutionSpNCHWOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, 1x1_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  ConvolutionSpNCHWOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, 1x1_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_width = 25; input_width <= 31; input_width++) {
    ConvolutionSpNCHWOperatorTester()
      .input_size(input_width, 29)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, 1x1_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_height = 27; input_height <= 33; input_height++) {
    ConvolutionSpNCHWOperatorTester()
      .input_size(27, input_height)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, 1x1_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    ConvolutionSpNCHWOperatorTester()
      .input_size(27, 29)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(19)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, 1x1_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t output_channels = 1; output_channels < 19; output_channels *= 2) {
    ConvolutionSpNCHWOperatorTester()
      .input_size(27, 29)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, 1x1_with_qmin) {
  ConvolutionSpNCHWOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, 1x1_with_qmax) {
  ConvolutionSpNCHWOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, 1x1_without_bias) {
  ConvolutionSpNCHWOperatorTester()
    .has_bias(false)
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestF32();
}

/**************************** SPMM path, batched ****************************/

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_1x1) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_1x1_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_1x1_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_width = 25; input_width <= 31; input_width++) {
    ConvolutionSpNCHWOperatorTester()
      .batch_size(2)
      .input_size(input_width, 29)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_1x1_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_height = 27; input_height <= 33; input_height++) {
    ConvolutionSpNCHWOperatorTester()
      .batch_size(2)
      .input_size(27, input_height)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_1x1_varying_input_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_channels = 1; input_channels <= 16; input_channels *= 4) {
    ConvolutionSpNCHWOperatorTester()
      .batch_size(2)
      .input_size(27, 29)
      .kernel_size(1, 1)
      .group_input_channels(input_channels)
      .group_output_channels(19)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_1x1_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t output_channels = 1; output_channels < 19; output_channels *= 2) {
    ConvolutionSpNCHWOperatorTester()
      .batch_size(2)
      .input_size(27, 29)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(output_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_1x1_with_input_stride) {
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(1, 1)
    .input_batch_stride(18013)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_1x1_with_output_stride) {
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(1, 1)
    .output_batch_stride(14879)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_1x1_with_qmin) {
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_1x1_with_qmax) {
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_1x1_without_bias) {
  ConvolutionSpNCHWOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestF32();
}

/**************************** DConv 3x3c3s2 HWC->SpCHW path ****************************/

TEST(CONVOLUTION_HWC2SpNHWC_OP_F32, 3x3c3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  ConvolutionSpNCHWOperatorTester()
    .input_size(27, 29)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .nhwc_input(true)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_HWC2SpNHWC_OP_F32, 3x3c3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_width = 25; input_width <= 31; input_width++) {
    ConvolutionSpNCHWOperatorTester()
      .input_size(input_width, 29)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(19)
      .nhwc_input(true)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_HWC2SpNHWC_OP_F32, 3x3c3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_height = 27; input_height <= 33; input_height++) {
    ConvolutionSpNCHWOperatorTester()
      .input_size(27, input_height)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(19)
      .nhwc_input(true)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_HWC2SpNHWC_OP_F32, 3x3c3s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t output_channels = 1; output_channels < 19; output_channels *= 2) {
    ConvolutionSpNCHWOperatorTester()
      .input_size(27, 29)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(output_channels)
      .nhwc_input(true)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_HWC2SpNHWC_OP_F32, 3x3c3s2_with_qmin) {
  ConvolutionSpNCHWOperatorTester()
    .input_size(27, 29)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .nhwc_input(true)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_HWC2SpNHWC_OP_F32, 3x3c3s2_with_qmax) {
  ConvolutionSpNCHWOperatorTester()
    .input_size(27, 29)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .nhwc_input(true)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_HWC2SpNHWC_OP_F32, 3x3c3s2_without_bias) {
  ConvolutionSpNCHWOperatorTester()
    .has_bias(false)
    .input_size(27, 29)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .nhwc_input(true)
    .iterations(3)
    .TestF32();
}

/**************************** DConv 3x3c3s2 HWC->SpCHW path, batched ****************************/

TEST(CONVOLUTION_HWC2SpNHWC_OP_F32, batched_3x3c3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .nhwc_input(true)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_HWC2SpNHWC_OP_F32, batched_3x3c3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_width = 25; input_width <= 31; input_width++) {
    ConvolutionSpNCHWOperatorTester()
      .batch_size(2)
      .input_size(input_width, 29)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(19)
      .nhwc_input(true)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_HWC2SpNHWC_OP_F32, batched_3x3c3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_height = 27; input_height <= 33; input_height++) {
    ConvolutionSpNCHWOperatorTester()
      .batch_size(2)
      .input_size(27, input_height)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(19)
      .nhwc_input(true)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_HWC2SpNHWC_OP_F32, batched_3x3c3s2_varying_output_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t output_channels = 1; output_channels < 19; output_channels *= 2) {
    ConvolutionSpNCHWOperatorTester()
      .batch_size(2)
      .input_size(27, 29)
      .padding(1)
      .kernel_size(3, 3)
      .subsampling(2)
      .group_input_channels(3)
      .group_output_channels(output_channels)
      .nhwc_input(true)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_HWC2SpNHWC_OP_F32, batched_3x3c3s2_with_output_stride) {
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .output_batch_stride(4001)
    .group_input_channels(3)
    .group_output_channels(19)
    .nhwc_input(true)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_HWC2SpNHWC_OP_F32, batched_3x3c3s2_with_qmin) {
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .nhwc_input(true)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_HWC2SpNHWC_OP_F32, batched_3x3c3s2_with_qmax) {
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .nhwc_input(true)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_HWC2SpNHWC_OP_F32, batched_3x3c3s2_without_bias) {
  ConvolutionSpNCHWOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 29)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(3)
    .group_output_channels(19)
    .nhwc_input(true)
    .iterations(3)
    .TestF32();
}

/**************************** DWCONV 3x3 path ****************************/

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  ConvolutionSpNCHWOperatorTester()
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .groups(19)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  ConvolutionSpNCHWOperatorTester()
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_width = 25; input_width <= 31; input_width++) {
    ConvolutionSpNCHWOperatorTester()
      .input_size(input_width, 29)
      .kernel_size(3, 3)
      .padding_width(1)
      .groups(19)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_height = 27; input_height <= 33; input_height++) {
    ConvolutionSpNCHWOperatorTester()
      .input_size(27, input_height)
      .kernel_size(3, 3)
      .padding_width(1)
      .groups(19)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionSpNCHWOperatorTester()
      .input_size(27, 29)
      .kernel_size(3, 3)
      .padding_width(1)
      .groups(channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3_with_qmin) {
  ConvolutionSpNCHWOperatorTester()
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3_with_qmax) {
  ConvolutionSpNCHWOperatorTester()
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3_without_bias) {
  ConvolutionSpNCHWOperatorTester()
    .has_bias(false)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .groups(19)
    .iterations(3)
    .TestF32();
}

/**************************** DWCONV 3x3 path, batched ****************************/

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .groups(19)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_width = 25; input_width <= 31; input_width++) {
    ConvolutionSpNCHWOperatorTester()
      .batch_size(2)
      .input_size(input_width, 29)
      .kernel_size(3, 3)
      .padding_width(1)
      .groups(19)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_height = 27; input_height <= 33; input_height++) {
    ConvolutionSpNCHWOperatorTester()
      .batch_size(2)
      .input_size(27, input_height)
      .kernel_size(3, 3)
      .padding_width(1)
      .groups(19)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionSpNCHWOperatorTester()
      .batch_size(2)
      .input_size(27, 29)
      .kernel_size(3, 3)
      .padding_width(1)
      .groups(channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3_with_input_stride) {
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .input_batch_stride(14879)
    .groups(19)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3_with_output_stride) {
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .output_batch_stride(13781)
    .groups(19)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3_with_qmin) {
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3_with_qmax) {
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3_without_bias) {
  ConvolutionSpNCHWOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .groups(19)
    .iterations(3)
    .TestF32();
}

/**************************** DWCONV 3x3 stride-2 path ****************************/

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  ConvolutionSpNCHWOperatorTester()
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3s2_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  ConvolutionSpNCHWOperatorTester()
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .subsampling(2)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_width = 25; input_width <= 31; input_width++) {
    ConvolutionSpNCHWOperatorTester()
      .input_size(input_width, 29)
      .kernel_size(3, 3)
      .padding_width(1)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_height = 27; input_height <= 33; input_height++) {
    ConvolutionSpNCHWOperatorTester()
      .input_size(27, input_height)
      .kernel_size(3, 3)
      .padding_width(1)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionSpNCHWOperatorTester()
      .input_size(27, 29)
      .kernel_size(3, 3)
      .padding_width(1)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3s2_with_qmin) {
  ConvolutionSpNCHWOperatorTester()
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .subsampling(2)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3s2_with_qmax) {
  ConvolutionSpNCHWOperatorTester()
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .subsampling(2)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, depthwise_3x3s2_without_bias) {
  ConvolutionSpNCHWOperatorTester()
    .has_bias(false)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestF32();
}

/**************************** DWCONV 3x3 stride-2 path, batched ****************************/

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3s2) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3s2_zero_weights) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .subsampling(2)
    .groups(19)
    .sparsity(1.0f)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3s2_varying_input_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_width = 25; input_width <= 31; input_width++) {
    ConvolutionSpNCHWOperatorTester()
      .batch_size(2)
      .input_size(input_width, 29)
      .kernel_size(3, 3)
      .padding_width(1)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3s2_varying_input_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t input_height = 27; input_height <= 33; input_height++) {
    ConvolutionSpNCHWOperatorTester()
      .batch_size(2)
      .input_size(27, input_height)
      .kernel_size(3, 3)
      .padding_width(1)
      .subsampling(2)
      .groups(19)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3s2_varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t channels = 1; channels <= 16; channels *= 4) {
    ConvolutionSpNCHWOperatorTester()
      .batch_size(2)
      .input_size(27, 29)
      .kernel_size(3, 3)
      .padding_width(1)
      .subsampling(2)
      .groups(channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3s2_with_input_stride) {
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .subsampling(2)
    .input_batch_stride(14879)
    .groups(19)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3s2_with_output_stride) {
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .subsampling(2)
    .output_batch_stride(13781)
    .groups(19)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3s2_with_qmin) {
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .subsampling(2)
    .groups(19)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3s2_with_qmax) {
  ConvolutionSpNCHWOperatorTester()
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .subsampling(2)
    .groups(19)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(CONVOLUTION_SpNHWC_OP_F32, batched_depthwise_3x3s2_without_bias) {
  ConvolutionSpNCHWOperatorTester()
    .has_bias(false)
    .batch_size(2)
    .input_size(27, 29)
    .kernel_size(3, 3)
    .padding_width(1)
    .subsampling(2)
    .groups(19)
    .iterations(3)
    .TestF32();
}
