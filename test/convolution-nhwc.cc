// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "convolution-operator-tester.h"


TEST(CONVOLUTION_NHWC_QU8, 1x1) {
  ConvolutionOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 1x1_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 1x1_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 1x1_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .input_channel_stride(28)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 1x1_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .output_channel_stride(29)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 1x1_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(13, 14)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 1x1_with_batch) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_size(13, 14)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_1x1) {
  ConvolutionOperatorTester()
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_1x1_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_1x1_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_1x1_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .input_channel_stride(37)
    .group_input_channels(17)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_1x1_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .output_channel_stride(41)
    .group_input_channels(17)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_1x1_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_1x1_with_batch) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 1x3) {
  ConvolutionOperatorTester()
    .input_size(20, 19)
    .padding_width(1)
    .kernel_size(1, 3)
    .group_input_channels(17)
    .group_output_channels(15)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_1x3) {
  ConvolutionOperatorTester()
    .input_size(20, 19)
    .padding_width(1)
    .kernel_size(1, 3)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(15)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x1) {
  ConvolutionOperatorTester()
    .input_size(19, 20)
    .padding_height(1)
    .kernel_size(3, 1)
    .group_input_channels(17)
    .group_output_channels(15)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_3x1) {
  ConvolutionOperatorTester()
    .input_size(19, 20)
    .padding_height(1)
    .kernel_size(3, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(15)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3_without_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3_with_left_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding_left(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3_with_right_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding_right(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3_with_top_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding_top(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3_with_bottom_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding_bottom(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding(1)
    .kernel_size(3, 3)
    .input_channel_stride(22)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding(1)
    .kernel_size(3, 3)
    .output_channel_stride(23)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(10, 9)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3_with_batch) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_size(10, 9)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_3x3) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_3x3_without_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_3x3_with_left_padding) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding_left(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_3x3_with_right_padding) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding_right(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_3x3_with_top_padding) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding_top(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_3x3_with_bottom_padding) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding_bottom(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_3x3_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .input_channel_stride(29)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_3x3_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .output_channel_stride(31)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_3x3_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(10, 11)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, grouped_3x3_with_batch) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_size(10, 11)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3s2) {
  ConvolutionOperatorTester()
    .input_size(14, 13)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3s2_with_tf_same_padding) {
  for (size_t input_height = 13; input_height <= 14; input_height++) {
    for (size_t input_width = 13; input_width <= 14; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(2)
        .group_input_channels(27)
        .group_output_channels(19)
        .iterations(3)
        .TestNHWCxQU8();
    }
  }
}

TEST(CONVOLUTION_NHWC_QU8, 3x3s1x2) {
  ConvolutionOperatorTester()
    .input_size(14, 13)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(1, 2)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3s1x2_with_tf_same_padding) {
  for (size_t input_height = 13; input_height <= 14; input_height++) {
    for (size_t input_width = 13; input_width <= 14; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(1, 2)
        .group_input_channels(27)
        .group_output_channels(19)
        .iterations(3)
        .TestNHWCxQU8();
    }
  }
}

TEST(CONVOLUTION_NHWC_QU8, 3x3s2x1) {
  ConvolutionOperatorTester()
    .input_size(14, 13)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2, 1)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3s2x1_with_tf_same_padding) {
  for (size_t input_height = 13; input_height <= 14; input_height++) {
    for (size_t input_width = 13; input_width <= 14; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(2, 1)
        .group_input_channels(27)
        .group_output_channels(19)
        .iterations(3)
        .TestNHWCxQU8();
    }
  }
}

TEST(CONVOLUTION_NHWC_QU8, 3x3d2) {
  ConvolutionOperatorTester()
    .input_size(14, 13)
    .padding(2)
    .kernel_size(3, 3)
    .dilation(2)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3d1x2) {
  ConvolutionOperatorTester()
    .input_size(14, 13)
    .padding(1, 2)
    .kernel_size(3, 3)
    .dilation(1, 2)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, 3x3d2x1) {
  ConvolutionOperatorTester()
    .input_size(14, 13)
    .padding(2, 1)
    .kernel_size(3, 3)
    .dilation(2, 1)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, depthwise_3x3) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .groups(27)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, depthwise_3x3_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .groups(27)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, depthwise_3x3s2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .subsampling(2)
    .groups(27)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, depthwise_3x3s2_with_tf_same_padding) {
  for (size_t input_height = 14; input_height <= 15; input_height++) {
    for (size_t input_width = 14; input_width <= 15; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(2)
        .groups(27)
        .iterations(3)
        .TestNHWCxQU8();
    }
  }
}

TEST(CONVOLUTION_NHWC_QU8, depthwise_3x3s1x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .subsampling(1, 2)
    .groups(27)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, depthwise_3x3s1x2_with_tf_same_padding) {
  for (size_t input_height = 14; input_height <= 15; input_height++) {
    for (size_t input_width = 14; input_width <= 15; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(1, 2)
        .groups(27)
        .iterations(3)
        .TestNHWCxQU8();
    }
  }
}

TEST(CONVOLUTION_NHWC_QU8, depthwise_3x3s2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .subsampling(2, 1)
    .groups(27)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, depthwise_3x3s2x1_with_tf_same_padding) {
  for (size_t input_height = 14; input_height <= 15; input_height++) {
    for (size_t input_width = 14; input_width <= 15; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(2, 1)
        .groups(27)
        .iterations(3)
        .TestNHWCxQU8();
    }
  }
}

TEST(CONVOLUTION_NHWC_QU8, depthwise_3x3d2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .dilation(2)
    .groups(27)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, depthwise_3x3d1x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .dilation(1, 2)
    .groups(27)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, depthwise_3x3d2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .dilation(2, 1)
    .groups(27)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_QU8, 1x1) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .kernel_size(1, 1)
    .groups(24)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_QU8, 1x1_with_depth_multiplier) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .kernel_size(1, 1)
    .groups(24)
    .group_output_channels(3)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_QU8, 1x1_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(15, 14)
    .kernel_size(1, 1)
    .groups(24)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_QU8, 3x3) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .groups(24)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_QU8, 3x3_with_depth_multiplier) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .groups(24)
    .group_output_channels(3)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_QU8, 3x3_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .groups(24)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_QU8, 3x3s2_with_tf_same_padding) {
  for (size_t input_height = 14; input_height <= 15; input_height++) {
    for (size_t input_width = 14; input_width <= 15; input_width++) {
      ConvolutionOperatorTester()
        .depthwise_layout(true)
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .groups(24)
        .iterations(3)
        .TestNHWCxQU8();
    }
  }
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_QU8, 5x5) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .groups(24)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_QU8, 5x5_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .groups(24)
    .iterations(3)
    .TestNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_changing_input_buffer) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_changing_input_buffer_grouped) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_changing_input_buffer_depthwise) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_increasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .next_batch_size(5)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_increasing_batch_grouped) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .next_batch_size(5)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_increasing_batch_depthwise) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .next_batch_size(5)
    .input_height(8)
    .input_width(8)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_decreasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(5)
    .next_batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_decreasing_batch_grouped) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(5)
    .next_batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_decreasing_batch_depthwise) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(5)
    .next_batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(9)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(7)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_changing_height_grouped) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(9)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(7)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_changing_height_depthwise) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(9)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxQU8();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(7)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(9)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(7)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_changing_width_grouped) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(9)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(7)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_changing_width_depthwise) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(9)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxQU8();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(7)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_swap_height_and_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(9)
    .input_width(8)
    .next_input_height(8)
    .next_input_width(9)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_swap_height_and_width_grouped) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(9)
    .input_width(8)
    .next_input_height(8)
    .next_input_width(9)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_QU8, setup_swap_height_and_width_depthwise) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(9)
    .input_width(8)
    .next_input_height(8)
    .next_input_width(9)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxQU8();
}

TEST(CONVOLUTION_NHWC_F32, 1x1) {
  ConvolutionOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 1x1_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 1x1_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 1x1_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .input_channel_stride(28)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 1x1_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .output_channel_stride(29)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 1x1_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(13, 14)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 1x1_with_batch) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_size(13, 14)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_1x1) {
  ConvolutionOperatorTester()
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_1x1_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_1x1_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_1x1_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .input_channel_stride(37)
    .group_input_channels(17)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_1x1_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .output_channel_stride(41)
    .group_input_channels(17)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_1x1_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_1x1_with_batch) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_1x1) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_1x1_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_1x1_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_1x1_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .input_channel_stride(28)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_1x1_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .output_channel_stride(29)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_1x1) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_1x1_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(13)
    .qmin(128)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_1x1_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(13)
    .qmax(128)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_1x1_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .groups(2)
    .input_channel_stride(37)
    .group_input_channels(17)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_1x1_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .groups(2)
    .output_channel_stride(41)
    .group_input_channels(17)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 1x3) {
  ConvolutionOperatorTester()
    .input_size(20, 19)
    .padding_width(1)
    .kernel_size(1, 3)
    .group_input_channels(17)
    .group_output_channels(15)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_1x3) {
  ConvolutionOperatorTester()
    .input_size(20, 19)
    .padding_width(1)
    .kernel_size(1, 3)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(15)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x1) {
  ConvolutionOperatorTester()
    .input_size(19, 20)
    .padding_height(1)
    .kernel_size(3, 1)
    .group_input_channels(17)
    .group_output_channels(15)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_3x1) {
  ConvolutionOperatorTester()
    .input_size(19, 20)
    .padding_height(1)
    .kernel_size(3, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(15)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3_without_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3_with_left_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding_left(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3_with_right_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding_right(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3_with_top_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding_top(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3_with_bottom_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding_bottom(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding(1)
    .kernel_size(3, 3)
    .input_channel_stride(22)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding(1)
    .kernel_size(3, 3)
    .output_channel_stride(23)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(10, 9)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3_with_batch) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_size(10, 9)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_3x3) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_3x3_without_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_3x3_with_left_padding) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding_left(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_3x3_with_right_padding) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding_right(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_3x3_with_top_padding) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding_top(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_3x3_with_bottom_padding) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding_bottom(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_3x3_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .input_channel_stride(29)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_3x3_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .output_channel_stride(31)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_3x3_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(10, 11)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, grouped_3x3_with_batch) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_size(10, 11)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3s2) {
  ConvolutionOperatorTester()
    .input_size(14, 13)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3s2_with_tf_same_padding) {
  for (size_t input_height = 13; input_height <= 14; input_height++) {
    for (size_t input_width = 13; input_width <= 14; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(2)
        .group_input_channels(27)
        .group_output_channels(19)
        .iterations(3)
        .TestNHWCxF32();
    }
  }
}

TEST(CONVOLUTION_NHWC_F32, 3x3s1x2) {
  ConvolutionOperatorTester()
    .input_size(14, 13)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(1, 2)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3s1x2_with_tf_same_padding) {
  for (size_t input_height = 13; input_height <= 14; input_height++) {
    for (size_t input_width = 13; input_width <= 14; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(1, 2)
        .group_input_channels(27)
        .group_output_channels(19)
        .iterations(3)
        .TestNHWCxF32();
    }
  }
}

TEST(CONVOLUTION_NHWC_F32, 3x3s2x1) {
  ConvolutionOperatorTester()
    .input_size(14, 13)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2, 1)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3s2x1_with_tf_same_padding) {
  for (size_t input_height = 13; input_height <= 14; input_height++) {
    for (size_t input_width = 13; input_width <= 14; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(2, 1)
        .group_input_channels(27)
        .group_output_channels(19)
        .iterations(3)
        .TestNHWCxF32();
    }
  }
}

TEST(CONVOLUTION_NHWC_F32, 3x3d2) {
  ConvolutionOperatorTester()
    .input_size(13, 14)
    .padding(2)
    .kernel_size(3, 3)
    .dilation(2)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3d1x2) {
  ConvolutionOperatorTester()
    .input_size(14, 15)
    .padding(1, 2)
    .kernel_size(3, 3)
    .dilation(1, 2)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, 3x3d2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 1)
    .kernel_size(3, 3)
    .dilation(2, 1)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_3x3) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_3x3_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_3x3_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_3x3_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .input_channel_stride(28)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_3x3_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .output_channel_stride(29)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_3x3) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_3x3_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(13)
    .qmin(128)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_3x3_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(13)
    .qmax(128)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_3x3_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .input_channel_stride(37)
    .group_input_channels(17)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_3x3_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .output_channel_stride(41)
    .group_input_channels(17)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_1x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .kernel_size(1, 1)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_1x1_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(15, 14)
    .kernel_size(1, 1)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2s2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .subsampling(2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2s1x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .subsampling(1, 2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2s2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .subsampling(2, 1)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2d2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .dilation(2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2d1x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .dilation(1, 2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2d2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .dilation(2, 1)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_3x3) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_3x3_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_3x3s2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .subsampling(2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_3x3s2_with_tf_same_padding) {
  for (size_t input_height = 14; input_height <= 15; input_height++) {
    for (size_t input_width = 14; input_width <= 15; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(2)
        .groups(27)
        .iterations(3)
        .TestNHWCxF32();
    }
  }
}

TEST(CONVOLUTION_NHWC_F32, depthwise_3x3s1x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .subsampling(1, 2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_3x3s1x2_with_tf_same_padding) {
  for (size_t input_height = 14; input_height <= 15; input_height++) {
    for (size_t input_width = 14; input_width <= 15; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(1, 2)
        .groups(27)
        .iterations(3)
        .TestNHWCxF32();
    }
  }
}

TEST(CONVOLUTION_NHWC_F32, depthwise_3x3s2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .subsampling(2, 1)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_3x3s2x1_with_tf_same_padding) {
  for (size_t input_height = 14; input_height <= 15; input_height++) {
    for (size_t input_width = 14; input_width <= 15; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(2, 1)
        .groups(27)
        .iterations(3)
        .TestNHWCxF32();
    }
  }
}

TEST(CONVOLUTION_NHWC_F32, depthwise_3x3d2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .dilation(2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_3x3d1x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .dilation(1, 2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_3x3d2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .dilation(2, 1)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5s2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .subsampling(2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5s1x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .subsampling(1, 2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5s2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .subsampling(2, 1)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5d2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .dilation(2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5d1x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .dilation(1, 2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5d2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .dilation(2, 1)
    .groups(27)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F32, 1x1) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .kernel_size(1, 1)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F32, 1x1_with_depth_multiplier) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .kernel_size(1, 1)
    .groups(24)
    .group_output_channels(3)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F32, 1x1_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(15, 14)
    .kernel_size(1, 1)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F32, 2x2) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F32, 2x2_with_depth_multiplier) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .groups(24)
    .group_output_channels(3)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F32, 2x2_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F32, 3x3) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F32, 3x3_with_depth_multiplier) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .groups(24)
    .group_output_channels(3)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F32, 3x3_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F32, 3x3s2_with_tf_same_padding) {
  for (size_t input_height = 14; input_height <= 15; input_height++) {
    for (size_t input_width = 14; input_width <= 15; input_width++) {
      ConvolutionOperatorTester()
        .depthwise_layout(true)
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .groups(24)
        .iterations(3)
        .TestNHWCxF32();
    }
  }
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F32, 5x5) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F32, 5x5_with_depth_multiplier) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .groups(24)
    .group_output_channels(3)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F32, 5x5_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F32, 7x7) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(3, 3)
    .kernel_size(7, 7)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F32, 7x7_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(15, 14)
    .padding(3, 3)
    .kernel_size(7, 7)
    .groups(24)
    .iterations(3)
    .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_changing_input_buffer) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_changing_input_buffer_grouped) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_changing_input_buffer_depthwise) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_increasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .next_batch_size(5)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_increasing_batch_grouped) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .next_batch_size(5)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_increasing_batch_depthwise) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .next_batch_size(5)
    .input_height(8)
    .input_width(8)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_decreasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(5)
    .next_batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_decreasing_batch_grouped) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(5)
    .next_batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_decreasing_batch_depthwise) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(5)
    .next_batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(9)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(7)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_changing_height_grouped) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(9)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(7)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_changing_height_depthwise) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(9)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF32();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(7)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(9)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(7)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_changing_width_grouped) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(9)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(7)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_changing_width_depthwise) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(9)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF32();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(7)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_swap_height_and_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(9)
    .input_width(8)
    .next_input_height(8)
    .next_input_width(9)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_swap_height_and_width_grouped) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(9)
    .input_width(8)
    .next_input_height(8)
    .next_input_width(9)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, setup_swap_height_and_width_depthwise) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(9)
    .input_width(8)
    .next_input_height(8)
    .next_input_width(9)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F16, 1x1) {
  ConvolutionOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 1x1_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 1x1_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 1x1_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .input_channel_stride(28)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 1x1_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(27, 29)
    .kernel_size(1, 1)
    .output_channel_stride(29)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 1x1_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(13, 14)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 1x1_with_batch) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_size(13, 14)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_1x1) {
  ConvolutionOperatorTester()
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_1x1_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_1x1_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_1x1_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .input_channel_stride(37)
    .group_input_channels(17)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_1x1_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .output_channel_stride(41)
    .group_input_channels(17)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_1x1_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_1x1_with_batch) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_size(24, 25)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_1x1) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_1x1_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_1x1_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_1x1_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .input_channel_stride(28)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_1x1_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .output_channel_stride(29)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_1x1) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_1x1_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(13)
    .qmin(128)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_1x1_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(13)
    .qmax(128)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_1x1_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .groups(2)
    .input_channel_stride(37)
    .group_input_channels(17)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_1x1_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(1, 1)
    .kernel_size(1, 1)
    .groups(2)
    .output_channel_stride(41)
    .group_input_channels(17)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 1x3) {
  ConvolutionOperatorTester()
    .input_size(20, 19)
    .padding_width(1)
    .kernel_size(1, 3)
    .group_input_channels(17)
    .group_output_channels(15)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_1x3) {
  ConvolutionOperatorTester()
    .input_size(20, 19)
    .padding_width(1)
    .kernel_size(1, 3)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(15)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x1) {
  ConvolutionOperatorTester()
    .input_size(19, 20)
    .padding_height(1)
    .kernel_size(3, 1)
    .group_input_channels(17)
    .group_output_channels(15)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_3x1) {
  ConvolutionOperatorTester()
    .input_size(19, 20)
    .padding_height(1)
    .kernel_size(3, 1)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(15)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3_without_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3_with_left_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding_left(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3_with_right_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding_right(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3_with_top_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding_top(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3_with_bottom_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding_bottom(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding(1)
    .kernel_size(3, 3)
    .input_channel_stride(22)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .padding(1)
    .kernel_size(3, 3)
    .output_channel_stride(23)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(10, 9)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3_with_batch) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_size(10, 9)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_3x3) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_3x3_without_padding) {
  ConvolutionOperatorTester()
    .input_size(13, 12)
    .kernel_size(3, 3)
    .group_input_channels(15)
    .group_output_channels(17)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_3x3_with_left_padding) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding_left(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_3x3_with_right_padding) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding_right(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_3x3_with_top_padding) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding_top(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_3x3_with_bottom_padding) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding_bottom(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_3x3_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .input_channel_stride(29)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_3x3_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(10, 11)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .output_channel_stride(31)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_3x3_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(10, 11)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_3x3_with_batch) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_size(10, 11)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(14)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3s2) {
  ConvolutionOperatorTester()
    .input_size(14, 13)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3s2_with_tf_same_padding) {
  for (size_t input_height = 13; input_height <= 14; input_height++) {
    for (size_t input_width = 13; input_width <= 14; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(2)
        .group_input_channels(27)
        .group_output_channels(19)
        .iterations(3)
        .TestNHWCxF16();
    }
  }
}

TEST(CONVOLUTION_NHWC_F16, 3x3s1x2) {
  ConvolutionOperatorTester()
    .input_size(14, 13)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(1, 2)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3s1x2_with_tf_same_padding) {
  for (size_t input_height = 13; input_height <= 14; input_height++) {
    for (size_t input_width = 13; input_width <= 14; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(1, 2)
        .group_input_channels(27)
        .group_output_channels(19)
        .iterations(3)
        .TestNHWCxF16();
    }
  }
}

TEST(CONVOLUTION_NHWC_F16, 3x3s2x1) {
  ConvolutionOperatorTester()
    .input_size(14, 13)
    .padding(1)
    .kernel_size(3, 3)
    .subsampling(2, 1)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3s2x1_with_tf_same_padding) {
  for (size_t input_height = 13; input_height <= 14; input_height++) {
    for (size_t input_width = 13; input_width <= 14; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(2, 1)
        .group_input_channels(27)
        .group_output_channels(19)
        .iterations(3)
        .TestNHWCxF16();
    }
  }
}

TEST(CONVOLUTION_NHWC_F16, 3x3d2) {
  ConvolutionOperatorTester()
    .input_size(13, 14)
    .padding(2)
    .kernel_size(3, 3)
    .dilation(2)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3d1x2) {
  ConvolutionOperatorTester()
    .input_size(14, 15)
    .padding(1, 2)
    .kernel_size(3, 3)
    .dilation(1, 2)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, 3x3d2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 1)
    .kernel_size(3, 3)
    .dilation(2, 1)
    .group_input_channels(27)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_3x3) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_3x3_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_3x3_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .group_input_channels(23)
    .group_output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_3x3_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .input_channel_stride(28)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_3x3_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .output_channel_stride(29)
    .group_input_channels(23)
    .group_output_channels(19)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_3x3) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_3x3_with_qmin) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(13)
    .qmin(128)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_3x3_with_qmax) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .group_input_channels(17)
    .group_output_channels(13)
    .qmax(128)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_3x3_with_input_stride) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .input_channel_stride(37)
    .group_input_channels(17)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_3x3_with_output_stride) {
  ConvolutionOperatorTester()
    .input_size(3, 3)
    .padding(1)
    .kernel_size(3, 3)
    .groups(2)
    .output_channel_stride(41)
    .group_input_channels(17)
    .group_output_channels(13)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_1x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .kernel_size(1, 1)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_1x1_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(15, 14)
    .kernel_size(1, 1)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2s2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .subsampling(2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2s1x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .subsampling(1, 2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2s2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .subsampling(2, 1)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2d2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .dilation(2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2d1x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .dilation(1, 2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2d2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .dilation(2, 1)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_3x3) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_3x3_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_3x3s2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .subsampling(2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_3x3s2_with_tf_same_padding) {
  for (size_t input_height = 14; input_height <= 15; input_height++) {
    for (size_t input_width = 14; input_width <= 15; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(2)
        .groups(27)
        .iterations(3)
        .TestNHWCxF16();
    }
  }
}

TEST(CONVOLUTION_NHWC_F16, depthwise_3x3s1x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .subsampling(1, 2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_3x3s1x2_with_tf_same_padding) {
  for (size_t input_height = 14; input_height <= 15; input_height++) {
    for (size_t input_width = 14; input_width <= 15; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(1, 2)
        .groups(27)
        .iterations(3)
        .TestNHWCxF16();
    }
  }
}

TEST(CONVOLUTION_NHWC_F16, depthwise_3x3s2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .subsampling(2, 1)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_3x3s2x1_with_tf_same_padding) {
  for (size_t input_height = 14; input_height <= 15; input_height++) {
    for (size_t input_width = 14; input_width <= 15; input_width++) {
      ConvolutionOperatorTester()
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .subsampling(2, 1)
        .groups(27)
        .iterations(3)
        .TestNHWCxF16();
    }
  }
}

TEST(CONVOLUTION_NHWC_F16, depthwise_3x3d2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .dilation(2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_3x3d1x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .dilation(1, 2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_3x3d2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .dilation(2, 1)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5_without_bias) {
  ConvolutionOperatorTester()
    .has_bias(false)
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5s2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .subsampling(2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5s1x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .subsampling(1, 2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5s2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .subsampling(2, 1)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5d2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .dilation(2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5d1x2) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .dilation(1, 2)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5d2x1) {
  ConvolutionOperatorTester()
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .dilation(2, 1)
    .groups(27)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, 1x1) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .kernel_size(1, 1)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, 1x1_with_depth_multiplier) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .kernel_size(1, 1)
    .groups(24)
    .group_output_channels(3)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, 1x1_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(15, 14)
    .kernel_size(1, 1)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, 2x2) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, 2x2_with_depth_multiplier) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .groups(24)
    .group_output_channels(3)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, 2x2_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(2, 2)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, 3x3) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, 3x3_with_depth_multiplier) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .groups(24)
    .group_output_channels(3)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, 3x3_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(15, 14)
    .padding(1, 1)
    .kernel_size(3, 3)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, 3x3s2_with_tf_same_padding) {
  for (size_t input_height = 14; input_height <= 15; input_height++) {
    for (size_t input_width = 14; input_width <= 15; input_width++) {
      ConvolutionOperatorTester()
        .depthwise_layout(true)
        .input_size(input_height, input_width)
        .padding_tf_same(true)
        .kernel_size(3, 3)
        .groups(24)
        .iterations(3)
        .TestNHWCxF16();
    }
  }
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, 5x5) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, 5x5_with_depth_multiplier) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .groups(24)
    .group_output_channels(3)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, 5x5_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(15, 14)
    .padding(2, 2)
    .kernel_size(5, 5)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, 7x7) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .input_size(15, 14)
    .padding(3, 3)
    .kernel_size(7, 7)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, 7x7_without_bias) {
  ConvolutionOperatorTester()
    .depthwise_layout(true)
    .has_bias(false)
    .input_size(15, 14)
    .padding(3, 3)
    .kernel_size(7, 7)
    .groups(24)
    .iterations(3)
    .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_changing_input_buffer) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_changing_input_buffer_grouped) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_changing_input_buffer_depthwise) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_increasing_batch) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .next_batch_size(5)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_increasing_batch_grouped) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .next_batch_size(5)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_increasing_batch_depthwise) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .next_batch_size(5)
    .input_height(8)
    .input_width(8)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_decreasing_batch) {
  ConvolutionOperatorTester()
    .batch_size(5)
    .next_batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_decreasing_batch_grouped) {
  ConvolutionOperatorTester()
    .batch_size(5)
    .next_batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_decreasing_batch_depthwise) {
  ConvolutionOperatorTester()
    .batch_size(5)
    .next_batch_size(3)
    .input_height(8)
    .input_width(8)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_changing_height) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(9)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(7)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_changing_height_grouped) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(9)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(7)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_changing_height_depthwise) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(9)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF16();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(7)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_changing_width) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(9)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(7)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_changing_width_grouped) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(9)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(7)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_changing_width_depthwise) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(9)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF16();
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(7)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_swap_height_and_width) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(9)
    .input_width(8)
    .next_input_height(8)
    .next_input_width(9)
    .kernel_height(5)
    .kernel_width(3)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_swap_height_and_width_grouped) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(9)
    .input_width(8)
    .next_input_height(8)
    .next_input_width(9)
    .kernel_height(5)
    .kernel_width(3)
    .groups(2)
    .group_input_channels(15)
    .group_output_channels(17)
    .TestSetupNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, setup_swap_height_and_width_depthwise) {
  ConvolutionOperatorTester()
    .batch_size(3)
    .input_height(9)
    .input_width(8)
    .next_input_height(8)
    .next_input_width(9)
    .kernel_height(3)
    .kernel_width(3)
    .groups(19)
    .group_input_channels(1)
    .group_output_channels(1)
    .TestSetupNHWCxF16();
}
