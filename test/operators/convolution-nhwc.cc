// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "test/operators/convolution-operator-tester.h"

namespace {

using ConvolutionTestCase =
    std::pair<const char*, std::vector<ConvolutionOperatorTester>>;

static const ConvolutionTestCase kConvolutionTests[] = {
    {"kernel_1x1",
     {ConvolutionOperatorTester()
          .input_size(27, 37)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(19)}},
    {"kernel_1x1_with_qmin",
     {ConvolutionOperatorTester()
          .input_size(27, 37)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(19)
          .qmin(128)}},
    {"kernel_1x1_with_qmax",
     {ConvolutionOperatorTester()
          .input_size(27, 37)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(19)
          .qmax(128)}},
    {"kernel_1x1_with_input_stride",
     {ConvolutionOperatorTester()
          .input_size(27, 37)
          .kernel_size(1, 1)
          .input_channel_stride(28)
          .group_input_channels(23)
          .group_output_channels(19)}},
    {"kernel_1x1_with_output_stride",
     {ConvolutionOperatorTester()
          .input_size(27, 37)
          .kernel_size(1, 1)
          .output_channel_stride(29)
          .group_input_channels(23)
          .group_output_channels(19)}},
    {"kernel_1x1_without_bias",
     {ConvolutionOperatorTester()
          .has_bias(false)
          .input_size(13, 14)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(19)}},
    {"kernel_1x1_with_batch",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_size(13, 14)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(19)}},
    {"grouped_1x1",
     {ConvolutionOperatorTester()
          .input_size(24, 25)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(19)}},
    {"grouped_1x1_with_qmin",
     {ConvolutionOperatorTester()
          .input_size(24, 25)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(19)
          .qmin(128)}},
    {"grouped_1x1_with_qmax",
     {ConvolutionOperatorTester()
          .input_size(24, 25)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(19)
          .qmax(128)}},
    {"grouped_1x1_with_input_stride",
     {ConvolutionOperatorTester()
          .input_size(24, 25)
          .kernel_size(1, 1)
          .groups(2)
          .input_channel_stride(37)
          .group_input_channels(17)
          .group_output_channels(19)}},
    {"grouped_1x1_with_output_stride",
     {ConvolutionOperatorTester()
          .input_size(24, 25)
          .kernel_size(1, 1)
          .groups(2)
          .output_channel_stride(41)
          .group_input_channels(17)
          .group_output_channels(19)}},
    {"grouped_1x1_without_bias",
     {ConvolutionOperatorTester()
          .has_bias(false)
          .input_size(24, 25)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(19)}},
    {"grouped_1x1_with_batch",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_size(24, 25)
          .kernel_size(1, 1)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(19)}},
    {"kernel_1x3",
     {ConvolutionOperatorTester()
          .input_size(20, 19)
          .padding_width(1)
          .kernel_size(1, 3)
          .group_input_channels(17)
          .group_output_channels(15)}},
    {"grouped_1x3",
     {ConvolutionOperatorTester()
          .input_size(20, 19)
          .padding_width(1)
          .kernel_size(1, 3)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(15)}},
    {"kernel_3x1",
     {ConvolutionOperatorTester()
          .input_size(19, 20)
          .padding_height(1)
          .kernel_size(3, 1)
          .group_input_channels(17)
          .group_output_channels(15)}},
    {"grouped_3x1",
     {ConvolutionOperatorTester()
          .input_size(19, 20)
          .padding_height(1)
          .kernel_size(3, 1)
          .groups(2)
          .group_input_channels(17)
          .group_output_channels(15)}},
    {"kernel_3x3",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_without_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_with_left_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding_left(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_with_right_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding_right(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_with_top_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding_top(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_with_bottom_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding_bottom(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_with_input_stride",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding(1)
          .kernel_size(3, 3)
          .input_channel_stride(22)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_with_output_stride",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding(1)
          .kernel_size(3, 3)
          .output_channel_stride(23)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_without_bias",
     {ConvolutionOperatorTester()
          .has_bias(false)
          .input_size(10, 9)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"kernel_3x3_with_batch",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_size(10, 9)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"grouped_3x3",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_transient_indirection",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)
          .transient_indirection_buffer(true)}},
    {"grouped_3x3_without_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"grouped_3x3_with_left_padding",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding_left(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_with_right_padding",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding_right(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_with_top_padding",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding_top(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_with_bottom_padding",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding_bottom(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_with_input_stride",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .input_channel_stride(29)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_with_output_stride",
     {ConvolutionOperatorTester()
          .input_size(10, 11)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .output_channel_stride(31)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_without_bias",
     {ConvolutionOperatorTester()
          .has_bias(false)
          .input_size(10, 11)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"grouped_3x3_with_batch",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_size(10, 11)
          .padding(1)
          .kernel_size(3, 3)
          .groups(2)
          .group_input_channels(14)
          .group_output_channels(13)}},
    {"kernel_3x3s2",
     {ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding(1)
          .kernel_size(3, 3)
          .subsampling(2)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"kernel_3x3s1x2",
     {ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding(1)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"kernel_3x3s2x1",
     {ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding(1)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"kernel_3x3d2",
     {ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding(2)
          .kernel_size(3, 3)
          .dilation(2)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"kernel_3x3d1x2",
     {ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding(1, 2)
          .kernel_size(3, 3)
          .dilation(1, 2)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"kernel_3x3d2x1",
     {ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding(2, 1)
          .kernel_size(3, 3)
          .dilation(2, 1)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"depthwise_2x5",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 5)
          .groups(27)}},
    {"depthwise_2x5_multithreaded",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 5)
          .groups(27)
          .multithreaded(true)}},
    {"depthwise_2x5s5x6",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 5)
          .subsampling(5, 6)
          .groups(27)}},
    {"depthwise_3x3",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .groups(27)}},
    {"depthwise_3x3_transient_indirection",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .groups(27)
          .transient_indirection_buffer(true)}},
    {"depthwise_3x3_without_bias",
     {ConvolutionOperatorTester()
          .has_bias(false)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .groups(27)}},
    {"depthwise_3x3s2",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .subsampling(2)
          .groups(27)}},
    {"depthwise_3x3s1x2",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .groups(27)}},
    {"depthwise_3x3s2x1",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .groups(27)}},
    {"depthwise_3x3d2",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .dilation(2)
          .groups(27)}},
    {"depthwise_3x3d1x2",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .dilation(1, 2)
          .groups(27)}},
    {"depthwise_3x3d2x1",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .dilation(2, 1)
          .groups(27)}},
    /* Tests GEMM microkernel with weights_cache.  */
    {"weights_cache_1x1",
     {ConvolutionOperatorTester()
          .input_size(27, 37)
          .kernel_size(1, 1)
          .group_input_channels(23)
          .group_output_channels(19)
          .use_weights_cache(true)}},
    /* Tests IGEMM microkernel with weights cache.  */
    {"weights_cache_3x3",
     {ConvolutionOperatorTester()
          .input_size(13, 12)
          .padding(1)
          .kernel_size(3, 3)
          .group_input_channels(15)
          .group_output_channels(17)
          .use_weights_cache(true)}},
    /* Tests vmulcaddc microkernel with weights cache.  */
    {"weights_cache_depthwise_1x1",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .kernel_size(1, 1)
          .groups(24)
          .use_weights_cache(true)}},
    /* Tests dwconv microkernel with weights cache.  */
    {"weights_cache_depthwise_2x2d2",
     {ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 2)
          .dilation(2)
          .groups(27)
          .use_weights_cache(true)}},
    {"kernel_3x3s2_with_tf_same_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 13)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(13, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(14, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"kernel_3x3s1x2_with_tf_same_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 13)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(13, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(14, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"kernel_3x3s2x1_with_tf_same_padding",
     {ConvolutionOperatorTester()
          .input_size(13, 13)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(13, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(14, 13)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .group_input_channels(27)
          .group_output_channels(19),
      ConvolutionOperatorTester()
          .input_size(14, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .group_input_channels(27)
          .group_output_channels(19)}},
    {"depthwise_3x3s2_with_tf_same_padding",
     {ConvolutionOperatorTester()
          .input_size(14, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(14, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(15, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2)
          .groups(27)}},
    {"depthwise_3x3s1x2_with_tf_same_padding",
     {ConvolutionOperatorTester()
          .input_size(14, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(14, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(15, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(1, 2)
          .groups(27)}},
    {"depthwise_3x3s2x1_with_tf_same_padding",
     {ConvolutionOperatorTester()
          .input_size(14, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(14, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(15, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .groups(27),
      ConvolutionOperatorTester()
          .input_size(15, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .subsampling(2, 1)
          .groups(27)}},
};

static const ConvolutionTestCase kDepthwiseConvolutionTests[] = {
    {"kernel_1x1",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .kernel_size(1, 1)
          .groups(24)}},
    {"kernel_1x1_with_depth_multiplier",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .kernel_size(1, 1)
          .groups(24)
          .group_output_channels(3)}},
    {"kernel_1x1_without_bias",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .has_bias(false)
          .input_size(15, 14)
          .kernel_size(1, 1)
          .groups(24)}},
    {"kernel_2x2",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 2)
          .groups(24)}},
    {"kernel_2x2_with_depth_multiplier",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 2)
          .groups(24)
          .group_output_channels(3)}},
    {"kernel_2x2_without_bias",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .has_bias(false)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 2)
          .groups(24)}},
    {"kernel_3x3",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .groups(24)}},
    {"kernel_3x3_with_depth_multiplier",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .groups(24)
          .group_output_channels(3)}},
    {"kernel_3x3_without_bias",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .has_bias(false)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(3, 3)
          .groups(24)}},
    {"kernel_5x5",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(2, 2)
          .kernel_size(5, 5)
          .groups(24)}},
    {"kernel_5x5_with_depth_multiplier",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(2, 2)
          .kernel_size(5, 5)
          .groups(24)
          .group_output_channels(3)}},
    {"kernel_5x5_without_bias",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .has_bias(false)
          .input_size(15, 14)
          .padding(2, 2)
          .kernel_size(5, 5)
          .groups(24)}},
    {"kernel_7x7",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(3, 3)
          .kernel_size(7, 7)
          .groups(24)}},
    {"kernel_7x7_without_bias",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .has_bias(false)
          .input_size(15, 14)
          .padding(3, 3)
          .kernel_size(7, 7)
          .groups(24)}},
    /* Tests dwconv microkernel with weights cache.  */
    {"weights_cache_1x1",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .kernel_size(1, 1)
          .groups(24)
          .use_weights_cache(true)}},
    /* Tests dwconv microkernel with non 1x1 kernel (dwconv_hwg packing). */
    {"weights_cache_2x2",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding(1, 1)
          .kernel_size(2, 2)
          .groups(24)
          .use_weights_cache(true)}},
    {"kernel_3x3s2_with_tf_same_padding",
     {ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(14, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .groups(24),
      ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(14, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .groups(24),
      ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 14)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .groups(24),
      ConvolutionOperatorTester()
          .depthwise_layout(true)
          .input_size(15, 15)
          .padding_tf_same(true)
          .kernel_size(3, 3)
          .groups(24)}},
};

static const ConvolutionTestCase kConvolutionSetupTests[] = {
    {"setup_changing_input_buffer",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_input_buffer_grouped",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_input_buffer_depthwise",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}},
    {"setup_increasing_batch",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .next_batch_size(5)
          .input_height(8)
          .input_width(8)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_increasing_batch_grouped",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .next_batch_size(5)
          .input_height(8)
          .input_width(8)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_increasing_batch_depthwise",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .next_batch_size(5)
          .input_height(8)
          .input_width(8)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}},
    {"setup_decreasing_batch",
     {ConvolutionOperatorTester()
          .batch_size(5)
          .next_batch_size(3)
          .input_height(8)
          .input_width(8)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_decreasing_batch_grouped",
     {ConvolutionOperatorTester()
          .batch_size(5)
          .next_batch_size(3)
          .input_height(8)
          .input_width(8)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_decreasing_batch_depthwise",
     {ConvolutionOperatorTester()
          .batch_size(5)
          .next_batch_size(3)
          .input_height(8)
          .input_width(8)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}},
    {"setup_changing_height_1",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_height(9)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_height_2",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_height(7)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_height_grouped_1",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_height(9)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_height_grouped_2",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_height(7)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_height_depthwise_1",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_height(9)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}},
    {"setup_changing_height_depthwise_2",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_height(7)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}},
    {"setup_changing_width_1",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_width(9)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_width_2",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_width(7)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_width_grouped_1",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_width(9)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_width_grouped_2",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_width(7)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_changing_width_depthwise_1",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_width(9)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}},
    {"setup_changing_width_depthwise_2",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(8)
          .input_width(8)
          .next_input_width(7)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}},
    {"setup_swap_height_and_width",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(9)
          .input_width(8)
          .next_input_height(8)
          .next_input_width(9)
          .kernel_height(5)
          .kernel_width(3)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_swap_height_and_width_grouped",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(9)
          .input_width(8)
          .next_input_height(8)
          .next_input_width(9)
          .kernel_height(5)
          .kernel_width(3)
          .groups(2)
          .group_input_channels(15)
          .group_output_channels(17)}},
    {"setup_swap_height_and_width_depthwise",
     {ConvolutionOperatorTester()
          .batch_size(3)
          .input_height(9)
          .input_width(8)
          .next_input_height(8)
          .next_input_width(9)
          .kernel_height(3)
          .kernel_width(3)
          .groups(19)
          .group_input_channels(1)
          .group_output_channels(1)}}};

#define CREATE_CONVOLUTION_TESTS(test_suite_name, test_fn)                \
  using test_suite_name = testing::TestWithParam<ConvolutionTestCase>;    \
  TEST_P(test_suite_name, ConvolutionTest) {                              \
    const ConvolutionTestCase& test_case = GetParam();                    \
    for (const ConvolutionOperatorTester& tester : test_case.second) {    \
      tester.test_fn();                                                   \
    }                                                                     \
  }                                                                       \
  INSTANTIATE_TEST_SUITE_P(                                               \
      test_suite_name, test_suite_name,                                   \
      testing::ValuesIn<ConvolutionTestCase>(kConvolutionTests),          \
      [](const testing::TestParamInfo<ConvolutionTestCase>& info) {       \
        return info.param.first;                                          \
      });                                                                 \
  INSTANTIATE_TEST_SUITE_P(                                               \
      DEPTHWISE_##test_suite_name, test_suite_name,                       \
      testing::ValuesIn<ConvolutionTestCase>(kDepthwiseConvolutionTests), \
      [](const testing::TestParamInfo<ConvolutionTestCase>& info) {       \
        return info.param.first;                                          \
      });

#define CREATE_CONVOLUTION_SETUP_TESTS(test_suite_name, setup_test_fn)         \
  using SETUP_##test_suite_name = testing::TestWithParam<ConvolutionTestCase>; \
  TEST_P(SETUP_##test_suite_name, ConvolutionSetupTest) {                      \
    const ConvolutionTestCase& test_case = GetParam();                         \
    for (const ConvolutionOperatorTester& tester : test_case.second) {         \
      tester.setup_test_fn();                                                  \
    }                                                                          \
  }                                                                            \
  INSTANTIATE_TEST_SUITE_P(                                                    \
      SETUP_##test_suite_name, SETUP_##test_suite_name,                        \
      testing::ValuesIn<ConvolutionTestCase>(kConvolutionSetupTests),          \
      [](const testing::TestParamInfo<ConvolutionTestCase>& info) {            \
        return info.param.first;                                               \
      });

CREATE_CONVOLUTION_TESTS(CONVOLUTION_NHWC_QC8, TestNHWCxQC8)
CREATE_CONVOLUTION_SETUP_TESTS(CONVOLUTION_NHWC_QC8, TestSetupNHWCxQC8)

CREATE_CONVOLUTION_TESTS(CONVOLUTION_NHWC_QD8_F16_QC8W, TestNHWCxQD8F16QC8W)

CREATE_CONVOLUTION_TESTS(CONVOLUTION_NHWC_QD8_F32_QC8W, TestNHWCxQD8F32QC8W)

CREATE_CONVOLUTION_TESTS(CONVOLUTION_NHWC_QS8, TestNHWCxQS8)
CREATE_CONVOLUTION_SETUP_TESTS(CONVOLUTION_NHWC_QS8, TestSetupNHWCxQS8)

CREATE_CONVOLUTION_TESTS(CONVOLUTION_NHWC_QU8, TestNHWCxQU8)
CREATE_CONVOLUTION_SETUP_TESTS(CONVOLUTION_NHWC_QU8, TestSetupNHWCxQU8)

CREATE_CONVOLUTION_TESTS(CONVOLUTION_NHWC_F32, TestNHWCxF32)
CREATE_CONVOLUTION_SETUP_TESTS(CONVOLUTION_NHWC_F32, TestSetupNHWCxF32)

CREATE_CONVOLUTION_TESTS(CONVOLUTION_NHWC_F16, TestNHWCxF16)
CREATE_CONVOLUTION_SETUP_TESTS(CONVOLUTION_NHWC_F16, TestSetupNHWCxF16)

TEST(CONVOLUTION_NHWC_F32, unioutput_1x1) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_1x1_with_qmin) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .qmin(128)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_1x1_with_qmax) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .qmax(128)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_1x1_with_input_stride) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .input_channel_stride(28)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_1x1_with_output_stride) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .output_channel_stride(29)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_grouped_1x1) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(13)
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
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, unioutput_3x3) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(19)
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
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_1x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .kernel_size(1, 1)
      .groups(24)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_1x1_without_bias) {
  ConvolutionOperatorTester()
      .has_bias(false)
      .input_size(15, 14)
      .kernel_size(1, 1)
      .groups(24)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .groups(24)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2_without_bias) {
  ConvolutionOperatorTester()
      .has_bias(false)
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .groups(24)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2s2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .subsampling(2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2s1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .subsampling(1, 2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2s2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .subsampling(2, 1)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2d2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .dilation(2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2d1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .dilation(1, 2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_2x2d2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .dilation(2, 1)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5_without_bias) {
  ConvolutionOperatorTester()
      .has_bias(false)
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5s2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .subsampling(2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5s1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .subsampling(1, 2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5s2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .subsampling(2, 1)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5d2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .dilation(2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5d1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .dilation(1, 2)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F32, depthwise_5x5d2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .dilation(2, 1)
      .groups(27)
      .TestNHWCxF32();
}

TEST(CONVOLUTION_NHWC_F16, grouped_1x1_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .input_size(24, 25)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(19)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_1x1) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_1x1_with_qmin) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .qmin(128)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_1x1_with_qmax) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .group_input_channels(23)
      .group_output_channels(19)
      .qmax(128)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_1x1_with_input_stride) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .input_channel_stride(28)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_1x1_with_output_stride) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .output_channel_stride(29)
      .group_input_channels(23)
      .group_output_channels(19)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_grouped_1x1) {
  ConvolutionOperatorTester()
      .input_size(1, 1)
      .kernel_size(1, 1)
      .groups(2)
      .group_input_channels(17)
      .group_output_channels(13)
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
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, kernel_3x3_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .input_size(13, 12)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(15)
      .group_output_channels(17)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, grouped_3x3_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .input_size(10, 11)
      .padding(1)
      .kernel_size(3, 3)
      .groups(2)
      .group_input_channels(14)
      .group_output_channels(13)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, unioutput_3x3) {
  ConvolutionOperatorTester()
      .input_size(3, 3)
      .padding(1)
      .kernel_size(3, 3)
      .group_input_channels(23)
      .group_output_channels(19)
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
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_1x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .kernel_size(1, 1)
      .groups(24)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_1x1_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .input_size(15, 14)
      .kernel_size(1, 1)
      .groups(24)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_1x1_without_bias) {
  ConvolutionOperatorTester()
      .has_bias(false)
      .input_size(15, 14)
      .kernel_size(1, 1)
      .groups(24)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .groups(24)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2_without_bias) {
  ConvolutionOperatorTester()
      .has_bias(false)
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .groups(24)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2s2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .subsampling(2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2s1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .subsampling(1, 2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2s2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .subsampling(2, 1)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2d2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .dilation(2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2d1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .dilation(1, 2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_2x2d2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(2, 2)
      .dilation(2, 1)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_3x3_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(3, 3)
      .groups(24)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5_without_bias) {
  ConvolutionOperatorTester()
      .has_bias(false)
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5s2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .subsampling(2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5s1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .subsampling(1, 2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5s2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .subsampling(2, 1)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5d2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .dilation(2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5d1x2) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .dilation(1, 2)
      .groups(27)
      .TestNHWCxF16();
}

TEST(CONVOLUTION_NHWC_F16, depthwise_5x5d2x1) {
  ConvolutionOperatorTester()
      .input_size(15, 14)
      .padding(2, 2)
      .kernel_size(5, 5)
      .dilation(2, 1)
      .groups(27)
      .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, kernel_1x1_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .depthwise_layout(true)
      .input_size(15, 14)
      .kernel_size(1, 1)
      .groups(24)
      .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16,
     kernel_1x1_with_depth_multiplier_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .depthwise_layout(true)
      .input_size(15, 14)
      .kernel_size(1, 1)
      .groups(24)
      .group_output_channels(3)
      .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16, kernel_3x3_weight_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .depthwise_layout(true)
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(3, 3)
      .groups(24)
      .TestNHWCxF16();
}

TEST(DEPTHWISE_CONVOLUTION_NHWC_F16,
     kernel_3x3_with_depth_multiplier_with_fp32_weights) {
  ConvolutionOperatorTester()
      .weights_type(ConvolutionOperatorTester::WeightsType::FP32)
      .depthwise_layout(true)
      .input_size(15, 14)
      .padding(1, 1)
      .kernel_size(3, 3)
      .groups(24)
      .group_output_channels(3)
      .TestNHWCxF16();
}
}  // namespace
