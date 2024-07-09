// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include "xnnpack/subgraph.h"
#include "subgraph-tester.h"

namespace xnnpack {

TEST(SUBGRAPH_NCHW, single_conv) {
  SubgraphTester tester(4);
  tester
    .AddDynamicTensorF32({1, 256, 256, 3}, 0)
    .AddStaticTensorF32({32, 3, 3, 3}, TensorType::kDense, 1)
    .AddStaticTensorF32({32}, TensorType::kDense, 2)
    .AddDynamicTensorF32({1, 128, 128, 32}, 3)
    .AddConvolution2D(
        ConvolutionParams{
          Padding{1, 1, 1, 1},
          Kernel{3, 3},
          Subsampling{2, 2},
          Dilation{1, 1},
          /*groups=*/ 1,
          /*group_input_channels=*/ 3,
          /*group_output_channels=*/ 32,
        }, 0, 1, 2, 3)
    .Optimize()
    .RewriteForNchw();

  ASSERT_EQ(tester.GetLayout(0), xnn_layout_type_nhwc);
  ASSERT_EQ(tester.GetLayout(3), xnn_layout_type_nhwc);
}

TEST(SUBGRAPH_NCHW, single_conv_and_global_average_pooling) {
  SubgraphTester tester(5);
  tester
    .AddDynamicTensorF32({1, 256, 256, 3}, 0)
    .AddStaticTensorF32({32, 3, 3, 3}, TensorType::kDense, 1)
    .AddStaticTensorF32({32}, TensorType::kDense, 2)
    .AddDynamicTensorF32({1, 128, 128, 32}, 3)
    .AddOutputTensorF32({32}, 4)
    .AddConvolution2D(
        ConvolutionParams{
          Padding{1, 1, 1, 1},
          Kernel{3, 3},
          Subsampling{2, 2},
          Dilation{1, 1},
          /*groups=*/ 1,
          /*group_input_channels=*/ 3,
          /*group_output_channels=*/ 32,
        }, 0, 1, 2, 3)
    .AddGlobalAveragePooling(3, 4)
    .Optimize()
    .RewriteForNchw();

  ASSERT_EQ(tester.GetLayout(0), xnn_layout_type_nhwc);
  ASSERT_EQ(tester.GetLayout(3), xnn_layout_type_nhwc);
  ASSERT_EQ(tester.GetLayout(4), xnn_layout_type_nhwc);
}

TEST(SUBGRAPH_NCHW, pixelwise_conv_sandwich) {
  SubgraphTester tester(8);
  tester
    .AddDynamicTensorF32({1, 256, 256, 3}, 0)
    .AddStaticTensorF32({8, 3, 3, 3}, TensorType::kDense, 1)
    .AddStaticTensorF32({8}, TensorType::kDense, 2)
    .AddDynamicTensorF32({1, 128, 128, 8}, 3)
    .AddStaticTensorF32({4, 1, 1, 8}, TensorType::kSparse, 4)
    .AddStaticTensorF32({4}, TensorType::kDense, 5)
    .AddDynamicTensorF32({1, 128, 128, 4}, 6)
    .AddOutputTensorF32({1, 4}, 7)
    .AddConvolution2D(
        ConvolutionParams{
          Padding{1, 1, 1, 1},
          Kernel{3, 3},
          Subsampling{2, 2},
          Dilation{1, 1},
          /*groups=*/ 1,
          /*group_input_channels=*/ 3,
          /*group_output_channels=*/ 8
        }, 0, 1, 2, 3)
    .AddConvolution2D(
        ConvolutionParams{
          Padding{0, 0, 0, 0},
          Kernel{1, 1},
          Subsampling{1, 1},
          Dilation{1, 1},
          /*groups=*/ 1,
          /*group_input_channels=*/ 8,
          /*group_output_channels=*/ 4
        }, 3, 4, 5, 6)
    .AddGlobalAveragePooling(6, 7)
    .Optimize()
    .RewriteForNchw();

  ASSERT_EQ(tester.GetLayout(0), xnn_layout_type_nhwc);
  ASSERT_EQ(tester.GetLayout(3), xnn_layout_type_nchw);
  ASSERT_EQ(tester.GetLayout(6), xnn_layout_type_nchw);
  ASSERT_EQ(tester.GetLayout(7), xnn_layout_type_nhwc);
}

TEST(SUBGRAPH_NCHW, bottleneck) {
  SubgraphTester tester(15);
  tester
    .AddDynamicTensorF32({1, 256, 256, 3}, 0)
    .AddStaticTensorF32({8, 3, 3, 3}, TensorType::kDense, 1)
    .AddStaticTensorF32({8}, TensorType::kDense, 2)
    .AddDynamicTensorF32({1, 128, 128, 8}, 3)
    .AddStaticTensorF32({4, 1, 1, 8}, TensorType::kSparse, 4)
    .AddStaticTensorF32({4}, TensorType::kDense, 5)
    .AddDynamicTensorF32({1, 128, 128, 4}, 6)
    .AddStaticTensorF32({1, 3, 3, 4}, TensorType::kDense, 7)
    .AddStaticTensorF32({4}, TensorType::kDense, 8)
    .AddDynamicTensorF32({1, 128, 128, 4}, 9)
    .AddStaticTensorF32({4, 1, 1, 4}, TensorType::kSparse, 10)
    .AddStaticTensorF32({8}, TensorType::kDense, 11)
    .AddDynamicTensorF32({1, 128, 128, 8}, 12)
    .AddDynamicTensorF32({1, 128, 128, 8}, 13)
    .AddDynamicTensorF32({1, 128, 128, 8}, 13)
    .AddOutputTensorF32({1, 8}, 14)
    .AddConvolution2D(
        ConvolutionParams{
          Padding{1, 1, 1, 1},
          Kernel{3, 3},
          Subsampling{2, 2},
          Dilation{1, 1},
          /*groups=*/ 1,
          /*group_input_channels=*/ 3,
          /*group_output_channels=*/ 8
        }, 0, 1, 2, 3)
    .AddConvolution2D(
        ConvolutionParams{
          Padding{0, 0, 0, 0},
          Kernel{1, 1},
          Subsampling{1, 1},
          Dilation{1, 1},
          /*groups=*/ 1,
          /*group_input_channels=*/ 8,
          /*group_output_channels=*/ 4
        }, 3, 4, 5, 6)
    .AddDepthwiseConvolution2D(
        DepthwiseConvolutionParams{
          Padding{1, 1, 1, 1},
          Kernel{3, 3},
          Subsampling{1, 1},
          Dilation{1, 1},
          /*depth_multiplier=*/ 1,
          /*input_channels=*/ 4
        }, 6, 7, 8, 9)
    .AddConvolution2D(
        ConvolutionParams{
          Padding{0, 0, 0, 0},
          Kernel{1, 1},
          Subsampling{1, 1},
          Dilation{1, 1},
          /*groups=*/ 1,
          /*group_input_channels=*/ 8,
          /*group_output_channels=*/ 4
        }, 9, 10, 11, 12)
    .AddAddition(3, 12, 13)
    .AddGlobalAveragePooling(13, 14)
    .Optimize()
    .RewriteForNchw();

  ASSERT_EQ(tester.GetLayout(0), xnn_layout_type_nhwc);
  ASSERT_EQ(tester.GetLayout(3), xnn_layout_type_nchw);
  ASSERT_EQ(tester.GetLayout(6), xnn_layout_type_nchw);
  ASSERT_EQ(tester.GetLayout(9), xnn_layout_type_nchw);
  ASSERT_EQ(tester.GetLayout(12), xnn_layout_type_nchw);
  ASSERT_EQ(tester.GetLayout(13), xnn_layout_type_nchw);
  ASSERT_EQ(tester.GetLayout(14), xnn_layout_type_nhwc);
}

}  // namespace xnnpack
