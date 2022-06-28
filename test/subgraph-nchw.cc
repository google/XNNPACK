// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include "subgraph-tester.h"
#include <gtest/gtest.h>

TEST(SUBGRAPH_NCHW, single_conv) {
  auto tester = SubgraphTester(4);
  tester
    .AddDynamicTensorF32({1, 256, 256, 3}, 0)
    .AddStaticTensorF32({32, 3, 3, 3}, kStaticDense, 1)
    .AddStaticTensorF32({32}, kStaticDense, 2)
    .AddDynamicTensorF32({1, 128, 128, 32}, 3)
    .AddConvolution2D(1, 1, 1, 1, 3, 3, 2, 2, 1, 1, 1, 3, 32, 0, 1, 2, 3)
    .Optimize()
    .Rewrite();

  ASSERT_EQ(tester.GetLayout(0), xnn_layout_type_nhwc);
  ASSERT_EQ(tester.GetLayout(3), xnn_layout_type_nhwc);
}

TEST(SUBGRAPH_NCHW, single_conv_and_global_average_pooling) {
  auto tester = SubgraphTester(5);
  tester
    .AddDynamicTensorF32({1, 256, 256, 3}, 0)
    .AddStaticTensorF32({32, 3, 3, 3}, kStaticDense, 1)
    .AddStaticTensorF32({32}, kStaticDense, 2)
    .AddDynamicTensorF32({1, 128, 128, 32}, 3)
    .AddDynamicTensorF32({32}, 4)
    .AddConvolution2D(1, 1, 1, 1, 3, 3, 2, 2, 1, 1, 1, 3, 32, 0, 1, 2, 3)
    .AddGlobalAveragePooling(3, 4)
    .Optimize()
    .Rewrite();

  ASSERT_EQ(tester.GetLayout(0), xnn_layout_type_nhwc);
  ASSERT_EQ(tester.GetLayout(3), xnn_layout_type_nhwc);
  ASSERT_EQ(tester.GetLayout(4), xnn_layout_type_nhwc);
}

TEST(SUBGRAPH_NCHW, pixelwise_conv_sandwich) {
  auto tester = SubgraphTester(8);
  tester
    .AddDynamicTensorF32({1, 256, 256, 3}, 0)
    .AddStaticTensorF32({8, 3, 3, 3}, kStaticDense, 1)
    .AddStaticTensorF32({8}, kStaticDense, 2)
    .AddDynamicTensorF32({1, 128, 128, 8}, 3)
    .AddStaticTensorF32({4, 1, 1, 8}, kStaticSparse, 4)
    .AddStaticTensorF32({4}, kStaticDense, 5)
    .AddDynamicTensorF32({1, 128, 128, 4}, 6)
    .AddDynamicTensorF32({1, 4}, 7)
    .AddConvolution2D(1, 1, 1, 1, 3, 3, 2, 2, 1, 1, 1, 3, 8, 0, 1, 2, 3)
    .AddConvolution2D(0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 8, 4, 3, 4, 5, 6)
    .AddGlobalAveragePooling(6, 7)
    .Optimize()
    .Rewrite();

  ASSERT_EQ(tester.GetLayout(0), xnn_layout_type_nhwc);
  ASSERT_EQ(tester.GetLayout(3), xnn_layout_type_nchw);
  ASSERT_EQ(tester.GetLayout(6), xnn_layout_type_nchw);
  ASSERT_EQ(tester.GetLayout(7), xnn_layout_type_nhwc);
}

TEST(SUBGRAPH_NCHW, bottleneck) {
  auto tester = SubgraphTester(15);
  tester
    .AddDynamicTensorF32({1, 256, 256, 3}, 0)
    .AddStaticTensorF32({8, 3, 3, 3}, kStaticDense, 1)
    .AddStaticTensorF32({8}, kStaticDense, 2)
    .AddDynamicTensorF32({1, 128, 128, 8}, 3)
    .AddStaticTensorF32({4, 1, 1, 8}, kStaticSparse, 4)
    .AddStaticTensorF32({4}, kStaticDense, 5)
    .AddDynamicTensorF32({1, 128, 128, 4}, 6)
    .AddStaticTensorF32({1, 3, 3, 4}, kStaticDense, 7)
    .AddStaticTensorF32({4}, kStaticDense, 8)
    .AddDynamicTensorF32({1, 128, 128, 4}, 9)
    .AddStaticTensorF32({8, 1, 1, 4}, kStaticSparse, 10)
    .AddStaticTensorF32({8}, kStaticDense, 11)
    .AddDynamicTensorF32({1, 128, 128, 8}, 12)
    .AddDynamicTensorF32({1, 128, 128, 8}, 13)
    .AddDynamicTensorF32({1, 128, 128, 8}, 13)
    .AddDynamicTensorF32({1, 8}, 14)
    .AddConvolution2D(1, 1, 1, 1, 3, 3, 2, 2, 1, 1, 1, 3, 8, 0, 1, 2, 3)
    .AddConvolution2D(0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 8, 4, 3, 4, 5, 6)
    .AddDepthwiseConv(1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 4, 6, 7, 8, 9)
    .AddConvolution2D(0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 8, 4, 9, 10, 11, 12)
    .AddAddition(3, 12, 13)
    .AddGlobalAveragePooling(13, 14)
    .Optimize()
    .Rewrite();

  ASSERT_EQ(tester.GetLayout(0), xnn_layout_type_nhwc);
  ASSERT_EQ(tester.GetLayout(3), xnn_layout_type_nchw);
  ASSERT_EQ(tester.GetLayout(6), xnn_layout_type_nchw);
  ASSERT_EQ(tester.GetLayout(9), xnn_layout_type_nchw);
  ASSERT_EQ(tester.GetLayout(12), xnn_layout_type_nchw);
  ASSERT_EQ(tester.GetLayout(13), xnn_layout_type_nchw);
  ASSERT_EQ(tester.GetLayout(14), xnn_layout_type_nhwc);
}
