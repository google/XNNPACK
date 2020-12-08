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
    .add_tensor({1, 256, 256, 3}, kDynamic, 0)
    .add_tensor({32, 3, 3, 3}, kStaticDense, 1)
    .add_tensor({32}, kStaticDense, 2)
    .add_tensor({1, 128, 128, 32}, kDynamic, 3)
    .add_conv(1, 1, 1, 1, 3, 3, 2, 2, 1, 1, 1, 3, 32, 0, 1, 2, 3)
    .optimize()
    .rewrite();

  ASSERT_EQ(tester.get_layout(0), xnn_layout_type_nhwc);
  ASSERT_EQ(tester.get_layout(3), xnn_layout_type_nhwc);
}

TEST(SUBGRAPH_NCHW, single_conv_and_global_average_pooling) {
  auto tester = SubgraphTester(5);
  tester
    .add_tensor({1, 256, 256, 3}, kDynamic, 0)
    .add_tensor({32, 3, 3, 3}, kStaticDense, 1)
    .add_tensor({32}, kStaticDense, 2)
    .add_tensor({1, 128, 128, 32}, kDynamic, 3)
    .add_tensor({32}, kDynamic, 4)
    .add_conv(1, 1, 1, 1, 3, 3, 2, 2, 1, 1, 1, 3, 32, 0, 1, 2, 3)
    .add_global_average_pooling(3, 4)
    .optimize()
    .rewrite();

  ASSERT_EQ(tester.get_layout(0), xnn_layout_type_nhwc);
  ASSERT_EQ(tester.get_layout(3), xnn_layout_type_nhwc);
  ASSERT_EQ(tester.get_layout(4), xnn_layout_type_nhwc);
}

TEST(SUBGRAPH_NCHW, pixelwise_conv_sandwich) {
  auto tester = SubgraphTester(8);
  tester
    .add_tensor({1, 256, 256, 3}, kDynamic, 0)
    .add_tensor({8, 3, 3, 3}, kStaticDense, 1)
    .add_tensor({8}, kStaticDense, 2)
    .add_tensor({1, 128, 128, 8}, kDynamic, 3)
    .add_tensor({4, 1, 1, 8}, kStaticSparse, 4)
    .add_tensor({4}, kStaticDense, 5)
    .add_tensor({1, 128, 128, 4}, kDynamic, 6)
    .add_tensor({1, 4}, kDynamic, 7)
    .add_conv(1, 1, 1, 1, 3, 3, 2, 2, 1, 1, 1, 3, 8, 0, 1, 2, 3)
    .add_conv(0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 8, 4, 3, 4, 5, 6)
    .add_global_average_pooling(6, 7)
    .optimize()
    .rewrite();

  ASSERT_EQ(tester.get_layout(0), xnn_layout_type_nhwc);
  ASSERT_EQ(tester.get_layout(3), xnn_layout_type_nchw);
  ASSERT_EQ(tester.get_layout(6), xnn_layout_type_nchw);
  ASSERT_EQ(tester.get_layout(7), xnn_layout_type_nhwc);
}

TEST(SUBGRAPH_NCHW, bottleneck) {
  auto tester = SubgraphTester(15);
  tester
    .add_tensor({1, 256, 256, 3}, kDynamic, 0)
    .add_tensor({8, 3, 3, 3}, kStaticDense, 1)
    .add_tensor({8}, kStaticDense, 2)
    .add_tensor({1, 128, 128, 8}, kDynamic, 3)
    .add_tensor({4, 1, 1, 8}, kStaticSparse, 4)
    .add_tensor({4}, kStaticDense, 5)
    .add_tensor({1, 128, 128, 4}, kDynamic, 6)
    .add_tensor({1, 3, 3, 4}, kStaticDense, 7)
    .add_tensor({4}, kStaticDense, 8)
    .add_tensor({1, 128, 128, 4}, kDynamic, 9)
    .add_tensor({8, 1, 1, 4}, kStaticSparse, 10)
    .add_tensor({8}, kStaticDense, 11)
    .add_tensor({1, 128, 128, 8}, kDynamic, 12)
    .add_tensor({1, 128, 128, 8}, kDynamic, 13)
    .add_tensor({1, 128, 128, 8}, kDynamic, 13)
    .add_tensor({1, 8}, kDynamic, 14)
    .add_conv(1, 1, 1, 1, 3, 3, 2, 2, 1, 1, 1, 3, 8, 0, 1, 2, 3)
    .add_conv(0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 8, 4, 3, 4, 5, 6)
    .add_depthwise_conv(1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 4, 6, 7, 8, 9)
    .add_conv(0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 8, 4, 9, 10, 11, 12)
    .add_addition(3, 12, 13)
    .add_global_average_pooling(13, 14)
    .optimize()
    .rewrite();

  ASSERT_EQ(tester.get_layout(0), xnn_layout_type_nhwc);
  ASSERT_EQ(tester.get_layout(3), xnn_layout_type_nchw);
  ASSERT_EQ(tester.get_layout(6), xnn_layout_type_nchw);
  ASSERT_EQ(tester.get_layout(9), xnn_layout_type_nchw);
  ASSERT_EQ(tester.get_layout(12), xnn_layout_type_nchw);
  ASSERT_EQ(tester.get_layout(13), xnn_layout_type_nchw);
  ASSERT_EQ(tester.get_layout(14), xnn_layout_type_nhwc);
}
