// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include "subgraph-tester.h"
#include <gtest/gtest.h>

TEST(SUBGRAPH_NCHW, one_layer_model) {
  std::map<uint32_t, std::pair<xnn_layout_type, xnn_layout_type>>
      expected_layouts = {
          {0, {xnn_layout_type_nhwc, xnn_layout_type_nhwc}},
      };

  SubgraphTester(4)
      .add_tensor({1, 256, 256, 3}, kDynamic, 0)
      .add_tensor({32, 3, 3, 3}, kStatic, 1)
      .add_tensor({32}, kStatic, 2)
      .add_tensor({1, 128, 128, 32}, kDynamic, 3)
      .add_conv(1, 1, 1, 1, 3, 3, 2, 2, 1, 1, 1, 3, 32, 0, 1, 2, 3)
      .optimize()
      .rewrite()
      .CheckLayouts(expected_layouts);
}

TEST(SUBGRAPH_NCHW, two_layers_model) {
  std::map<uint32_t, std::pair<xnn_layout_type, xnn_layout_type>>
      expected_layouts = {
          {0, {xnn_layout_type_nhwc, xnn_layout_type_nchw}},
          {1, {xnn_layout_type_nchw, xnn_layout_type_nhwc}},
      };
  SubgraphTester(5)
      .add_tensor({1, 256, 256, 3}, kDynamic, 0)
      .add_tensor({32, 3, 3, 3}, kStatic, 1)
      .add_tensor({32}, kStatic, 2)
      .add_tensor({1, 128, 128, 32}, kDynamic, 3)
      .add_conv(1, 1, 1, 1, 3, 3, 2, 2, 1, 1, 1, 3, 32, 0, 1, 2, 3)
      .add_tensor({32}, kDynamic, 4)
      .add_global_average_pooling(3, 4)
      .optimize()
      .rewrite()
      .CheckLayouts(expected_layouts);
}

TEST(SUBGRAPH_NCHW, two_layers_with_bottleneck_model) {
  std::map<uint32_t, std::pair<xnn_layout_type, xnn_layout_type>>
      expected_layouts = {
          {0, {xnn_layout_type_nhwc, xnn_layout_type_nchw}},
          {1, {xnn_layout_type_nchw, xnn_layout_type_nchw}},
          {2, {xnn_layout_type_nchw, xnn_layout_type_nchw}},
          {3, {xnn_layout_type_nchw, xnn_layout_type_nhwc}},
      };
  SubgraphTester(9)
      .add_tensor({1, 256, 256, 3}, kDynamic, 0)
      .add_tensor({32, 3, 3, 3}, kStatic, 1)
      .add_tensor({32}, kStatic, 2)
      .add_tensor({1, 128, 128, 32}, kDynamic, 3)
      .add_conv(1, 1, 1, 1, 3, 3, 2, 2, 1, 1, 1, 3, 32, 0, 1, 2, 3)
      .add_tensor({1, 3, 3, 2}, kStatic, 4)
      .add_tensor({32}, kStatic, 5)
      .add_tensor({1, 128, 128, 32}, kDynamic, 6)
      .add_depthwise_conv(1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 32, 3, 4, 5, 6)
      .add_tensor({1, 128, 128, 32}, kDynamic, 7)
      .add_addition(3, 6, 7)
      .add_tensor({32}, kDynamic, 8)
      .add_global_average_pooling(7, 8)
      .optimize()
      .rewrite()
      .CheckLayouts(expected_layouts);
}
