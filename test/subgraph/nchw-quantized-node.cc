// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "include/xnnpack.h"
#include "src/xnnpack/node-type.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

// Regression test for a trap in xnn_subgraph_rewrite_for_nchw().
//
// The pass that measures filter sparsity walks every node and skips only the
// ones whose *cluster* carries XNN_LAYOUT_FLAG_INCOMPATIBLE_CLUSTER. Unlike the
// two loops around it, it never checks whether the node itself is
// NCHW-compatible. A node whose Values are quantized gets no layout flags at all
// from xnn_check_nchw_compatibility(), and it leads its own cluster, so it
// reaches the `switch (filter->datatype)` further down - whose `default` arm is
// XNN_UNREACHABLE, i.e. undefined behaviour, and a trap in practice.
//
// Any mixed-precision model reaches this, as long as it also holds one
// NCHW-compatible chain: that chain is what sets `update`, which is the only
// thing keeping the pass from returning before the switch.
TEST(SUBGRAPH_NCHW, quantized_node_does_not_reach_unreachable) {
  SubgraphTester tester(11);
  const float filter_scales[4] = {0.5f, 0.5f, 0.5f, 0.5f};
  tester.AddDynamicTensorF32({1, 64, 64, 3}, 0)
      .AddStaticTensorF32({8, 3, 3, 3}, TensorType::kDense, 1)
      .AddStaticTensorF32({8}, TensorType::kDense, 2)
      .AddDynamicTensorF32({1, 32, 32, 8}, 3)
      // A 90%-sparse 1x1 filter, so the FP32 cluster stays profitable enough to
      // keep the sparsity scan running.
      .AddStaticTensorF32({4, 1, 1, 8}, TensorType::kSparse, 4)
      .AddStaticTensorF32({4}, TensorType::kDense, 5)
      .AddDynamicTensorF32({1, 32, 32, 4}, 6)
      .AddOutputTensorF32({1, 4}, 7)
      .AddConvolution2D(
          ConvolutionParams{Padding{1, 1, 1, 1}, Kernel{3, 3}, Subsampling{2, 2},
                            Dilation{1, 1}, /*groups=*/1,
                            /*group_input_channels=*/3,
                            /*group_output_channels=*/8},
          /*input_id=*/0, /*filter_id=*/1, /*bias_id=*/2, /*output_id=*/3)
      .AddConvolution2D(
          ConvolutionParams{Padding{0, 0, 0, 0}, Kernel{1, 1}, Subsampling{1, 1},
                            Dilation{1, 1}, /*groups=*/1,
                            /*group_input_channels=*/8,
                            /*group_output_channels=*/4},
          /*input_id=*/3, /*filter_id=*/4, /*bias_id=*/5, /*output_id=*/6)
      .AddGlobalAveragePooling(6, 7)
      // An unrelated quantized branch: a channelwise qint8 fully-connected node,
      // whose datatype the sparsity switch does not handle.
      .AddDynamicTensorQS8(/*zero_point=*/0, /*scale=*/1.0f, {1, 4, 4, 8}, 8)
      .AddStaticTensorQS8({4, 8}, /*channel_dim=*/0, TensorType::kDense,
                          /*scale=*/filter_scales, /*external_id=*/9)
      .AddOutputTensor({1, 4}, xnn_datatype_qint8, {0, 1.0f}, 10)
      .AddFullyConnected(/*output_min=*/-128.0f, /*output_max=*/127.0f,
                         /*input_id=*/8, /*filter_id=*/9,
                         /*bias_id=*/XNN_INVALID_VALUE_ID, /*output_id=*/10)
      // XNN_FLAG_HINT_SPARSE_INFERENCE is a documented public flag, and
      // xnn_create_runtime_v4() forwards it to xnn_subgraph_optimize(), which
      // calls xnn_subgraph_rewrite_for_nchw(). Before the fix this traps.
      .Optimize(XNN_FLAG_HINT_SPARSE_INFERENCE);

  // The quantized fully-connected node must not have been pulled into the
  // sparse cluster, which is what let it reach the unreachable arm.
  const struct xnn_node* fc_node = nullptr;
  for (int i = 0; i < tester.NumNodes(); i++) {
    const struct xnn_node* node = tester.Node(i);
    if (node->type == xnn_node_type_fully_connected &&
        node->num_inputs >= 2 &&
        tester.Value(node->inputs[0])->datatype == xnn_datatype_qint8) {
      fc_node = node;
    }
  }
  ASSERT_NE(nullptr, fc_node);
  EXPECT_EQ(0u, fc_node->layout_flags);
}

}  // namespace xnnpack