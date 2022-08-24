// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <array>

#include <xnnpack.h>
#include <xnnpack/node-type.h>

#include "subgraph-tester.h"
#include <gtest/gtest.h>

namespace xnnpack {

TEST(SUBGRAPH_FP16, value_both_external_output_and_input) {
  auto tester = SubgraphTester(4);
  std::array<size_t, 4> pre_paddings = {0,1,0,0};
  std::array<size_t, 4> post_paddings = {0,1,0,0};
  // external input[0]
  //      /
  // [constant pad]
  //     /
  //  external     dynamic[1]
  //  output[2]     /
  //           \   /
  //           [add]
  //             |
  //         external
  //         output[3]
  tester
      .AddInputTensorF32({1, 2, 2, 3}, 0)
      .AddDynamicTensorF32({1, 1, 1, 3}, 1)
      .AddOutputTensorF32({1, 4, 2, 3}, 2)
      .AddOutputTensorF32({1, 4, 2, 3}, 3)
      .AddConstantPad(pre_paddings.data(), post_paddings.data(), 0.0f, 0, 2)
      .AddAddition(2, 1, 3)
      .Optimize()
      .RewriteForFp16();

  // After rewriting for FP16, the graph should look like this, with * indicating new operators and values created:
  //
  //   external input[0]
  //        |
  //    [convert]*
  //        |
  //     input[4]*
  //       /
  // [constant pad]
  //     /
  //   fp16 value[5]*
  //    |       \
  //  [convert]* \
  //    |         \
  //  external     \    dynamic[1] converted in-place
  //  output[2]     \     /
  //                 \   /
  //                 [add]
  //                   |
  //                fp16 value[6]*
  //                   |
  //                [convert]*
  //                   |
  //               external
  //               output[3]

  // We should have 3 convert nodes, one for external input, 2 for external
  // outputs, so 5 in total, including the pad and add in the original graph.
  ASSERT_EQ(tester.NumNodes(), 5);

  const xnn_node* output_convert_node = tester.Node(4);
  ASSERT_EQ(output_convert_node->type, xnn_node_type_convert);
  ASSERT_EQ(output_convert_node->compute_type, xnn_compute_type_fp16_to_fp32);

  // Check that Addition node refers to the FP16 value before conversion.
  const xnn_node* addition_node = tester.Node(3);
  ASSERT_EQ(addition_node->type, xnn_node_type_add2);
  ASSERT_EQ(addition_node->inputs[0], 5);
  ASSERT_EQ(addition_node->inputs[1], 1);
  ASSERT_EQ(tester.Value(5)->datatype, xnn_datatype_fp16);
  ASSERT_EQ(tester.Value(1)->datatype, xnn_datatype_fp16);

  ASSERT_EQ(tester.Node(2)->type, xnn_node_type_convert);
  ASSERT_EQ(tester.Node(2)->compute_type, xnn_compute_type_fp16_to_fp32);
  ASSERT_EQ(tester.Node(1)->type, xnn_node_type_static_constant_pad);
  ASSERT_EQ(tester.Node(0)->type, xnn_node_type_convert);
  ASSERT_EQ(tester.Node(0)->compute_type, xnn_compute_type_fp32_to_fp16);
}

}  // namespace xnnpack
