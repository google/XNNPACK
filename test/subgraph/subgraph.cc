// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/subgraph.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "test/subgraph/runtime-tester.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

TEST(SUBGRAPH, hanging_nodes) {
  SubgraphTester tester(6);
  tester.AddDynamicTensorF32({1, 256, 256, 3}, 0)
      .AddStaticTensorF32({32, 3, 3, 3}, TensorType::kDense, 1)
      .AddStaticTensorF32({32}, TensorType::kDense, 2)
      .AddDynamicTensorF32({1, 128, 128, 32}, 3)
      .AddOutputTensorF32({32}, 4)
      .AddDynamicTensorF32({32}, 5)
      .AddConvolution2D(
          ConvolutionParams{
              Padding{1, 1, 1, 1},
              Kernel{3, 3},
              Subsampling{2, 2},
              Dilation{1, 1},
              /*groups=*/1,
              /*group_input_channels=*/3,
              /*group_output_channels=*/32,
          },
          0, 1, 2, 3)
      .AddGlobalAveragePooling(3, 4)
      // Add hanging node
      .AddGlobalAveragePooling(3, 5)
      .Optimize();

  // The hanging node is no longer there.
  ASSERT_EQ(tester.NumNodes(), 2);
}

TEST(SUBGRAPH, multiple_outputs_with_hanging_nodes) {
  SubgraphTester tester(4);
  tester.AddDynamicTensorF32({96}, 0)
      .AddDynamicTensorF32({32}, 1)
      .AddDynamicTensorF32({32}, 2)
      .AddOutputTensorF32({32}, 3)
      // Add split3 with 1 consumed output and two unconsumed outputs.
      .AddEvenSplit(0, 0, {1, 2, 3})
      .Optimize();

  // The node is still there.
  ASSERT_EQ(tester.NumNodes(), 1);
  // And all four values also.
  ASSERT_EQ(tester.NumValues(), 4);
  // The first two outputs are optimized away.
  ASSERT_EQ(tester.Value(1)->type, xnn_value_type_invalid);
  ASSERT_EQ(tester.Value(2)->type, xnn_value_type_invalid);
  // The last output is consumed.
  ASSERT_EQ(tester.Value(3)->type, xnn_value_type_dense_tensor);
}

TEST(SUBGRAPH, even_split3_first_two_outputs_optimized_away) {
  RuntimeTester tester(5);
  constexpr size_t size = 9;
  float inputs[size] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  tester.AddStaticTensorF32({size}, TensorType::kDense, 0, 0, inputs)
      .AddDynamicTensorF32({3}, 1)
      .AddDynamicTensorF32({3}, 2)
      .AddOutputTensorF32({3}, 3)
      // Add split3 with 1 consumed output and two unconsumed outputs.
      .AddEvenSplit(0, 0, {1, 2, 3});
  // Regression test for a crash where we could not deal with a split where the
  // 0th output is not used (and optimized away).
  auto output = tester.RunWithFusion<float>();
  xnnpack::Buffer<float> expected = {6, 7, 8};
  ASSERT_EQ(expected, output);
}

TEST(SUBGRAPH, create_subgraph_invalid_external_value_ids) {
  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);
  xnn_subgraph_t subgraph = nullptr;
  EXPECT_EQ(xnn_create_subgraph(XNN_INVALID_VALUE_ID, /*flags=*/0, &subgraph),
            xnn_status_invalid_parameter);
  EXPECT_EQ(subgraph, nullptr);
}

TEST(SUBGRAPH, reserve_values_overflow) {
  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_create_subgraph(/*external_value_ids=*/1, /*flags=*/0,
                                &subgraph),
            xnn_status_success);
  EXPECT_EQ(xnn_subgraph_reserve_values(subgraph, SIZE_MAX),
            xnn_status_out_of_memory);
  EXPECT_EQ(xnn_subgraph_reserve_values(
                subgraph, static_cast<size_t>(XNN_INVALID_VALUE_ID)),
            xnn_status_out_of_memory);
  ASSERT_EQ(xnn_delete_subgraph(subgraph), xnn_status_success);
}

TEST(SUBGRAPH, reserve_nodes_overflow) {
  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_create_subgraph(/*external_value_ids=*/1, /*flags=*/0,
                                &subgraph),
            xnn_status_success);
  EXPECT_EQ(xnn_subgraph_reserve_nodes(subgraph, SIZE_MAX),
            xnn_status_out_of_memory);
  EXPECT_EQ(xnn_subgraph_reserve_nodes(
                subgraph, static_cast<size_t>(XNN_INVALID_NODE_ID)),
            xnn_status_out_of_memory);
  ASSERT_EQ(xnn_delete_subgraph(subgraph), xnn_status_success);
}

TEST(SUBGRAPH, widen_fp16_accumulators_converts_static_value_correctly) {
  ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0,
                                &subgraph),
            xnn_status_success);

  // Input tensor: 1D FP16 external input (id 0).
  const size_t input_dims[1] = {4};
  uint32_t input_id = 0;
  ASSERT_EQ(xnn_define_tensor_value(
                subgraph, xnn_datatype_fp16, 1, input_dims, nullptr,
                /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
            xnn_status_success);

  // Intermediate reduced tensor: 1D FP16 internal (id 2).
  const size_t reduced_dims[1] = {1};
  uint32_t reduced_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_define_tensor_value(
                subgraph, xnn_datatype_fp16, 1, reduced_dims, nullptr,
                /*external_id=*/XNN_INVALID_VALUE_ID, 0, &reduced_id),
            xnn_status_success);

  // Output tensor: 1D FP16 external output (id 1).
  uint32_t output_id = 1;
  ASSERT_EQ(xnn_define_tensor_value(
                subgraph, xnn_datatype_fp16, 1, reduced_dims, nullptr,
                /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
            xnn_status_success);

  // Static scalar FP16 multiplier (value = 0.5f).
  const uint16_t static_fp16_val = 0x3800;  // 0.5 in FP16
  const size_t scalar_dims[1] = {1};
  uint32_t static_scalar_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_define_tensor_value(
                subgraph, xnn_datatype_fp16, 1, scalar_dims, &static_fp16_val,
                /*external_id=*/XNN_INVALID_VALUE_ID, 0, &static_scalar_id),
            xnn_status_success);

  // Static sum reduction node.
  const size_t reduction_axes[1] = {0};
  ASSERT_EQ(xnn_define_static_reduce(
                subgraph, xnn_reduce_sum, 1, reduction_axes, input_id,
                reduced_id, /*flags=*/XNN_FLAG_KEEP_DIMS),
            xnn_status_success);

  // Binary multiply node: reduced_id * static_scalar_id -> output_id.
  ASSERT_EQ(xnn_define_binary(
                subgraph, xnn_binary_multiply, nullptr, reduced_id,
                static_scalar_id, output_id, /*flags=*/0),
            xnn_status_success);

  // Run subgraph optimization.
  ASSERT_EQ(xnn_subgraph_optimize(subgraph, /*flags=*/0), xnn_status_success);

  // Verify that the graph contains 3 nodes: reduce, binary op, and convert.
  ASSERT_EQ(subgraph->num_nodes, 3);
  EXPECT_EQ(subgraph->nodes[0].type, xnn_node_type_static_sum);
  EXPECT_EQ(subgraph->nodes[1].type, xnn_node_type_binary_elementwise);
  EXPECT_EQ(subgraph->nodes[1].binary_operator, xnn_binary_multiply);
  EXPECT_EQ(subgraph->nodes[2].type, xnn_node_type_unary_elementwise);
  EXPECT_EQ(subgraph->nodes[2].unary_operator, xnn_unary_convert);

  // Verify that static_scalar_id was rewritten to FP32 and equals 0.5f.
  const struct xnn_value* scalar_val = &subgraph->values[static_scalar_id];
  EXPECT_EQ(scalar_val->datatype, xnn_datatype_fp32);
  ASSERT_NE(scalar_val->data, nullptr);
  EXPECT_EQ(*reinterpret_cast<const float*>(scalar_val->data), 0.5f);

  ASSERT_EQ(xnn_delete_subgraph(subgraph), xnn_status_success);
}

}  // namespace xnnpack
