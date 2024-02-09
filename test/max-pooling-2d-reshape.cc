// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/node-type.h>
#include <xnnpack/operator-utils.h>
#include <xnnpack/operator.h>
#include <xnnpack/subgraph.h>

TEST(MaxPooling2DTestF32, Reshape)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call Subgraph APIs
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<size_t> input_dims{2, 3, 4, 5};
  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  std::vector<size_t> output_dims{2, 1, 2, 5};
  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  const size_t pooling_height = 2;
  const size_t pooling_width = 2;
  const size_t stride_height = 2;
  const size_t stride_width = 2;
  const size_t dilation_height = 1;
  const size_t dilation_width = 1;
  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_max_pooling_2d(
        subgraph,/*input_padding_top=*/0, /*input_padding_right=*/0, /*input_padding_bottom=*/0, /*input_padding_left=*/0, pooling_height,
      pooling_width, stride_height, stride_width, dilation_height, dilation_width, output_min, output_max, input_id, output_id,
      /*flags=*/0
    ));
  ASSERT_EQ(subgraph->num_nodes, 1);
  struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_max_pooling_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  ASSERT_EQ(node->reshape(&runtime->opdata[0], subgraph->values, subgraph->num_values, /*threadpool=*/nullptr), xnn_status_success);

  input_dims[0] += 1;
  input_dims[1] += 1;
  input_dims[2] += 1;
  input_dims[3] += 1;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, 0, input_dims.size(), input_dims.data()));

  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;

  ASSERT_EQ(output_shape->dim[0], input_dims[0]);
  ASSERT_EQ(output_shape->dim[1], 2);
  ASSERT_EQ(output_shape->dim[2], 2);
  ASSERT_EQ(output_shape->dim[3], input_dims[3]);
}


TEST(MaxPooling2DTestF32, ReshapeWithPadding)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call Subgraph APIs
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<size_t> input_dims{2, 3, 4, 5};
  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  std::vector<size_t> output_dims{2, 3, 5, 5};
  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  const size_t pooling_height = 2;
  const size_t pooling_width = 2;
  const size_t stride_height = 2;
  const size_t stride_width = 2;
  const size_t dilation_height = 1;
  const size_t dilation_width = 1;
  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_max_pooling_2d(
        subgraph,/*input_padding_top=*/3, /*input_padding_right=*/2, /*input_padding_bottom=*/1, /*input_padding_left=*/4, pooling_height,
      pooling_width, stride_height, stride_width, dilation_height, dilation_width, output_min, output_max, input_id, output_id,
      /*flags=*/0
    ));
  ASSERT_EQ(subgraph->num_nodes, 1);
  struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_max_pooling_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  ASSERT_EQ(node->reshape(&runtime->opdata[0], subgraph->values, subgraph->num_values, /*threadpool=*/nullptr), xnn_status_success);

  input_dims[0] = 2;
  input_dims[1] = 2;
  input_dims[2] = 8;
  input_dims[3] = 17;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, 0, input_dims.size(), input_dims.data()));

  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;

  ASSERT_EQ(output_shape->dim[0], input_dims[0]);
  ASSERT_EQ(output_shape->dim[1], 3);
  ASSERT_EQ(output_shape->dim[2], 7);
  ASSERT_EQ(output_shape->dim[3], input_dims[3]);
}

TEST(MaxPooling2DTestF32, ReshapeWithPaddingAndDilation)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call Subgraph APIs
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<size_t> input_dims{2, 3, 4, 5};
  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  std::vector<size_t> output_dims{2, 3, 4, 5};
  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  const size_t pooling_height = 2;
  const size_t pooling_width = 2;
  const size_t stride_height = 2;
  const size_t stride_width = 2;
  const size_t dilation_height = 2;
  const size_t dilation_width = 3;
  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_max_pooling_2d(
        subgraph,/*input_padding_top=*/3, /*input_padding_right=*/2, /*input_padding_bottom=*/1, /*input_padding_left=*/4, pooling_height,
      pooling_width, stride_height, stride_width, dilation_height, dilation_width, output_min, output_max, input_id, output_id,
      /*flags=*/0
    ));
  ASSERT_EQ(subgraph->num_nodes, 1);
  struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_max_pooling_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  ASSERT_EQ(node->reshape(&runtime->opdata[0], subgraph->values, subgraph->num_values, /*threadpool=*/nullptr), xnn_status_success);

  input_dims[0] = 2;
  input_dims[1] = 2;
  input_dims[2] = 8;
  input_dims[3] = 17;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, 0, input_dims.size(), input_dims.data()));

  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;

  ASSERT_EQ(output_shape->dim[0], input_dims[0]);
  ASSERT_EQ(output_shape->dim[1], 2);
  ASSERT_EQ(output_shape->dim[2], 6);
  ASSERT_EQ(output_shape->dim[3], input_dims[3]);
}
