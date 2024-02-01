// Copyright 2022 Google LLC
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
#include <xnnpack/operator.h>
#include <xnnpack/subgraph.h>

namespace {
inline size_t compute_output_dimension(size_t padded_input_dimension, size_t kernel_dimension)
{
  return padded_input_dimension / kernel_dimension;
}
}  // namespace

class ArgmaxPoolingTestF32 : public ::testing::Test {
protected:
  ArgmaxPoolingTestF32()
  {
    random_device = std::make_unique<std::random_device>();
    rng = std::mt19937((*random_device)());
    input_size_dist = std::uniform_int_distribution<uint32_t>(10, 15);
    pooling_size_dist = std::uniform_int_distribution<uint32_t>(2, 5);
    batch_size = input_size_dist(rng);
    input_height = input_size_dist(rng);
    input_width = input_size_dist(rng);
    channels = input_size_dist(rng);
    pooling_height = pooling_size_dist(rng);
    pooling_width = pooling_size_dist(rng);
    input_padding_top = input_size_dist(rng);
    input_padding_right = input_size_dist(rng);
    input_padding_bottom = input_size_dist(rng);
    input_padding_left = input_size_dist(rng);
    output_height = compute_output_dimension(input_height + input_padding_top + input_padding_bottom, pooling_height);
    output_width = compute_output_dimension(input_width + input_padding_left + input_padding_right, pooling_width);
    input_dims = {batch_size, input_height, input_width, channels};
    output_dims = {batch_size, output_height, output_width, channels};
    input = std::vector<float>(XNN_EXTRA_BYTES / sizeof(float) + batch_size * input_height * input_width * channels);
    operator_output = std::vector<float>(batch_size * output_height * output_width * channels);
    operator_output_index = std::vector<uint32_t>(batch_size * output_height * output_width * channels);
    subgraph_output = std::vector<float>(batch_size * output_height * output_width * channels);
    subgraph_output_index = std::vector<uint32_t>(batch_size * output_height * output_width * channels);
  }

  std::unique_ptr<std::random_device> random_device;
  std::mt19937 rng;
  std::uniform_int_distribution<uint32_t> input_size_dist;
  std::uniform_int_distribution<uint32_t> pooling_size_dist;
  uint32_t batch_size;
  uint32_t input_height;
  uint32_t input_width;
  uint32_t channels;
  uint32_t pooling_height;
  uint32_t pooling_width;
  uint32_t output_height;
  uint32_t output_width;
  std::array<size_t, 4> input_dims;
  std::array<size_t, 4> output_dims;
  uint32_t input_padding_top;
  uint32_t input_padding_right;
  uint32_t input_padding_bottom;
  uint32_t input_padding_left;

  uint32_t input_id;
  uint32_t output_value_id;
  uint32_t output_index_id;

  std::vector<float> input;
  std::vector<float> operator_output;
  std::vector<uint32_t> operator_output_index;
  std::vector<float> subgraph_output;
  std::vector<uint32_t> subgraph_output_index;
};

TEST_F(ArgmaxPoolingTestF32, define)
{

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_value_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_value_id));
  ASSERT_NE(output_value_id, XNN_INVALID_NODE_ID);

  output_index_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, 2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_index_id));
  ASSERT_NE(output_index_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success, xnn_define_argmax_pooling_2d(
                          subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
                          pooling_height, pooling_width, input_id, output_value_id, output_index_id,
                          /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_argmax_pooling_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->params.pooling_2d.padding_top, input_padding_top);
  ASSERT_EQ(node->params.pooling_2d.padding_right, input_padding_right);
  ASSERT_EQ(node->params.pooling_2d.padding_bottom, input_padding_bottom);
  ASSERT_EQ(node->params.pooling_2d.padding_left, input_padding_left);
  ASSERT_EQ(node->params.pooling_2d.pooling_height, pooling_height);
  ASSERT_EQ(node->params.pooling_2d.pooling_width, pooling_width);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 2);
  ASSERT_EQ(node->outputs[0], output_value_id);
  ASSERT_EQ(node->outputs[1], output_index_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(ArgmaxPoolingTestF32, matches_operator_api)
{
  std::uniform_real_distribution<float> f32dist(-255.0f, 255.0f);
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_argmax_pooling2d_nhwc_f32(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, pooling_height, pooling_width,
    /*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  ASSERT_EQ(
    xnn_status_success,
    xnn_reshape_argmax_pooling2d_nhwc_f32(
      op, batch_size, input_height, input_width,
      /*channels=*/channels,
      /*input_pixel_stride=*/channels,
      /*output_pixel_stride=*/channels,
      &workspace_size, &workspace_alignment,
      /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
      /*threadpool=*/nullptr));

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
  ASSERT_EQ(
    xnn_status_success, xnn_setup_argmax_pooling2d_nhwc_f32(
                          op, workspace.data(), input.data(), operator_output.data(), operator_output_index.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr, /*external_id=*/0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_value_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, /*external_id=*/1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_value_id));
  ASSERT_NE(output_value_id, XNN_INVALID_NODE_ID);

  output_index_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, /*external_id=*/2,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_index_id));
  ASSERT_NE(output_index_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(
    xnn_status_success, xnn_define_argmax_pooling_2d(
                          subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
                          pooling_height, pooling_width, input_id, output_value_id, output_index_id,
                          /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 3> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_value_id, subgraph_output.data()},
    xnn_external_value{output_index_id, subgraph_output_index.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(ArgmaxPoolingTestF32, reshape_output)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<size_t> dims{2, 3, 4, 5};
  std::vector<size_t> output_dims{2, 3, 5, 5};
  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_value_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_value_id));
  ASSERT_NE(output_value_id, XNN_INVALID_NODE_ID);

  output_index_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, 2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_index_id));
  ASSERT_NE(output_index_id, XNN_INVALID_NODE_ID);
  const size_t pooling_height = 2;
  const size_t pooling_width = 2;
  ASSERT_EQ(xnn_status_success, xnn_define_argmax_pooling_2d(
      subgraph, /*input_padding_top=*/3, /*input_padding_right=*/2, /*input_padding_bottom=*/1, /*input_padding_left=*/4,
      pooling_height, pooling_width, input_id, output_value_id, output_index_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_argmax_pooling_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 2);
  ASSERT_EQ(node->outputs[0], output_value_id);
  ASSERT_EQ(node->outputs[1], output_index_id);
  ASSERT_EQ(node->flags, 0);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  ASSERT_EQ(node->reshape(&runtime->opdata[0], subgraph->values, subgraph->num_values, /*threadpool=*/nullptr), xnn_status_success);

  dims[0] = 2;
  dims[1] = 2;
  dims[2] = 8;
  dims[3] = 17;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, 0, dims.size(), dims.data()));

  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  ASSERT_EQ(output_shape->dim[0], dims[0]);
  ASSERT_EQ(output_shape->dim[1], 3);
  ASSERT_EQ(output_shape->dim[2], 7);
  ASSERT_EQ(output_shape->dim[3], dims[3]);
}
