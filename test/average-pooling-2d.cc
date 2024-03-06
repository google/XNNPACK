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

#include <fp16/fp16.h>
#include <gtest/gtest.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/node-type.h>
#include <xnnpack/operator-utils.h>
#include <xnnpack/operator.h>
#include <xnnpack/subgraph.h>

template <
  typename InputType,
  typename OutputType = InputType>
class AveragePoolingTest : public ::testing::Test {
protected:
  AveragePoolingTest()
  {
    random_device = std::make_unique<std::random_device>();
    rng = std::mt19937((*random_device)());
    input_size_dist = std::uniform_int_distribution<uint32_t>(10, 15);
    pooling_size_dist = std::uniform_int_distribution<uint32_t>(2, 5);
    stride_dist = std::uniform_int_distribution<uint32_t>(1, 2);
    batch_size = input_size_dist(rng);
    input_height = input_size_dist(rng);
    input_width = input_size_dist(rng);
    channels = input_size_dist(rng);
    pooling_height = pooling_size_dist(rng);
    pooling_width = pooling_size_dist(rng);
    // Avoid padding == pooling dimension because it will result in NaNs and cause comparison to fail.
    input_padding_top = std::uniform_int_distribution<uint32_t>(0, pooling_height - 1)(rng);
    input_padding_right = std::uniform_int_distribution<uint32_t>(0, pooling_width - 1)(rng);
    input_padding_bottom = std::uniform_int_distribution<uint32_t>(0, pooling_height - 1)(rng);
    input_padding_left = std::uniform_int_distribution<uint32_t>(0, pooling_width - 1)(rng);
    stride_height = stride_dist(rng);
    stride_width = stride_dist(rng);
    output_height = xnn_compute_convolution_output_dimension(
      input_padding_top + input_height + input_padding_bottom, pooling_height, 1, stride_height);
    output_width = xnn_compute_convolution_output_dimension(
      input_padding_left + input_width + input_padding_right, pooling_width, 1, stride_width);
    output_min = std::uniform_real_distribution<float>(-255.0f, 0.0f)(rng);
    output_max = std::uniform_real_distribution<float>(0.1f, 255.0f)(rng);
    input_dims = {batch_size, input_height, input_width, channels};
    output_dims = {batch_size, output_height, output_width, channels};
    input = std::vector<InputType>(XNN_EXTRA_BYTES / sizeof(InputType) + batch_size * input_height * input_width * channels);
    operator_output = std::vector<OutputType>(batch_size * output_height * output_width * channels);
    subgraph_output = std::vector<OutputType>(batch_size * output_height * output_width * channels);
  }

  std::unique_ptr<std::random_device> random_device;
  std::mt19937 rng;
  std::uniform_int_distribution<uint32_t> input_size_dist;
  std::uniform_int_distribution<uint32_t> pooling_size_dist;
  std::uniform_int_distribution<uint32_t> stride_dist;
  uint32_t batch_size;
  uint32_t input_height;
  uint32_t input_width;
  uint32_t channels;
  uint32_t pooling_height;
  uint32_t pooling_width;
  uint32_t output_height;
  uint32_t output_width;
  uint32_t stride_height;
  uint32_t stride_width;
  std::array<size_t, 4> input_dims;
  std::array<size_t, 4> output_dims;
  uint32_t input_padding_top;
  uint32_t input_padding_right;
  uint32_t input_padding_bottom;
  uint32_t input_padding_left;
  float output_min;
  float output_max;

  uint32_t input_id;
  uint32_t output_id;

  std::vector<InputType> input;
  std::vector<OutputType> operator_output;
  std::vector<OutputType> subgraph_output;
};

using AveragePoolingTestF16 = AveragePoolingTest<uint16_t>;
using AveragePoolingTestF32 = AveragePoolingTest<float>;

TEST_F(AveragePoolingTestF16, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input_dims.size(), input_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_average_pooling_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, pooling_height,
      pooling_width, stride_height, stride_width, output_min, output_max, input_id, output_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_average_pooling_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp16);

  ASSERT_EQ(node->params.pooling_2d.padding_top, input_padding_top);
  ASSERT_EQ(node->params.pooling_2d.padding_right, input_padding_right);
  ASSERT_EQ(node->params.pooling_2d.padding_bottom, input_padding_bottom);
  ASSERT_EQ(node->params.pooling_2d.padding_left, input_padding_left);
  ASSERT_EQ(node->params.pooling_2d.pooling_height, pooling_height);
  ASSERT_EQ(node->params.pooling_2d.pooling_width, pooling_width);
  ASSERT_EQ(node->params.pooling_2d.stride_height, stride_height);
  ASSERT_EQ(node->params.pooling_2d.stride_width, stride_width);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(AveragePoolingTestF32, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_average_pooling_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, pooling_height,
      pooling_width, stride_height, stride_width, output_min, output_max, input_id, output_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_average_pooling_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);

  ASSERT_EQ(node->params.pooling_2d.padding_top, input_padding_top);
  ASSERT_EQ(node->params.pooling_2d.padding_right, input_padding_right);
  ASSERT_EQ(node->params.pooling_2d.padding_bottom, input_padding_bottom);
  ASSERT_EQ(node->params.pooling_2d.padding_left, input_padding_left);
  ASSERT_EQ(node->params.pooling_2d.pooling_height, pooling_height);
  ASSERT_EQ(node->params.pooling_2d.pooling_width, pooling_width);
  ASSERT_EQ(node->params.pooling_2d.stride_height, stride_height);
  ASSERT_EQ(node->params.pooling_2d.stride_width, stride_width);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(AveragePoolingTestF16, matches_operator_api)
{
  std::uniform_real_distribution<float> f32dist;
  std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::fill(operator_output.begin(), operator_output.end(), fp16_ieee_from_fp32_value(nanf("")));
  std::fill(subgraph_output.begin(), subgraph_output.end(), fp16_ieee_from_fp32_value(nanf("")));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_average_pooling2d_nhwc_f16(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, pooling_height, pooling_width,
    stride_height, stride_width, output_min, output_max, /*flags=*/0, &op);
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
    xnn_reshape_average_pooling2d_nhwc_f16(
      op, batch_size, input_height, input_width,
      channels, /*input_pixel_stride=*/channels, /*output_pixel_stride=*/channels,
      &workspace_size, &workspace_alignment,
      /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
      /*threadpool=*/nullptr));
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
  ASSERT_EQ(xnn_status_success, xnn_setup_average_pooling2d_nhwc_f16(op, workspace.data(), input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input_dims.size(), input_dims.data(), nullptr, /*external_id=*/0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr, /*external_id=*/1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_average_pooling_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, pooling_height,
      pooling_width, stride_height, stride_width, output_min, output_max, input_id, output_id,
      /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(AveragePoolingTestF32, matches_operator_api)
{
  std::uniform_real_distribution<float> f32dist;
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_average_pooling2d_nhwc_f32(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, pooling_height, pooling_width,
    stride_height, stride_width, output_min, output_max, /*flags=*/0, &op);
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
    xnn_reshape_average_pooling2d_nhwc_f32(
      op, batch_size, input_height, input_width,
      channels, /*input_pixel_stride=*/channels, /*output_pixel_stride=*/channels,
      &workspace_size, &workspace_alignment,
      /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
      /*threadpool=*/nullptr));
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
  ASSERT_EQ(xnn_status_success, xnn_setup_average_pooling2d_nhwc_f32(op, workspace.data(), input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr, /*external_id=*/0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, /*external_id=*/1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_average_pooling_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, pooling_height,
      pooling_width, stride_height, stride_width, output_min, output_max, input_id, output_id,
      /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}
