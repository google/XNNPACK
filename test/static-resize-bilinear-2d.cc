// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>  // For std::generate, std::min.
#include <array>      // For std::array.
#include <cmath>      // For std::lrintf.
#include <cstddef>    // For size_t.
#include <cstdint>    // For uint32_t.
#include <limits>     // For std::numeric_limits.
#include <memory>     // For std::unique_ptr.
#include <random>     // For std::random_device, std::mt19937, std::uniform_real_distribution.
#include <vector>     // For std::vector.

#include <fp16/fp16.h>
#include <gtest/gtest.h>

#include <xnnpack.h>
#include <xnnpack/operator.h>
#include <xnnpack/subgraph.h>

template <class T> class StaticResizeBilinear2DTestBase : public ::testing::Test {
protected:
  StaticResizeBilinear2DTestBase()
  {
    random_device = std::make_unique<std::random_device>();
    rng = std::mt19937((*random_device)());
    input_size_dist = std::uniform_int_distribution<uint32_t>(10, 15);
    kernel_size_dist = std::uniform_int_distribution<uint32_t>(2, 5);
    stride_dist = std::uniform_int_distribution<uint32_t>(1, 2);
    f32dist = std::uniform_real_distribution<float>();
    scale_dist = std::uniform_real_distribution<float>(1.0f, 5.0f);
    i32dist = std::uniform_int_distribution<int32_t>(-10000, 10000);
    dilation_dist = std::uniform_int_distribution<uint32_t>(1, 2);
    i8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    u8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    batch_size = input_size_dist(rng);
    input_height = input_size_dist(rng);
    input_width = input_size_dist(rng);
    channels = input_size_dist(rng);
    output_height = input_size_dist(rng);
    output_width = input_size_dist(rng);

    input_dims = {{batch_size, input_height, input_width, channels}};
    output_dims = {{batch_size, output_height, output_width, channels}};

    input = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + batch_size * input_height * input_width * channels);
    operator_output = std::vector<T>(batch_size * output_height * output_width * channels);
    subgraph_output = std::vector<T>(batch_size * output_height * output_width * channels);
  }

  std::unique_ptr<std::random_device> random_device;
  std::mt19937 rng;
  std::uniform_int_distribution<uint32_t> input_size_dist;
  std::uniform_int_distribution<uint32_t> kernel_size_dist;
  std::uniform_int_distribution<uint32_t> stride_dist;
  std::uniform_int_distribution<int32_t> i32dist;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_real_distribution<float> scale_dist;
  std::uniform_int_distribution<uint32_t> dilation_dist;
  std::uniform_int_distribution<int32_t> i8dist;
  std::uniform_int_distribution<int32_t> u8dist;

  uint32_t batch_size;
  uint32_t input_height;
  uint32_t input_width;
  uint32_t channels;
  uint32_t output_height;
  uint32_t output_width;

  std::array<size_t, 4> input_dims;
  std::array<size_t, 4> output_dims;

  std::vector<T> input;
  std::vector<T> operator_output;
  std::vector<T> subgraph_output;
};

using StaticResizeBilinear2DTestQS8 = StaticResizeBilinear2DTestBase<int8_t>;
using StaticResizeBilinear2DTestQU8 = StaticResizeBilinear2DTestBase<uint8_t>;
using StaticResizeBilinear2DTestF16 = StaticResizeBilinear2DTestBase<uint16_t>;
using StaticResizeBilinear2DTestF32 = StaticResizeBilinear2DTestBase<float>;

TEST_F(StaticResizeBilinear2DTestQS8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, 0, 1.0f, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, 0, 1.0f, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/1, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_resize_bilinear_2d(subgraph, output_height, output_width, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_resize_bilinear_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qs8);
  ASSERT_EQ(node->params.static_resize.new_height, output_height);
  ASSERT_EQ(node->params.static_resize.new_width, output_width);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticResizeBilinear2DTestQU8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, 0, 1.0f, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, 0, 1.0f, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/1, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_resize_bilinear_2d(subgraph, output_height, output_width, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_resize_bilinear_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qu8);
  ASSERT_EQ(node->params.static_resize.new_height, output_height);
  ASSERT_EQ(node->params.static_resize.new_width, output_width);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticResizeBilinear2DTestF16, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/1, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_resize_bilinear_2d(subgraph, output_height, output_width, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_resize_bilinear_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp16);
  ASSERT_EQ(node->params.static_resize.new_height, output_height);
  ASSERT_EQ(node->params.static_resize.new_width, output_width);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticResizeBilinear2DTestF32, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/1, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_resize_bilinear_2d(subgraph, output_height, output_width, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_resize_bilinear_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->params.static_resize.new_height, output_height);
  ASSERT_EQ(node->params.static_resize.new_width, output_width);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticResizeBilinear2DTestQS8, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), INT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));
  const int8_t input_zero_point = i8dist(rng);
  const float input_scale = scale_dist(rng);
  const float output_zero_point = input_zero_point;
  const float output_scale = input_scale;

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_resize_bilinear2d_nhwc_s8(output_height, output_width, /*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;
  ASSERT_EQ(
    xnn_status_success,
    xnn_reshape_resize_bilinear2d_nhwc_s8(
      op, batch_size, input_height, input_width, channels, channels, channels,
      &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
  ASSERT_EQ(workspace_size, 0);
  ASSERT_EQ(workspace_alignment, 1);
  ASSERT_EQ(xnn_status_success, xnn_setup_resize_bilinear2d_nhwc_s8(op, /*workspace=*/nullptr, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, input_zero_point, input_scale, input_dims.size(),
                          input_dims.data(), nullptr, /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, output_zero_point, output_scale, output_dims.size(),
                          output_dims.data(), nullptr, /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_resize_bilinear_2d(subgraph, output_height, output_width, input_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(StaticResizeBilinear2DTestQU8, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), UINT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT8_C(0xA5));
  const uint8_t input_zero_point = u8dist(rng);
  const float input_scale = scale_dist(rng);
  const uint8_t output_zero_point = input_zero_point;
  const float output_scale = input_scale;

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_resize_bilinear2d_nhwc_u8(output_height, output_width, /*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;
  ASSERT_EQ(
    xnn_status_success,
    xnn_reshape_resize_bilinear2d_nhwc_u8(
      op, batch_size, input_height, input_width, channels, channels, channels,
      &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
  ASSERT_EQ(workspace_size, 0);
  ASSERT_EQ(workspace_alignment, 1);
  ASSERT_EQ(xnn_status_success, xnn_setup_resize_bilinear2d_nhwc_u8(op, /*workspace=*/nullptr, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, input_zero_point, input_scale, input_dims.size(),
                          input_dims.data(), nullptr, /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, output_zero_point, output_scale, output_dims.size(),
                          output_dims.data(), nullptr, /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_resize_bilinear_2d(subgraph, output_height, output_width, input_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(StaticResizeBilinear2DTestF16, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::fill(operator_output.begin(), operator_output.end(), fp16_ieee_from_fp32_value(std::nanf("")));
  std::fill(subgraph_output.begin(), subgraph_output.end(), fp16_ieee_from_fp32_value(std::nanf("")));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_resize_bilinear2d_nhwc_f16(output_height, output_width, /*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;
  ASSERT_EQ(
    xnn_status_success,
    xnn_reshape_resize_bilinear2d_nhwc_f16(
      op, batch_size, input_height, input_width, channels, channels, channels,
      &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
  ASSERT_EQ(workspace_size, 0);
  ASSERT_EQ(workspace_alignment, 1);
  ASSERT_EQ(xnn_status_success, xnn_setup_resize_bilinear2d_nhwc_f16(op, /*workspace=*/nullptr, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input_dims.size(), input_dims.data(), nullptr, /*external_id=*/0,
                          XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_resize_bilinear_2d(subgraph, output_height, output_width, input_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(StaticResizeBilinear2DTestF32, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), std::nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), std::nanf(""));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_resize_bilinear2d_nhwc_f32(output_height, output_width, /*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;
  ASSERT_EQ(
    xnn_status_success,
    xnn_reshape_resize_bilinear2d_nhwc_f32(
      op, batch_size, input_height, input_width, channels, channels, channels,
      &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
  ASSERT_EQ(workspace_size, 0);
  ASSERT_EQ(workspace_alignment, 1);
  ASSERT_EQ(xnn_status_success, xnn_setup_resize_bilinear2d_nhwc_f32(op, /*workspace=*/nullptr, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr, /*external_id=*/0,
                          XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_resize_bilinear_2d(subgraph, output_height, output_width, input_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(StaticResizeBilinear2DTestF32, reshape_output)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr, /*external_id=*/0,
                          XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_resize_bilinear_2d(subgraph, output_height, output_width, input_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Channels and batch size can change.
  input_dims[0] += 2;
  input_dims[3] += 7;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input_id, input_dims.size(), input_dims.data()));
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  ASSERT_EQ(output_shape->dim[0], input_dims[0]);
  ASSERT_EQ(output_shape->dim[1], output_dims[1]);
  ASSERT_EQ(output_shape->dim[2], output_dims[2]);
  ASSERT_EQ(output_shape->dim[3], input_dims[3]);

  input_dims[0] -= 1;
  input_dims[1] -= 1;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input_id, input_dims.size(), input_dims.data()));
  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_success);
  ASSERT_EQ(output_shape->dim[0], input_dims[0]);
  ASSERT_EQ(output_shape->dim[1], output_dims[1]);
  ASSERT_EQ(output_shape->dim[2], output_dims[2]);
  ASSERT_EQ(output_shape->dim[3], input_dims[3]);
}
