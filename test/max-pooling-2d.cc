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
#include <random>     // For std::uniform_real_distribution.
#include <vector>     // For std::vector.

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/operator.h"
#include "xnnpack/requantization.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"

template <class T> class MaxPooling2DTestBase : public ::testing::Test {
 protected:
  MaxPooling2DTestBase() {
    input_size_dist = std::uniform_int_distribution<uint32_t>(10, 15);
    kernel_size_dist = std::uniform_int_distribution<uint32_t>(2, 5);
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
    pooling_height = kernel_size_dist(rng);
    pooling_width = kernel_size_dist(rng);
    padding_top = std::uniform_int_distribution<uint32_t>(0, pooling_height - 1)(rng);
    padding_bottom = std::uniform_int_distribution<uint32_t>(0, pooling_height - 1)(rng);
    padding_left = std::uniform_int_distribution<uint32_t>(0, pooling_width - 1)(rng);
    padding_right = std::uniform_int_distribution<uint32_t>(0, pooling_width - 1)(rng);
    dilation_height = dilation_dist(rng);
    dilation_width = dilation_height;
    // stride dimension must be <= filter dimension
    stride_height = std::uniform_int_distribution<uint32_t>(1, pooling_height)(rng);
    stride_width = std::uniform_int_distribution<uint32_t>(1, pooling_width)(rng);
    output_min = -std::numeric_limits<float>::infinity();
    output_max = std::numeric_limits<float>::infinity();
    output_height = xnn_compute_convolution_output_dimension(
      padding_top + input_height + padding_bottom, pooling_height, dilation_height, stride_height);
    output_width = xnn_compute_convolution_output_dimension(
      padding_left + input_width + padding_right, pooling_width, dilation_width, stride_width);

    input_dims = {{batch_size, input_height, input_width, channels}};
    output_dims = {{batch_size, output_height, output_width, channels}};

    input = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + batch_size * input_height * input_width * channels);
    operator_output =
      std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + batch_size * output_height * output_width * channels);
    subgraph_output =
      std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + batch_size * output_height * output_width * channels);
  }

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<uint32_t> input_size_dist;
  std::uniform_int_distribution<uint32_t> kernel_size_dist;
  std::uniform_int_distribution<int32_t> i32dist;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_real_distribution<float> scale_dist;
  std::uniform_int_distribution<uint32_t> dilation_dist;
  std::uniform_int_distribution<int32_t> i8dist;
  std::uniform_int_distribution<int32_t> u8dist;

  uint32_t padding_top;
  uint32_t padding_right;
  uint32_t padding_bottom;
  uint32_t padding_left;
  uint32_t batch_size;
  uint32_t input_height;
  uint32_t input_width;
  uint32_t pooling_height;
  uint32_t pooling_width;
  uint32_t stride_height;
  uint32_t stride_width;
  uint32_t dilation_height;
  uint32_t dilation_width;
  uint32_t channels;
  float output_min;
  float output_max;
  uint32_t output_height;
  uint32_t output_width;

  std::array<size_t, 4> input_dims;
  std::array<size_t, 4> output_dims;

  std::vector<T> input;
  std::vector<T> operator_output;
  std::vector<T> subgraph_output;
};

using MaxPooling2DTestQS8 = MaxPooling2DTestBase<int8_t>;
using MaxPooling2DTestQU8 = MaxPooling2DTestBase<uint8_t>;
using MaxPooling2DTestF16 = MaxPooling2DTestBase<uint16_t>;
using MaxPooling2DTestF32 = MaxPooling2DTestBase<float>;

TEST_F(MaxPooling2DTestQS8, define)
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
    xnn_define_max_pooling_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, pooling_height, pooling_width, stride_height,
      stride_width, dilation_height, dilation_width, output_min, output_max, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_max_pooling_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qs8);
  ASSERT_EQ(node->params.pooling_2d.padding_top, padding_top);
  ASSERT_EQ(node->params.pooling_2d.padding_right, padding_right);
  ASSERT_EQ(node->params.pooling_2d.padding_bottom, padding_bottom);
  ASSERT_EQ(node->params.pooling_2d.padding_left, padding_left);
  ASSERT_EQ(node->params.pooling_2d.pooling_height, pooling_height);
  ASSERT_EQ(node->params.pooling_2d.pooling_width, pooling_width);
  ASSERT_EQ(node->params.pooling_2d.stride_height, stride_height);
  ASSERT_EQ(node->params.pooling_2d.stride_width, stride_width);
  ASSERT_EQ(node->params.pooling_2d.dilation_height, dilation_height);
  ASSERT_EQ(node->params.pooling_2d.dilation_width, dilation_width);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(MaxPooling2DTestQU8, define)
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
    xnn_define_max_pooling_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, pooling_height, pooling_width, stride_height,
      stride_width, dilation_height, dilation_width, output_min, output_max, input_id, output_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_max_pooling_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qu8);
  ASSERT_EQ(node->params.pooling_2d.padding_top, padding_top);
  ASSERT_EQ(node->params.pooling_2d.padding_right, padding_right);
  ASSERT_EQ(node->params.pooling_2d.padding_bottom, padding_bottom);
  ASSERT_EQ(node->params.pooling_2d.padding_left, padding_left);
  ASSERT_EQ(node->params.pooling_2d.pooling_height, pooling_height);
  ASSERT_EQ(node->params.pooling_2d.pooling_width, pooling_width);
  ASSERT_EQ(node->params.pooling_2d.stride_height, stride_height);
  ASSERT_EQ(node->params.pooling_2d.stride_width, stride_width);
  ASSERT_EQ(node->params.pooling_2d.dilation_height, dilation_height);
  ASSERT_EQ(node->params.pooling_2d.dilation_width, dilation_width);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(MaxPooling2DTestF16, define)
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
    xnn_define_max_pooling_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, pooling_height, pooling_width, stride_height,
      stride_width, dilation_height, dilation_width, output_min, output_max, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_max_pooling_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp16);
  ASSERT_EQ(node->params.pooling_2d.padding_top, padding_top);
  ASSERT_EQ(node->params.pooling_2d.padding_right, padding_right);
  ASSERT_EQ(node->params.pooling_2d.padding_bottom, padding_bottom);
  ASSERT_EQ(node->params.pooling_2d.padding_left, padding_left);
  ASSERT_EQ(node->params.pooling_2d.pooling_height, pooling_height);
  ASSERT_EQ(node->params.pooling_2d.pooling_width, pooling_width);
  ASSERT_EQ(node->params.pooling_2d.stride_height, stride_height);
  ASSERT_EQ(node->params.pooling_2d.stride_width, stride_width);
  ASSERT_EQ(node->params.pooling_2d.dilation_height, dilation_height);
  ASSERT_EQ(node->params.pooling_2d.dilation_width, dilation_width);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(MaxPooling2DTestF32, define)
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
    xnn_define_max_pooling_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, pooling_height, pooling_width, stride_height,
      stride_width, dilation_height, dilation_width, output_min, output_max, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_max_pooling_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->params.pooling_2d.padding_top, padding_top);
  ASSERT_EQ(node->params.pooling_2d.padding_right, padding_right);
  ASSERT_EQ(node->params.pooling_2d.padding_bottom, padding_bottom);
  ASSERT_EQ(node->params.pooling_2d.padding_left, padding_left);
  ASSERT_EQ(node->params.pooling_2d.pooling_height, pooling_height);
  ASSERT_EQ(node->params.pooling_2d.pooling_width, pooling_width);
  ASSERT_EQ(node->params.pooling_2d.stride_height, stride_height);
  ASSERT_EQ(node->params.pooling_2d.stride_width, stride_width);
  ASSERT_EQ(node->params.pooling_2d.dilation_height, dilation_height);
  ASSERT_EQ(node->params.pooling_2d.dilation_width, dilation_width);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(MaxPooling2DTestQS8, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), INT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));
  const int8_t input_zero_point = i8dist(rng);
  const float input_scale = scale_dist(rng);
  const int8_t output_zero_point = input_zero_point;
  const float output_scale = input_scale;
  const int8_t quantized_output_min = xnn_qs8_quantize(output_min, output_scale, output_zero_point);
  const int8_t quantized_output_max = xnn_qs8_quantize(output_max, output_scale, output_zero_point);

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_max_pooling2d_nhwc_s8(
    padding_top, padding_right, padding_bottom, padding_left, pooling_height, pooling_width, stride_height,
    stride_width, dilation_height, dilation_width, quantized_output_min, quantized_output_max, /*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_max_pooling2d_nhwc_s8(
                          op, batch_size, input_height, input_width, channels, /*input_pixel_stride=*/channels,
                          /*output_pixel_stride=*/channels, /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_max_pooling2d_nhwc_s8(op, input.data(), operator_output.data()));

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
    xnn_define_max_pooling_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, pooling_height, pooling_width, stride_height,
      stride_width, dilation_height, dilation_width, output_min, output_max, input_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  for (size_t i = 0; i < batch_size * output_height * output_width * channels; i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
  }
}

TEST_F(MaxPooling2DTestQU8, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), UINT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT8_C(0xA5));
  const uint8_t input_zero_point = u8dist(rng);
  const float input_scale = scale_dist(rng);
  const uint8_t output_zero_point = input_zero_point;
  const float output_scale = input_scale;
  const uint8_t quantized_output_min = xnn_qu8_quantize(output_min, output_scale, output_zero_point);
  const uint8_t quantized_output_max = xnn_qu8_quantize(output_max, output_scale, output_zero_point);

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_max_pooling2d_nhwc_u8(
    padding_top, padding_right, padding_bottom, padding_left, pooling_height, pooling_width, stride_height,
    stride_width, dilation_height, dilation_width, quantized_output_min, quantized_output_max, /*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_max_pooling2d_nhwc_u8(
                          op, batch_size, input_height, input_width, channels, /*input_pixel_stride=*/channels,
                          /*output_pixel_stride=*/channels, /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_max_pooling2d_nhwc_u8(op, input.data(), operator_output.data()));

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
    xnn_define_max_pooling_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, pooling_height, pooling_width, stride_height,
      stride_width, dilation_height, dilation_width, output_min, output_max, input_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  for (size_t i = 0; i < batch_size * output_height * output_width * channels; i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
  }
}

TEST_F(MaxPooling2DTestF16, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::fill(operator_output.begin(), operator_output.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT16_C(0x7E00) /* NaN */);

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_max_pooling2d_nhwc_f16(
    padding_top, padding_right, padding_bottom, padding_left, pooling_height, pooling_width, stride_height,
    stride_width, dilation_height, dilation_width, output_min, output_max, /*flags=*/0,
    &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_max_pooling2d_nhwc_f16(
                          op, batch_size, input_height, input_width, channels, channels, channels,
                          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_max_pooling2d_nhwc_f16(op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_max_pooling_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, pooling_height, pooling_width, stride_height,
      stride_width, dilation_height, dilation_width, output_min, output_max, input_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  for (size_t i = 0; i < batch_size * output_height * output_width * channels; i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
  }
}

TEST_F(MaxPooling2DTestF32, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_max_pooling2d_nhwc_f32(
    padding_top, padding_right, padding_bottom, padding_left, pooling_height, pooling_width, stride_height,
    stride_width, dilation_height, dilation_width, output_min, output_max, /*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_max_pooling2d_nhwc_f32(
                          op, batch_size, input_height, input_width, channels, /*input_pixel_stride=*/channels,
                          /*output_pixel_stride=*/channels, /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                          /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_max_pooling2d_nhwc_f32(op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_max_pooling_2d(
      subgraph, padding_top, padding_right, padding_bottom, padding_left, pooling_height, pooling_width, stride_height,
      stride_width, dilation_height, dilation_width, output_min, output_max, input_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  for (size_t i = 0; i < batch_size * output_height * output_width * channels; i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
  }
}

TEST_F(MaxPooling2DTestF32, Reshape)
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


TEST_F(MaxPooling2DTestF32, ReshapeWithPadding)
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

TEST_F(MaxPooling2DTestF32, ReshapeWithPaddingAndDilation)
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
