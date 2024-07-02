// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>  // For std::generate, std::shuffle.
#include <array>      // For std::array.
#include <cmath>
#include <cstddef>  // For size_t.
#include <cstdint>
#include <memory>  // For std::unique_ptr.
#include <random>  // For std::uniform_real_distribution.
#include <vector>  // For std::vector.

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/math.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/requantization.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"
#include "subgraph-unary-tester.h"

using StaticConstantPadTestInt8 = UnaryTest<int8_t>;
using StaticConstantPadTestUint8 = UnaryTest<uint8_t>;
using StaticConstantPadTestF16 = UnaryTest<uint16_t>;
using StaticConstantPadTestF32 = UnaryTest<float>;

TEST_F(StaticConstantPadTestInt8, define)
{
  const int32_t zero_point = i8dist(rng);
  const float scale = scale_dist(rng);
  std::array<size_t, XNN_MAX_TENSOR_DIMS> pre_paddings;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> post_paddings;
  std::fill(pre_paddings.begin(), pre_paddings.begin() + dims.size(), dim_dist(rng));
  std::fill(post_paddings.begin(), post_paddings.begin() + dims.size(), dim_dist(rng));
  float padding_value = f32dist(rng);
  uint32_t quantized_padding_value = xnn_qs8_quantize(padding_value, scale, zero_point);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, zero_point, scale, dims.size(), dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, zero_point, scale, dims.size(), dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_constant_pad(
      subgraph, pre_paddings.data(), post_paddings.data(), padding_value, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_constant_pad);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qs8);
  for (size_t i = 0; i < dims.size(); i++) {
    ASSERT_EQ(node->params.static_pad.pre_paddings[i], pre_paddings[i]);
    ASSERT_EQ(node->params.static_pad.post_paddings[i], post_paddings[i]);
  }
  ASSERT_EQ(node->params.static_pad.padding_value, quantized_padding_value);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticConstantPadTestUint8, define)
{
  const int32_t zero_point = u8dist(rng);
  const float scale = scale_dist(rng);
  std::array<size_t, XNN_MAX_TENSOR_DIMS> pre_paddings;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> post_paddings;
  std::fill(pre_paddings.begin(), pre_paddings.begin() + dims.size(), dim_dist(rng));
  std::fill(post_paddings.begin(), post_paddings.begin() + dims.size(), dim_dist(rng));
  float padding_value = f32dist(rng);
  uint32_t quantized_padding_value = xnn_qu8_quantize(padding_value, scale, zero_point);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, zero_point, scale, dims.size(), dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, zero_point, scale, dims.size(), dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_constant_pad(
      subgraph, pre_paddings.data(), post_paddings.data(), padding_value, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_constant_pad);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qu8);
  for (size_t i = 0; i < dims.size(); i++) {
    ASSERT_EQ(node->params.static_pad.pre_paddings[i], pre_paddings[i]);
    ASSERT_EQ(node->params.static_pad.post_paddings[i], post_paddings[i]);
  }
  ASSERT_EQ(node->params.static_pad.padding_value, quantized_padding_value);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticConstantPadTestF16, define)
{
  std::array<size_t, XNN_MAX_TENSOR_DIMS> pre_paddings;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> post_paddings;
  std::fill(pre_paddings.begin(), pre_paddings.begin() + dims.size(), dim_dist(rng));
  std::fill(post_paddings.begin(), post_paddings.begin() + dims.size(), dim_dist(rng));
  uint16_t padding_value = f32dist(rng);
  uint32_t padding_value_as_bits = fp16_ieee_from_fp32_value(padding_value);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, dims.size(), dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, dims.size(), dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_constant_pad(
      subgraph, pre_paddings.data(), post_paddings.data(), padding_value, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_constant_pad);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp16);
  for (size_t i = 0; i < dims.size(); i++) {
    ASSERT_EQ(node->params.static_pad.pre_paddings[i], pre_paddings[i]);
    ASSERT_EQ(node->params.static_pad.post_paddings[i], post_paddings[i]);
  }
  ASSERT_EQ(node->params.static_pad.padding_value, padding_value_as_bits);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticConstantPadTestF32, define)
{
  std::array<size_t, XNN_MAX_TENSOR_DIMS> pre_paddings;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> post_paddings;
  std::fill(pre_paddings.begin(), pre_paddings.begin() + dims.size(), dim_dist(rng));
  std::fill(post_paddings.begin(), post_paddings.begin() + dims.size(), dim_dist(rng));
  float padding_value = f32dist(rng);
  uint32_t padding_value_as_bits = float_as_uint32(padding_value);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_constant_pad(
      subgraph, pre_paddings.data(), post_paddings.data(), padding_value, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_constant_pad);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  for (size_t i = 0; i < dims.size(); i++) {
    ASSERT_EQ(node->params.static_pad.pre_paddings[i], pre_paddings[i]);
    ASSERT_EQ(node->params.static_pad.post_paddings[i], post_paddings[i]);
  }
  ASSERT_EQ(node->params.static_pad.padding_value, padding_value_as_bits);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticConstantPadTestInt8, matches_operator_api)
{
  const int32_t zero_point = i8dist(rng);
  const float scale = scale_dist(rng);
  std::array<size_t, XNN_MAX_TENSOR_DIMS> pre_paddings;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> post_paddings;
  std::fill(pre_paddings.begin(), pre_paddings.begin() + dims.size(), dim_dist(rng));
  std::fill(post_paddings.begin(), post_paddings.begin() + dims.size(), dim_dist(rng));
  float padding_value = f32dist(rng);
  uint32_t quantized_padding_value = xnn_qs8_quantize(padding_value, scale, zero_point);
  std::vector<size_t> output_dims = dims;
  for (size_t i = 0; i < dims.size(); i++) {
    output_dims[i] = pre_paddings[i] + output_dims[i] + post_paddings[i];
  }
  // Output sizes
  operator_output = std::vector<int8_t>(NumElements(output_dims));
  subgraph_output = std::vector<int8_t>(operator_output.size());
  std::fill(operator_output.begin(), operator_output.end(), INT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_constant_pad_nd_x8(&quantized_padding_value, /*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);
  ASSERT_EQ(
    xnn_status_success,
    xnn_reshape_constant_pad_nd_x8(
      op, dims.size(), dims.data(), pre_paddings.data(), post_paddings.data(), /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_constant_pad_nd_x8(op, input.data(), operator_output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, zero_point, scale, dims.size(), dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, zero_point, scale, output_dims.size(), output_dims.data(), nullptr, 1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_constant_pad(
      subgraph, pre_paddings.data(), post_paddings.data(), padding_value, input_id, output_id, /*flags=*/0));

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

TEST_F(StaticConstantPadTestUint8, matches_operator_api)
{
  const int32_t zero_point = u8dist(rng);
  const float scale = scale_dist(rng);
  std::array<size_t, XNN_MAX_TENSOR_DIMS> pre_paddings;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> post_paddings;
  std::fill(pre_paddings.begin(), pre_paddings.begin() + dims.size(), dim_dist(rng));
  std::fill(post_paddings.begin(), post_paddings.begin() + dims.size(), dim_dist(rng));
  float padding_value = f32dist(rng);
  uint32_t quantized_padding_value = xnn_qu8_quantize(padding_value, scale, zero_point);
  std::vector<size_t> output_dims = dims;
  for (size_t i = 0; i < dims.size(); i++) {
    output_dims[i] = pre_paddings[i] + output_dims[i] + post_paddings[i];
  }
  // Output sizes
  operator_output = std::vector<uint8_t>(NumElements(output_dims));
  subgraph_output = std::vector<uint8_t>(operator_output.size());
  std::fill(operator_output.begin(), operator_output.end(), UINT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT8_C(0xA5));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_constant_pad_nd_x8(&quantized_padding_value, /*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);
  ASSERT_EQ(
    xnn_status_success,
    xnn_reshape_constant_pad_nd_x8(
      op, dims.size(), dims.data(), pre_paddings.data(), post_paddings.data(), /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_constant_pad_nd_x8(op, input.data(), operator_output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, zero_point, scale, dims.size(), dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, zero_point, scale, output_dims.size(), output_dims.data(), nullptr, 1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_constant_pad(
      subgraph, pre_paddings.data(), post_paddings.data(), padding_value, input_id, output_id, /*flags=*/0));

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

TEST_F(StaticConstantPadTestF16, matches_operator_api)
{
  std::array<size_t, XNN_MAX_TENSOR_DIMS> pre_paddings;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> post_paddings;
  std::fill(pre_paddings.begin(), pre_paddings.begin() + dims.size(), dim_dist(rng));
  std::fill(post_paddings.begin(), post_paddings.begin() + dims.size(), dim_dist(rng));
  float padding_value = f32dist(rng);
  uint32_t padding_value_as_u32 = fp16_ieee_from_fp32_value(padding_value);
  std::vector<size_t> output_dims = dims;
  for (size_t i = 0; i < dims.size(); i++) {
    output_dims[i] = pre_paddings[i] + output_dims[i] + post_paddings[i];
  }
  // Output sizes
  operator_output = std::vector<uint16_t>(NumElements(output_dims));
  subgraph_output = std::vector<uint16_t>(operator_output.size());
  std::fill(operator_output.begin(), operator_output.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT16_C(0x7E00) /* NaN */);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_constant_pad_nd_x16(&padding_value_as_u32, /*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);
  ASSERT_EQ(
    xnn_status_success,
    xnn_reshape_constant_pad_nd_x16(
      op, dims.size(), dims.data(), pre_paddings.data(), post_paddings.data(), /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_constant_pad_nd_x16(op, input.data(), operator_output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, dims.size(), dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr, 1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_constant_pad(
      subgraph, pre_paddings.data(), post_paddings.data(), padding_value, input_id, output_id, /*flags=*/0));

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

TEST_F(StaticConstantPadTestF32, matches_operator_api)
{
  std::array<size_t, XNN_MAX_TENSOR_DIMS> pre_paddings;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> post_paddings;
  std::fill(pre_paddings.begin(), pre_paddings.begin() + dims.size(), dim_dist(rng));
  std::fill(post_paddings.begin(), post_paddings.begin() + dims.size(), dim_dist(rng));
  float padding_value = f32dist(rng);
  uint32_t padding_value_as_u32 = float_as_uint32(padding_value);
  std::vector<size_t> output_dims = dims;
  for (size_t i = 0; i < dims.size(); i++) {
    output_dims[i] = pre_paddings[i] + output_dims[i] + post_paddings[i];
  }
  // Output sizes
  operator_output = std::vector<float>(NumElements(output_dims));
  subgraph_output = std::vector<float>(operator_output.size());
  std::fill(operator_output.begin(), operator_output.end(), std::nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), std::nanf(""));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_constant_pad_nd_x32(&padding_value_as_u32, /*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);
  ASSERT_EQ(
    xnn_status_success,
    xnn_reshape_constant_pad_nd_x32(
      op, dims.size(), dims.data(), pre_paddings.data(), post_paddings.data(), /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_constant_pad_nd_x32(op, input.data(), operator_output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, 1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_constant_pad(
      subgraph, pre_paddings.data(), post_paddings.data(), padding_value, input_id, output_id, /*flags=*/0));

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

TEST_F(StaticConstantPadTestF32, reshape_output)
{
  std::array<size_t, XNN_MAX_TENSOR_DIMS> pre_paddings;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> post_paddings;
  std::fill(pre_paddings.begin(), pre_paddings.begin() + dims.size(), dim_dist(rng));
  std::fill(post_paddings.begin(), post_paddings.begin() + dims.size(), dim_dist(rng));
  float padding_value = f32dist(rng);
  std::vector<size_t> output_dims = dims;
  for (size_t i = 0; i < dims.size(); i++) {
    output_dims[i] = pre_paddings[i] + output_dims[i] + post_paddings[i];
  }
  subgraph_output = std::vector<float>(NumElements(output_dims));
  std::fill(subgraph_output.begin(), subgraph_output.end(), std::nanf(""));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, 1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_constant_pad(
      subgraph, pre_paddings.data(), post_paddings.data(), padding_value, input_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  if (!dims.empty()) {
    dims[0] += 2;
    ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input_id, dims.size(), dims.data()));
    ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));
    const struct xnn_node* node = &subgraph->nodes[0];
    const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
    for (size_t i = 0; i < output_shape->num_dims; ++i) {
      ASSERT_EQ(output_shape->dim[i], dims[i] + pre_paddings[i] + post_paddings[i]);
    }

    dims[0] -= 1;
    ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input_id, dims.size(), dims.data()));
    ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_success);
    for (size_t i = 0; i < output_shape->num_dims; ++i) {
      ASSERT_EQ(output_shape->dim[i], dims[i] + pre_paddings[i] + post_paddings[i]);
    }
  }
}
