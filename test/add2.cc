// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/node-type.h>
#include <xnnpack/operator.h>
#include <xnnpack/subgraph.h>

#include "subgraph-binary-tester.h"
#include <gtest/gtest.h>

using Add2TestQS8 = BinaryTest<int8_t>;
using Add2TestQU8 = BinaryTest<uint8_t>;
using Add2TestF32 = BinaryTest<float>;

TEST_F(Add2TestQS8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  const int32_t zero_point = i8dist(rng);
  const float scale = scale_dist(rng);

  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, zero_point, scale, input1_dims.size(), input1_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, zero_point, scale, input2_dims.size(), input2_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, zero_point, scale, output_dims.size(), output_dims.data(), nullptr,
                          XNN_INVALID_VALUE_ID, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_add2(subgraph, output_min, output_max, input1_id, input2_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_add2);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qs8);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input1_id);
  ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(Add2TestQU8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  const int32_t zero_point = u8dist(rng);
  const float scale = scale_dist(rng);

  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, zero_point, scale, input1_dims.size(), input1_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, zero_point, scale, input2_dims.size(), input2_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, zero_point, scale, output_dims.size(), output_dims.data(), nullptr,
                          XNN_INVALID_VALUE_ID, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_add2(subgraph, output_min, output_max, input1_id, input2_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_add2);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qu8);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input1_id);
  ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(Add2TestF32, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input1_dims.size(), input1_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input2_dims.size(), input2_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_add2(subgraph, output_min, output_max, input1_id, input2_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_add2);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input1_id);
  ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(Add2TestQS8, matches_operator_api)
{
  std::generate(input1.begin(), input1.end(), [&]() { return i8dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return i8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), INT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));

  const int32_t input1_zero_point = i8dist(rng);
  const float input1_scale = scale_dist(rng);
  const int32_t input2_zero_point = i8dist(rng);
  const float input2_scale = scale_dist(rng);
  const int32_t output_zero_point = i8dist(rng);
  const float output_scale = scale_dist(rng);
  const int8_t quantized_output_min = xnn_qs8_quantize(output_min, output_scale, output_zero_point);
  const int8_t quantized_output_max = xnn_qs8_quantize(output_max, output_scale, output_zero_point);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  // Call operator API.
  ASSERT_EQ(
    xnn_status_success, xnn_create_add_nd_qs8(
                          input1_zero_point, input1_scale, input2_zero_point, input2_scale, output_zero_point,
                          output_scale, quantized_output_min, quantized_output_max, /*flags=*/0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  ASSERT_EQ(
    xnn_status_success, xnn_reshape_add_nd_qs8(
                          op, input1_dims.size(), input1_dims.data(), input2_dims.size(), input2_dims.data(),
                          /*threadpool=*/nullptr));

  ASSERT_EQ(
    xnn_status_success, xnn_setup_add_nd_qs8(op, input1.data(), input2.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, input1_zero_point, input1_scale, input1_dims.size(), input1_dims.data(), nullptr,
      /*external_id=*/0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, input2_zero_point, input2_scale, input2_dims.size(), input2_dims.data(), nullptr,
      /*external_id=*/1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, output_zero_point, output_scale, output_dims.size(),
                          output_dims.data(), nullptr, /*external_id=*/2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_add2(subgraph, output_min, output_max, input1_id, input2_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 3> external = {
    xnn_external_value{input1_id, input1.data()}, xnn_external_value{input2_id, input2.data()},
    xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check output shape matches.
  size_t observed_output_num_dims = 0;
  std::vector<size_t> observed_output_dims(XNN_MAX_TENSOR_DIMS, 0);
  ASSERT_EQ(
    xnn_status_success,
    xnn_get_external_value_shape(runtime, output_id, &observed_output_num_dims, observed_output_dims.data()));
  ASSERT_EQ(output_dims.size(), observed_output_num_dims);
  for (size_t i = 0; i < observed_output_num_dims; i++) {
    ASSERT_EQ(output_dims[i], observed_output_dims[i]);
  }

  // Check outputs match.
  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(Add2TestQU8, matches_operator_api)
{
  std::generate(input1.begin(), input1.end(), [&]() { return u8dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return u8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), UINT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT8_C(0xA5));

  const int32_t input1_zero_point = u8dist(rng);
  const float input1_scale = scale_dist(rng);
  const int32_t input2_zero_point = u8dist(rng);
  const float input2_scale = scale_dist(rng);
  const int32_t output_zero_point = u8dist(rng);
  const float output_scale = scale_dist(rng);
  const uint8_t quantized_output_min = xnn_qu8_quantize(output_min, output_scale, output_zero_point);
  const uint8_t quantized_output_max = xnn_qu8_quantize(output_max, output_scale, output_zero_point);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  // Call operator API.
  ASSERT_EQ(
    xnn_status_success, xnn_create_add_nd_qu8(
                          input1_zero_point, input1_scale, input2_zero_point, input2_scale, output_zero_point,
                          output_scale, quantized_output_min, quantized_output_max, /*flags=*/0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  ASSERT_EQ(
    xnn_status_success, xnn_reshape_add_nd_qu8(
                          op, input1_dims.size(), input1_dims.data(), input2_dims.size(), input2_dims.data(),
                          /*threadpool=*/nullptr));

  ASSERT_EQ(
    xnn_status_success, xnn_setup_add_nd_qu8(op, input1.data(), input2.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, input1_zero_point, input1_scale, input1_dims.size(), input1_dims.data(), nullptr,
      /*external_id=*/0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, input2_zero_point, input2_scale, input2_dims.size(), input2_dims.data(), nullptr,
      /*external_id=*/1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, output_zero_point, output_scale, output_dims.size(),
                          output_dims.data(), nullptr, /*external_id=*/2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_add2(subgraph, output_min, output_max, input1_id, input2_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 3> external = {
    xnn_external_value{input1_id, input1.data()}, xnn_external_value{input2_id, input2.data()},
    xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check output shape matches.
  size_t observed_output_num_dims = 0;
  std::vector<size_t> observed_output_dims(XNN_MAX_TENSOR_DIMS, 0);
  ASSERT_EQ(
    xnn_status_success,
    xnn_get_external_value_shape(runtime, output_id, &observed_output_num_dims, observed_output_dims.data()));
  ASSERT_EQ(output_dims.size(), observed_output_num_dims);
  for (size_t i = 0; i < observed_output_num_dims; i++) {
    ASSERT_EQ(output_dims[i], observed_output_dims[i]);
  }

  // Check outputs match.
  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(Add2TestF32, matches_operator_api)
{
  std::generate(input1.begin(), input1.end(), [&]() { return f32dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  // Call operator API.
  ASSERT_EQ(xnn_status_success, xnn_create_add_nd_f32(output_min, output_max, 0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  ASSERT_EQ(
    xnn_status_success, xnn_reshape_add_nd_f32(
                          op, input1_dims.size(), input1_dims.data(), input2_dims.size(), input2_dims.data(),
                          /*threadpool=*/nullptr));

  ASSERT_EQ(
    xnn_status_success, xnn_setup_add_nd_f32(op, input1.data(), input2.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input1_dims.size(), input1_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input2_dims.size(), input2_dims.data(), nullptr,
                          /*external_id=*/1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, /*external_id=*/2,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_add2(subgraph, output_min, output_max, input1_id, input2_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 3> external = {
    xnn_external_value{input1_id, input1.data()}, xnn_external_value{input2_id, input2.data()},
    xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check output shape matches.
  size_t observed_output_num_dims = 0;
  std::vector<size_t> observed_output_dims(XNN_MAX_TENSOR_DIMS, 0);

  // Check Invalid calls for this API
  ASSERT_EQ(xnn_status_invalid_parameter,
    xnn_get_external_value_shape(runtime, XNN_INVALID_VALUE_ID, &observed_output_num_dims, observed_output_dims.data()));

  ASSERT_EQ(xnn_status_invalid_parameter,
    xnn_get_external_value_shape(runtime, output_id, nullptr, observed_output_dims.data()));

  ASSERT_EQ(xnn_status_invalid_parameter,
    xnn_get_external_value_shape(runtime, output_id, &observed_output_num_dims, nullptr));

  // Actually get the valid shape and check it matches
  ASSERT_EQ(
    xnn_status_success,
    xnn_get_external_value_shape(runtime, output_id, &observed_output_num_dims, observed_output_dims.data()));
  ASSERT_EQ(output_dims.size(), observed_output_num_dims);
  for (size_t i = 0; i < observed_output_num_dims; i++) {
    ASSERT_EQ(output_dims[i], observed_output_dims[i]);
  }

  // Check outputs match.
  ASSERT_EQ(subgraph_output, operator_output);
}
