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
#include <numeric>
#include <random>

#include <fp16/fp16.h>
#include <gtest/gtest.h>

#include <xnnpack.h>
#include <xnnpack/node-type.h>
#include <xnnpack/operator.h>
#include <xnnpack/subgraph.h>

#include "subgraph-unary-tester.h"

using StaticTransposeTestQS8 = UnaryTest<int8_t, int8_t, /*min_dim=*/1>;
using StaticTransposeTestQU8 = UnaryTest<uint8_t, uint8_t, /*min_dim=*/1>;
using StaticTransposeTestF16 = UnaryTest<uint16_t, uint16_t, /*min_dim=*/1>;
using StaticTransposeTestF32 = UnaryTest<float, float, /*min_dim=*/1>;

namespace {
template <typename Rng> std::vector<size_t> RandomPermutation(const std::vector<size_t>& input, Rng rng)
{
  std::vector<size_t> perm = std::vector<size_t>(input);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  return perm;
}

std::vector<size_t> PermuteInputDimensions(const std::vector<size_t>& input, std::vector<size_t> perm)
{
  std::vector<size_t> output = input;
  for (size_t i = 0; i < input.size(); i++) {
    output[i] = input[perm[i]];
  }
  return output;
}
}  // namespace

TEST_F(StaticTransposeTestQS8, define)
{
  const int32_t input_zero_point = i8dist(rng);
  const float input_scale = scale_dist(rng);
  const int32_t output_zero_point = input_zero_point;
  const float output_scale = input_scale;
  std::vector<size_t> perm = RandomPermutation(dims, rng);
  std::vector<size_t> output_dims = PermuteInputDimensions(dims, perm);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, input_zero_point, input_scale, dims.size(), dims.data(),
                          nullptr, 0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, output_zero_point, output_scale, output_dims.size(),
                          output_dims.data(), nullptr, 1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_transpose(subgraph, perm.size(), perm.data(), input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_transpose);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qs8);
  ASSERT_EQ(node->params.transpose.num_dims, dims.size());
  for (size_t i = 0; i < dims.size(); i++) {
    ASSERT_EQ(node->params.transpose.perm[i], perm[i]);
  }
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticTransposeTestQU8, define)
{
  const int32_t input_zero_point = u8dist(rng);
  const float input_scale = scale_dist(rng);
  const int32_t output_zero_point = input_zero_point;
  const float output_scale = input_scale;
  std::vector<size_t> perm = RandomPermutation(dims, rng);
  std::vector<size_t> output_dims = PermuteInputDimensions(dims, perm);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, input_zero_point, input_scale, dims.size(), dims.data(),
                          nullptr, 0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, output_zero_point, output_scale, output_dims.size(),
                          output_dims.data(), nullptr, 1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_transpose(subgraph, perm.size(), perm.data(), input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_transpose);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qu8);
  for (size_t i = 0; i < dims.size(); i++) {
    ASSERT_EQ(node->params.transpose.perm[i], perm[i]);
  }
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticTransposeTestF16, define)
{
  std::vector<size_t> perm = RandomPermutation(dims, rng);
  std::vector<size_t> output_dims = PermuteInputDimensions(dims, perm);

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
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_transpose(subgraph, perm.size(), perm.data(), input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_transpose);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp16);
  for (size_t i = 0; i < dims.size(); i++) {
    ASSERT_EQ(node->params.transpose.perm[i], perm[i]);
  }
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticTransposeTestF32, define)
{
  std::vector<size_t> perm = RandomPermutation(dims, rng);
  std::vector<size_t> output_dims = PermuteInputDimensions(dims, perm);

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
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_transpose(subgraph, perm.size(), perm.data(), input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_transpose);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  for (size_t i = 0; i < dims.size(); i++) {
    ASSERT_EQ(node->params.transpose.perm[i], perm[i]);
  }
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticTransposeTestQS8, matches_operator_api)
{
  const int32_t input_zero_point = i8dist(rng);
  const float input_scale = scale_dist(rng);
  const int32_t output_zero_point = input_zero_point;
  const float output_scale = input_scale;
  std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), INT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));
  std::vector<size_t> perm = RandomPermutation(dims, rng);
  std::vector<size_t> output_dims = PermuteInputDimensions(dims, perm);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_transpose_nd_x8(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  ASSERT_EQ(
    xnn_status_success, xnn_reshape_transpose_nd_x8(op, dims.size(), dims.data(), perm.data(), /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_transpose_nd_x8(op, input.data(), operator_output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, input_zero_point, input_scale, dims.size(), dims.data(),
                          nullptr, /*external_id=*/0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, output_zero_point, output_scale, dims.size(), dims.data(),
                          nullptr, /*external_id=*/1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_transpose(subgraph, perm.size(), perm.data(), input_id, output_id, /*flags=*/0));

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

TEST_F(StaticTransposeTestQU8, matches_operator_api)
{
  const int32_t input_zero_point = u8dist(rng);
  const float input_scale = scale_dist(rng);
  const int32_t output_zero_point = input_zero_point;
  const float output_scale = input_scale;
  std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), UINT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT8_C(0xA5));
  std::vector<size_t> perm = RandomPermutation(dims, rng);
  std::vector<size_t> output_dims = PermuteInputDimensions(dims, perm);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_transpose_nd_x8(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  ASSERT_EQ(
    xnn_status_success, xnn_reshape_transpose_nd_x8(op, dims.size(), dims.data(), perm.data(), /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_transpose_nd_x8(op, input.data(), operator_output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, input_zero_point, input_scale, dims.size(), dims.data(),
                          nullptr, /*external_id=*/0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, output_zero_point, output_scale, dims.size(), dims.data(),
                          nullptr, /*external_id=*/1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_transpose(subgraph, perm.size(), perm.data(), input_id, output_id, /*flags=*/0));

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

TEST_F(StaticTransposeTestF16, matches_operator_api)
{
  std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::fill(operator_output.begin(), operator_output.end(), fp16_ieee_from_fp32_value(nanf("")));
  std::fill(subgraph_output.begin(), subgraph_output.end(), fp16_ieee_from_fp32_value(nanf("")));
  std::vector<size_t> perm = RandomPermutation(dims, rng);
  std::vector<size_t> output_dims = PermuteInputDimensions(dims, perm);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_transpose_nd_x16(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  ASSERT_EQ(
    xnn_status_success,
    xnn_reshape_transpose_nd_x16(op, dims.size(), dims.data(), perm.data(), /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_transpose_nd_x16(op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, dims.size(), dims.data(), nullptr, /*external_id=*/0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, dims.size(), dims.data(), nullptr, /*external_id=*/1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_transpose(subgraph, perm.size(), perm.data(), input_id, output_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(StaticTransposeTestF32, matches_operator_api)
{
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));
  std::vector<size_t> perm = RandomPermutation(dims, rng);
  std::vector<size_t> output_dims = PermuteInputDimensions(dims, perm);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_transpose_nd_x32(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  ASSERT_EQ(
    xnn_status_success,
    xnn_reshape_transpose_nd_x32(op, dims.size(), dims.data(), perm.data(), /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_transpose_nd_x32(op, input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, /*external_id=*/0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, /*external_id=*/1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_transpose(subgraph, perm.size(), perm.data(), input_id, output_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}
