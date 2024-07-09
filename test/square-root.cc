// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph.h"
#include "subgraph-unary-tester.h"

using SquareRootTestF16 = UnaryTest<uint16_t>;
using SquareRootTestF32 = UnaryTest<float>;

TEST_F(SquareRootTestF16, define)
{
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

  ASSERT_EQ(xnn_status_success, xnn_define_square_root(subgraph, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_square_root);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp16);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(SquareRootTestF32, define)
{
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

  ASSERT_EQ(xnn_status_success, xnn_define_square_root(subgraph, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_square_root);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(SquareRootTestF16, matches_operator_api)
{
  std::uniform_real_distribution<float> f32dist(0.1f, 5.0f);
  std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::fill(operator_output.begin(), operator_output.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT16_C(0x7E00) /* NaN */);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status =
    xnn_create_square_root_nc_f16(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  ASSERT_EQ(xnn_status_success, xnn_reshape_square_root_nc_f16(op, batch_size, channels, channels, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_square_root_nc_f16(op, input.data(), operator_output.data()));
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
  ASSERT_EQ(xnn_status_success, xnn_define_square_root(subgraph, input_id, output_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(SquareRootTestF32, matches_operator_api)
{
  std::uniform_real_distribution<float> f32dist(0.1f, 5.0f);
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status =
    xnn_create_square_root_nc_f32(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  ASSERT_EQ(xnn_status_success, xnn_reshape_square_root_nc_f32(op, batch_size, channels, channels, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_square_root_nc_f32(op, input.data(), operator_output.data()));
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
  ASSERT_EQ(xnn_status_success, xnn_define_square_root(subgraph, input_id, output_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}
