// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/node-type.h>
#include <xnnpack/operator.h>
#include <xnnpack/subgraph.h>

#include "subgraph-unary-tester.h"
#include <gtest/gtest.h>

TEST(DyanmicSubgraph, Test1)
{
  std::uniform_real_distribution<float> f32dist(-255.0f, 255.0f);
  std::vector<float> input0(8+XNN_EXTRA_BYTES/sizeof(float));
  std::vector<float> input1(8+XNN_EXTRA_BYTES/sizeof(float));
  std::iota(input0.begin(), input0.end(), -17);
  std::iota(input1.begin(), input1.end(), 11);
  std::vector<float> expected_output(8);
  for (size_t i = 0; i < expected_output.size(); ++i) {
    expected_output[i] = std::abs(input0[i] + input1[i]);
  }
  std::vector<float> subgraph_output(8);
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  uint32_t input0_id = XNN_INVALID_NODE_ID;
  uint32_t input1_id = XNN_INVALID_NODE_ID;
  size_t h = 2, w = 4;
  std::vector<size_t> dims{h,w};
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, /*external_id=*/0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input0_id));
  ASSERT_NE(input0_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, /*external_id=*/1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  std::vector<size_t> max_dims{4,8};
  ASSERT_EQ(
    xnn_status_success, xnn_tensor_set_max_shape(subgraph, input0_id, max_dims.size(), max_dims.data()));
  ASSERT_EQ(
    xnn_status_success, xnn_tensor_set_max_shape(subgraph, input1_id, max_dims.size(), max_dims.data()));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, /*external_id=*/2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success, xnn_tensor_set_max_shape(subgraph, output_id, max_dims.size(), max_dims.data()));
  uint32_t inter0_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, /*external_id=*/XNN_INVALID_NODE_ID,
                          /*flags=*/0, &inter0_id));
  ASSERT_NE(inter0_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(
    xnn_status_success, xnn_tensor_set_max_shape(subgraph, inter0_id, max_dims.size(), max_dims.data()));
  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_define_add2(subgraph, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), input0_id, input1_id, inter0_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_define_abs(subgraph, inter0_id, output_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 3> external = {
    xnn_external_value{input0_id, input0.data()}, xnn_external_value{input1_id, input1.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  for (size_t i = 0; i < expected_output.size(); ++i) {
    ASSERT_EQ(expected_output[i], subgraph_output[i]) << " i " << i;
  }

  h = 3;
  w = 4;
  std::vector<size_t> new_dims{h, w};
  input0.resize(h * w + XNN_EXTRA_BYTES/sizeof(float));
  input1.resize(h * w + XNN_EXTRA_BYTES/sizeof(float));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));
  subgraph_output.resize(h * w);
  expected_output.resize(h * w);
  std::iota(input0.begin(), input0.end(), -28);
  std::iota(input1.begin(), input1.end(), 1);
  for (size_t i = 0; i < expected_output.size(); ++i) {
    expected_output[i] = std::abs(input0[i] + input1[i]);
  }
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_external_value(runtime, input0_id, new_dims.size(), new_dims.data()));
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_external_value(runtime, input1_id, new_dims.size(), new_dims.data()));
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_external_value(runtime, output_id, new_dims.size(), new_dims.data()));
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_runtime(runtime));
  external = {
    xnn_external_value{input0_id, input0.data()}, xnn_external_value{input1_id, input1.data()}, xnn_external_value{output_id, subgraph_output.data()}};

  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  for (size_t i = 0; i < expected_output.size(); ++i) {
    ASSERT_EQ(expected_output[i], subgraph_output[i]) << " i " << i;
  }
}
