// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "subgraph-unary-tester.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

template <typename T> class StaticSliceTest : public UnaryTest<T, T, /*min_dim=*/1> {
public:
  StaticSliceTest()
      : UnaryTest<T, T, /*min_dim=*/1>{}
  {
    offsets = RandomOffsets(this->dims);
    sizes = RandomSizes(this->dims, offsets);
    // Overwrite outputs since slice output size is different from input.
    this->operator_output = std::vector<T>(this->NumElements(sizes));
    this->subgraph_output = std::vector<T>(this->NumElements(sizes));
  }

private:
  std::vector<size_t> RandomOffsets(const std::vector<size_t>& input_dims)
  {
    std::vector<size_t> offsets(input_dims.size());
    for (size_t i = 0; i < input_dims.size(); i++) {
      auto offset_dist = std::uniform_int_distribution<size_t>(0, input_dims[i] - 1);
      offsets[i] = offset_dist(this->rng);
    }
    return offsets;
  }

  std::vector<size_t> RandomSizes(const std::vector<size_t>& input_dims, const std::vector<size_t>& offsets)
  {
    std::vector<size_t> sizes(input_dims.size());
    for (size_t i = 0; i < input_dims.size(); i++) {
      auto size_dist = std::uniform_int_distribution<size_t>(1, input_dims[i] - offsets[i]);
      sizes[i] = size_dist(this->rng);
    }
    return sizes;
  }

protected:
  std::vector<size_t> offsets;
  std::vector<size_t> sizes;
};

using StaticSliceTestQS8 = StaticSliceTest<int8_t>;
using StaticSliceTestQU8 = StaticSliceTest<uint8_t>;
using StaticSliceTestF32 = StaticSliceTest<float>;

TEST_F(StaticSliceTestQS8, define)
{
  const int32_t zero_point = i8dist(rng);
  const float scale = scale_dist(rng);

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
                          subgraph, xnn_datatype_qint8, zero_point, scale, sizes.size(), sizes.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_slice(subgraph, dims.size(), offsets.data(), sizes.data(), input_id, output_id, /*flags=*/0));

  EXPECT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  EXPECT_EQ(node->type, xnn_node_type_static_slice);
  EXPECT_EQ(node->compute_type, xnn_compute_type_qs8);
  EXPECT_EQ(node->num_inputs, 1);
  EXPECT_EQ(node->inputs[0], input_id);
  EXPECT_EQ(node->num_outputs, 1);
  EXPECT_EQ(node->outputs[0], output_id);
  EXPECT_EQ(node->flags, 0);
  EXPECT_EQ(node->params.slice.num_dims, dims.size());
  EXPECT_THAT(offsets, testing::ElementsAreArray(node->params.slice.offsets, dims.size()));
  EXPECT_THAT(sizes, testing::ElementsAreArray(node->params.slice.sizes, dims.size()));
}

TEST_F(StaticSliceTestQU8, define)
{
  const int32_t zero_point = u8dist(rng);
  const float scale = scale_dist(rng);

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
                          subgraph, xnn_datatype_quint8, zero_point, scale, sizes.size(), sizes.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_slice(subgraph, dims.size(), offsets.data(), sizes.data(), input_id, output_id, /*flags=*/0));

  EXPECT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  EXPECT_EQ(node->type, xnn_node_type_static_slice);
  EXPECT_EQ(node->compute_type, xnn_compute_type_qu8);
  EXPECT_EQ(node->num_inputs, 1);
  EXPECT_EQ(node->inputs[0], input_id);
  EXPECT_EQ(node->num_outputs, 1);
  EXPECT_EQ(node->outputs[0], output_id);
  EXPECT_EQ(node->flags, 0);
  EXPECT_EQ(node->params.slice.num_dims, dims.size());
  EXPECT_THAT(offsets, testing::ElementsAreArray(node->params.slice.offsets, dims.size()));
  EXPECT_THAT(sizes, testing::ElementsAreArray(node->params.slice.sizes, dims.size()));
}

TEST_F(StaticSliceTestF32, define)
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
                          subgraph, xnn_datatype_fp32, sizes.size(), sizes.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_slice(subgraph, dims.size(), offsets.data(), sizes.data(), input_id, output_id, /*flags=*/0));

  EXPECT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  EXPECT_EQ(node->type, xnn_node_type_static_slice);
  EXPECT_EQ(node->compute_type, xnn_compute_type_fp32);
  EXPECT_EQ(node->num_inputs, 1);
  EXPECT_EQ(node->inputs[0], input_id);
  EXPECT_EQ(node->num_outputs, 1);
  EXPECT_EQ(node->outputs[0], output_id);
  EXPECT_EQ(node->flags, 0);
  EXPECT_EQ(node->params.slice.num_dims, dims.size());
  EXPECT_THAT(offsets, testing::ElementsAreArray(node->params.slice.offsets, dims.size()));
  EXPECT_THAT(sizes, testing::ElementsAreArray(node->params.slice.sizes, dims.size()));
}

TEST_F(StaticSliceTestQS8, matches_operator_api)
{
  const int32_t zero_point = i8dist(rng);
  const float scale = scale_dist(rng);

  std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), INT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  xnn_status status = xnn_create_slice_nd_x8(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);
  ASSERT_EQ(
    xnn_status_success, xnn_setup_slice_nd_x8(
                          op, dims.size(), dims.data(), offsets.data(), sizes.data(), input.data(),
                          operator_output.data(), /*threadpool=*/nullptr));
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
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, zero_point, scale, sizes.size(), sizes.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_slice(subgraph, dims.size(), offsets.data(), sizes.data(), input_id, output_id, /*flags=*/0));
  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  EXPECT_EQ(subgraph_output, operator_output);
}

TEST_F(StaticSliceTestQU8, matches_operator_api)
{
  const int32_t zero_point = u8dist(rng);
  const float scale = scale_dist(rng);

  std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), INT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  xnn_status status = xnn_create_slice_nd_x8(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);
  ASSERT_EQ(
    xnn_status_success, xnn_setup_slice_nd_x8(
                          op, dims.size(), dims.data(), offsets.data(), sizes.data(), input.data(),
                          operator_output.data(), /*threadpool=*/nullptr));
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
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, zero_point, scale, sizes.size(), sizes.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_slice(subgraph, dims.size(), offsets.data(), sizes.data(), input_id, output_id, /*flags=*/0));
  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  EXPECT_EQ(subgraph_output, operator_output);
}

TEST_F(StaticSliceTestF32, matches_operator_api)
{
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  xnn_status status = xnn_create_slice_nd_x32(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);
  ASSERT_EQ(
    xnn_status_success, xnn_setup_slice_nd_x32(
                          op, dims.size(), dims.data(), offsets.data(), sizes.data(), input.data(),
                          operator_output.data(), /*threadpool=*/nullptr));
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
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, sizes.size(), sizes.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_slice(subgraph, dims.size(), offsets.data(), sizes.data(), input_id, output_id, /*flags=*/0));
  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  EXPECT_EQ(subgraph_output, operator_output);
}
