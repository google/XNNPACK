// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/math.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"
#include "subgraph-unary-tester.h"
#include "runtime-flags.h"

template <typename T> class StaticSliceTest : public UnaryTest<T, T, /*min_dim=*/1> {
public:
  StaticSliceTest()
      : UnaryTest<T, T, /*min_dim=*/1>{}
  {
    std::tie(begins, offsets) = RandomBegins(this->dims);
    std::tie(ends, sizes) = RandomEnds(this->dims);

    // Overwrite outputs since slice output size is different from input.
    this->operator_output = xnnpack::Buffer<T>(this->NumElements(sizes));
    this->subgraph_output = xnnpack::Buffer<T>(this->NumElements(sizes));
  }

private:
  std::tuple<std::vector<int64_t>, std::vector<size_t>> RandomBegins(const std::vector<size_t>& input_dims)
  {
    std::vector<int64_t> begins(input_dims.size());
    std::vector<size_t> offsets(input_dims.size());
    for (size_t i = 0; i < input_dims.size(); i++) {
      const int64_t range = input_dims[i];
      auto offset_dist = std::uniform_int_distribution<int64_t>(-range, range - 1);
      begins[i] = offset_dist(this->rng);
      offsets[i] = begins[i] < 0 ? input_dims[i] + begins[i] : begins[i];
    }
    return {begins, offsets};
  }

  std::tuple<std::vector<int64_t>, std::vector<size_t>> RandomEnds(
      const std::vector<size_t>& input_dims)
  {
    std::vector<int64_t> ends(input_dims.size());
    std::vector<size_t> sizes(input_dims.size());
    for (size_t i = 0; i < input_dims.size(); i++) {
      const int64_t range = input_dims[i] - offsets[i];
      auto size_dist = std::uniform_int_distribution<int64_t>(-range + 1, range);
      int64_t r = size_dist(this->rng);
      // ends[i] == 0 means "infer end as largest interval"
      ends[i] = r <= 0 ? r : offsets[i] + r;
      sizes[i] = ends[i] <= 0 ? input_dims[i] + ends[i] - offsets[i] : ends[i] - offsets[i];
    }
    return {ends, sizes};
  }

protected:
  std::vector<int64_t> begins;
  std::vector<int64_t> ends;
  std::vector<size_t> offsets;
  std::vector<size_t> sizes;
};

using StaticSliceTestQS8 = StaticSliceTest<int8_t>;
using StaticSliceTestQU8 = StaticSliceTest<uint8_t>;
using StaticSliceTestF16 = StaticSliceTest<xnn_float16>;
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
    xnn_define_static_slice_v3(subgraph, dims.size(), begins.data(), ends.data(), /*strides*/nullptr, input_id, output_id, /*flags=*/0));

  EXPECT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  EXPECT_EQ(node->type, xnn_node_type_static_slice);
  EXPECT_EQ(node->num_inputs, 1);
  EXPECT_EQ(node->inputs[0], input_id);
  EXPECT_EQ(node->num_outputs, 1);
  EXPECT_EQ(node->outputs[0], output_id);
  EXPECT_EQ(node->flags, 0);
  EXPECT_EQ(node->params.slice.num_dims, dims.size());
  EXPECT_THAT(begins, testing::ElementsAreArray(node->params.slice.begins, dims.size()));
  EXPECT_THAT(ends, testing::ElementsAreArray(node->params.slice.ends, dims.size()));
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
    xnn_define_static_slice_v3(subgraph, dims.size(), begins.data(), ends.data(), /*strides*/nullptr, input_id, output_id, /*flags=*/0));

  EXPECT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  EXPECT_EQ(node->type, xnn_node_type_static_slice);
  EXPECT_EQ(node->num_inputs, 1);
  EXPECT_EQ(node->inputs[0], input_id);
  EXPECT_EQ(node->num_outputs, 1);
  EXPECT_EQ(node->outputs[0], output_id);
  EXPECT_EQ(node->flags, 0);
  EXPECT_EQ(node->params.slice.num_dims, dims.size());
  EXPECT_THAT(begins, testing::ElementsAreArray(node->params.slice.begins, dims.size()));
  EXPECT_THAT(ends, testing::ElementsAreArray(node->params.slice.ends, dims.size()));
}

TEST_F(StaticSliceTestF16, define)
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
                          subgraph, xnn_datatype_fp16, sizes.size(), sizes.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_slice_v3(subgraph, dims.size(), begins.data(), ends.data(), /*strides*/nullptr, input_id, output_id, /*flags=*/0));

  EXPECT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  EXPECT_EQ(node->type, xnn_node_type_static_slice);
  EXPECT_EQ(node->num_inputs, 1);
  EXPECT_EQ(node->inputs[0], input_id);
  EXPECT_EQ(node->num_outputs, 1);
  EXPECT_EQ(node->outputs[0], output_id);
  EXPECT_EQ(node->flags, 0);
  EXPECT_EQ(node->params.slice.num_dims, dims.size());
  EXPECT_THAT(begins, testing::ElementsAreArray(node->params.slice.begins, dims.size()));
  EXPECT_THAT(ends, testing::ElementsAreArray(node->params.slice.ends, dims.size()));
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
    xnn_define_static_slice_v3(subgraph, dims.size(), begins.data(), ends.data(), /*strides*/nullptr, input_id, output_id, /*flags=*/0));

  EXPECT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  EXPECT_EQ(node->type, xnn_node_type_static_slice);
  EXPECT_EQ(node->num_inputs, 1);
  EXPECT_EQ(node->inputs[0], input_id);
  EXPECT_EQ(node->num_outputs, 1);
  EXPECT_EQ(node->outputs[0], output_id);
  EXPECT_EQ(node->flags, 0);
  EXPECT_EQ(node->params.slice.num_dims, dims.size());
  EXPECT_THAT(begins, testing::ElementsAreArray(node->params.slice.begins, dims.size()));
  EXPECT_THAT(ends, testing::ElementsAreArray(node->params.slice.ends, dims.size()));
}

TEST_F(StaticSliceTestQS8, matches_operator_api)
{
  const int32_t zero_point = i8dist(rng);
  const float scale = scale_dist(rng);

  std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });

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
    xnn_status_success,
    xnn_reshape_slice_nd_x8(op, dims.size(), dims.data(), offsets.data(), sizes.data(), /*threadpool=*/nullptr));
  ASSERT_EQ(
    xnn_status_success, xnn_setup_slice_nd_x8(op, input.data(), operator_output.data()));
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
    xnn_define_static_slice_v3(subgraph, dims.size(), begins.data(), ends.data(), /*strides*/nullptr, input_id, output_id, /*flags=*/0));
  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, xnn_test_runtime_flags(), &runtime));
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
    xnn_status_success,
    xnn_reshape_slice_nd_x8(op, dims.size(), dims.data(), offsets.data(), sizes.data(), /*threadpool=*/nullptr));
  ASSERT_EQ(
    xnn_status_success, xnn_setup_slice_nd_x8(op, input.data(), operator_output.data()));
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
    xnn_define_static_slice_v3(subgraph, dims.size(), begins.data(), ends.data(), /*strides*/nullptr, input_id, output_id, /*flags=*/0));
  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  EXPECT_EQ(subgraph_output, operator_output);
}

TEST_F(StaticSliceTestF16, matches_operator_api)
{
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  xnn_status status = xnn_create_slice_nd_x16(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);
  ASSERT_EQ(
    xnn_status_success,
    xnn_reshape_slice_nd_x16(op, dims.size(), dims.data(), offsets.data(), sizes.data(), /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_slice_nd_x16(op, input.data(), operator_output.data()));
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
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, sizes.size(), sizes.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_static_slice_v3(subgraph, dims.size(), begins.data(), ends.data(), /*strides*/nullptr, input_id, output_id, /*flags=*/0));
  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, xnn_test_runtime_flags(), &runtime));
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
    xnn_status_success,
    xnn_reshape_slice_nd_x32(op, dims.size(), dims.data(), offsets.data(), sizes.data(), /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_slice_nd_x32(op, input.data(), operator_output.data()));
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
    xnn_define_static_slice_v3(subgraph, dims.size(), begins.data(), ends.data(), /*strides*/nullptr, input_id, output_id, /*flags=*/0));
  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, xnn_test_runtime_flags(), &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  EXPECT_EQ(subgraph_output, operator_output);
}

TEST_F(StaticSliceTestF32, illegal_stride_values)
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

  std::vector<int64_t> strides(this->dims.size(), 1);
  strides[0] = 2;

  ASSERT_EQ(
    xnn_status_invalid_parameter,
    xnn_define_static_slice_v3(subgraph, dims.size(), begins.data(), ends.data(), strides.data(), input_id, output_id, /*flags=*/0));
}

