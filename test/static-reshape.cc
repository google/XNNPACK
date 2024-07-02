// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>  // For std::generate, std::shuffle.
#include <array>      // For std::array.
#include <cmath>
#include <cstddef>  // For size_t.
#include <cstdint>
#include <functional>  // For std::multiplies.
#include <memory>      // For std::unique_ptr.
#include <numeric>     // For std::accumulate.
#include <random>      // For std::uniform_real_distribution.
#include <vector>      // For std::vector.

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"
#include "subgraph-unary-tester.h"

template <typename InputType, typename OutputType = InputType,
          size_t min_dim = 0, size_t max_dim = XNN_MAX_TENSOR_DIMS,
          bool pad_output = false>
class StaticReshapeTest
    : public UnaryTest<InputType, OutputType, min_dim, max_dim, pad_output> {
 protected:
  void SetUp() override {
    new_dims_hint = PosiblySetOneDimsionToZero(this->dims);
  }

  // Set exactly one element of `given_dims` to zero.
  std::vector<size_t> RandomSetOneDimsionToZero(
      const std::vector<size_t> given_dims) {
    std::vector<size_t> result = given_dims;
    if (result.empty()) {
      return result;
    }
    // Randomly set one dimension to zero.
    auto dynamic_axis_dist =
        std::uniform_int_distribution<size_t>(0, result.size() - 1);
    const size_t dynamic_axis = dynamic_axis_dist(this->rng);
    result[dynamic_axis] = 0;
    return result;
  }

  // Set at most one element of `given_dims` to zero.
  std::vector<size_t> PosiblySetOneDimsionToZero(
      const std::vector<size_t> given_dims) {
    std::vector<size_t> result = given_dims;
    // Randomly set one dimension to zero.
    auto dynamic_axis_dist =
        std::uniform_int_distribution<size_t>(0, result.size());
    const size_t dynamic_axis = dynamic_axis_dist(this->rng);
    if (dynamic_axis == result.size()) {
      return result;
    }
    result[dynamic_axis] = 0;
    return result;
  }

  std::vector<size_t> new_dims_hint;
};

using StaticReshapeTestInt8 = StaticReshapeTest<int8_t>;
using StaticReshapeTestUint8 = StaticReshapeTest<uint8_t>;
using StaticReshapeTestF16 = StaticReshapeTest<uint16_t>;
using StaticReshapeTestF32 = StaticReshapeTest<float>;

TEST_F(StaticReshapeTestInt8, define)
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
                          subgraph, xnn_datatype_qint8, zero_point, scale, dims.size(), dims.data(),
                          nullptr, 0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, zero_point, scale, dims.size(), dims.data(),
                          nullptr, 1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_static_reshape(subgraph, new_dims_hint.size(), new_dims_hint.data(), input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_reshape);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qs8);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticReshapeTestUint8, define)
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
                          subgraph, xnn_datatype_quint8, zero_point, scale, dims.size(), dims.data(),
                          nullptr, 0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, zero_point, scale, dims.size(), dims.data(),
                          nullptr, 1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_static_reshape(subgraph, new_dims_hint.size(), new_dims_hint.data(), input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_reshape);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qu8);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticReshapeTestF16, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(0, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_static_reshape(subgraph, dims.size(), dims.data(), input_id, output_id, 0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_reshape);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp16);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticReshapeTestF32, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(0, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_static_reshape(subgraph, new_dims_hint.size(), new_dims_hint.data(), input_id, output_id, 0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_reshape);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(StaticReshapeTestInt8, matches_operator_api)
{
  const int32_t zero_point = i8dist(rng);
  const float scale = scale_dist(rng);
  std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), INT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));

  std::vector<size_t> output_dims = dims;
  std::shuffle(output_dims.begin(), output_dims.end(), rng);
  new_dims_hint = PosiblySetOneDimsionToZero(output_dims);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status =
    xnn_create_copy_nc_x8(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  size_t batch_size = NumElements(dims);
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op, batch_size, 1, 1, 1, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_copy_nc_x8(op, input.data(), operator_output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, zero_point, scale, dims.size(), dims.data(),
                          nullptr, /*external_id=*/0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, zero_point, scale, output_dims.size(), output_dims.data(),
                          nullptr, /*external_id=*/1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_static_reshape(subgraph, new_dims_hint.size(), new_dims_hint.data(), input_id, output_id, /*flags=*/0));

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

TEST_F(StaticReshapeTestUint8, matches_operator_api)
{
  const int32_t zero_point = u8dist(rng);
  const float scale = scale_dist(rng);
  std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), UINT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT8_C(0xA5));

  std::vector<size_t> output_dims = dims;
  std::shuffle(output_dims.begin(), output_dims.end(), rng);
  new_dims_hint = PosiblySetOneDimsionToZero(output_dims);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status =
    xnn_create_copy_nc_x8(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  size_t batch_size = NumElements(dims);
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op, batch_size, 1, 1, 1, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_copy_nc_x8(op, input.data(), operator_output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, zero_point, scale, dims.size(), dims.data(),
                          nullptr, /*external_id=*/0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, zero_point, scale, output_dims.size(), output_dims.data(),
                          nullptr, /*external_id=*/1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_static_reshape(subgraph, new_dims_hint.size(), new_dims_hint.data(), input_id, output_id, /*flags=*/0));

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

TEST_F(StaticReshapeTestF16, matches_operator_api)
{
  std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::fill(operator_output.begin(), operator_output.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT16_C(0x7E00) /* NaN */);

  std::vector<size_t> output_dims = dims;
  std::shuffle(output_dims.begin(), output_dims.end(), rng);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  xnn_status status = xnn_create_copy_nc_x16(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);
  size_t batch_size = NumElements(dims);
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x16(op, batch_size, 1, 1, 1, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_copy_nc_x16(op, input.data(), operator_output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  ASSERT_NE(nullptr, subgraph);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, dims.size(), dims.data(), nullptr, /*external_id=*/0,
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
    xnn_define_static_reshape(subgraph, output_dims.size(), output_dims.data(), input_id, output_id, /*flags=*/0));
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

TEST_F(StaticReshapeTestF32, matches_operator_api)
{
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  std::vector<size_t> output_dims = dims;
  std::shuffle(output_dims.begin(), output_dims.end(), rng);
  new_dims_hint = PosiblySetOneDimsionToZero(output_dims);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  xnn_status status = xnn_create_copy_nc_x32(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);
  size_t batch_size = NumElements(dims);
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x32(op, batch_size, 1, 1, 1, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_copy_nc_x32(op, input.data(), operator_output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph));
  ASSERT_NE(nullptr, subgraph);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, dims.size(), dims.data(), nullptr, /*external_id=*/0,
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
    xnn_define_static_reshape(subgraph, new_dims_hint.size(), new_dims_hint.data(), input_id, output_id, /*flags=*/0));
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

TEST_F(StaticReshapeTestF32, reshape_output) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  std::vector<size_t> output_dims = dims;
  std::shuffle(output_dims.begin(), output_dims.end(), rng);
  new_dims_hint = RandomSetOneDimsionToZero(output_dims);

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, dims.size(),
                dims.data(), nullptr, /*external_id=*/0,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr, /*external_id=*/1,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_define_static_reshape(subgraph, new_dims_hint.size(),
                                      new_dims_hint.data(), input_id, output_id,
                                      /*flags=*/0));
  ASSERT_EQ(
      xnn_status_success,
      xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Change the input shape (make it large enough to trigger a reallocation).
  if (!dims.empty()) {
    dims.front() *= 2;
    dims.back() *= 3;
  }
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, input_id, dims.size(),
                                       dims.data()));
  const struct xnn_node* node = &subgraph->nodes[0];
  xnn_status status =
      node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values,
                    /*threadpool=*/nullptr);
  ASSERT_EQ(status, dims.empty() ? xnn_status_success
                                 : xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  EXPECT_EQ(xnn_shape_multiply_all_dims(output_shape),
            std::accumulate(dims.begin(), dims.end(), size_t(1),
                            std::multiplies<size_t>()));

  // Change the input shape (make it a bit smaller again).
  if (!dims.empty()) {
    dims.front() /= 2;
  }
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, input_id, dims.size(),
                                       dims.data()));
  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values,
                          runtime->num_values, /*threadpool=*/nullptr),
            xnn_status_success);
  EXPECT_EQ(xnn_shape_multiply_all_dims(output_shape),
            std::accumulate(dims.begin(), dims.end(), size_t(1),
                            std::multiplies<size_t>()));
}
