// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"

template <typename T> class Concatenate2Test : public ::testing::Test {
 protected:
  Concatenate2Test() {
    shape_dist = std::uniform_int_distribution<size_t>(1, XNN_MAX_TENSOR_DIMS);
    dim_dist = std::uniform_int_distribution<size_t>(1, 9);
    f32dist = std::uniform_real_distribution<float>();
    i8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    u8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    scale_dist = std::uniform_real_distribution<float>(0.1f, 5.0f);

    input1_dims = RandomShape();
    axis = RandomAxis(input1_dims);
    input2_dims = RandomShape(input1_dims, axis);
    output_dims = input1_dims;
    output_dims[axis] = input1_dims[axis] + input2_dims[axis];

    input1 = std::vector<T>(NumElements(input1_dims));
    input2 = std::vector<T>(NumElements(input2_dims));
    operator_output = std::vector<T>(NumElements(output_dims));
    subgraph_output = std::vector<T>(NumElements(output_dims));

    signed_zero_point = i8dist(rng);
    unsigned_zero_point = u8dist(rng);
    scale = scale_dist(rng);

    batch_size = 1;
    channels_1 = 1;
    channels_2 = 1;
    for (size_t i = 0; i < axis; i++) {
      batch_size *= output_dims[i];
    }

    for (size_t i = axis; i < input1_dims.size(); i++) {
      channels_1 *= input1_dims[i];
      channels_2 *= input2_dims[i];
    }
    output_stride = channels_1 + channels_2;
  }

  std::vector<size_t> RandomShape()
  {
    std::vector<size_t> dims(shape_dist(rng));
    std::generate(dims.begin(), dims.end(), [&] { return dim_dist(rng); });
    return dims;
  }

  std::vector<size_t> RandomShape(const std::vector<size_t> base_dims, size_t axis)
  {
    auto dims = base_dims;
    dims[axis] = dim_dist(rng);
    return dims;
  }

  size_t RandomAxis(const std::vector<size_t>& dims)
  {
    return std::uniform_int_distribution<size_t>(0, dims.size() - 1)(rng);
  }

  size_t NumElements(const std::vector<size_t>& dims)
  {
    return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
  }

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<size_t> shape_dist;
  std::uniform_int_distribution<size_t> dim_dist;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_int_distribution<int32_t> i8dist;
  std::uniform_int_distribution<int32_t> u8dist;
  std::uniform_real_distribution<float> scale_dist;

  uint32_t input1_id;
  uint32_t input2_id;
  uint32_t output_id;

  std::vector<size_t> input1_dims;
  std::vector<size_t> input2_dims;
  std::vector<size_t> output_dims;

  size_t axis;
  size_t batch_size;
  size_t channels_1;
  size_t channels_2;
  size_t output_stride;

  int32_t signed_zero_point;
  int32_t unsigned_zero_point;
  float scale;

  std::vector<T> input1;
  std::vector<T> input2;
  std::vector<T> operator_output;
  std::vector<T> subgraph_output;
};

using Concatenate2TestQS8 = Concatenate2Test<int8_t>;
using Concatenate2TestQU8 = Concatenate2Test<uint8_t>;
using Concatenate2TestF16 = Concatenate2Test<uint16_t>;
using Concatenate2TestF32 = Concatenate2Test<float>;

TEST_F(Concatenate2TestQS8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, input1_dims.size(), input1_dims.data(), nullptr, 0,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, input2_dims.size(), input2_dims.data(), nullptr, 1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, output_dims.size(), output_dims.data(), nullptr, 2,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_concatenate2(subgraph, axis, input1_id, input2_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_concatenate2);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qs8);
  ASSERT_EQ(node->params.concatenate.axis, axis);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input1_id);
  ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(Concatenate2TestQU8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, input1_dims.size(), input1_dims.data(), nullptr, 0,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, input2_dims.size(), input2_dims.data(), nullptr, 1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, output_dims.size(), output_dims.data(), nullptr, 2,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_concatenate2(subgraph, axis, input1_id, input2_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_concatenate2);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qu8);
  ASSERT_EQ(node->params.concatenate.axis, axis);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input1_id);
  ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(Concatenate2TestF16, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input1_dims.size(), input1_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input2_dims.size(), input2_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr, 2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_concatenate2(subgraph, axis, input1_id, input2_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_concatenate2);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp16);
  ASSERT_EQ(node->params.concatenate.axis, axis);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input1_id);
  ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(Concatenate2TestF32, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input1_dims.size(), input1_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input2_dims.size(), input2_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, 2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_concatenate2(subgraph, axis, input1_id, input2_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_concatenate2);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->params.concatenate.axis, axis);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input1_id);
  ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(Concatenate2TestQS8, matches_operator_api)
{
  std::generate(input1.begin(), input1.end(), [&]() { return i8dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return i8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), INT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op1 = nullptr;
  xnn_operator_t op2 = nullptr;

  // Call operator API.
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &op1));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op1(op1, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &op2));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op2(op2, xnn_delete_operator);

  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op1, batch_size, channels_1, channels_1, output_stride, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op2, batch_size, channels_2, channels_2, output_stride, /*threadpool=*/nullptr));

  ASSERT_EQ(xnn_status_success, xnn_setup_copy_nc_x8(op1, input1.data(), operator_output.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x8(op2, input2.data(), (uint8_t*) operator_output.data() + op1->channels));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op1, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op2, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, input1_dims.size(), input1_dims.data(), nullptr, 0,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, input2_dims.size(), input2_dims.data(), nullptr, 1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, output_dims.size(), output_dims.data(), nullptr, 2,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_concatenate2(subgraph, axis, input1_id, input2_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 3> external = {
    xnn_external_value{input1_id, input1.data()}, xnn_external_value{input2_id, input2.data()},
    xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(Concatenate2TestQU8, matches_operator_api)
{
  std::generate(input1.begin(), input1.end(), [&]() { return u8dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return u8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), UINT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT8_C(0xA5));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op1 = nullptr;
  xnn_operator_t op2 = nullptr;

  // Call operator API.
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &op1));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op1(op1, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &op2));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op2(op2, xnn_delete_operator);

  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op1, batch_size, channels_1, channels_1, output_stride, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op2, batch_size, channels_2, channels_2, output_stride, /*threadpool=*/nullptr));

  ASSERT_EQ(xnn_status_success, xnn_setup_copy_nc_x8(op1, input1.data(), operator_output.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x8(op2, input2.data(), (uint8_t*) operator_output.data() + op1->channels));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op1, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op2, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, input1_dims.size(), input1_dims.data(), nullptr, 0,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, input2_dims.size(), input2_dims.data(), nullptr, 1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, output_dims.size(), output_dims.data(), nullptr, 2,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_concatenate2(subgraph, axis, input1_id, input2_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 3> external = {
    xnn_external_value{input1_id, input1.data()}, xnn_external_value{input2_id, input2.data()},
    xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(Concatenate2TestF16, matches_operator_api)
{
  std::generate(input1.begin(), input1.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::generate(input2.begin(), input2.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::fill(operator_output.begin(), operator_output.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT16_C(0x7E00) /* NaN */);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op1 = nullptr;
  xnn_operator_t op2 = nullptr;

  // Call operator API.
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x16(/*flags=*/0, &op1));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op1(op1, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x16(/*flags=*/0, &op2));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op2(op2, xnn_delete_operator);

  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x16(op1, batch_size, channels_1, channels_1, output_stride, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x16(op2, batch_size, channels_2, channels_2, output_stride, /*threadpool=*/nullptr));

  ASSERT_EQ(xnn_status_success, xnn_setup_copy_nc_x16(op1, input1.data(), operator_output.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x16( op2, input2.data(), (uint16_t*) operator_output.data() + op1->channels));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op1, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op2, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input1_dims.size(), input1_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input2_dims.size(), input2_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr, 2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_concatenate2(subgraph, axis, input1_id, input2_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 3> external = {
    xnn_external_value{input1_id, input1.data()}, xnn_external_value{input2_id, input2.data()},
    xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(Concatenate2TestF32, matches_operator_api)
{
  std::generate(input1.begin(), input1.end(), [&]() { return f32dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), std::nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), std::nanf(""));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op1 = nullptr;
  xnn_operator_t op2 = nullptr;

  // Call operator API.
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x32(/*flags=*/0, &op1));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op1(op1, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x32(/*flags=*/0, &op2));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op2(op2, xnn_delete_operator);

  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x32(op1, batch_size, channels_1, channels_1, output_stride, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x32(op2, batch_size, channels_2, channels_2, output_stride, /*threadpool=*/nullptr));

  ASSERT_EQ(xnn_status_success, xnn_setup_copy_nc_x32(op1, input1.data(), operator_output.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x32( op2, input2.data(), (float*) operator_output.data() + op1->channels));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op1, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op2, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input1_dims.size(), input1_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input2_dims.size(), input2_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, 2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success, xnn_define_concatenate2(subgraph, axis, input1_id, input2_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 3> external = {
    xnn_external_value{input1_id, input1.data()}, xnn_external_value{input2_id, input2.data()},
    xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(Concatenate2TestF32, Reshape)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input1_dims.size(), input1_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input2_dims.size(), input2_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, 2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_concatenate2(subgraph, axis, input1_id, input2_id, output_id, /*flags=*/0));


  ASSERT_EQ(subgraph->num_nodes, 1);
  struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_concatenate2);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input1_id);
  ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  ASSERT_EQ(node->reshape(&runtime->opdata[0], subgraph->values, subgraph->num_values, /*threadpool=*/nullptr), xnn_status_success);

  input1_dims[axis] += 1;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input1_id, input1_dims.size(), input1_dims.data()));

  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  ASSERT_EQ(output_shape->dim[axis], input1_dims[axis] + input2_dims[axis]);
  for (size_t i = 0; i < input1_dims.size(); ++i) {
    if (i == axis) continue;
    ASSERT_EQ(output_shape->dim[i], input1_dims[i]);
  }

  for (size_t i = 0; i < input1_dims.size(); ++i) {
    if (i == axis) continue;
    input1_dims[i] += 1;
    input2_dims[i] += 1;
    ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input1_id, input1_dims.size(), input1_dims.data()));
    ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input2_id, input2_dims.size(), input2_dims.data()));

    ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);
    ASSERT_EQ(output_shape->dim[axis], input1_dims[axis] + input2_dims[axis]);
    for (size_t i = 0; i < input1_dims.size(); ++i) {
      if (i == axis) continue;
      ASSERT_EQ(output_shape->dim[i], input1_dims[i]);
    }
  }
  for (size_t i = 0; i < input1_dims.size(); ++i) {
    if (i == axis) continue;
    input1_dims[i] -= 1;
    input2_dims[i] -= 1;
    ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input1_id, input1_dims.size(), input1_dims.data()));
    ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input2_id, input2_dims.size(), input2_dims.data()));

    ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_success);
    ASSERT_EQ(output_shape->dim[axis], input1_dims[axis] + input2_dims[axis]);
    for (size_t i = 0; i < input1_dims.size(); ++i) {
      if (i == axis) continue;
      ASSERT_EQ(output_shape->dim[i], input1_dims[i]);
    }
  }
}
