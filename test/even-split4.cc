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

template <typename T> class EvenSplit4Test : public ::testing::Test {
 protected:
  EvenSplit4Test() {
    shape_dist = std::uniform_int_distribution<size_t>(1, XNN_MAX_TENSOR_DIMS);
    dim_dist = std::uniform_int_distribution<size_t>(1, 9);
    f32dist = std::uniform_real_distribution<float>();
    i8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    u8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    scale_dist = std::uniform_real_distribution<float>(0.1f, 5.0f);

    output1_dims = RandomShape();
    output2_dims = output1_dims;
    output3_dims = output1_dims;
    output4_dims = output1_dims;
    input_dims = output1_dims;
    axis = RandomAxis(output1_dims);
    input_dims[axis] = output1_dims[axis] + output2_dims[axis] + output3_dims[axis] + output4_dims[axis];

    input = std::vector<T>(NumElements(input_dims));
    operator_output1 = std::vector<T>(NumElements(output1_dims));
    operator_output2 = std::vector<T>(NumElements(output2_dims));
    operator_output3 = std::vector<T>(NumElements(output3_dims));
    operator_output4 = std::vector<T>(NumElements(output4_dims));
    subgraph_output1 = std::vector<T>(NumElements(output1_dims));
    subgraph_output2 = std::vector<T>(NumElements(output2_dims));
    subgraph_output3 = std::vector<T>(NumElements(output3_dims));
    subgraph_output4 = std::vector<T>(NumElements(output4_dims));

    signed_zero_point = i8dist(rng);
    unsigned_zero_point = u8dist(rng);
    scale = scale_dist(rng);

    batch_size = 1;
    input_stride = 1;
    for (size_t i = 0; i < axis; i++) {
      batch_size *= input_dims[i];
    }

    for (size_t i = axis; i < input_dims.size(); i++) {
      input_stride *= input_dims[i];
    }
    channels = input_stride / 4;
  }

  std::vector<size_t> RandomShape()
  {
    std::vector<size_t> dims(shape_dist(rng));
    std::generate(dims.begin(), dims.end(), [&] { return dim_dist(rng); });
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

  uint32_t output1_id;
  uint32_t output2_id;
  uint32_t output3_id;
  uint32_t output4_id;
  uint32_t input_id;

  std::vector<size_t> output1_dims;
  std::vector<size_t> output2_dims;
  std::vector<size_t> output3_dims;
  std::vector<size_t> output4_dims;
  std::vector<size_t> input_dims;

  size_t axis;
  size_t batch_size;
  size_t channels;
  size_t input_stride;

  int32_t signed_zero_point;
  int32_t unsigned_zero_point;
  float scale;

  std::vector<T> operator_output1;
  std::vector<T> operator_output2;
  std::vector<T> operator_output3;
  std::vector<T> operator_output4;
  std::vector<T> subgraph_output1;
  std::vector<T> subgraph_output2;
  std::vector<T> subgraph_output3;
  std::vector<T> subgraph_output4;
  std::vector<T> input;
};

using EvenSplit4TestQS8 = EvenSplit4Test<int8_t>;
using EvenSplit4TestQU8 = EvenSplit4Test<uint8_t>;
using EvenSplit4TestF16 = EvenSplit4Test<uint16_t>;
using EvenSplit4TestF32 = EvenSplit4Test<float>;

TEST_F(EvenSplit4TestQS8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/5, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, input_dims.size(), input_dims.data(), nullptr, 0,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, output1_dims.size(), output1_dims.data(), nullptr, 1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output1_id));
  ASSERT_NE(output1_id, XNN_INVALID_NODE_ID);

  output2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, output2_dims.size(), output2_dims.data(), nullptr, 2,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output2_id));
  ASSERT_NE(output2_id, XNN_INVALID_NODE_ID);

  output3_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, output3_dims.size(), output3_dims.data(), nullptr, 3,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output3_id));
  ASSERT_NE(output3_id, XNN_INVALID_NODE_ID);

  output4_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, output4_dims.size(), output4_dims.data(), nullptr, 4,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output4_id));
  ASSERT_NE(output4_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split4(subgraph, axis, input_id, output1_id, output2_id, output3_id, output4_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_even_split4);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qs8);
  ASSERT_EQ(node->params.even_split.axis, axis);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 4);
  ASSERT_EQ(node->outputs[0], output1_id);
  ASSERT_EQ(node->outputs[1], output2_id);
  ASSERT_EQ(node->outputs[2], output3_id);
  ASSERT_EQ(node->outputs[3], output4_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(EvenSplit4TestQU8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/5, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, input_dims.size(), input_dims.data(), nullptr, 0,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, output1_dims.size(), output1_dims.data(), nullptr, 1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output1_id));
  ASSERT_NE(output1_id, XNN_INVALID_NODE_ID);

  output2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, output2_dims.size(), output2_dims.data(), nullptr, 2,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output2_id));
  ASSERT_NE(output2_id, XNN_INVALID_NODE_ID);

  output3_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, output3_dims.size(), output3_dims.data(), nullptr, 3,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output3_id));
  ASSERT_NE(output3_id, XNN_INVALID_NODE_ID);

  output4_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, output4_dims.size(), output4_dims.data(), nullptr, 4,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output4_id));
  ASSERT_NE(output4_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split4(subgraph, axis, input_id, output1_id, output2_id, output3_id, output4_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_even_split4);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qu8);
  ASSERT_EQ(node->params.even_split.axis, axis);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 4);
  ASSERT_EQ(node->outputs[0], output1_id);
  ASSERT_EQ(node->outputs[1], output2_id);
  ASSERT_EQ(node->outputs[2], output3_id);
  ASSERT_EQ(node->outputs[3], output4_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(EvenSplit4TestF16, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/5, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input_dims.size(), input_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output1_dims.size(), output1_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output1_id));
  ASSERT_NE(output1_id, XNN_INVALID_NODE_ID);

  output2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output2_dims.size(), output2_dims.data(), nullptr, 2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output2_id));
  ASSERT_NE(output2_id, XNN_INVALID_NODE_ID);

  output3_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output3_dims.size(), output3_dims.data(), nullptr, 3,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output3_id));
  ASSERT_NE(output3_id, XNN_INVALID_NODE_ID);

  output4_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output4_dims.size(), output4_dims.data(), nullptr, 4,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output4_id));
  ASSERT_NE(output4_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split4(subgraph, axis, input_id, output1_id, output2_id, output3_id, output4_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_even_split4);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp16);
  ASSERT_EQ(node->params.even_split.axis, axis);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 4);
  ASSERT_EQ(node->outputs[0], output1_id);
  ASSERT_EQ(node->outputs[1], output2_id);
  ASSERT_EQ(node->outputs[2], output3_id);
  ASSERT_EQ(node->outputs[3], output4_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(EvenSplit4TestF32, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/5, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output1_dims.size(), output1_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output1_id));
  ASSERT_NE(output1_id, XNN_INVALID_NODE_ID);

  output2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output2_dims.size(), output2_dims.data(), nullptr, 2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output2_id));
  ASSERT_NE(output2_id, XNN_INVALID_NODE_ID);

  output3_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output3_dims.size(), output3_dims.data(), nullptr, 3,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output3_id));
  ASSERT_NE(output3_id, XNN_INVALID_NODE_ID);

  output4_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output4_dims.size(), output4_dims.data(), nullptr, 4,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output4_id));
  ASSERT_NE(output4_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split4(subgraph, axis, input_id, output1_id, output2_id, output3_id, output4_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_even_split4);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->params.even_split.axis, axis);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 4);
  ASSERT_EQ(node->outputs[0], output1_id);
  ASSERT_EQ(node->outputs[1], output2_id);
  ASSERT_EQ(node->outputs[2], output3_id);
  ASSERT_EQ(node->outputs[3], output4_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(EvenSplit4TestQS8, matches_operator_api)
{
  std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
  std::fill(operator_output1.begin(), operator_output1.end(), INT8_C(0xA5));
  std::fill(operator_output2.begin(), operator_output2.end(), INT8_C(0xA5));
  std::fill(operator_output3.begin(), operator_output3.end(), INT8_C(0xA5));
  std::fill(operator_output4.begin(), operator_output4.end(), INT8_C(0xA5));
  std::fill(subgraph_output1.begin(), subgraph_output1.end(), INT8_C(0xA5));
  std::fill(subgraph_output2.begin(), subgraph_output2.end(), INT8_C(0xA5));
  std::fill(subgraph_output3.begin(), subgraph_output3.end(), INT8_C(0xA5));
  std::fill(subgraph_output4.begin(), subgraph_output4.end(), INT8_C(0xA5));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op1 = nullptr;
  xnn_operator_t op2 = nullptr;
  xnn_operator_t op3 = nullptr;
  xnn_operator_t op4 = nullptr;

  // Call operator API.
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &op1));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op1(op1, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &op2));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op2(op2, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &op3));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op3(op3, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &op4));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op4(op4, xnn_delete_operator);

  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op1, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op2, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op3, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op4, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));

  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x8(op1, input.data(), operator_output1.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x8(op2, (uint8_t*) input.data() + op1->channels, operator_output2.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x8(op3, (uint8_t*) input.data() + op1->channels * 2, operator_output3.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x8(op4, (uint8_t*) input.data() + op1->channels * 3, operator_output4.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op1, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op2, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op3, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op4, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/5, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, input_dims.size(), input_dims.data(), nullptr, 0,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, output1_dims.size(), output1_dims.data(), nullptr, 1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output1_id));
  ASSERT_NE(output1_id, XNN_INVALID_NODE_ID);

  output2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, output2_dims.size(), output2_dims.data(), nullptr, 2,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output2_id));
  ASSERT_NE(output2_id, XNN_INVALID_NODE_ID);

  output3_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, output3_dims.size(), output3_dims.data(), nullptr, 3,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output3_id));
  ASSERT_NE(output2_id, XNN_INVALID_NODE_ID);

  output4_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, output4_dims.size(), output4_dims.data(), nullptr, 4,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output4_id));
  ASSERT_NE(output2_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split4(subgraph, axis, input_id, output1_id, output2_id, output3_id, output4_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 5> external = {
    xnn_external_value{input_id, input.data()},
    xnn_external_value{output1_id, subgraph_output1.data()},
    xnn_external_value{output2_id, subgraph_output2.data()},
    xnn_external_value{output3_id, subgraph_output3.data()},
    xnn_external_value{output4_id, subgraph_output4.data()},
  };
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output1, operator_output1);
  ASSERT_EQ(subgraph_output2, operator_output2);
  ASSERT_EQ(subgraph_output3, operator_output3);
  ASSERT_EQ(subgraph_output4, operator_output4);
}

TEST_F(EvenSplit4TestQU8, matches_operator_api)
{
  std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
  std::fill(operator_output1.begin(), operator_output1.end(), UINT8_C(0xA5));
  std::fill(operator_output2.begin(), operator_output2.end(), UINT8_C(0xA5));
  std::fill(operator_output3.begin(), operator_output3.end(), UINT8_C(0xA5));
  std::fill(operator_output4.begin(), operator_output4.end(), UINT8_C(0xA5));
  std::fill(subgraph_output1.begin(), subgraph_output1.end(), UINT8_C(0xA5));
  std::fill(subgraph_output2.begin(), subgraph_output2.end(), UINT8_C(0xA5));
  std::fill(subgraph_output3.begin(), subgraph_output3.end(), UINT8_C(0xA5));
  std::fill(subgraph_output4.begin(), subgraph_output4.end(), UINT8_C(0xA5));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op1 = nullptr;
  xnn_operator_t op2 = nullptr;
  xnn_operator_t op3 = nullptr;
  xnn_operator_t op4 = nullptr;

  // Call operator API.
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &op1));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op1(op1, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &op2));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op2(op2, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &op3));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op3(op3, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &op4));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op4(op4, xnn_delete_operator);

  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op1, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op2, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op3, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op4, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));

  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x8(op1, input.data(), operator_output1.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x8(op2, (uint8_t*) input.data() + op1->channels, operator_output2.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x8(op3, (uint8_t*) input.data() + op1->channels * 2, operator_output3.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x8(op4, (uint8_t*) input.data() + op1->channels * 3, operator_output4.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op1, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op2, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op3, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op4, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/5, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, input_dims.size(), input_dims.data(), nullptr, 0,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, output1_dims.size(), output1_dims.data(), nullptr, 1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output1_id));
  ASSERT_NE(output1_id, XNN_INVALID_NODE_ID);

  output2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, output2_dims.size(), output2_dims.data(), nullptr, 2,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output2_id));
  ASSERT_NE(output2_id, XNN_INVALID_NODE_ID);

  output3_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, output3_dims.size(), output3_dims.data(), nullptr, 3,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output3_id));
  ASSERT_NE(output3_id, XNN_INVALID_NODE_ID);

  output4_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, output4_dims.size(), output4_dims.data(), nullptr, 4,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output4_id));
  ASSERT_NE(output4_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split4(subgraph, axis, input_id, output1_id, output2_id, output3_id, output4_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 5> external = {
    xnn_external_value{input_id, input.data()},
    xnn_external_value{output1_id, subgraph_output1.data()},
    xnn_external_value{output2_id, subgraph_output2.data()},
    xnn_external_value{output3_id, subgraph_output3.data()},
    xnn_external_value{output4_id, subgraph_output4.data()},
  };
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output1, operator_output1);
  ASSERT_EQ(subgraph_output2, operator_output2);
  ASSERT_EQ(subgraph_output3, operator_output3);
  ASSERT_EQ(subgraph_output4, operator_output4);
}

TEST_F(EvenSplit4TestF16, matches_operator_api)
{
  std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::fill(operator_output1.begin(), operator_output1.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(operator_output2.begin(), operator_output2.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(operator_output3.begin(), operator_output3.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(operator_output4.begin(), operator_output4.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(subgraph_output1.begin(), subgraph_output1.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(subgraph_output2.begin(), subgraph_output2.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(subgraph_output3.begin(), subgraph_output3.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(subgraph_output4.begin(), subgraph_output4.end(), UINT16_C(0x7E00) /* NaN */);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op1 = nullptr;
  xnn_operator_t op2 = nullptr;
  xnn_operator_t op3 = nullptr;
  xnn_operator_t op4 = nullptr;

  // Call operator API.
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x16(/*flags=*/0, &op1));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op1(op1, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x16(/*flags=*/0, &op2));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op2(op2, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x16(/*flags=*/0, &op3));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op3(op3, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x16(/*flags=*/0, &op4));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op4(op4, xnn_delete_operator);

  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x16(op1, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x16(op2, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x16(op3, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x16(op4, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));

  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x16(op1, input.data(), operator_output1.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x16(op2, (uint16_t*) input.data() + op1->channels, operator_output2.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x16(op3, (uint16_t*) input.data() + op1->channels * 2, operator_output3.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x16(op4, (uint16_t*) input.data() + op1->channels * 3, operator_output4.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op1, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op2, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op3, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op4, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/5, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input_dims.size(), input_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output1_dims.size(), output1_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output1_id));
  ASSERT_NE(output1_id, XNN_INVALID_NODE_ID);

  output2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output2_dims.size(), output2_dims.data(), nullptr, 2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output2_id));
  ASSERT_NE(output2_id, XNN_INVALID_NODE_ID);

  output3_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output3_dims.size(), output3_dims.data(), nullptr, 3,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output3_id));
  ASSERT_NE(output3_id, XNN_INVALID_NODE_ID);

  output4_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output4_dims.size(), output4_dims.data(), nullptr, 4,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output4_id));
  ASSERT_NE(output4_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split4(subgraph, axis, input_id, output1_id, output2_id, output3_id, output4_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 5> external = {
    xnn_external_value{input_id, input.data()},
    xnn_external_value{output1_id, subgraph_output1.data()},
    xnn_external_value{output2_id, subgraph_output2.data()},
    xnn_external_value{output3_id, subgraph_output3.data()},
    xnn_external_value{output4_id, subgraph_output4.data()},
  };
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output1, operator_output1);
  ASSERT_EQ(subgraph_output2, operator_output2);
  ASSERT_EQ(subgraph_output3, operator_output3);
  ASSERT_EQ(subgraph_output4, operator_output4);
}

TEST_F(EvenSplit4TestF32, matches_operator_api)
{
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output1.begin(), operator_output1.end(), std::nanf(""));
  std::fill(operator_output2.begin(), operator_output2.end(), std::nanf(""));
  std::fill(operator_output3.begin(), operator_output3.end(), std::nanf(""));
  std::fill(operator_output4.begin(), operator_output4.end(), std::nanf(""));
  std::fill(subgraph_output1.begin(), subgraph_output1.end(), std::nanf(""));
  std::fill(subgraph_output2.begin(), subgraph_output2.end(), std::nanf(""));
  std::fill(subgraph_output3.begin(), subgraph_output3.end(), std::nanf(""));
  std::fill(subgraph_output4.begin(), subgraph_output4.end(), std::nanf(""));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op1 = nullptr;
  xnn_operator_t op2 = nullptr;
  xnn_operator_t op3 = nullptr;
  xnn_operator_t op4 = nullptr;

  // Call operator API.
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x32(/*flags=*/0, &op1));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op1(op1, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x32(/*flags=*/0, &op2));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op2(op2, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x32(/*flags=*/0, &op3));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op3(op3, xnn_delete_operator);
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x32(/*flags=*/0, &op4));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op4(op4, xnn_delete_operator);

  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x32(op1, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x32(op2, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x32(op3, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x32(op4, batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));

  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x32(op1, input.data(), operator_output1.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x32(op2, (uint32_t*) input.data() + op1->channels, operator_output2.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x32(op3, (uint32_t*) input.data() + op1->channels * 2, operator_output3.data()));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_copy_nc_x32(op4, (uint32_t*) input.data() + op1->channels * 3, operator_output4.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op1, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op2, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op3, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op4, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/5, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output1_dims.size(), output1_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output1_id));
  ASSERT_NE(output1_id, XNN_INVALID_NODE_ID);

  output2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output2_dims.size(), output2_dims.data(), nullptr, 2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output2_id));
  ASSERT_NE(output2_id, XNN_INVALID_NODE_ID);

  output3_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output3_dims.size(), output3_dims.data(), nullptr, 3,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output3_id));
  ASSERT_NE(output3_id, XNN_INVALID_NODE_ID);

  output4_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output4_dims.size(), output4_dims.data(), nullptr, 4,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output4_id));
  ASSERT_NE(output4_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split4(subgraph, axis, input_id, output1_id, output2_id, output3_id, output4_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 5> external = {
    xnn_external_value{input_id, input.data()},
    xnn_external_value{output1_id, subgraph_output1.data()},
    xnn_external_value{output2_id, subgraph_output2.data()},
    xnn_external_value{output3_id, subgraph_output3.data()},
    xnn_external_value{output4_id, subgraph_output4.data()},
  };
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output1, operator_output1);
  ASSERT_EQ(subgraph_output2, operator_output2);
  ASSERT_EQ(subgraph_output3, operator_output3);
  ASSERT_EQ(subgraph_output4, operator_output4);
}

TEST_F(EvenSplit4TestF32, reshape_output)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/5, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output1_dims.size(), output1_dims.data(), nullptr, 1,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output1_id));
  ASSERT_NE(output1_id, XNN_INVALID_NODE_ID);

  output2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output2_dims.size(), output2_dims.data(), nullptr, 2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output2_id));
  ASSERT_NE(output2_id, XNN_INVALID_NODE_ID);

  output3_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output3_dims.size(), output3_dims.data(), nullptr, 3,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output3_id));
  ASSERT_NE(output3_id, XNN_INVALID_NODE_ID);

  output4_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output4_dims.size(), output4_dims.data(), nullptr, 4,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output4_id));
  ASSERT_NE(output4_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split4(subgraph, axis, input_id, output1_id, output2_id, output3_id, output4_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 5> external = {
    xnn_external_value{input_id, input.data()},
    xnn_external_value{output1_id, subgraph_output1.data()},
    xnn_external_value{output2_id, subgraph_output2.data()},
    xnn_external_value{output3_id, subgraph_output3.data()},
    xnn_external_value{output4_id, subgraph_output4.data()},
  };
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  input_dims[axis] += 4;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input_id, input_dims.size(), input_dims.data()));
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);
  for (size_t i = 0; i < 4; ++i) {
    const xnn_shape* output_n_shape = &runtime->values[node->outputs[i]].shape;
    ASSERT_EQ(output_n_shape->dim[axis], input_dims[axis] / 4);
    for (size_t i = 0; i < input_dims.size(); ++i) {
      if (i == axis) continue;
      ASSERT_EQ(output_n_shape->dim[i], input_dims[i]);
    }
  }

  input_dims[axis] -= 8;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input_id, input_dims.size(), input_dims.data()));
  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_success);
  for (size_t i = 0; i < 4; ++i) {
    const xnn_shape* output_n_shape = &runtime->values[node->outputs[i]].shape;
    ASSERT_EQ(output_n_shape->dim[axis], input_dims[axis] / 4);
    for (size_t i = 0; i < input_dims.size(); ++i) {
      if (i == axis) continue;
      ASSERT_EQ(output_n_shape->dim[i], input_dims[i]);
    }
  }
}
