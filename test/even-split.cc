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
#include "xnnpack.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"

template <typename T> class EvenSplitNTest : public ::testing::Test {
protected:
  EvenSplitNTest()
  {
    shape_dist = std::uniform_int_distribution<size_t>(1, XNN_MAX_TENSOR_DIMS);
    dim_dist = std::uniform_int_distribution<size_t>(1, 9);
    f32dist = std::uniform_real_distribution<float>();
    i8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    u8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    scale_dist = std::uniform_real_distribution<float>(0.1f, 5.0f);

    num_outputs = RandomNumOutputs();
    output_dims.resize(num_outputs);
    output_dims[0] = RandomShape();
    output_id.resize(num_outputs);
    for (int i = 1; i < num_outputs; i++) {
      output_dims[i] = output_dims[0];
    }
    input_dims = output_dims[0];
    axis = RandomAxis(output_dims[0]);
    for (int i = 1; i < num_outputs; i++) {
      input_dims[axis] += output_dims[i][axis];
    }

    input = std::vector<T>(NumElements(input_dims));
    operator_outputs.resize(num_outputs);
    subgraph_outputs.resize(num_outputs);
    for (int i = 0; i < num_outputs; i++) {
      operator_outputs[i] = std::vector<T>(NumElements(output_dims[i]));
      subgraph_outputs[i] = std::vector<T>(NumElements(output_dims[i]));
    }

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
    channels = input_stride / num_outputs;

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

  size_t RandomNumOutputs() { return std::uniform_int_distribution<size_t>(1, XNN_MAX_OUTPUTS)(rng); }

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

  std::vector<uint32_t> output_id;
  uint32_t input_id;

  std::vector<std::vector<size_t>> output_dims;
  std::vector<size_t> input_dims;

  size_t axis;
  size_t num_outputs;
  size_t batch_size;
  size_t channels;
  size_t input_stride;

  int32_t signed_zero_point;
  int32_t unsigned_zero_point;
  float scale;

  std::vector<std::vector<T>> operator_outputs;
  std::vector<std::vector<T>> subgraph_outputs;
  std::vector<T> input;
};

using EvenSplitNTestQS8 = EvenSplitNTest<int8_t>;
using EvenSplitNTestQU8 = EvenSplitNTest<uint8_t>;
using EvenSplitNTestF16 = EvenSplitNTest<xnn_float16>;
using EvenSplitNTestF32 = EvenSplitNTest<float>;

TEST_F(EvenSplitNTestQS8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_outputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, input_dims.size(), input_dims.data(), nullptr, 0,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  for (size_t i = 0; i < num_outputs; ++i) {
    output_id[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success, xnn_define_quantized_tensor_value(
                            subgraph, xnn_datatype_qint8, signed_zero_point, scale, output_dims[i].size(),
                            output_dims[i].data(), nullptr, 1,
                            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id[i]));
    ASSERT_NE(output_id[i], XNN_INVALID_NODE_ID);
  }
  int32_t split_dim = axis;
  uint32_t output_ddds[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    output_ddds[i] = output_id[i];
  }
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split(subgraph, split_dim, input_id, num_outputs, output_ddds, /*flags=*/0));
  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_even_split);
  ASSERT_EQ(node->params.even_split.axis, axis);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    ASSERT_EQ(node->outputs[i], output_ddds[i]);
  }
  ASSERT_EQ(node->flags, 0);
}

TEST_F(EvenSplitNTestQU8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_outputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, input_dims.size(), input_dims.data(), nullptr, 0,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);
  for (size_t i = 0; i < num_outputs; ++i) {
    output_id[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success, xnn_define_quantized_tensor_value(
                            subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, output_dims[i].size(),
                            output_dims[i].data(), nullptr, 1,
                            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id[i]));
    ASSERT_NE(output_id[i], XNN_INVALID_NODE_ID);
  }
  uint32_t output_ddds[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    output_ddds[i] = output_id[i];
  }
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split(subgraph, axis, input_id, num_outputs, output_ddds, /*flags=*/0));
  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_even_split);
  ASSERT_EQ(node->params.even_split.axis, axis);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    ASSERT_EQ(node->outputs[i], output_ddds[i]);
  }
  ASSERT_EQ(node->flags, 0);
}

TEST_F(EvenSplitNTestF16, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_outputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input_dims.size(), input_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);
  for (int i = 0; i < num_outputs; ++i) {
    output_id[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success, xnn_define_tensor_value(
                            subgraph, xnn_datatype_fp16, output_dims[i].size(), output_dims[i].data(), nullptr, 1,
                            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id[i]));
    ASSERT_NE(output_id[i], XNN_INVALID_NODE_ID);
  }
  uint32_t output_ddds[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    output_ddds[i] = output_id[i];
  }
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split(subgraph, axis, input_id, num_outputs, output_ddds, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_even_split);
  ASSERT_EQ(node->params.even_split.axis, axis);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    ASSERT_EQ(node->outputs[i], output_ddds[i]);
  }
  ASSERT_EQ(node->flags, 0);
}

TEST_F(EvenSplitNTestF32, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_outputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);
  for (size_t i = 0; i < num_outputs; ++i) {
    output_id[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success, xnn_define_tensor_value(
                            subgraph, xnn_datatype_fp32, output_dims[i].size(), output_dims[i].data(), nullptr, 1,
                            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id[i]));
    ASSERT_NE(output_id[i], XNN_INVALID_NODE_ID);
  }
  uint32_t output_ddds[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    output_ddds[i] = output_id[i];
  }
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split(subgraph, axis, input_id, num_outputs, output_ddds, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_even_split);
  ASSERT_EQ(node->params.even_split.axis, axis);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    ASSERT_EQ(node->outputs[i], output_ddds[i]);
  }
  ASSERT_EQ(node->flags, 0);
}

TEST_F(EvenSplitNTestQS8, matches_operator_api)
{
  for (int i = 0; i < input.size(); ++i) {
    input = std::vector<int8_t>(NumElements(input_dims), static_cast<int8_t>(i + 1));
  }
  for (int i = 0; i < num_outputs; i++) {
    std::fill(operator_outputs[i].begin(), operator_outputs[i].end(), INT8_C(0xA5));
    std::fill(subgraph_outputs[i].begin(), subgraph_outputs[i].end(), INT8_C(0xA5));
  }

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  std::vector<xnn_operator_t> ops(num_outputs, nullptr);

  // Call operator API.
  for (int i = 0; i < num_outputs; ++i) {
    ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &ops[i]));
    std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op1(ops[i], xnn_delete_operator);
    ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_copy_nc_x8(ops[i], batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
    if (i == 0) {
      ASSERT_EQ(
        xnn_status_success,
        xnn_setup_copy_nc_x8(ops[i], input.data() + (i * (ops[i]->channels)), operator_outputs[i].data()));
    }
    else {
      ASSERT_EQ(
        xnn_status_success,
        xnn_setup_copy_nc_x8(ops[i], (uint8_t*) input.data() + (i * (ops[i]->channels)), operator_outputs[i].data()));
    }
    ASSERT_EQ(xnn_status_success, xnn_run_operator(ops[i], /*threadpool=*/nullptr));
  }


  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_outputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, signed_zero_point, scale, input_dims.size(), input_dims.data(), nullptr, 0,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);
  uint32_t output_ddds[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    output_ddds[i] = output_id[i];
  }

  for (int i = 0; i < num_outputs; ++i) {
    output_ddds[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success, xnn_define_quantized_tensor_value(
                            subgraph, xnn_datatype_qint8, signed_zero_point, scale, output_dims[i].size(),
                            output_dims[i].data(), nullptr, i + 1,
                            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_ddds[i]));
    ASSERT_NE(output_ddds[i], XNN_INVALID_NODE_ID);
  }

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split(subgraph, axis, input_id, num_outputs, output_ddds, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::vector<xnn_external_value> external;
  external.reserve(1 + num_outputs);  // Reserve space for input + outputs

  // Add the input value
  external.emplace_back(xnn_external_value{input_id, input.data()});

  // Loop to add output values
  for (int i = 0; i < num_outputs; ++i) {
    external.emplace_back(xnn_external_value{output_ddds[i], subgraph_outputs[i].data()});
  }

  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
  for (int i = 0; i < num_outputs; ++i) {
    ASSERT_EQ(subgraph_outputs[i], operator_outputs[i]);
  }
}

TEST_F(EvenSplitNTestQU8, matches_operator_api)
{
  for (int i = 0; i < input.size(); ++i) {
    input = std::vector<uint8_t>(NumElements(input_dims), static_cast<uint8_t>(i + 1));
  }
  for (int i = 0; i < num_outputs; i++) {
    std::fill(operator_outputs[i].begin(), operator_outputs[i].end(), UINT8_C(0xA5));
    std::fill(subgraph_outputs[i].begin(), subgraph_outputs[i].end(), UINT8_C(0xA5));
  }

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  std::vector<xnn_operator_t> ops(num_outputs, nullptr);

  for (int i = 0; i < num_outputs; ++i) {
    ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &ops[i]));
    std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op1(ops[i], xnn_delete_operator);
    ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_copy_nc_x8(ops[i], batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
    if (i == 0) {
      ASSERT_EQ(xnn_status_success, xnn_setup_copy_nc_x8(ops[i], input.data(), operator_outputs[i].data()));
    }
    else {
      ASSERT_EQ(
        xnn_status_success,
        xnn_setup_copy_nc_x8(ops[i], (uint8_t*) input.data() + (i * (ops[i]->channels)), operator_outputs[i].data()));
    }
    ASSERT_EQ(xnn_status_success, xnn_run_operator(ops[i], /*threadpool=*/nullptr));
  }

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_outputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, input_dims.size(), input_dims.data(), nullptr, 0,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);
  uint32_t output_ddds[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    output_ddds[i] = output_id[i];
  }

  for (int i = 0; i < num_outputs; ++i) {
    output_ddds[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success, xnn_define_quantized_tensor_value(
                            subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, output_dims[i].size(),
                            output_dims[i].data(), nullptr, i + 1,
                            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_ddds[i]));
    ASSERT_NE(output_ddds[i], XNN_INVALID_NODE_ID);
  }

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split(subgraph, axis, input_id, num_outputs, output_ddds, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::vector<xnn_external_value> external;
  external.reserve(1 + num_outputs);  // Reserve space for input + outputs

  // Add the input value
  external.emplace_back(xnn_external_value{input_id, input.data()});

  // Loop to add output values
  for (int i = 0; i < num_outputs; ++i) {
    external.emplace_back(xnn_external_value{output_ddds[i], subgraph_outputs[i].data()});
  }

  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
  for (int i = 0; i < num_outputs; ++i) {
    ASSERT_EQ(subgraph_outputs[i], operator_outputs[i]);
  }
}

TEST_F(EvenSplitNTestF16, matches_operator_api)
{
  for (int i = 0; i < input.size(); ++i) {
    input = std::vector<xnn_float16>(NumElements(input_dims), static_cast<xnn_float16>(i + 1));
  }
  for (int i = 0; i < num_outputs; i++) {
    std::fill(operator_outputs[i].begin(), operator_outputs[i].end(), std::nanf(""));
    std::fill(subgraph_outputs[i].begin(), subgraph_outputs[i].end(), std::nanf(""));
  }

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  std::vector<xnn_operator_t> ops(num_outputs, nullptr);

  // Call operator API.
  for (int i = 0; i < num_outputs; ++i) {
    ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x16(/*flags=*/0, &ops[i]));
    std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op1(ops[i], xnn_delete_operator);

    ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_copy_nc_x16(ops[i], batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
    if (i == 0) {
      ASSERT_EQ(xnn_status_success, xnn_setup_copy_nc_x16(ops[i], input.data(), operator_outputs[i].data()));
    }
    else {
      ASSERT_EQ(
        xnn_status_success,
        xnn_setup_copy_nc_x16(
          ops[i], (xnn_float16*) input.data() + (i * (ops[i]->channels)), operator_outputs[i].data()));
    }
    ASSERT_EQ(xnn_status_success, xnn_run_operator(ops[i], /*threadpool=*/nullptr));
  }

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_outputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input_dims.size(), input_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_ddds[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    output_ddds[i] = output_id[i];
  }
  for (int i = 0; i < num_outputs; ++i) {
    output_ddds[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success, xnn_define_tensor_value(
                            subgraph, xnn_datatype_fp16, output_dims[i].size(), output_dims[i].data(), nullptr, i + 1,
                            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_ddds[i]));
    ASSERT_NE(output_ddds[i], XNN_INVALID_NODE_ID);
  }

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split(subgraph, axis, input_id, num_outputs, output_ddds, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::vector<xnn_external_value> external;
  external.reserve(1 + num_outputs);  // Reserve space for input + outputs

  // Add the input value
  external.emplace_back(xnn_external_value{input_id, input.data()});

  // Loop to add output values
  for (int i = 0; i < num_outputs; ++i) {
    external.emplace_back(xnn_external_value{output_ddds[i], subgraph_outputs[i].data()});
  }
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  for (int i = 0; i < num_outputs; ++i) {
    ASSERT_EQ(subgraph_outputs[i], operator_outputs[i]);
  }
}

TEST_F(EvenSplitNTestF32, matches_operator_api)
{
  for (int i = 0; i < input.size(); ++i) {
    input = std::vector<float>(NumElements(input_dims), static_cast<float>(i + 1));
  }
  for (int i = 0; i < num_outputs; i++) {
    std::fill(operator_outputs[i].begin(), operator_outputs[i].end(), std::nanf(""));
    std::fill(subgraph_outputs[i].begin(), subgraph_outputs[i].end(), std::nanf(""));
  }

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  std::vector<xnn_operator_t> ops(num_outputs, nullptr);

  // Call operator API.
  for (int i = 0; i < num_outputs; ++i) {
    ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x32(/*flags=*/0, &ops[i]));
    std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op1(ops[i], xnn_delete_operator);

    ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_copy_nc_x32(ops[i], batch_size, channels, input_stride, channels, /*threadpool=*/nullptr));
    if (i == 0) {
      ASSERT_EQ(xnn_status_success, xnn_setup_copy_nc_x32(ops[i], input.data(), operator_outputs[i].data()));
    }
    else {
      ASSERT_EQ(
        xnn_status_success,
        xnn_setup_copy_nc_x32(ops[i], (uint32_t*) input.data() + (i * ops[i]->channels), operator_outputs[i].data()));
    }
    ASSERT_EQ(xnn_status_success, xnn_run_operator(ops[i], /*threadpool=*/nullptr));
  }

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_outputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_ddds[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    output_ddds[i] = output_id[i];
  }
  for (int i = 0; i < num_outputs; ++i) {
    output_ddds[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success, xnn_define_tensor_value(
                            subgraph, xnn_datatype_fp32, output_dims[i].size(), output_dims[i].data(), nullptr, i + 1,
                            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_ddds[i]));
    ASSERT_NE(output_ddds[i], XNN_INVALID_NODE_ID);
  }

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split(subgraph, axis, input_id, num_outputs, output_ddds, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::vector<xnn_external_value> external;
  external.reserve(1 + num_outputs);  // Reserve space for input + outputs

  // Add the input value
  external.emplace_back(xnn_external_value{input_id, input.data()});

  // Loop to add output values
  for (int i = 0; i < num_outputs; ++i) {
    external.emplace_back(xnn_external_value{output_ddds[i], subgraph_outputs[i].data()});
  }
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  for (int i = 0; i < num_outputs; ++i) {
    ASSERT_EQ(subgraph_outputs[i], operator_outputs[i]);
  }
}

TEST_F(EvenSplitNTestF32, reshape_output)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_outputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_dims.size(), input_dims.data(), nullptr, 0,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_ddds[num_outputs];
  for (int i = 0; i < num_outputs; i++) {
    output_ddds[i] = output_id[i];
  }
  for (int i = 0; i < num_outputs; ++i) {
    output_ddds[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success, xnn_define_tensor_value(
                            subgraph, xnn_datatype_fp32, output_dims[i].size(), output_dims[i].data(), nullptr, i + 1,
                            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_ddds[i]));
    ASSERT_NE(output_ddds[i], XNN_INVALID_NODE_ID);
  }

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_even_split(subgraph, axis, input_id, num_outputs, output_ddds, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::vector<xnn_external_value> external;
  external.reserve(1 + num_outputs);  // Reserve space for input + outputs

  // Add the input value
  external.emplace_back(xnn_external_value{input_id, input.data()});

  // Loop to add output values
  for (int i = 0; i < num_outputs; ++i) {
    external.emplace_back(xnn_external_value{output_ddds[i], subgraph_outputs[i].data()});
  }
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  input_dims[axis] += num_outputs;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input_id, input_dims.size(), input_dims.data()));
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(
    node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr),
    xnn_status_reallocation_required);
  for (size_t i = 0; i < num_outputs; ++i) {
    const xnn_shape* output_n_shape = &runtime->values[node->outputs[i]].shape;
    ASSERT_EQ(output_n_shape->dim[axis], input_dims[axis] / num_outputs);
    for (size_t i = 0; i < input_dims.size(); ++i) {
      if (i == axis)
        continue;
      ASSERT_EQ(output_n_shape->dim[i], input_dims[i]);
    }
  }

  input_dims[axis] -= 2 * num_outputs;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input_id, input_dims.size(), input_dims.data()));
  ASSERT_EQ(
    node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr),
    xnn_status_success);
  for (size_t i = 0; i < num_outputs; ++i) {
    const xnn_shape* output_n_shape = &runtime->values[node->outputs[i]].shape;
    ASSERT_EQ(output_n_shape->dim[axis], input_dims[axis] / num_outputs);
    for (size_t i = 0; i < input_dims.size(); ++i) {
      if (i == axis)
        continue;
      ASSERT_EQ(output_n_shape->dim[i], input_dims[i]);
    }
  }
}
