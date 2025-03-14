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
#include <stdint.h>
#include <vector>

#include <gtest/gtest.h>

#include "test/replicable_random_device.h"
#include "include/xnnpack.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/subgraph.h"

template <typename T> class ConcatenateTest : public ::testing::Test {
protected:
  ConcatenateTest()
  {
    shape_dist = std::uniform_int_distribution<size_t>(1, XNN_MAX_TENSOR_DIMS);
    dim_dist = std::uniform_int_distribution<size_t>(1, 9);
    f32dist = std::uniform_real_distribution<float>();
    i8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    u8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    scale_dist = std::uniform_real_distribution<float>(0.1f, 5.0f);

    num_inputs = RandomNumInputs();
    input_dims.resize(num_inputs);
    input_dims[0] = RandomShape();
    axis = RandomAxis(input_dims[0]);
    for (int i = 1; i < num_inputs; i++) {
      input_dims[i] = RandomShape(input_dims[0], axis);
    }
    output_dims = input_dims[0];
    for (int i = 1; i < num_inputs; i++) {
      output_dims[axis] += input_dims[i][axis];
    }
    inputs.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
      inputs[i] = std::vector<T>(NumElements(input_dims[i]));
    }
    operator_output = std::vector<T>(NumElements(output_dims));
    subgraph_output = std::vector<T>(NumElements(output_dims));

    signed_zero_point = i8dist(rng);
    unsigned_zero_point = u8dist(rng);
    scale = scale_dist(rng);
    channels.resize(num_inputs);
    batch_size = 1;

    // Compute the batch size up to the axis.
    for (size_t i = 0; i < axis; i++) {
      batch_size *= output_dims[i];
    }

    // Initialize channels for each input tensor.
    for (size_t j = 0; j < num_inputs; j++) {
      channels[j] = 1;  // Reset or initialize channels for each input.
      for (size_t i = axis; i < input_dims[j].size(); i++) {
        channels[j] *= input_dims[j][i];
      }
    }

    output_stride = 0;
    for (int i = 0; i < num_inputs; i++) {
      output_stride += channels[i];
    }
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

  size_t RandomNumInputs()
  {
    return std::uniform_int_distribution<size_t>(2, XNN_MAX_OPERATOR_OBJECTS)(rng);  // You can adjust the range
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

  uint32_t output_id;

  std::vector<std::vector<size_t>> input_dims;
  std::vector<size_t> output_dims;

  size_t axis;
  size_t num_inputs;
  size_t batch_size;
  std::vector<size_t> channels;
  size_t output_stride;

  int32_t signed_zero_point;
  int32_t unsigned_zero_point;
  float scale;

  std::vector<std::vector<T>> inputs;
  std::vector<T> operator_output;
  std::vector<T> subgraph_output;
};

using ConcatenateTestQS8 = ConcatenateTest<int8_t>;
using ConcatenateTestQU8 = ConcatenateTest<uint8_t>;
using ConcatenateTestF16 = ConcatenateTest<xnn_float16>;
using ConcatenateTestF32 = ConcatenateTest<float>;


TEST_F(ConcatenateTestQS8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_inputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  std::vector<uint32_t> input_ids(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    input_ids[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success,
      xnn_define_quantized_tensor_value(
        subgraph, xnn_datatype_qint8, signed_zero_point, scale, input_dims[i].size(), input_dims[i].data(), nullptr, i,
        /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_ids[i]));
    ASSERT_NE(input_ids[i], XNN_INVALID_NODE_ID);
  }

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, signed_zero_point, scale, output_dims.size(),
                          output_dims.data(), nullptr, num_inputs,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_concatenate(subgraph, axis,num_inputs, input_ids.data(), output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_concatenate);
  ASSERT_EQ(node->params.concatenate.axis, axis);
  ASSERT_EQ(node->num_inputs, num_inputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    ASSERT_EQ(node->inputs[i], input_ids[i]);
  }

  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(ConcatenateTestQU8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_inputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<uint32_t> input_ids(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    input_ids[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success, xnn_define_quantized_tensor_value(
                            subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, input_dims[i].size(),
                            input_dims[i].data(), nullptr, i,
                            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_ids[i]));
    ASSERT_NE(input_ids[i], XNN_INVALID_NODE_ID);
  }

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, unsigned_zero_point, scale, output_dims.size(),
                          output_dims.data(), nullptr, num_inputs,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_concatenate(subgraph, axis,num_inputs, input_ids.data(), output_id, /*flags=*/0));

  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_concatenate);
  ASSERT_EQ(node->params.concatenate.axis, axis);
  ASSERT_EQ(node->num_inputs, num_inputs);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(ConcatenateTestF16, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_inputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<uint32_t> input_ids(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    input_ids[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success, xnn_define_tensor_value(
                            subgraph, xnn_datatype_fp16, input_dims[i].size(), input_dims[i].data(), nullptr, i,
                            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_ids[i]));
    ASSERT_NE(input_ids[i], XNN_INVALID_NODE_ID);
  }
  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr, num_inputs,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_concatenate(subgraph, axis,num_inputs, input_ids.data(), output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_concatenate);
  ASSERT_EQ(node->params.concatenate.axis, axis);
  ASSERT_EQ(node->num_inputs, num_inputs);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(ConcatenateTestF32, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_inputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<uint32_t> input_ids(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    input_ids[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success, xnn_define_tensor_value(
                            subgraph, xnn_datatype_fp32, input_dims[i].size(), input_dims[i].data(), nullptr, i,
                            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_ids[i]));
    ASSERT_NE(input_ids[i], XNN_INVALID_NODE_ID);
  }

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, num_inputs,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_concatenate(subgraph, axis,num_inputs, input_ids.data(), output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_concatenate);
  ASSERT_EQ(node->params.concatenate.axis, axis);
  ASSERT_EQ(node->num_inputs, num_inputs);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(ConcatenateTestQS8, matches_operator_api)
{
  inputs.resize(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    inputs[i] =
      std::vector<int8_t>(NumElements(input_dims[i]), static_cast<int8_t>(i + 1));  // For example, fill with i + 1
  }
  std::fill(operator_output.begin(), operator_output.end(), INT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  std::vector<xnn_operator_t> operators(num_inputs);
  std::vector<std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>> auto_op;
  auto_op.reserve(num_inputs);
  size_t offset = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &operators[i]));
    auto_op.emplace_back(
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>(operators[i], xnn_delete_operator));
    ASSERT_EQ(
      xnn_status_success, xnn_reshape_copy_nc_x8(
                            operators[i], batch_size, channels[i], channels[i], output_stride, /*threadpool=*/nullptr));
    ASSERT_EQ(
      xnn_status_success, xnn_setup_copy_nc_x8(operators[i], inputs[i].data(), operator_output.data() + offset));
    offset += channels[i];
    ASSERT_EQ(xnn_status_success, xnn_run_operator(operators[i], /*threadpool=*/nullptr));
  }

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_inputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<uint32_t> input_ids(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    input_ids[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success,
      xnn_define_quantized_tensor_value(
        subgraph, xnn_datatype_qint8, signed_zero_point, scale, input_dims[i].size(), input_dims[i].data(), nullptr, i,
        /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_ids[i]));
    ASSERT_NE(input_ids[i], XNN_INVALID_NODE_ID);
  }

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, signed_zero_point, scale, output_dims.size(),
                          output_dims.data(), nullptr, num_inputs,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_concatenate(subgraph, axis,num_inputs, input_ids.data(), output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  std::vector<xnn_external_value> external(1 + num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    external[i] = xnn_external_value{input_ids[i], inputs[i].data()};
  }
  external[num_inputs] = xnn_external_value{output_id, subgraph_output.data()};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(ConcatenateTestQU8, matches_operator_api)
{
  inputs.resize(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    inputs[i] =
      std::vector<uint8_t>(NumElements(input_dims[i]), static_cast<uint8_t>(i + 1));  // For example, fill with i + 1
  }
  std::fill(operator_output.begin(), operator_output.end(), UINT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT8_C(0xA5));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  std::vector<xnn_operator_t> operators(num_inputs);
  std::vector<std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>> auto_op;
  auto_op.reserve(num_inputs);
  size_t offset = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x8(/*flags=*/0, &operators[i]));
    auto_op.emplace_back(
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>(operators[i], xnn_delete_operator));
    ASSERT_EQ(
      xnn_status_success, xnn_reshape_copy_nc_x8(
                            operators[i], batch_size, channels[i], channels[i], output_stride, /*threadpool=*/nullptr));
    ASSERT_EQ(
      xnn_status_success, xnn_setup_copy_nc_x8(operators[i], inputs[i].data(), operator_output.data() + offset));
    offset += channels[i];
    ASSERT_EQ(xnn_status_success, xnn_run_operator(operators[i], /*threadpool=*/nullptr));
  }

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_inputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<uint32_t> input_ids(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    input_ids[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success,
      xnn_define_quantized_tensor_value(
        subgraph, xnn_datatype_qint8, signed_zero_point, scale, input_dims[i].size(), input_dims[i].data(), nullptr, i,
        /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_ids[i]));
    ASSERT_NE(input_ids[i], XNN_INVALID_NODE_ID);
  }

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, signed_zero_point, scale, output_dims.size(),
                          output_dims.data(), nullptr, num_inputs,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_concatenate(subgraph, axis,num_inputs, input_ids.data(), output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  std::vector<xnn_external_value> external(1 + num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    external[i] = xnn_external_value{input_ids[i], inputs[i].data()};
  }
  external[num_inputs] = xnn_external_value{output_id, subgraph_output.data()};

  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(ConcatenateTestF16, matches_operator_api)
{
  inputs.resize(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    inputs[i] = std::vector<xnn_float16>(
      NumElements(input_dims[i]), static_cast<xnn_float16>(i + 1));  // For example, fill with i + 1
  }
  std::fill(operator_output.begin(), operator_output.end(), std::nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), std::nanf(""));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  std::vector<xnn_operator_t> operators(num_inputs);
  std::vector<std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>> auto_op;
  auto_op.reserve(num_inputs);
  size_t offset = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x16(/*flags=*/0, &operators[i]));
    auto_op.emplace_back(
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>(operators[i], xnn_delete_operator));
    ASSERT_EQ(
      xnn_status_success, xnn_reshape_copy_nc_x16(
                            operators[i], batch_size, channels[i], channels[i], output_stride, /*threadpool=*/nullptr));
    ASSERT_EQ(
      xnn_status_success,
      xnn_setup_copy_nc_x16(operators[i], inputs[i].data(), (xnn_float16*) operator_output.data() + offset));
    offset += channels[i];
    ASSERT_EQ(xnn_status_success, xnn_run_operator(operators[i], /*threadpool=*/nullptr));
  }

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_inputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<uint32_t> input_ids(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    input_ids[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success, xnn_define_tensor_value(
                            subgraph, xnn_datatype_fp16, input_dims[i].size(), input_dims[i].data(), nullptr, i,
                            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_ids[i]));
    ASSERT_NE(input_ids[i], XNN_INVALID_NODE_ID);
  }

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr, num_inputs,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_concatenate(subgraph, axis,num_inputs, input_ids.data(), output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  std::vector<xnn_external_value> external(1 + num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    external[i] = xnn_external_value{input_ids[i], inputs[i].data()};
  }
  external[num_inputs] = xnn_external_value{output_id, subgraph_output.data()};

  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
  ASSERT_EQ(subgraph_output, operator_output);
}


TEST_F(ConcatenateTestF32, matches_operator_api)
{
  inputs.resize(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    inputs[i] =
      std::vector<float>(NumElements(input_dims[i]), static_cast<float>(i + 1));  // For example, fill with i + 1
  }
  std::fill(operator_output.begin(), operator_output.end(), std::nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), std::nanf(""));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  std::vector<xnn_operator_t> operators(num_inputs);
  std::vector<std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>> auto_op;
  auto_op.reserve(num_inputs);
  size_t offset = 0;
  for (size_t i = 0; i < num_inputs; ++i) {
    ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x32(/*flags=*/0, &operators[i]));
    auto_op.emplace_back(
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>(operators[i], xnn_delete_operator));
    ASSERT_EQ(
      xnn_status_success, xnn_reshape_copy_nc_x32(
                            operators[i], batch_size, channels[i], channels[i], output_stride, /*threadpool=*/nullptr));
    ASSERT_EQ(
      xnn_status_success, xnn_setup_copy_nc_x32(operators[i], inputs[i].data(), operator_output.data() + offset));
    offset += channels[i];
    ASSERT_EQ(xnn_status_success, xnn_run_operator(operators[i], /*threadpool=*/nullptr));
  }

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_inputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<uint32_t> input_ids(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    input_ids[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success, xnn_define_tensor_value(
                            subgraph, xnn_datatype_fp32, input_dims[i].size(), input_dims[i].data(), nullptr, i,
                            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_ids[i]));
    ASSERT_NE(input_ids[i], XNN_INVALID_NODE_ID);
  }

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, num_inputs,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_concatenate(subgraph, axis,num_inputs, input_ids.data(), output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  std::vector<xnn_external_value> external(1 + num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    external[i] = xnn_external_value{input_ids[i], inputs[i].data()};
  }
  external[num_inputs] = xnn_external_value{output_id, subgraph_output.data()};

  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(ConcatenateTestF32, Reshape)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/num_inputs + 1, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  std::vector<uint32_t> input_ids(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    input_ids[i] = XNN_INVALID_NODE_ID;
    ASSERT_EQ(
      xnn_status_success, xnn_define_tensor_value(
                            subgraph, xnn_datatype_fp32, input_dims[i].size(), input_dims[i].data(), nullptr, i,
                            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_ids[i]));
    ASSERT_NE(input_ids[i], XNN_INVALID_NODE_ID);
  }

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, num_inputs,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_concatenate(subgraph, axis,num_inputs, input_ids.data(), output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_concatenate);
  ASSERT_EQ(node->num_inputs, num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    ASSERT_EQ(node->inputs[i], input_ids[i]);
  }
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  ASSERT_EQ(
    node->reshape(&runtime->opdata[0], subgraph->values, subgraph->num_values, /*threadpool=*/nullptr),
    xnn_status_success);

  input_dims[0][axis] += 1;
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_external_value(runtime, input_ids[0], input_dims[0].size(), input_dims[0].data()));

  ASSERT_EQ(
    node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr),
    xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  int Dimensions = 0;
  for (int i = 0; i < num_inputs; ++i) {
    Dimensions += input_dims[i][axis];
  }
  ASSERT_EQ(output_shape->dim[axis], Dimensions);
  for (size_t i = 0; i < input_dims[0].size(); ++i) {
    if (i == axis)
      continue;
    ASSERT_EQ(output_shape->dim[i], input_dims[0][i]);
  }

  for (size_t i = 0; i < input_dims[0].size(); ++i) {
    if (i == axis)
      continue;
    for (size_t j = 0; j < num_inputs; j++) {
      input_dims[j][i] += 1;
      ASSERT_EQ(
        xnn_status_success,
        xnn_reshape_external_value(runtime, input_ids[j], input_dims[j].size(), input_dims[j].data()));
    }

    ASSERT_EQ(
      node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr),
      xnn_status_reallocation_required);
    ASSERT_EQ(output_shape->dim[axis], Dimensions);
    for (size_t i = 0; i < input_dims[0].size(); ++i) {
      if (i == axis)
        continue;
      ASSERT_EQ(output_shape->dim[i], input_dims[0][i]);
    }
  }
  for (size_t i = 0; i < input_dims[0].size(); ++i) {
    if (i == axis)
      continue;
    for (size_t j = 0; j < num_inputs; j++) {
      input_dims[j][i] -= 1;
      ASSERT_EQ(
        xnn_status_success,
        xnn_reshape_external_value(runtime, input_ids[j], input_dims[j].size(), input_dims[j].data()));
    }
    ASSERT_EQ(
      node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr),
      xnn_status_success);
    ASSERT_EQ(output_shape->dim[axis], Dimensions);
    for (size_t i = 0; i < input_dims[0].size(); ++i) {
      if (i == axis)
        continue;
      ASSERT_EQ(output_shape->dim[i], input_dims[0][i]);
    }
  }
}
