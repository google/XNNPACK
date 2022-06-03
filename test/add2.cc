// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/node-type.h>
#include <xnnpack/operator.h>
#include <xnnpack/requantization.h>
#include <xnnpack/subgraph.h>

#include <gtest/gtest.h>

template <typename T> class Add2Test : public ::testing::Test {
protected:
  Add2Test()
  {
    random_device = std::unique_ptr<std::random_device>(new std::random_device());
    rng = std::mt19937((*random_device)());
    shape_dist = std::uniform_int_distribution<size_t>(0, XNN_MAX_TENSOR_DIMS);
    dim_dist = std::uniform_int_distribution<size_t>(1, 9);
    f32dist = std::uniform_real_distribution<float>(0.01f, 1.0f);
    i8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    u8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    scale_dist = std::uniform_real_distribution<float>(0.1f, 10.0f);
  }

  void SetUp() override
  {
    std::vector<size_t> input1_shape = RandomShape();
    std::vector<size_t> input2_shape;
    std::vector<size_t> output_shape;
    // Create input dimensions.
    // Create input 2 with an equal or larger number of dimensions.
    const size_t input2_num_dims = std::uniform_int_distribution<size_t>(input1_shape.size(), XNN_MAX_TENSOR_DIMS)(rng);
    input2_shape = RandomShape(input2_num_dims);
    // Ensure that the inputs dimensions match.
    std::copy_backward(input1_shape.begin(), input1_shape.end(), input2_shape.end());

    // Choose a random dimension to broadcast for each input.
    const size_t input1_broadcast_dim = std::uniform_int_distribution<size_t>(0, input1_shape.size())(rng);
    if (input1_broadcast_dim < input1_shape.size()) {
      input1_shape[input1_broadcast_dim] = 1;
    }
    const size_t input2_broadcast_dim = std::uniform_int_distribution<size_t>(0, input2_shape.size())(rng);
    if (input2_broadcast_dim < input2_shape.size()) {
      input2_shape[input2_broadcast_dim] = 1;
    }
    // Calculate generalized shapes.
    std::fill(input1_dims.begin(), input1_dims.end(), 1);
    std::fill(input2_dims.begin(), input2_dims.end(), 1);
    std::fill(output_dims.begin(), output_dims.end(), 1);
    std::copy_backward(input1_shape.cbegin(), input1_shape.cend(), input1_dims.end());
    std::copy_backward(input2_shape.cbegin(), input2_shape.cend(), input2_dims.end());
    for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS; i++) {
      if (input1_dims[i] != 1 && input2_dims[i] != 1) {
        ASSERT_EQ(input1_dims[i], input2_dims[i]) << "i: " << i;
      }
      output_dims[i] = std::max(input1_dims[i], input2_dims[i]);
    }

    input1 = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(input1_shape));
    input2 = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(input2_shape));
    operator_output = std::vector<T>(NumElements(output_dims));
    subgraph_output = std::vector<T>(operator_output.size());
  }

  std::vector<size_t> RandomShape(size_t num_dims)
  {
    std::vector<size_t> dims(num_dims);
    std::generate(dims.begin(), dims.end(), [&] { return dim_dist(rng); });
    return dims;
  }

  std::vector<size_t> RandomShape() { return RandomShape(shape_dist(rng)); }

  size_t NumElements(std::vector<size_t>& dims)
  {
    return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
  }

  size_t NumElements(std::array<size_t, XNN_MAX_TENSOR_DIMS>& dims)
  {
    return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
  }

  std::unique_ptr<std::random_device> random_device;
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> shape_dist;
  std::uniform_int_distribution<size_t> dim_dist;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_real_distribution<float> scale_dist;
  std::uniform_int_distribution<int32_t> i8dist;
  std::uniform_int_distribution<int32_t> u8dist;

  float output_min = -std::numeric_limits<float>::infinity();
  float output_max = std::numeric_limits<float>::infinity();

  std::array<size_t, XNN_MAX_TENSOR_DIMS> input1_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> input2_dims;
  std::array<size_t, XNN_MAX_TENSOR_DIMS> output_dims;

  std::vector<T> input1;
  std::vector<T> input2;
  std::vector<T> operator_output;
  std::vector<T> subgraph_output;
};

using Add2TestQS8 = Add2Test<int8_t>;
using Add2TestQU8 = Add2Test<uint8_t>;
using Add2TestF32 = Add2Test<float>;

TEST_F(Add2TestQS8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  const int32_t zero_point = i8dist(rng);
  const float scale = scale_dist(rng);

  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, zero_point, scale, input1_dims.size(), input1_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, zero_point, scale, input2_dims.size(), input2_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, zero_point, scale, output_dims.size(), output_dims.data(), nullptr,
                          XNN_INVALID_VALUE_ID, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_add2(subgraph, output_min, output_max, input1_id, input2_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_add2);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qs8);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input1_id);
  ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(Add2TestQU8, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  const int32_t zero_point = u8dist(rng);
  const float scale = scale_dist(rng);

  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, zero_point, scale, input1_dims.size(), input1_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, zero_point, scale, input2_dims.size(), input2_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, zero_point, scale, output_dims.size(), output_dims.data(), nullptr,
                          XNN_INVALID_VALUE_ID, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_add2(subgraph, output_min, output_max, input1_id, input2_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_add2);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qu8);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input1_id);
  ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(Add2TestF32, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input1_dims.size(), input1_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input2_dims.size(), input2_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_add2(subgraph, output_min, output_max, input1_id, input2_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_add2);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->activation.output_min, output_min);
  ASSERT_EQ(node->activation.output_max, output_max);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input1_id);
  ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(Add2TestQS8, matches_operator_api)
{
  std::generate(input1.begin(), input1.end(), [&]() { return i8dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return i8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), 0xAA);
  std::fill(subgraph_output.begin(), subgraph_output.end(), 0xAA);

  const int32_t input1_zero_point = i8dist(rng);
  const float input1_scale = scale_dist(rng);
  const int32_t input2_zero_point = i8dist(rng);
  const float input2_scale = scale_dist(rng);
  const int32_t output_zero_point = i8dist(rng);
  const float output_scale = scale_dist(rng);
  const int8_t quantized_output_min = xnn_qs8_quantize(output_min, output_scale, output_zero_point);
  const int8_t quantized_output_max = xnn_qs8_quantize(output_max, output_scale, output_zero_point);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  // Call operator API.
  ASSERT_EQ(
    xnn_status_success, xnn_create_add_nd_qs8(
                          input1_zero_point, input1_scale, input2_zero_point, input2_scale, output_zero_point,
                          output_scale, quantized_output_min, quantized_output_max, /*flags=*/0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  ASSERT_EQ(
    xnn_status_success, xnn_setup_add_nd_qs8(
                          op, input1_dims.size(), input1_dims.data(), input2_dims.size(), input2_dims.data(),
                          input1.data(), input2.data(), operator_output.data(), nullptr /* thread pool */));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, nullptr /* thread pool */));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, input1_zero_point, input1_scale, input1_dims.size(), input1_dims.data(), nullptr,
      /*external_id=*/0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_qint8, input2_zero_point, input2_scale, input2_dims.size(), input2_dims.data(), nullptr,
      /*external_id=*/1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_qint8, output_zero_point, output_scale, output_dims.size(),
                          output_dims.data(), nullptr, /*external_id=*/2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_add2(subgraph, output_min, output_max, input1_id, input2_id, output_id, /*flags=*/0));

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

TEST_F(Add2TestQU8, matches_operator_api)
{
  std::generate(input1.begin(), input1.end(), [&]() { return u8dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return u8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), 0xAA);
  std::fill(subgraph_output.begin(), subgraph_output.end(), 0xAA);

  const int32_t input1_zero_point = u8dist(rng);
  const float input1_scale = scale_dist(rng);
  const int32_t input2_zero_point = u8dist(rng);
  const float input2_scale = scale_dist(rng);
  const int32_t output_zero_point = u8dist(rng);
  const float output_scale = scale_dist(rng);
  const uint8_t quantized_output_min = xnn_qu8_quantize(output_min, output_scale, output_zero_point);
  const uint8_t quantized_output_max = xnn_qu8_quantize(output_max, output_scale, output_zero_point);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  // Call operator API.
  ASSERT_EQ(
    xnn_status_success, xnn_create_add_nd_qu8(
                          input1_zero_point, input1_scale, input2_zero_point, input2_scale, output_zero_point,
                          output_scale, quantized_output_min, quantized_output_max, /*flags=*/0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  ASSERT_EQ(
    xnn_status_success, xnn_setup_add_nd_qu8(
                          op, input1_dims.size(), input1_dims.data(), input2_dims.size(), input2_dims.data(),
                          input1.data(), input2.data(), operator_output.data(), nullptr /* thread pool */));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, nullptr /* thread pool */));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, input1_zero_point, input1_scale, input1_dims.size(), input1_dims.data(), nullptr,
      /*external_id=*/0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_quantized_tensor_value(
      subgraph, xnn_datatype_quint8, input2_zero_point, input2_scale, input2_dims.size(), input2_dims.data(), nullptr,
      /*external_id=*/1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_quantized_tensor_value(
                          subgraph, xnn_datatype_quint8, output_zero_point, output_scale, output_dims.size(),
                          output_dims.data(), nullptr, /*external_id=*/2,
                          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_add2(subgraph, output_min, output_max, input1_id, input2_id, output_id, /*flags=*/0));

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

TEST_F(Add2TestF32, matches_operator_api)
{
  std::generate(input1.begin(), input1.end(), [&]() { return f32dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  // Call operator API.
  ASSERT_EQ(xnn_status_success, xnn_create_add_nd_f32(output_min, output_max, 0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  ASSERT_EQ(
    xnn_status_success, xnn_setup_add_nd_f32(
                          op, input1_dims.size(), input1_dims.data(), input2_dims.size(), input2_dims.data(),
                          input1.data(), input2.data(), operator_output.data(), nullptr /* thread pool */));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, nullptr /* thread pool */));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input1_dims.size(), input1_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t input2_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input2_dims.size(), input2_dims.data(), nullptr,
                          /*external_id=*/1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr, /*external_id=*/2,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_add2(subgraph, output_min, output_max, input1_id, input2_id, output_id, /*flags=*/0));

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
