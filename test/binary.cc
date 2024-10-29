// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/buffer.h"
#include "xnnpack/datatype.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph.h"
#include "operator-test-utils.h"
#include "replicable_random_device.h"

using ::testing::Combine;
using ::testing::Values;

template <typename Rng>
size_t RandomRank(Rng& rng) {
  return std::uniform_int_distribution<size_t>(0, XNN_MAX_TENSOR_DIMS)(rng);
}

template <typename Rng>
std::vector<size_t> RandomShape(Rng& rng, size_t rank) {
  std::uniform_int_distribution<size_t> dims_dist(1, 9);
  std::vector<size_t> dims(rank);
  std::generate(dims.begin(), dims.end(), [&]() { return dims_dist(rng); });
  return dims;
}

template <typename Rng>
std::vector<size_t> RandomShape(Rng& rng) {
  return RandomShape(rng, RandomRank(rng));
}

template <typename Rng>
xnn_quantization_params RandomQuantization(xnn_datatype datatype, Rng& rng) {
  if (datatype == xnn_datatype_qint8) {
    std::uniform_int_distribution<int> dist{std::numeric_limits<int8_t>::min(),
                                            std::numeric_limits<int8_t>::max()};
    return {
        static_cast<int32_t>(dist(rng)),
        std::uniform_real_distribution<float>(0.1f, 5.0f)(rng),
    };
  } else if (datatype == xnn_datatype_quint8) {
    std::uniform_int_distribution<int> dist{
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max()};
    return {
        static_cast<int32_t>(dist(rng)),
        std::uniform_real_distribution<float>(0.1f, 5.0f)(rng),
    };
  } else {
    return {0, 1.0f};
  }
}

void RemoveLeadingOnes(std::vector<size_t>& dims) {
  while (!dims.empty()) {
    if (dims.front() == 1) {
      dims.erase(dims.begin());
    } else {
      break;
    }
  }
}

size_t NumElements(const std::vector<size_t>& dims) {
  return std::accumulate(dims.begin(), dims.end(), size_t(1),
                         std::multiplies<size_t>());
}

void MatchesOperatorApi(xnn_datatype datatype, xnn_binary_operator binary_op) {
  xnnpack::ReplicableRandomDevice rng;

  std::vector<size_t> input0_dims = RandomShape(rng);
  std::vector<size_t> input1_dims;
  std::vector<size_t> output_dims;
  // Create input dimensions.
  // Create input 2 with an equal or larger number of dimensions.
  const size_t input1_num_dims = std::uniform_int_distribution<size_t>(
      input0_dims.size(), XNN_MAX_TENSOR_DIMS)(rng);
  input1_dims = RandomShape(rng, input1_num_dims);
  // Ensure that the inputs dimensions match.
  std::copy_backward(input0_dims.begin(), input0_dims.end(), input1_dims.end());

  // Choose a random dimension to broadcast for each input.
  const size_t input0_broadcast_dim =
      std::uniform_int_distribution<size_t>(0, input0_dims.size())(rng);
  if (input0_broadcast_dim < input0_dims.size()) {
    input0_dims[input0_broadcast_dim] = 1;
  }
  const size_t input1_broadcast_dim =
      std::uniform_int_distribution<size_t>(0, input1_dims.size())(rng);
  if (input1_broadcast_dim < input1_dims.size()) {
    input1_dims[input1_broadcast_dim] = 1;
  }
  input0_dims.resize(XNN_MAX_TENSOR_DIMS);
  input1_dims.resize(XNN_MAX_TENSOR_DIMS);
  output_dims.resize(XNN_MAX_TENSOR_DIMS);

  // Calculate generalized shapes.
  std::fill(input0_dims.begin(), input0_dims.end(), 1);
  std::fill(input1_dims.begin(), input1_dims.end(), 1);
  std::copy_backward(input0_dims.cbegin(), input0_dims.cend(),
                     input0_dims.end());
  std::copy_backward(input1_dims.cbegin(), input1_dims.cend(),
                     input1_dims.end());
  for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS; i++) {
    if (input0_dims[i] != 1 && input1_dims[i] != 1) {
      ASSERT_EQ(input0_dims[i], input1_dims[i]) << "i: " << i;
    }
    output_dims[i] = std::max(input0_dims[i], input1_dims[i]);
  }

  if (rng() % 2 == 0) {
    RemoveLeadingOnes(input0_dims);
  }
  if (rng() % 2 == 0) {
    RemoveLeadingOnes(input1_dims);
  }
  while (output_dims.size() >
         std::max(input0_dims.size(), input1_dims.size())) {
    output_dims.erase(output_dims.begin());
  }

  size_t datatype_size = xnn_datatype_size_bytes(datatype);
  xnnpack::Buffer<char, XNN_ALLOCATION_ALIGNMENT> input0(
      NumElements(input0_dims) * datatype_size +
      XNN_EXTRA_BYTES / sizeof(char));
  xnnpack::Buffer<char, XNN_ALLOCATION_ALIGNMENT> input1(
      NumElements(input1_dims) * datatype_size +
      XNN_EXTRA_BYTES / sizeof(char));
  xnnpack::Buffer<char, XNN_ALLOCATION_ALIGNMENT> operator_output(
      NumElements(output_dims) * datatype_size);
  xnnpack::Buffer<char, XNN_ALLOCATION_ALIGNMENT> subgraph_output(
      NumElements(output_dims) * datatype_size);

  double datatype_min, datatype_max;
  switch (datatype) {
    case xnn_datatype_quint8:
      datatype_min = std::numeric_limits<uint8_t>::min();
      datatype_max = std::numeric_limits<uint8_t>::max();
      break;
    case xnn_datatype_qint8:
      datatype_min = std::numeric_limits<int8_t>::min();
      datatype_max = std::numeric_limits<int8_t>::max();
      break;
    case xnn_datatype_int32:
      datatype_min = std::numeric_limits<int32_t>::min();
      datatype_max = std::numeric_limits<int32_t>::max();
      break;
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      datatype_min = -10.0;
      datatype_max = 10.0;
      break;
    default:
      datatype_min = 0;
      datatype_max = 0;
      assert(false);
      break;
  }
  std::uniform_real_distribution<double> dist(datatype_min, datatype_max);
  randomize_buffer(datatype, rng, dist, input0);
  randomize_buffer(datatype, rng, dist, input1);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  bool quantized = xnn_datatype_is_quantized(datatype);
  xnn_quantization_params input0_quantization =
      RandomQuantization(datatype, rng);
  xnn_quantization_params input1_quantization =
      RandomQuantization(datatype, rng);
  xnn_quantization_params output_quantization =
      RandomQuantization(datatype, rng);

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input0_id = XNN_INVALID_NODE_ID;
  uint32_t input1_id = XNN_INVALID_NODE_ID;
  uint32_t output_id = XNN_INVALID_NODE_ID;
  if (quantized) {
    ASSERT_EQ(xnn_status_success,
              xnn_define_quantized_tensor_value(
                  subgraph, datatype, input0_quantization.zero_point,
                  input0_quantization.scale, input0_dims.size(),
                  input0_dims.data(), nullptr,
                  /*external_id=*/0, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                  &input0_id));

    ASSERT_EQ(xnn_status_success,
              xnn_define_quantized_tensor_value(
                  subgraph, datatype, input1_quantization.zero_point,
                  input1_quantization.scale, input1_dims.size(),
                  input1_dims.data(), nullptr,
                  /*external_id=*/1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                  &input1_id));

    ASSERT_EQ(xnn_status_success,
              xnn_define_quantized_tensor_value(
                  subgraph, datatype, output_quantization.zero_point,
                  output_quantization.scale, output_dims.size(),
                  output_dims.data(), nullptr, /*external_id=*/2,
                  /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  } else {
    ASSERT_EQ(xnn_status_success,
              xnn_define_tensor_value(subgraph, datatype, input0_dims.size(),
                                      input0_dims.data(), nullptr,
                                      /*external_id=*/0,
                                      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                      &input0_id));

    ASSERT_EQ(xnn_status_success,
              xnn_define_tensor_value(subgraph, datatype, input1_dims.size(),
                                      input1_dims.data(), nullptr,
                                      /*external_id=*/1,
                                      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                      &input1_id));

    ASSERT_EQ(xnn_status_success,
              xnn_define_tensor_value(
                  subgraph, datatype, output_dims.size(), output_dims.data(),
                  nullptr, /*external_id=*/2,
                  /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  }
  ASSERT_NE(input0_id, XNN_INVALID_NODE_ID);
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_binary(subgraph, binary_op, nullptr, input0_id,
                              input1_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  xnn_status status =
      xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 3> external = {
      xnn_external_value{input0_id, input0.data()},
      xnn_external_value{input1_id, input1.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Call operator API.
  xnn_operator_t op = nullptr;

  if (quantized) {
    ASSERT_EQ(xnn_status_success, xnn_create_binary_elementwise_nd(
                                      binary_op, datatype, &input0_quantization,
                                      &input1_quantization,
                                      &output_quantization, /*flags=*/0, &op));
  } else {
    ASSERT_EQ(xnn_status_success, xnn_create_binary_elementwise_nd(
                                      binary_op, datatype, &input0_quantization,
                                      &input1_quantization,
                                      &output_quantization, /*flags=*/0, &op));
  }
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  ASSERT_EQ(xnn_status_success, xnn_reshape_binary_elementwise_nd(
                                    op, input0_dims.size(), input0_dims.data(),
                                    input1_dims.size(), input1_dims.data(),
                                    /*threadpool=*/nullptr));

  ASSERT_EQ(xnn_status_success,
            xnn_setup_binary_elementwise_nd(op, input0.data(), input1.data(),
                                            operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Check output shape matches.
  size_t observed_output_num_dims = 0;
  std::vector<size_t> observed_output_dims(XNN_MAX_TENSOR_DIMS, 0);
  ASSERT_EQ(
    xnn_status_success,
    xnn_get_external_value_shape(runtime, output_id, &observed_output_num_dims, observed_output_dims.data()));
  ASSERT_EQ(output_dims.size(), observed_output_num_dims);
  for (size_t i = 0; i < observed_output_num_dims; i++) {
    ASSERT_EQ(output_dims[i], observed_output_dims[i]);
  }

  // Check outputs match.
  ASSERT_EQ(subgraph_output, operator_output);
}

void Reshape(xnn_datatype datatype, xnn_binary_operator binary_op) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  std::vector<size_t> dims{2, 3, 4};
  uint32_t input0_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, datatype, dims.size(),
                                    dims.data(), nullptr, /*external_id=*/0,
                                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                    &input0_id));
  ASSERT_NE(input0_id, XNN_INVALID_NODE_ID);
  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, datatype, dims.size(),
                                    dims.data(), nullptr, /*external_id=*/1,
                                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                    &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, datatype, dims.size(),
                                    dims.data(), nullptr, /*external_id=*/2,
                                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                                    &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_binary(subgraph, binary_op, nullptr, input0_id,
                              input1_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_binary_elementwise);
  ASSERT_EQ(node->binary_operator, binary_op);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input0_id);
  ASSERT_EQ(node->inputs[1], input1_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);

  xnn_runtime_t runtime = nullptr;
  xnn_status status =
      xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  ASSERT_EQ(node->reshape(&runtime->opdata[0], subgraph->values,
                          subgraph->num_values, /*threadpool=*/nullptr),
            xnn_status_success);

  dims[0] = 7;
  ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_external_value(runtime, input0_id, dims.size(), dims.data()));
  ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_external_value(runtime, input1_id, dims.size(), dims.data()));

  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values,
                          runtime->num_values, /*threadpool=*/nullptr),
            xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  const size_t num_input_elements = std::accumulate(
      dims.cbegin(), dims.cend(), size_t{1}, std::multiplies<size_t>());
  ASSERT_EQ(output_shape->dim[0], dims[0]);
  ASSERT_EQ(runtime->values[node->outputs[0]].size,
            num_input_elements * xnn_datatype_size_bytes(datatype));
}

void ReshapeBroadcastDim0(xnn_datatype datatype,
                          xnn_binary_operator binary_op) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<size_t> dim0{1, 3, 4};
  std::vector<size_t> dim1{5, 3, 4};
  uint32_t input0_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, datatype, dim0.size(),
                                    dim0.data(), nullptr, /*external_id=*/0,
                                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                    &input0_id));
  ASSERT_NE(input0_id, XNN_INVALID_NODE_ID);
  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, datatype, dim1.size(),
                                    dim1.data(), nullptr, /*external_id=*/1,
                                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                    &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  // Output dims will be correctly set by reshape.
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, datatype, dim1.size(),
                                    dim1.data(), nullptr, /*external_id=*/2,
                                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                                    &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_binary(subgraph, binary_op, nullptr, input0_id,
                              input1_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_binary_elementwise);
  ASSERT_EQ(node->binary_operator, binary_op);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input0_id);
  ASSERT_EQ(node->inputs[1], input1_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);

  xnn_runtime_t runtime = nullptr;
  xnn_status status =
      xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  ASSERT_EQ(node->reshape(&runtime->opdata[0], subgraph->values,
                          subgraph->num_values, /*threadpool=*/nullptr),
            xnn_status_success);

  dim0[0] = 7;
  dim1[0] = 1;
  ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_external_value(runtime, input0_id, dim0.size(), dim0.data()));
  ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_external_value(runtime, input1_id, dim1.size(), dim1.data()));

  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values,
                          runtime->num_values, /*threadpool=*/nullptr),
            xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  const size_t num_input_elements = std::accumulate(
      dim0.cbegin(), dim0.cend(), size_t{1}, std::multiplies<size_t>());
  ASSERT_EQ(output_shape->dim[0], dim0[0]);
  ASSERT_EQ(runtime->values[node->outputs[0]].size,
            num_input_elements * xnn_datatype_size_bytes(datatype));
}

void ReshapeBroadcast1D(xnn_datatype datatype, xnn_binary_operator binary_op) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  std::vector<size_t> dim0{1, 20, 80, 32};
  std::vector<size_t> dim1{32};
  uint32_t input0_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, datatype, dim0.size(),
                                    dim0.data(), nullptr, /*external_id=*/0,
                                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                    &input0_id));
  ASSERT_NE(input0_id, XNN_INVALID_NODE_ID);
  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, datatype, dim1.size(),
                                    dim1.data(), nullptr, /*external_id=*/1,
                                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                    &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, datatype, dim0.size(),
                                    dim0.data(), nullptr, /*external_id=*/2,
                                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                                    &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_binary(subgraph, binary_op, nullptr, input0_id,
                              input1_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_binary_elementwise);
  ASSERT_EQ(node->binary_operator, binary_op);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input0_id);
  ASSERT_EQ(node->inputs[1], input1_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);

  xnn_runtime_t runtime = nullptr;
  xnn_status status =
      xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  ASSERT_EQ(node->reshape(&runtime->opdata[0], subgraph->values,
                          subgraph->num_values, /*threadpool=*/nullptr),
            xnn_status_success);

  dim0[0] = 7;
  dim1[0] = 1;
  ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_external_value(runtime, input0_id, dim0.size(), dim0.data()));
  ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_external_value(runtime, input1_id, dim1.size(), dim1.data()));

  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values,
                          runtime->num_values, /*threadpool=*/nullptr),
            xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  const size_t num_input_elements = std::accumulate(
      dim0.cbegin(), dim0.cend(), size_t{1}, std::multiplies<size_t>());
  ASSERT_EQ(output_shape->dim[0], dim0[0]);
  ASSERT_EQ(runtime->values[node->outputs[0]].size,
            num_input_elements * xnn_datatype_size_bytes(datatype));
}

void ReshapeBroadcast2D(xnn_datatype datatype, xnn_binary_operator binary_op) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  std::vector<size_t> dim0{1, 20, 80, 32};
  std::vector<size_t> dim1{80, 32};
  uint32_t input0_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, datatype, dim0.size(),
                                    dim0.data(), nullptr, /*external_id=*/0,
                                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                    &input0_id));
  ASSERT_NE(input0_id, XNN_INVALID_NODE_ID);
  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, datatype, dim1.size(),
                                    dim1.data(), nullptr, /*external_id=*/1,
                                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                    &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, datatype, dim0.size(),
                                    dim0.data(), nullptr, /*external_id=*/2,
                                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                                    &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_binary(subgraph, binary_op, nullptr, input0_id,
                              input1_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_binary_elementwise);
  ASSERT_EQ(node->binary_operator, binary_op);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input0_id);
  ASSERT_EQ(node->inputs[1], input1_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);

  xnn_runtime_t runtime = nullptr;
  xnn_status status =
      xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  ASSERT_EQ(node->reshape(&runtime->opdata[0], subgraph->values,
                          subgraph->num_values, /*threadpool=*/nullptr),
            xnn_status_success);

  dim0[0] = 7;
  dim1[0] = 1;
  ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_external_value(runtime, input0_id, dim0.size(), dim0.data()));
  ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_external_value(runtime, input1_id, dim1.size(), dim1.data()));

  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values,
                          runtime->num_values, /*threadpool=*/nullptr),
            xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  const size_t num_input_elements = std::accumulate(
      dim0.cbegin(), dim0.cend(), size_t{1}, std::multiplies<size_t>());
  ASSERT_EQ(output_shape->dim[0], dim0[0]);
  ASSERT_EQ(runtime->values[node->outputs[0]].size,
            num_input_elements * xnn_datatype_size_bytes(datatype));
}

void DegenerateDimension(xnn_datatype datatype, xnn_binary_operator binary_op) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/3,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  std::vector<size_t> dim0{0, 32};
  std::vector<size_t> dim1{2, 0, 32};
  uint32_t input0_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, datatype, dim0.size(),
                                    dim0.data(), nullptr, /*external_id=*/0,
                                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                    &input0_id));
  ASSERT_NE(input0_id, XNN_INVALID_NODE_ID);
  uint32_t input1_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, datatype, dim1.size(),
                                    dim1.data(), nullptr, /*external_id=*/1,
                                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                    &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, datatype, dim1.size(),
                                    dim1.data(), nullptr, /*external_id=*/2,
                                    /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                                    &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_binary(subgraph, binary_op, nullptr, input0_id,
                              input1_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_binary_elementwise);
  ASSERT_EQ(node->binary_operator, binary_op);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input0_id);
  ASSERT_EQ(node->inputs[1], input1_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);

  xnn_runtime_t runtime = nullptr;
  xnn_status status =
      xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  ASSERT_EQ(node->reshape(&runtime->opdata[0], subgraph->values,
                          subgraph->num_values, /*threadpool=*/nullptr),
            xnn_status_success);
}

struct Param {
  using TupleT = std::tuple<xnn_datatype, xnn_binary_operator>;
  explicit Param(TupleT p)
      : datatype(std::get<0>(p)), binary_operator(std::get<1>(p)) {}

  std::string Name() const {
    std::stringstream sstr;
    sstr << xnn_datatype_to_string(datatype) << "_"
         << xnn_binary_operator_to_string(binary_operator);
    return sstr.str();
  }

  xnn_datatype datatype;
  xnn_binary_operator binary_operator;
};

class BinaryTest : public testing::TestWithParam<Param> {};

// Some combinations aren't implemented.
bool SupportedBinaryTest(xnn_datatype datatype, xnn_binary_operator binary_op) {
  switch (datatype) {
    case xnn_datatype_quint8:
    case xnn_datatype_qint8:
      switch (binary_op) {
        case xnn_binary_add:
        case xnn_binary_multiply:
        case xnn_binary_subtract:
          return true;
        default:
          return false;
      }
    case xnn_datatype_int32:
      switch (binary_op) {
        case xnn_binary_multiply:
          return true;
        default:
          return false;
      }
    case xnn_datatype_fp16:
#ifdef XNN_EXCLUDE_F16_TESTS
      return false;
#else
      switch (binary_op) {
        case xnn_binary_add:
        case xnn_binary_divide:
        case xnn_binary_maximum:
        case xnn_binary_minimum:
        case xnn_binary_multiply:
        case xnn_binary_prelu:
        case xnn_binary_squared_difference:
        case xnn_binary_subtract:
          return true;
        default:
          return false;
      }
#endif
    case xnn_datatype_fp32:
      switch (binary_op) {
        case xnn_binary_add:
        case xnn_binary_copysign:
        case xnn_binary_divide:
        case xnn_binary_maximum:
        case xnn_binary_minimum:
        case xnn_binary_multiply:
        case xnn_binary_prelu:
        case xnn_binary_subtract:
        case xnn_binary_squared_difference:
          return true;
        default:
          return false;
      }
    default:
      return false;
  }
}

TEST_P(BinaryTest, matches_operator_api) {
  if (!SupportedBinaryTest(GetParam().datatype, GetParam().binary_operator)) {
    GTEST_SKIP();
  }
  MatchesOperatorApi(GetParam().datatype, GetParam().binary_operator);
}

TEST_P(BinaryTest, reshape) {
  if (!SupportedBinaryTest(GetParam().datatype, GetParam().binary_operator)) {
    GTEST_SKIP();
  }
  if (xnn_datatype_is_quantized(GetParam().datatype)) {
    GTEST_SKIP();
  }
  Reshape(GetParam().datatype, GetParam().binary_operator);
}

TEST_P(BinaryTest, reshape_broadcast_dim0) {
  if (!SupportedBinaryTest(GetParam().datatype, GetParam().binary_operator)) {
    GTEST_SKIP();
  }
  if (xnn_datatype_is_quantized(GetParam().datatype)) {
    GTEST_SKIP();
  }
  ReshapeBroadcastDim0(GetParam().datatype, GetParam().binary_operator);
}

TEST_P(BinaryTest, reshape_broadcast_1d) {
  if (!SupportedBinaryTest(GetParam().datatype, GetParam().binary_operator)) {
    GTEST_SKIP();
  }
  if (xnn_datatype_is_quantized(GetParam().datatype)) {
    GTEST_SKIP();
  }
  ReshapeBroadcast1D(GetParam().datatype, GetParam().binary_operator);
}

TEST_P(BinaryTest, reshape_broadcast_2d) {
  if (!SupportedBinaryTest(GetParam().datatype, GetParam().binary_operator)) {
    GTEST_SKIP();
  }
  if (xnn_datatype_is_quantized(GetParam().datatype)) {
    GTEST_SKIP();
  }
  ReshapeBroadcast2D(GetParam().datatype, GetParam().binary_operator);
}

INSTANTIATE_TEST_SUITE_P(
    BinaryTest, BinaryTest,
    testing::ConvertGenerator<Param::TupleT>(Combine(
        Values(xnn_datatype_quint8, xnn_datatype_qint8, xnn_datatype_fp16,
               xnn_datatype_fp32, xnn_datatype_int32),
        Values(xnn_binary_add, xnn_binary_subtract, xnn_binary_multiply,
               xnn_binary_divide, xnn_binary_maximum, xnn_binary_minimum,
               xnn_binary_copysign, xnn_binary_squared_difference,
               xnn_binary_prelu))),
    [](const auto& info) { return info.param.Name(); });
