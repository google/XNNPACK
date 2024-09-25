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
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/math.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"

template <typename T>
class NumericLimits {
 public:
  static constexpr T min() { return std::numeric_limits<T>::min(); }
  static constexpr T max() { return std::numeric_limits<T>::max(); }
};

template <>
class NumericLimits<xnn_float16> {
 public:
  static xnn_float16 min() { return -std::numeric_limits<float>::infinity(); }
  static xnn_float16 max() { return +std::numeric_limits<float>::infinity(); }
};

template <typename T>
struct UniformDistribution {
  std::uniform_real_distribution<T> dist{-10.0f, 10.0f};

  template <class Generator>
  T operator()(Generator& g) {
    return dist(g);
  }
};

template <>
struct UniformDistribution<xnn_float16> {
  std::uniform_real_distribution<float> dist{-10.0f, 10.0f};

  template <class Generator>
  xnn_float16 operator()(Generator& g) {
    return dist(g);
  }
};

template <>
struct UniformDistribution<int8_t> {
  std::uniform_int_distribution<int> dist{std::numeric_limits<int8_t>::min(),
                                          std::numeric_limits<int8_t>::max()};

  template <class Generator>
  int8_t operator()(Generator& g) {
    return dist(g);
  }
};

template <>
struct UniformDistribution<uint8_t> {
  std::uniform_int_distribution<int> dist{
      std::numeric_limits<uint8_t>::min(),
      std::numeric_limits<uint8_t>::max()};

  template <class Generator>
  uint8_t operator()(Generator& g) {
    return dist(g);
  }
};

template <>
struct UniformDistribution<int32_t> {
  std::uniform_int_distribution<int32_t> dist{
      std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::max()};

  template <class Generator>
  int32_t operator()(Generator& g) {
    return dist(g);
  }
};

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

template <typename T, typename Rng>
xnn_quantization_params RandomQuantization(Rng& rng) {
  if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
    return {
        static_cast<int32_t>(UniformDistribution<T>()(rng)),
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

bool is_quantized(xnn_datatype t) {
  switch (t) {
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
    case xnn_datatype_qint32:
      return true;
    default:
      return false;
  }
}

static const char* binary_operator_to_string(
    xnn_binary_operator operation_type) {
  switch (operation_type) {
    case xnn_binary_add:
      return "Add";
    case xnn_binary_copysign:
      return "CopySign";
    case xnn_binary_divide:
      return "Divide";
    case xnn_binary_maximum:
      return "Maximum";
    case xnn_binary_minimum:
      return "Minimum";
    case xnn_binary_multiply:
      return "Multiply";
    case xnn_binary_subtract:
      return "Subtract";
    case xnn_binary_squared_difference:
      return "SquaredDifference";
    default:
      return "Unknown";
  }
}

template <typename T>
xnn_datatype datatype_of() {
  if (std::is_same<T, uint8_t>::value) {
    return xnn_datatype_quint8;
  } else if (std::is_same<T, int8_t>::value) {
    return xnn_datatype_qint8;
  } else if (std::is_same<T, xnn_float16>::value) {
    return xnn_datatype_fp16;
  } else if (std::is_same<T, float>::value) {
    return xnn_datatype_fp32;
  } else if (std::is_same<T, int32_t>::value) {
    return xnn_datatype_int32;
  } else {
    XNN_UNREACHABLE;
  }
}

size_t xnn_datatype_size(xnn_datatype datatype) {
  switch (datatype) {
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
      return sizeof(int8_t);
    case xnn_datatype_fp16:
      return sizeof(xnn_float16);
    case xnn_datatype_fp32:
      return sizeof(float);
    case xnn_datatype_int32:
      return sizeof(int32_t);
    default:
      XNN_UNREACHABLE;
  }
}

// TODO(dsharlet): We need a place to put helper functions like this.
// XNNPACK's built-in equivalent helpers are not implemented in release
// builds...
const char* datatype_to_string(xnn_datatype datatype) {
  switch (datatype) {
    case xnn_datatype_qint8:
      return "qint8";
    case xnn_datatype_quint8:
      return "quint8";
    case xnn_datatype_fp16:
      return "fp16";
    case xnn_datatype_fp32:
      return "fp32";
    case xnn_datatype_int32:
      return "int32";
    default:
      XNN_UNREACHABLE;
  }
}

template <typename T>
void MatchesOperatorApi(xnn_binary_operator binary_op) {
  xnn_datatype datatype = datatype_of<T>();
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

  std::vector<T, AlignedAllocator<T, 64>> input0(NumElements(input0_dims) +
                                                 XNN_EXTRA_BYTES / sizeof(T));
  std::vector<T, AlignedAllocator<T, 64>> input1(NumElements(input1_dims) +
                                                 XNN_EXTRA_BYTES / sizeof(T));
  std::vector<T, AlignedAllocator<T, 64>> operator_output(
      NumElements(output_dims));
  std::vector<T, AlignedAllocator<T, 64>> subgraph_output(
      NumElements(output_dims));
  UniformDistribution<T> dist;
  std::generate(input0.begin(), input0.end(), [&]() { return dist(rng); });
  std::generate(input1.begin(), input1.end(), [&]() { return dist(rng); });

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  bool quantized = is_quantized(datatype);
  xnn_quantization_params input0_quantization = RandomQuantization<T>(rng);
  xnn_quantization_params input1_quantization = RandomQuantization<T>(rng);
  xnn_quantization_params output_quantization = RandomQuantization<T>(rng);

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
  ASSERT_EQ(node->type, xnn_binary_operator_to_node_type(binary_op));
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
            num_input_elements * xnn_datatype_size(datatype));
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
  ASSERT_EQ(node->type, xnn_binary_operator_to_node_type(binary_op));
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
            num_input_elements * xnn_datatype_size(datatype));
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
  ASSERT_EQ(node->type, xnn_binary_operator_to_node_type(binary_op));
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
            num_input_elements * xnn_datatype_size(datatype));
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
  ASSERT_EQ(node->type, xnn_binary_operator_to_node_type(binary_op));
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
            num_input_elements * xnn_datatype_size(datatype));
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
  ASSERT_EQ(node->type, xnn_binary_operator_to_node_type(binary_op));
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

template <typename T>
class BinaryTest : public testing::TestWithParam<xnn_binary_operator> {};

using BinaryTestQS8 = BinaryTest<int8_t>;
using BinaryTestQU8 = BinaryTest<uint8_t>;
#ifndef XNN_EXCLUDE_F16_TESTS
using BinaryTestF16 = BinaryTest<xnn_float16>;
#endif  // XNN_EXCLUDE_F16_TESTS
using BinaryTestF32 = BinaryTest<float>;
using BinaryTestS32 = BinaryTest<int32_t>;

TEST_P(BinaryTestQS8, matches_operator_api) {
  MatchesOperatorApi<int8_t>(GetParam());
}
TEST_P(BinaryTestQU8, matches_operator_api) {
  MatchesOperatorApi<uint8_t>(GetParam());
}
#ifndef XNN_EXCLUDE_F16_TESTS
TEST_P(BinaryTestF16, matches_operator_api) {
  MatchesOperatorApi<xnn_float16>(GetParam());
}
#endif  // XNN_EXCLUDE_F16_TESTS
TEST_P(BinaryTestF32, matches_operator_api) {
  MatchesOperatorApi<float>(GetParam());
}
TEST_P(BinaryTestS32, matches_operator_api) {
  MatchesOperatorApi<int32_t>(GetParam());
}

#ifndef XNN_EXCLUDE_F16_TESTS
TEST_P(BinaryTestF16, reshape) { Reshape(xnn_datatype_fp16, GetParam()); }
#endif  // XNN_EXCLUDE_F16_TESTS
TEST_P(BinaryTestF32, reshape) { Reshape(xnn_datatype_fp32, GetParam()); }
TEST_P(BinaryTestS32, reshape) { Reshape(xnn_datatype_int32, GetParam()); }

#ifndef XNN_EXCLUDE_F16_TESTS
TEST_P(BinaryTestF16, reshape_broadcast_dim0) {
  ReshapeBroadcastDim0(xnn_datatype_fp16, GetParam());
}
#endif  // XNN_EXCLUDE_F16_TESTS
TEST_P(BinaryTestF32, reshape_broadcast_dim0) {
  ReshapeBroadcastDim0(xnn_datatype_fp32, GetParam());
}
TEST_P(BinaryTestS32, reshape_broadcast_dim0) {
  ReshapeBroadcastDim0(xnn_datatype_int32, GetParam());
}

#ifndef XNN_EXCLUDE_F16_TESTS
TEST_P(BinaryTestF16, reshape_broadcast_1d) {
  ReshapeBroadcast1D(xnn_datatype_fp16, GetParam());
}
#endif  // XNN_EXCLUDE_F16_TESTS
TEST_P(BinaryTestF32, reshape_broadcast_1d) {
  ReshapeBroadcast1D(xnn_datatype_fp32, GetParam());
}
TEST_P(BinaryTestS32, reshape_broadcast_1d) {
  ReshapeBroadcast1D(xnn_datatype_int32, GetParam());
}

#ifndef XNN_EXCLUDE_F16_TESTS
TEST_P(BinaryTestF16, reshape_broadcast_2d) {
  ReshapeBroadcast2D(xnn_datatype_fp16, GetParam());
}
#endif  // XNN_EXCLUDE_F16_TESTS
TEST_P(BinaryTestF32, reshape_broadcast_2d) {
  ReshapeBroadcast2D(xnn_datatype_fp32, GetParam());
}
TEST_P(BinaryTestS32, reshape_broadcast_2d) {
  ReshapeBroadcast2D(xnn_datatype_int32, GetParam());
}

#ifndef XNN_EXCLUDE_F16_TESTS
TEST_P(BinaryTestF16, degenerate_dimension) {
  DegenerateDimension(xnn_datatype_fp16, GetParam());
}
#endif  // XNN_EXCLUDE_F16_TESTS
TEST_P(BinaryTestF32, degenerate_dimension) {
  DegenerateDimension(xnn_datatype_fp32, GetParam());
}
TEST_P(BinaryTestS32, degenerate_dimension) {
  DegenerateDimension(xnn_datatype_int32, GetParam());
}

std::string ToString(xnn_binary_operator op) {
  return binary_operator_to_string(op);
}

INSTANTIATE_TEST_SUITE_P(test, BinaryTestQS8,
                         testing::Values(xnn_binary_add, xnn_binary_subtract,
                                         xnn_binary_multiply),
                         [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(test, BinaryTestQU8,
                         testing::Values(xnn_binary_add, xnn_binary_subtract,
                                         xnn_binary_multiply),
                         [](const auto& info) { return ToString(info.param); });
#ifndef XNN_EXCLUDE_F16_TESTS
INSTANTIATE_TEST_SUITE_P(test, BinaryTestF16,
                         testing::Values(xnn_binary_add, xnn_binary_subtract,
                                         xnn_binary_multiply, xnn_binary_divide,
                                         xnn_binary_maximum, xnn_binary_minimum,
                                         xnn_binary_squared_difference),
                         [](const auto& info) { return ToString(info.param); });
#endif
INSTANTIATE_TEST_SUITE_P(test, BinaryTestF32,
                         testing::Values(xnn_binary_add, xnn_binary_subtract,
                                         xnn_binary_multiply, xnn_binary_divide,
                                         xnn_binary_maximum, xnn_binary_minimum,
                                         xnn_binary_copysign,
                                         xnn_binary_squared_difference),
                         [](const auto& info) { return ToString(info.param); });
INSTANTIATE_TEST_SUITE_P(test, BinaryTestS32,
                         testing::Values(xnn_binary_multiply),
                         [](const auto& info) { return ToString(info.param); });
