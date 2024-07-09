// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/reshape-helpers.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"

xnn_runtime_t SetupUnary(const std::vector<size_t> &dims) {
  if (xnn_initialize(/*allocator=*/nullptr) != xnn_status_success) {
    return nullptr;
  }

  xnn_subgraph_t subgraph = nullptr;
  if (xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0, &subgraph) !=
      xnn_status_success) {
    return nullptr;
  }
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  if (xnn_define_tensor_value(subgraph, xnn_datatype_fp32, dims.size(),
                              dims.data(), nullptr, 0,
                              /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                              &input_id) != xnn_status_success) {
    return nullptr;
  }
  if (input_id == XNN_INVALID_NODE_ID) {
    return nullptr;
  }

  uint32_t output_id = XNN_INVALID_NODE_ID;
  if (xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 0, dims.data(),
                              nullptr, 1,
                              /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                              &output_id) != xnn_status_success) {
    return nullptr;
  }
  if (output_id == XNN_INVALID_NODE_ID) {
    return nullptr;
  }

  if (xnn_define_abs(subgraph, input_id, output_id, /*flags=*/0) !=
      xnn_status_success) {
    return nullptr;
  }

  xnn_runtime_t runtime = nullptr;
  if (xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0,
                            &runtime) != xnn_status_success) {
    return nullptr;
  }

  return runtime;
}

xnn_runtime_t SetupBinary(const std::vector<size_t> &input0_dims,
                          const std::vector<size_t> &input1_dims) {
  if (xnn_initialize(/*allocator=*/nullptr) != xnn_status_success) {
    return nullptr;
  }

  xnn_subgraph_t subgraph = nullptr;
  if (xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph) !=
      xnn_status_success) {
    return nullptr;
  }
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input0_id = XNN_INVALID_NODE_ID;
  if (xnn_define_tensor_value(subgraph, xnn_datatype_fp32, input0_dims.size(),
                              input0_dims.data(), nullptr, 0,
                              /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                              &input0_id) != xnn_status_success) {
    return nullptr;
  }
  if (input0_id == XNN_INVALID_NODE_ID) {
    return nullptr;
  }
  uint32_t input1_id = XNN_INVALID_NODE_ID;
  if (xnn_define_tensor_value(subgraph, xnn_datatype_fp32, input1_dims.size(),
                              input1_dims.data(), nullptr, 1,
                              /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                              &input1_id) != xnn_status_success) {
    return nullptr;
  }
  if (input1_id == XNN_INVALID_NODE_ID) {
    return nullptr;
  }
  uint32_t output_id = XNN_INVALID_NODE_ID;
  if (xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 0,
                              input1_dims.data(), nullptr, 2,
                              /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                              &output_id) != xnn_status_success) {
    return nullptr;
  }
  if (output_id == XNN_INVALID_NODE_ID) {
    return nullptr;
  }

  const float output_min = -std::numeric_limits<float>::infinity();
  const float output_max = std::numeric_limits<float>::infinity();
  if (xnn_define_add2(subgraph, output_min, output_max, input0_id, input1_id,
                      output_id, /*flags=*/0) != xnn_status_success) {
    return nullptr;
  }

  xnn_runtime_t runtime = nullptr;
  if (xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0,
                            &runtime) != xnn_status_success) {
    return nullptr;
  }

  return runtime;
}

TEST(ReshapeHelpersTest, Unary3D) {
  std::vector<size_t> dims{2, 3, 4};
  xnn_runtime_t runtime = SetupUnary(dims);
  ASSERT_NE(runtime, nullptr);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));

  size_t num_output_dims;
  std::vector<size_t> output_dims(XNN_MAX_TENSOR_DIMS);
  ASSERT_EQ(xnn_status_success,
            xnn_get_external_value_shape(runtime, /*external_id=*/1,
                                         &num_output_dims, &output_dims[0]));
  for (size_t i = 0; i < num_output_dims; ++i) {
    ASSERT_EQ(output_dims[i], dims[i]);
  }
}

TEST(ReshapeHelpersTest, UnaryScalar) {
  std::vector<size_t> dims{};
  xnn_runtime_t runtime = SetupUnary(dims);
  ASSERT_NE(runtime, nullptr);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));

  size_t num_output_dims;
  std::vector<size_t> output_dims(XNN_MAX_TENSOR_DIMS);
  ASSERT_EQ(xnn_status_success,
            xnn_get_external_value_shape(runtime, /*external_id=*/1,
                                         &num_output_dims, &output_dims[0]));
  std::vector<float> input(1 + XNN_EXTRA_BYTES / sizeof(float));
  input[0] = -7;
  std::vector<float> output(1);
  ASSERT_EQ(num_output_dims, 0);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{0, input.data()},
      xnn_external_value{1, output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime_v2(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
  ASSERT_EQ(output[0], std::abs(input[0]));
}

TEST(ReshapeHelpersTest, BinaryScalarLHSRHS) {
  std::vector<size_t> dims{};
  xnn_runtime_t runtime = SetupBinary(dims, dims);
  ASSERT_NE(runtime, nullptr);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));

  size_t num_output_dims;
  std::vector<size_t> output_dims(XNN_MAX_TENSOR_DIMS);
  ASSERT_EQ(xnn_status_success,
            xnn_get_external_value_shape(runtime, /*external_id=*/1,
                                         &num_output_dims, &output_dims[0]));
  std::vector<float> input0(1 + XNN_EXTRA_BYTES / sizeof(float));
  input0[0] = 2;
  std::vector<float> input1(1 + XNN_EXTRA_BYTES / sizeof(float));
  input1[0] = 3;
  std::vector<float> output(1);
  ASSERT_EQ(num_output_dims, 0);
  std::array<xnn_external_value, 3> external = {
      xnn_external_value{0, input0.data()},
      xnn_external_value{1, input1.data()},
      xnn_external_value{2, output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime_v2(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
  ASSERT_EQ(output[0], input0[0] + input1[0]);
}

TEST(ReshapeHelpersTest, BinaryScalarLHS3DRHS) {
  std::vector<size_t> input0_dims{};
  std::vector<size_t> input1_dims{3, 4, 5};
  xnn_runtime_t runtime = SetupBinary(input0_dims, input1_dims);
  ASSERT_NE(runtime, nullptr);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));

  size_t num_output_dims;
  std::vector<size_t> output_dims(input1_dims.size());
  ASSERT_EQ(xnn_status_success,
            xnn_get_external_value_shape(runtime, /*external_id=*/2,
                                         &num_output_dims, &output_dims[0]));
  ASSERT_EQ(num_output_dims, input1_dims.size());
  ASSERT_EQ(output_dims, input1_dims);

  const size_t num_input0_elements =
      std::accumulate(input0_dims.begin(), input0_dims.end(), size_t(1),
                      std::multiplies<size_t>());
  const size_t num_input1_elements =
      std::accumulate(input1_dims.begin(), input1_dims.end(), size_t(1),
                      std::multiplies<size_t>());
  const size_t num_output_elements =
      std::max(num_input0_elements, num_input1_elements);
  std::vector<float> input0(num_input0_elements);
  std::vector<float> input1(num_input1_elements);
  std::vector<float> output(num_output_elements);

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(-1.f, 1.f);
  std::generate(input0.begin(), input0.end(), [&]() { return f32dist(rng); });
  std::generate(input1.begin(), input1.end(), [&]() { return f32dist(rng); });

  std::array<xnn_external_value, 3> external = {
      xnn_external_value{0, input0.data()},
      xnn_external_value{1, input1.data()},
      xnn_external_value{2, output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime_v2(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
  for (int i = 0; i < num_output_elements; ++i) {
    ASSERT_EQ(output[i], input0[0] + input1[i]);
  }
}

TEST(ReshapeHelpersTest, Binary3DLHSScalarRHS) {
  std::vector<size_t> input0_dims{3, 4, 5};
  std::vector<size_t> input1_dims{};
  xnn_runtime_t runtime = SetupBinary(input0_dims, input1_dims);
  ASSERT_NE(runtime, nullptr);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));

  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));

  size_t num_output_dims;
  std::vector<size_t> output_dims(input0_dims.size());
  ASSERT_EQ(xnn_status_success,
            xnn_get_external_value_shape(runtime, /*external_id=*/2,
                                         &num_output_dims, &output_dims[0]));
  ASSERT_EQ(num_output_dims, input0_dims.size());
  ASSERT_EQ(output_dims, input0_dims);

  const size_t num_input0_elements =
      std::accumulate(input0_dims.begin(), input0_dims.end(), size_t(1),
                      std::multiplies<size_t>());
  const size_t num_input1_elements =
      std::accumulate(input1_dims.begin(), input1_dims.end(), size_t(1),
                      std::multiplies<size_t>());
  const size_t num_output_elements =
      std::max(num_input0_elements, num_input1_elements);
  std::vector<float> input0(num_input0_elements);
  std::vector<float> input1(num_input1_elements);
  std::vector<float> output(num_output_elements);

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(-1.f, 1.f);
  std::generate(input0.begin(), input0.end(), [&]() { return f32dist(rng); });
  std::generate(input1.begin(), input1.end(), [&]() { return f32dist(rng); });

  std::array<xnn_external_value, 3> external = {
      xnn_external_value{0, input0.data()},
      xnn_external_value{1, input1.data()},
      xnn_external_value{2, output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime_v2(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
  for (int i = 0; i < num_output_elements; ++i) {
    ASSERT_EQ(output[i], input0[i] + input1[0]);
  }
}

TEST(ReshapeHelpersTest, BinaryBroadcasting) {
  std::vector<size_t> input0_dims{3, 4, 5};
  std::vector<size_t> input1_dims{1, 5};
  xnn_runtime_t runtime = SetupBinary(input0_dims, input1_dims);
  ASSERT_NE(runtime, nullptr);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  ASSERT_EQ(xnn_status_success, xnn_reshape_runtime(runtime));

  size_t num_output_dims;
  std::vector<size_t> output_dims(input0_dims.size());
  ASSERT_EQ(xnn_status_success,
            xnn_get_external_value_shape(runtime, /*external_id=*/2,
                                         &num_output_dims, &output_dims[0]));
  ASSERT_EQ(num_output_dims, input0_dims.size());
  ASSERT_EQ(output_dims, input0_dims);
}
