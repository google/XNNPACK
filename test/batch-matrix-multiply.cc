// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>  // For std::generate.
#include <array>      // For std::array.
#include <cassert>
#include <cmath>
#include <cstddef>  // For size_t.
#include <cstdint>  // For uint32_t.
#include <functional>
#include <iterator>
#include <limits>
#include <memory>   // For std::unique_ptr.
#include <numeric>  // For std::accumulate.
#include <ostream>
#include <random>  // For std::uniform_real_distribution.
#include <string>
#include <vector>  // For std::vector.

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"

template <class InputType, class OutputType>
class BatchMatrixMultiplyTestBase : public ::testing::Test {
 protected:
  BatchMatrixMultiplyTestBase() {
    f32dist = std::uniform_real_distribution<float>(0.1f, 1.0f);
    i8dist = std::uniform_int_distribution<int32_t>(
        std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    w8dist = std::uniform_int_distribution<int32_t>(
        -std::numeric_limits<uint8_t>::max(),
        std::numeric_limits<uint8_t>::max());
    auto shape_dist =
        std::uniform_int_distribution<size_t>(4, XNN_MAX_TENSOR_DIMS);
    dim_dist = std::uniform_int_distribution<size_t>(5, 15);

    // input1: B x G x M x K
    // input2: B x H x K x N or B x H x N x K (transposed)
    // output: B x G x M x N
    // where G is an integer multiple of H.
    size_t num_input_dims = shape_dist(rng);
    input1_dims = RandomShape(num_input_dims);
    assert(input1_dims.size() >= 3);
    m = input1_dims[num_input_dims - 2];

    k = input1_dims.back();
    n = dim_dist(rng);
    input2_dims = input1_dims;
    input2_dims[num_input_dims - 2] = k;
    input2_dims[num_input_dims - 1] = n;
    input2_t_dims = input1_dims;
    input2_t_dims[num_input_dims - 2] = n;

    output_dims = input1_dims;
    output_dims[num_input_dims - 2] = m;
    output_dims[num_input_dims - 1] = n;

    input1 = std::vector<InputType>(XNN_EXTRA_BYTES / sizeof(InputType) +
                                    NumElements(input1_dims));
    input2 = std::vector<InputType>(XNN_EXTRA_BYTES / sizeof(InputType) +
                                    NumElements(input2_dims));
    operator_output = std::vector<OutputType>(NumElements(output_dims));
    subgraph_output = std::vector<OutputType>(operator_output.size());
  }

  std::vector<size_t> RandomShape(size_t num_dims)
  {
    std::vector<size_t> dims(num_dims);
    std::generate(dims.begin(), dims.end(), [&] { return dim_dist(rng); });
    return dims;
  }

  size_t NumElements(std::vector<size_t>& dims)
  {
    return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
  }

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_int_distribution<int32_t> i8dist;
  std::uniform_int_distribution<int32_t> w8dist;
  std::uniform_int_distribution<size_t> dim_dist;

  size_t m;
  size_t k;
  size_t n;

  std::vector<size_t> input1_dims;
  std::vector<size_t> input2_dims;
  std::vector<size_t> input2_t_dims;  // input2 transposed.
  std::vector<size_t> output_dims;

  std::vector<InputType> input1;
  std::vector<InputType> input2;
  std::vector<OutputType> operator_output;
  std::vector<OutputType> subgraph_output;
};

using BatchMatrixMultiplyTestF16 =
    BatchMatrixMultiplyTestBase<uint16_t, uint16_t>;
using BatchMatrixMultiplyTestF32 = BatchMatrixMultiplyTestBase<float, float>;
using BatchMatrixMultiplyTestQD8ToF32 =
    BatchMatrixMultiplyTestBase<int8_t, float>;

TEST_F(BatchMatrixMultiplyTestF16, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input1_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input1_dims.size(), input1_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_VALUE_ID);

  uint32_t input2_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp16, input2_dims.size(), input2_dims.data(), input2.data(), /*external_id=*/1,
      /*flags=*/0, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/2, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_batch_matrix_multiply(subgraph, input1_id, input2_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_batch_matrix_multiply);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp16);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input1_id);
  ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(BatchMatrixMultiplyTestF32, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input1_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input1_dims.size(), input1_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_VALUE_ID);

  uint32_t input2_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, input2_dims.size(), input2_dims.data(), input2.data(), /*external_id=*/1,
      /*flags=*/0, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/2, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_batch_matrix_multiply(subgraph, input1_id, input2_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_batch_matrix_multiply);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input1_id);
  ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(BatchMatrixMultiplyTestF32, define_transposed)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input1_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input1_dims.size(), input1_dims.data(), nullptr,
                          /*external_id=*/0, /*flags=*/0, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_VALUE_ID);

  uint32_t input2_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, input2_t_dims.size(), input2_t_dims.data(), input2.data(), /*external_id=*/1,
      /*flags=*/0, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/2, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(
    xnn_status_success,
    xnn_define_batch_matrix_multiply(subgraph, input1_id, input2_id, output_id, /*flags=*/XNN_FLAG_TRANSPOSE_B));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_batch_matrix_multiply);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input1_id);
  ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, XNN_FLAG_TRANSPOSE_B);
}

TEST_F(BatchMatrixMultiplyTestF16, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input1.begin(), input1.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::generate(input2.begin(), input2.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::fill(operator_output.begin(), operator_output.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT16_C(0x7E00) /* NaN */);

  // Call operator API.
  const xnn_status status = xnn_create_batch_matrix_multiply_nc_f16(/*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_batch_matrix_multiply_nc_f16(
          op, /*num_batch_dims=*/input1_dims.size() - 2,
          /*batch_dims_a=*/input1_dims.data(),
          /*batch_dims_b=*/input2_dims.data(), m, k, n,
          &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
  ASSERT_NE(workspace_size, 0);
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(
      workspace_size + XNN_EXTRA_BYTES);
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_batch_matrix_multiply_nc_f16(op, workspace.data(), input1.data(), input2.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input1_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input1_dims.size(), input1_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_VALUE_ID);

  uint32_t input2_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, input2_dims.size(), input2_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp16, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_batch_matrix_multiply(subgraph, input1_id, input2_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 3> external = {
    xnn_external_value{input1_id, input1.data()},
    xnn_external_value{input2_id, input2.data()},
    xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i])
        << " at index " << i << " of " << operator_output.size();
  }
}

TEST_F(BatchMatrixMultiplyTestF32, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input1.begin(), input1.end(), [&]() { return f32dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  // Call operator API.
  const xnn_status status = xnn_create_batch_matrix_multiply_nc_f32(/*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_batch_matrix_multiply_nc_f32(
          op, /*num_batch_dims=*/input1_dims.size() - 2,
          /*batch_dims_a=*/input1_dims.data(),
          /*batch_dims_b=*/input2_dims.data(), m, k, n,
          &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
  ASSERT_NE(workspace_size, 0);
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(
      workspace_size + XNN_EXTRA_BYTES);
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_batch_matrix_multiply_nc_f32(op, workspace.data(), input1.data(), input2.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input1_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input1_dims.size(), input1_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_VALUE_ID);

  uint32_t input2_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input2_dims.size(), input2_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_batch_matrix_multiply(subgraph, input1_id, input2_id, output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 3> external = {
    xnn_external_value{input1_id, input1.data()},
    xnn_external_value{input2_id, input2.data()},
    xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i])
        << " at index " << i << " of " << operator_output.size();
  }
}

TEST_F(BatchMatrixMultiplyTestF32, matches_operator_api_transposed)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input1.begin(), input1.end(), [&]() { return f32dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  // Call operator API.
  const xnn_status status = xnn_create_batch_matrix_multiply_nc_f32(/*flags=*/XNN_FLAG_TRANSPOSE_B, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_batch_matrix_multiply_nc_f32(
          op, /*num_batch_dims=*/input1_dims.size() - 2,
          /*batch_dims_a=*/input1_dims.data(),
          /*batch_dims_b=*/input2_dims.data(), m, k, n,
          &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
  ASSERT_NE(workspace_size, 0);
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(
      workspace_size + XNN_EXTRA_BYTES);
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_batch_matrix_multiply_nc_f32(op, workspace.data(), input1.data(), input2.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input1_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input1_dims.size(), input1_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_VALUE_ID);

  uint32_t input2_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input2_t_dims.size(), input2_t_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);
  ASSERT_EQ(
    xnn_status_success,
    xnn_define_batch_matrix_multiply(subgraph, input1_id, input2_id, output_id, /*flags=*/XNN_FLAG_TRANSPOSE_B));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 3> external = {
    xnn_external_value{input1_id, input1.data()},
    xnn_external_value{input2_id, input2.data()},
    xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i])
        << " at index " << i << " of " << operator_output.size();
  }
}

TEST_F(BatchMatrixMultiplyTestQD8ToF32, define) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input1_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, input1_dims.size(),
                /*num_nonbatch_dims=*/2, input1_dims.data(),
                /*external_id=*/0, /*flags=*/0, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_VALUE_ID);

  uint32_t input2_id = XNN_INVALID_VALUE_ID;
  std::vector<float> channelwise_scale(
      NumElements(input2_dims) / input2_dims.back(),
      1.0f / std::numeric_limits<int8_t>::max());
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, channelwise_scale.data(),
                input2_dims.size(), /*channel_dim=*/input2_dims.size() - 1,
                input2_dims.data(), input2.data(), /*external_id=*/1,
                /*flags=*/0, &input2_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_dims.size(),
                              output_dims.data(), nullptr,
                              /*external_id=*/3, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_batch_matrix_multiply(subgraph, input1_id, input2_id,
                                             output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_batch_matrix_multiply);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qd8_to_fp32);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input1_id);
  ASSERT_EQ(node->inputs[1], input2_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(BatchMatrixMultiplyTestQD8ToF32, matches_operator_api) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  std::vector<float> channelwise_scale(
      NumElements(input2_dims) / input2_dims[input2_dims.size() - 2],
      1.0f / std::numeric_limits<int8_t>::max());
  std::generate(input2.begin(), input2.end(), [&]() { return i8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  // Create the dynamically quantized input data with the corresponding
  // `quantization_params`.
  const size_t input_batch_size = NumElements(input1_dims) / m / k;
  std::vector<xnn_dynamic_quantization_params> quantization_params(
      input_batch_size * m + XNN_EXTRA_QUANTIZATION_PARAMS);
  std::vector<float> input1_f32(NumElements(input1_dims) +
                                XNN_EXTRA_BYTES / sizeof(float));
  std::generate(input1_f32.begin(), input1_f32.end(),
                [&]() { return f32dist(rng); });

  xnn_operator_t convert_op = nullptr;
  xnn_status status = xnn_create_convert_nc_f32_qd8(
      /*flags=*/0, &convert_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(
      convert_op, xnn_delete_operator);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, convert_op);
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_convert_nc_f32_qd8(convert_op, input_batch_size * m, k,
                                           k, k, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success, xnn_setup_convert_nc_f32_qd8(
                                    convert_op, input1_f32.data(),
                                    input1.data(), quantization_params.data()));
  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(convert_op, /*threadpool=*/nullptr));

  // Create a BatchMatrixMultiply with the operator API.
  xnn_operator_t batch_matrix_multiply_op = nullptr;
  size_t batch_size_b = 1;
  for (size_t i = 0; i < input2_dims.size() - 2; i++) {
    batch_size_b *= input2_dims[i];
  }
  status = xnn_create_batch_matrix_multiply_nc_qd8_f32_qc8w(
      batch_size_b, /*k=*/input2_dims[input2_dims.size() - 2],
      /*n=*/input2_dims[input2_dims.size() - 1], input2.data(),
      channelwise_scale.data(),
      /*flags=*/0, &batch_matrix_multiply_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      batch_matrix_multiply_op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, batch_matrix_multiply_op);

  ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_batch_matrix_multiply_nc_qd8_f32_qc8w(
          batch_matrix_multiply_op, /*num_batch_dims=*/input1_dims.size() - 2,
          /*batch_dims_a=*/input1_dims.data(),
          /*batch_dims_b=*/input2_dims.data(), m, k, n,
          /*threadpool=*/nullptr));

  ASSERT_EQ(xnn_status_success,
            xnn_setup_batch_matrix_multiply_nc_qd8_f32_qc8w(
                batch_matrix_multiply_op, input1.data(),
                quantization_params.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success,
            xnn_run_operator(batch_matrix_multiply_op, /*threadpool=*/nullptr));

  // Create a BatchMatrixMultiply with the subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input1_f32_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, input1_dims.size(),
                input1_dims.data(), nullptr, /*external_id=*/0,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_f32_id));
  ASSERT_NE(input1_f32_id, XNN_INVALID_NODE_ID);

  uint32_t input1_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_dynamically_quantized_tensor_value(
                subgraph, xnn_datatype_qdint8, input1_dims.size(),
                /*num_nonbatch_dims=*/1, input1_dims.data(),
                /*external_id=*/XNN_INVALID_VALUE_ID, /*flags=*/0, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_VALUE_ID);

  uint32_t input2_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_channelwise_quantized_tensor_value(
                subgraph, xnn_datatype_qcint8, channelwise_scale.data(),
                input2_dims.size(), /*channel_dim=*/input2_dims.size() - 1,
                input2_dims.data(), input2.data(), /*external_id=*/1,
                /*flags=*/0, &input2_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  // Define the ops.
  ASSERT_EQ(xnn_status_success, xnn_define_convert(subgraph, input1_f32_id,
                                                   input1_id, /*flags=*/0));
  ASSERT_EQ(xnn_status_success,
            xnn_define_batch_matrix_multiply(subgraph, input1_id, input2_id,
                                             output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(
      xnn_status_success,
      xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input1_f32_id, input1_f32.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i])
        << " at index " << i << " of " << operator_output.size();
  }
}

namespace {

// Define a subgraph with a single batch matrix multiply node with 2 inputs and
// 1 output of the specified dimensions.
// Returns the result of defining the node, a xnn_status.
void DefineBatchMatrixMultiplySubgraphHelper(
    xnn_status* status_out, std::vector<size_t> input1_dims,
    std::vector<size_t> input2_dims, std::vector<size_t> output_dims,
    xnn_subgraph_t* subgraph_out, uint32_t batch_matrix_multiply_flags = 0) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(3, /*flags=*/0, &subgraph));

  uint32_t input1_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input1_dims.size(), input1_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id));
  ASSERT_NE(input1_id, XNN_INVALID_VALUE_ID);

  uint32_t input2_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input2_dims.size(), input2_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id));
  ASSERT_NE(input2_id, XNN_INVALID_VALUE_ID);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_VALUE_ID);

  *status_out = xnn_define_batch_matrix_multiply(subgraph, input1_id, input2_id, output_id, batch_matrix_multiply_flags);
  *subgraph_out = subgraph;
}

void DefineAndReshapeBatchMatrixMultiplySubgraph(
    xnn_status* status_out, std::vector<size_t> input1_dims,
    std::vector<size_t> input2_dims, std::vector<size_t> expected_output_dims,
    uint32_t batch_matrix_multiply_flags = 0) {
  xnn_subgraph_t subgraph = nullptr;
  DefineBatchMatrixMultiplySubgraphHelper(status_out, input1_dims, input2_dims,
                                          expected_output_dims, &subgraph,
                                          batch_matrix_multiply_flags);
  if (*status_out != xnn_status_success) {
    return;
  }

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(
      xnn_status_success,
      xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)>
      clean_up_subgraph(subgraph, xnn_delete_subgraph);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> clean_up_runtime(
      runtime, xnn_delete_runtime);

  *status_out = xnn_reshape_runtime(runtime);

  // Check whether the output shape is as expected.
  const xnn_shape* output_shape =
      &runtime->values[subgraph->nodes[0].outputs[0]].shape;
  std::vector<size_t> output_dims_vector(
      output_shape->dim, output_shape->dim + output_shape->num_dims);
  EXPECT_EQ(output_dims_vector.size(), expected_output_dims.size());
  for (size_t i = 0; i < output_dims_vector.size(); i++) {
    EXPECT_EQ(output_dims_vector[i], expected_output_dims[i]);
  }
}

}  // namespace

TEST(BatchMatrixMultiplyReshapeTest, reshape_input1) {
  std::vector<size_t> input1_dims = {2, 3, 4};
  std::vector<size_t> input2_dims = {2, 4, 5};
  std::vector<size_t> output_dims = {2, 3, 5};
  xnn_status status = xnn_status_success;
  xnn_subgraph_t subgraph;
  DefineBatchMatrixMultiplySubgraphHelper(&status, input1_dims, input2_dims, output_dims, &subgraph);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);
  ASSERT_EQ(xnn_status_success, status);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  ASSERT_EQ(subgraph->num_nodes, 1);
  struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);

  input1_dims[2] = 7;
  input2_dims[1] = 7;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, /*external_id=*/0, input1_dims.size(), input1_dims.data()));
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, /*external_id=*/1, input2_dims.size(), input2_dims.data()));

  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  ASSERT_EQ(output_shape->dim[0], input1_dims[0]);
  ASSERT_EQ(output_shape->dim[1], input1_dims[1]);
  ASSERT_EQ(output_shape->dim[2], input2_dims[2]);

  input1_dims[1] = 19;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, /*external_id=*/0, input1_dims.size(), input1_dims.data()));
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, /*external_id=*/1, input2_dims.size(), input2_dims.data()));

  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);
  ASSERT_EQ(output_shape->dim[0], input1_dims[0]);
  ASSERT_EQ(output_shape->dim[1], input1_dims[1]);
  ASSERT_EQ(output_shape->dim[2], input2_dims[2]);

  input2_dims[2] = 4;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, /*external_id=*/0, input1_dims.size(), input1_dims.data()));
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, /*external_id=*/1, input2_dims.size(), input2_dims.data()));

  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_success);
  ASSERT_EQ(output_shape->dim[0], input1_dims[0]);
  ASSERT_EQ(output_shape->dim[1], input1_dims[1]);
  ASSERT_EQ(output_shape->dim[2], input2_dims[2]);

  input1_dims[0] = 4;
  input2_dims[0] = 4;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, /*external_id=*/0, input1_dims.size(), input1_dims.data()));
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, /*external_id=*/1, input2_dims.size(), input2_dims.data()));

  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_reallocation_required);
  ASSERT_EQ(output_shape->dim[0], input1_dims[0]);
  ASSERT_EQ(output_shape->dim[1], input1_dims[1]);
  ASSERT_EQ(output_shape->dim[2], input2_dims[2]);
}

struct BatchMatrixMultiplyTestParams {
  std::string name;
  std::vector<size_t> input_a_dims;
  std::vector<size_t> input_b_dims;
  std::vector<size_t> expected_output_dims;
  uint32_t flags = 0;
  enum xnn_status expected_status = xnn_status_success;
};

template <typename T>
std::ostream& PrintVector(std::ostream& os, const std::vector<T>& v) {
  os << "[";
  std::copy(v.begin(), v.end() - 1, std::ostream_iterator<T>(os, ", "));
  os << v.back() << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const BatchMatrixMultiplyTestParams& params) {
  os << "{input_a_dims=";
  PrintVector(os, params.input_a_dims);
  os << ", input_b_dims=";
  PrintVector(os, params.input_b_dims);
  os << ", expected_output_dims=";
  PrintVector(os, params.input_a_dims);
  os << ", flags=" << params.flags
     << ", expected_status=" << params.expected_status << "}";
  return os;
}

using BatchMatrixMultiplyTest =
    testing::TestWithParam<BatchMatrixMultiplyTestParams>;

TEST_P(BatchMatrixMultiplyTest, DefineAndReshape) {
  const BatchMatrixMultiplyTestParams& params = GetParam();
  xnn_status status = xnn_status_success;
  DefineAndReshapeBatchMatrixMultiplySubgraph(
      &status, params.input_a_dims, params.input_b_dims,
      params.expected_output_dims, params.flags);
  ASSERT_EQ(params.expected_status, status);
}

INSTANTIATE_TEST_SUITE_P(
    BMM, BatchMatrixMultiplyTest,
    testing::ValuesIn<BatchMatrixMultiplyTestParams>({
        {/*.name = */"input_a_num_dim_less_than_3",
         /*.input_a_dims = */{3, 7},
         /*.input_b_dims = */{2, 7, 5},
         /*.expected_output_dims = */{2, 3, 5}},

        {/*.name = */"input_b_num_dim_less_than_3",
         /*.input_a_dims = */{2, 3, 5},
         /*.input_b_dims = */{5, 7},
         /*.expected_output_dims = */{2, 3, 7}},

        {/*.name = */"output_num_dim_less_than_3_fails",
         /*.input_a_dims = */{2, 3, 5},
         /*.input_b_dims = */{2, 7, 5},
         /*.expected_output_dims = */{2, 3},
         /*.flags = */0,
         /*.expected_status = */xnn_status_invalid_parameter},

        {/*.name = */"input_a_num_dim_ne_input2_num_dim",
         /*.input_a_dims = */{2, 3, 5},
         /*.input_b_dims = */{2, 7, 5, 5},
         /*.expected_output_dims = */{2, 3, 7},
         /*.flags = */0,
         /*.expected_status = */xnn_status_invalid_parameter},

        {/*.name = */"input_a_k_dim_ne_input2_dim",
         /*.input_a_dims = */{2, 3, 7},
         /*.input_b_dims = */{2, 5, 7},
         /*.expected_output_dims = */{2, 3, 7},
         /*.flags = */0,
         /*.expected_status = */xnn_status_invalid_parameter},

        {/*.name = */"input_a_k_dim_ne_transposed_input2_dim",
         /*.input_a_dims = */{2, 3, 7},
         /*.input_b_dims = */{2, 7, 5},
         /*.expected_output_dims = */{2, 3, 7},
         /*.flags = */XNN_FLAG_TRANSPOSE_B,
         /*.expected_status = */xnn_status_invalid_parameter},

        {/*.name = */"output_num_dim_ne_input1_num_dim",
         /*.input_a_dims = */{2, 3, 5},
         /*.input_b_dims = */{2, 7, 5},
         /*.expected_output_dims = */{2, 3, 7, 5},
         /*.flags = */0,
         /*.expected_status = */xnn_status_invalid_parameter},

        {/*.name = */"output_m_ne_input_m",
         /*.input_a_dims = */{2, 3, 5},
         /*.input_b_dims = */{2, 7, 5},
         /*.expected_output_dims = */{2, 5, 7},
         /*.flags = */0,
         /*.expected_status = */xnn_status_invalid_parameter},

        {/*.name = */"output_shape",
         /*.input_a_dims = */{2, 3, 5},
         /*.input_b_dims = */{2, 5, 7},
         /*.expected_output_dims = */{2, 3, 7}},

        {/*.name = */"output_shape_transposed",
         /*.input_a_dims = */{2, 3, 5},
         /*.input_b_dims = */{2, 7, 5},
         /*.expected_output_dims = */{2, 3, 7},
         /*.flags = */XNN_FLAG_TRANSPOSE_B},

        // Test broadcasting in the batch dimensions of the first input.
        {/*.name = */"input_a_batch_dim_ne_input2_dim",
         /*.input_a_dims = */{3, 3, 5},
         /*.input_b_dims = */{2, 7, 5},
         /*.expected_output_dims = */{2, 3, 7},
         /*.flags = */0,
         /*.expected_status = */xnn_status_invalid_parameter},

        {/*.name = */"input_a_batch_dim_bcast_one",
         /*.input_a_dims = */{1, 3, 5},
         /*.input_b_dims = */{2, 5, 7},
         /*.expected_output_dims = */{2, 3, 7}},

        {/*.name = */"input_a_batch_dim_bcast_mult",
         /*.input_a_dims = */{2, 3, 5},
         /*.input_b_dims = */{6, 5, 7},
         /*.expected_output_dims = */{6, 3, 7},
         /*.flags = */0,
         /*.expected_status = */xnn_status_invalid_parameter},

        {/*.name = */"input_b_batch_dim_bcast_one",
         /*.input_a_dims = */{2, 3, 5},
         /*.input_b_dims = */{1, 5, 7},
         /*.expected_output_dims = */{2, 3, 7}},

        {/*.name = */"input_b_batch_dim_bcast_mult",
         /*.input_a_dims = */{6, 3, 5},
         /*.input_b_dims = */{2, 5, 7},
         /*.expected_output_dims = */{6, 3, 7},
         /*.flags = */0,
         /*.expected_status = */xnn_status_invalid_parameter},

        {/*.name = */"both_inputs_batch_dim_bcast_mult",
         /*.input_a_dims = */{2, 6, 3, 5},
         /*.input_b_dims = */{4, 2, 5, 7},
         /*.expected_output_dims = */{4, 6, 3, 7},
         /*.flags = */0,
         /*.expected_status = */xnn_status_invalid_parameter},

        {/*.name = */"both_inputs_batch_dim_bcast_one",
         /*.input_a_dims = */{1, 6, 3, 5},
         /*.input_b_dims = */{4, 1, 5, 7},
         /*.expected_output_dims = */{4, 6, 3, 7}},

        {/*.name = */"input_a_missing_batch_dim",
         /*.input_a_dims = */{6, 3, 5},
         /*.input_b_dims = */{4, 1, 5, 7},
         /*.expected_output_dims = */{4, 6, 3, 7}},

        {/*.name = */"input_b_missing_batch_dim",
         /*.input_a_dims = */{4, 1, 3, 5},
         /*.input_b_dims = */{6, 5, 7},
         /*.expected_output_dims = */{4, 6, 3, 7}},
    }),
    [](const testing::TestParamInfo<BatchMatrixMultiplyTest::ParamType>& info) {
      return info.param.name;
    });
