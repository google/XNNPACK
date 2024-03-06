// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>  // For std::generate.
#include <array>      // For std::array.
#include <cstddef>    // For size_t.
#include <cstdint>    // For uint32_t.
#include <memory>     // For std::unique_ptr.
#include <numeric>    // For std::accumulate.
#include <random>     // For std::random_device, std::mt19937, std::uniform_real_distribution.
#include <vector>     // For std::vector.

#include <fp16/fp16.h>
#include <gtest/gtest.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/operator.h>
#include <xnnpack/subgraph.h>

template <class T, class BiasType = T> class BatchMatrixMultiplyTestBase : public ::testing::Test {
protected:
  BatchMatrixMultiplyTestBase()
  {
    random_device = std::make_unique<std::random_device>();
    rng = std::mt19937((*random_device)());
    f32dist = std::uniform_real_distribution<float>(0.1f, 1.0f);
    auto shape_dist = std::uniform_int_distribution<size_t>(3, XNN_MAX_TENSOR_DIMS);
    dim_dist = std::uniform_int_distribution<size_t>(5, 15);

    // input1: B x M x K
    // input2: B x K x N or B x N x K (transposed)
    // output: B x M x N
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

    batch_size = NumElements(input1_dims) / k / m;

    input1 = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(input1_dims));
    input2 = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(input2_dims));
    operator_output = std::vector<T>(NumElements(output_dims));
    subgraph_output = std::vector<T>(operator_output.size());
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

  std::unique_ptr<std::random_device> random_device;
  std::mt19937 rng;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_int_distribution<size_t> dim_dist;

  uint32_t batch_size;
  size_t m;
  size_t k;
  size_t n;
  size_t output_channels;

  std::vector<size_t> input1_dims;
  std::vector<size_t> input2_dims;
  std::vector<size_t> input2_t_dims;  // input2 transposed.
  std::vector<size_t> output_dims;

  std::vector<T> input1;
  std::vector<T> input2;
  std::vector<T> operator_output;
  std::vector<T> subgraph_output;
};

using BatchMatrixMultiplyTestF16 = BatchMatrixMultiplyTestBase<uint16_t>;
using BatchMatrixMultiplyTestF32 = BatchMatrixMultiplyTestBase<float>;

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
  std::fill(operator_output.begin(), operator_output.end(), fp16_ieee_from_fp32_value(nanf("")));
  std::fill(subgraph_output.begin(), subgraph_output.end(), fp16_ieee_from_fp32_value(nanf("")));

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
    xnn_status_success, xnn_reshape_batch_matrix_multiply_nc_f16(
                          op, batch_size, m, k, n,
                          &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
  ASSERT_NE(workspace_size, 0);
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
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
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
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
    xnn_status_success, xnn_reshape_batch_matrix_multiply_nc_f32(
                          op, batch_size, m, k, n,
                          &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
  ASSERT_NE(workspace_size, 0);
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
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
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
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
    xnn_status_success, xnn_reshape_batch_matrix_multiply_nc_f32(
                          op, batch_size, m, k, n,
                          &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
  ASSERT_NE(workspace_size, 0);
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
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
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
  }
}

// Define a subgraph with a single batch matrix multiply node with 2 inputs and 1 output of the specified dimensions.
// Returns the result of defining the node, a xnn_status.
namespace {
void DefineBatchMatrixMultiplySubgraphHelper(
    xnn_status* status_out,
    std::vector<size_t> input1_dims,
    std::vector<size_t> input2_dims,
    std::vector<size_t> output_dims,
    xnn_subgraph_t *subgraph_out,
    uint32_t batch_matrix_multiply_flags = 0) {
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

void DefineBatchMatrixMultiplySubgraph(
    xnn_status* status_out,
    std::vector<size_t> input1_dims,
    std::vector<size_t> input2_dims,
    std::vector<size_t> output_dims,
    uint32_t batch_matrix_multiply_flags = 0) {
  xnn_subgraph_t subgraph = nullptr;
  DefineBatchMatrixMultiplySubgraphHelper(status_out, input1_dims, input2_dims, output_dims,
                                     &subgraph, batch_matrix_multiply_flags);
  xnn_delete_subgraph(subgraph);
}
}  // namespace

TEST(BatchMatrixMultiplyTest, reshape_input1) {
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

TEST(BatchMatrixMultiplyTest, input1_num_dim_less_than_3_fails) {
  std::vector<size_t> input1_dims = {2, 3};
  std::vector<size_t> input2_dims = {2, 7, 5};
  std::vector<size_t> output_dims = {2, 3, 7};
  xnn_status status = xnn_status_success;
  DefineBatchMatrixMultiplySubgraph(&status, input1_dims, input2_dims, output_dims);
  ASSERT_EQ(xnn_status_invalid_parameter, status);
}

TEST(BatchMatrixMultiplyTest, input2_num_dim_less_than_3_fails) {
  std::vector<size_t> input1_dims = {2, 3, 5};
  std::vector<size_t> input2_dims = {2, 7};
  std::vector<size_t> output_dims = {2, 3, 7};
  xnn_status status = xnn_status_success;
  DefineBatchMatrixMultiplySubgraph(&status, input1_dims, input2_dims, output_dims);
  ASSERT_EQ(xnn_status_invalid_parameter, status);
}

TEST(BatchMatrixMultiplyTest, output_num_dim_less_than_3_fails)
{
  std::vector<size_t> input1_dims = {2, 3, 5};
  std::vector<size_t> input2_dims = {2, 7, 5};
  std::vector<size_t> output_dims = {2, 3};
  xnn_status status = xnn_status_success;
  DefineBatchMatrixMultiplySubgraph(&status, input1_dims, input2_dims, output_dims);
  ASSERT_EQ(xnn_status_invalid_parameter, status);
}

TEST(BatchMatrixMultiplyTest, input1_num_dim_ne_input2_num_dim) {
  std::vector<size_t> input1_dims = {2, 3, 5};
  std::vector<size_t> input2_dims = {2, 7, 5, 5};
  std::vector<size_t> output_dims = {2, 3, 7};
  xnn_status status = xnn_status_success;
  DefineBatchMatrixMultiplySubgraph(&status, input1_dims, input2_dims, output_dims);
  ASSERT_EQ(xnn_status_invalid_parameter, status);
}

TEST(BatchMatrixMultiplyTest, input1_batch_dim_ne_input2_dim) {
  std::vector<size_t> input1_dims = {3, 3, 5};
  std::vector<size_t> input2_dims = {2, 7, 5};
  std::vector<size_t> output_dims = {2, 3, 7};
  xnn_status status = xnn_status_success;
  DefineBatchMatrixMultiplySubgraph(&status, input1_dims, input2_dims, output_dims);
  ASSERT_EQ(xnn_status_invalid_parameter, status);
}

TEST(BatchMatrixMultiplyTest, input1_k_dim_ne_input2_dim) {
  std::vector<size_t> input1_dims = {2, 3, 7};
  std::vector<size_t> input2_dims = {2, 5, 7};
  std::vector<size_t> output_dims = {2, 3, 7};
  xnn_status status = xnn_status_success;
  DefineBatchMatrixMultiplySubgraph(&status, input1_dims, input2_dims, output_dims);
  ASSERT_EQ(xnn_status_invalid_parameter, status);
}

TEST(BatchMatrixMultiplyTest, input1_k_dim_ne_transposed_input2_dim) {
  std::vector<size_t> input1_dims = {2, 3, 7};
  std::vector<size_t> input2_dims = {2, 7, 5};
  std::vector<size_t> output_dims = {2, 3, 7};
  xnn_status status = xnn_status_success;
  DefineBatchMatrixMultiplySubgraph(&status, input1_dims, input2_dims, output_dims, XNN_FLAG_TRANSPOSE_B);
  ASSERT_EQ(xnn_status_invalid_parameter, status);
}

TEST(BatchMatrixMultiplyTest, output_num_dim_ne_input1_num_dim) {
  std::vector<size_t> input1_dims = {2, 3, 5};
  std::vector<size_t> input2_dims = {2, 7, 5};
  std::vector<size_t> output_dims = {2, 3, 7, 5};
  xnn_status status = xnn_status_success;
  DefineBatchMatrixMultiplySubgraph(&status, input1_dims, input2_dims, output_dims);
  ASSERT_EQ(xnn_status_invalid_parameter, status);
}

TEST(BatchMatrixMultiplyTest, output_m_ne_input_m) {
  std::vector<size_t> input1_dims = {2, 3, 5};
  std::vector<size_t> input2_dims = {2, 7, 5};
  std::vector<size_t> output_dims = {2, 5, 7};
  xnn_status status = xnn_status_success;
  DefineBatchMatrixMultiplySubgraph(&status, input1_dims, input2_dims, output_dims);
  ASSERT_EQ(xnn_status_invalid_parameter, status);
}

TEST(BatchMatrixMultiplyTest, output_n_ne_input1_n) {
  std::vector<size_t> input1_dims = {2, 3, 5};
  std::vector<size_t> input2_dims = {2, 5, 7};
  std::vector<size_t> output_dims = {2, 3, 5};
  xnn_status status = xnn_status_success;
  DefineBatchMatrixMultiplySubgraph(&status, input1_dims, input2_dims, output_dims);
  ASSERT_EQ(xnn_status_invalid_parameter, status);
}

TEST(BatchMatrixMultiplyTest, output_n_ne_transposed_input2_n) {
  std::vector<size_t> input1_dims = {2, 3, 5};
  std::vector<size_t> input2_dims = {2, 7, 5};
  std::vector<size_t> output_dims = {2, 3, 5};
  xnn_status status = xnn_status_success;
  DefineBatchMatrixMultiplySubgraph(&status, input1_dims, input2_dims, output_dims, XNN_FLAG_TRANSPOSE_B);
  ASSERT_EQ(xnn_status_invalid_parameter, status);
}
