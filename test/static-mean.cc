// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>  // For std::generate, std::min.
#include <array>      // For std::array.
#include <cmath>      // For std::lrintf.
#include <cstddef>    // For size_t.
#include <cstdint>    // For uint32_t.
#include <functional> // For std::multiplies.
#include <memory>     // For std::unique_ptr.
#include <numeric>    // For std::accumulate.
#include <random>     // For std::random_device, std::mt19937, std::uniform_real_distribution.
#include <vector>     // For std::vector.

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/node-type.h>
#include <xnnpack/operator.h>
#include <xnnpack/subgraph.h>

#include <gtest/gtest.h>


namespace xnnpack {
template <class T> class MeanTestBase : public ::testing::Test {
protected:
  MeanTestBase()
  {
    random_device = std::make_unique<std::random_device>();
    rng = std::mt19937((*random_device)());
    f32dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);

    auto num_input_dim_dist = std::uniform_int_distribution<size_t>(2, XNN_MAX_TENSOR_DIMS);
    const size_t num_input_dims = num_input_dim_dist(rng);

    auto reduction_axes_seq_start_dist = std::uniform_int_distribution<size_t>(0, num_input_dims - 1);
    const size_t reduction_axes_seq_start = reduction_axes_seq_start_dist(rng);
    auto reduction_axes_seq_end_dist = std::uniform_int_distribution<size_t>(reduction_axes_seq_start + 1, num_input_dims);
    const size_t reduction_axes_seq_end = reduction_axes_seq_end_dist(rng);

    reduction_axes.clear();
    for (size_t axis = reduction_axes_seq_start; axis < reduction_axes_seq_end; axis++) {
      reduction_axes.push_back(axis);
    }

    auto shape_dist = std::uniform_int_distribution<size_t>(2, 15);
    input_shape.resize(num_input_dims);
    std::generate(input_shape.begin(), input_shape.end(), [&]() { return shape_dist(rng); });
    num_input_elements = std::accumulate(input_shape.cbegin(), input_shape.cend(), size_t(1), std::multiplies<size_t>());

    output_shape = input_shape;
    for (size_t axis : reduction_axes) {
      output_shape[axis] = 1;
    }
    num_output_elements = std::accumulate(output_shape.cbegin(), output_shape.cend(), size_t(1), std::multiplies<size_t>());

    input = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + num_input_elements);
    operator_output = std::vector<T>(num_output_elements);
    subgraph_output = std::vector<T>(num_output_elements);
  }

  std::unique_ptr<std::random_device> random_device;
  std::mt19937 rng;
  std::uniform_real_distribution<float> f32dist;

  std::vector<size_t> reduction_axes;
  std::vector<size_t> input_shape;
  size_t num_input_elements;
  std::vector<size_t> output_shape;
  size_t num_output_elements;

  std::vector<T> input;
  std::vector<T> operator_output;
  std::vector<T> subgraph_output;
};

using MeanTestF32 = MeanTestBase<float>;

TEST_F(MeanTestF32, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
    xnn_define_tensor_value(subgraph, xnn_datatype_fp32, input_shape.size(), input_shape.data(),
                            nullptr, /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
    xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_shape.size(), output_shape.data(),
                            nullptr, /*external_id=*/1, /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success,
    xnn_define_static_mean(
      subgraph,
      reduction_axes.size(), reduction_axes.data(),
      input_id, output_id,
      /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_static_mean);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->params.reduce.num_reduction_axes, reduction_axes.size());
  for (size_t i = 0; i < reduction_axes.size(); i++) {
    ASSERT_EQ(node->params.reduce.reduction_axes[i], reduction_axes[i]);
  }
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(MeanTestF32, matches_operator_api)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  // Call operator API.
  const xnn_status status = xnn_create_mean_nd_f32(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);

  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  ASSERT_EQ(xnn_status_success,
    xnn_reshape_mean_nd_f32(op,
      reduction_axes.size(), reduction_axes.data(),
      input_shape.size(), input_shape.data(),
      &workspace_size, &workspace_alignment,
      /*threadpool=*/nullptr));
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);

  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
  ASSERT_EQ(xnn_status_success, xnn_setup_mean_nd_f32(op, workspace.data(), input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
    xnn_define_tensor_value(subgraph, xnn_datatype_fp32, input_shape.size(), input_shape.data(),
                            nullptr, /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
    xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_shape.size(), output_shape.data(),
                            nullptr, /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success,
    xnn_define_static_mean(
      subgraph,
      reduction_axes.size(), reduction_axes.data(),
      input_id, output_id,
      /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);

  const std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data()},
    xnn_external_value{output_id, subgraph_output.data()}
  };
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Check outputs match.
  for (size_t i = 0; i < operator_output.size(); i++) {
    ASSERT_EQ(subgraph_output[i], operator_output[i]);
  }
}

}  // namespace xnnpack
