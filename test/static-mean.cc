// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>   // For std::generate, std::min.
#include <array>       // For std::array.
#include <cmath>       // For std::lrintf.
#include <cstddef>     // For size_t.
#include <cstdint>     // For uint32_t.
#include <functional>  // For std::multiplies.
#include <memory>      // For std::unique_ptr.
#include <numeric>     // For std::accumulate.
#include <random>      // For std::uniform_real_distribution.
#include <vector>      // For std::vector.

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"

namespace xnnpack {
template <class T>
class MeanTestBase : public ::testing::TestWithParam<bool> {
 protected:
  MeanTestBase() {
    f32dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);

    auto num_input_dim_dist = std::uniform_int_distribution<size_t>(2, XNN_MAX_TENSOR_DIMS);
    const size_t num_input_dims = num_input_dim_dist(rng);
    auto num_reduction_axes_dist = std::uniform_int_distribution<size_t>(1, num_input_dims);
    const size_t num_reduction_axes = num_reduction_axes_dist(rng);

    auto axes_dist = std::uniform_int_distribution<size_t>(0, num_input_dims - 1);
    reduction_axes.resize(num_reduction_axes);
    std::generate(reduction_axes.begin(), reduction_axes.end(), [&]() { return axes_dist(rng); });
    std::sort(reduction_axes.begin(), reduction_axes.end());
    auto end = std::unique(reduction_axes.begin(), reduction_axes.end());
    reduction_axes.erase(end, reduction_axes.end());

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

  xnnpack::ReplicableRandomDevice rng;
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

using MeanTestF16 = MeanTestBase<uint16_t>;
using MeanTestF32 = MeanTestBase<float>;

INSTANTIATE_TEST_SUITE_P(KeepDims, MeanTestF16, testing::Bool());
INSTANTIATE_TEST_SUITE_P(KeepDims, MeanTestF32, testing::Bool());

TEST_F(MeanTestF16, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
    xnn_define_tensor_value(subgraph, xnn_datatype_fp16, input_shape.size(), input_shape.data(),
                            nullptr, /*external_id=*/0, /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
    xnn_define_tensor_value(subgraph, xnn_datatype_fp16, output_shape.size(), output_shape.data(),
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
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp16);
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

TEST_P(MeanTestF16, matches_operator_api) {
  bool keep_dims = GetParam();
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
  std::fill(operator_output.begin(), operator_output.end(), UINT16_C(0x7E00) /* NaN */);
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT16_C(0x7E00) /* NaN */);

  uint32_t flags = keep_dims ? XNN_FLAG_KEEP_DIMS : 0;
  // Call operator API.
  const xnn_status status = xnn_create_mean_nd_f16(flags, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);

  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;
  ASSERT_EQ(xnn_status_success,
    xnn_reshape_mean_nd_f16(op,
      reduction_axes.size(), reduction_axes.data(),
      input_shape.size(), input_shape.data(),
      &workspace_size, &workspace_alignment,
      /*threadpool=*/nullptr));

  ASSERT_NE(workspace_size, SIZE_MAX);
  ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
  ASSERT_EQ(xnn_status_success, xnn_setup_mean_nd_f16(op, workspace.data(), input.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
    xnn_define_tensor_value(subgraph, xnn_datatype_fp16, input_shape.size(), input_shape.data(),
                            nullptr, /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  int output_num_dims = input_shape.size();
  if (!keep_dims) {
    output_num_dims -= reduction_axes.size();
  }
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp16, output_num_dims,
                              output_shape.data(), nullptr, /*external_id=*/1,
                              XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_static_mean(subgraph, reduction_axes.size(),
                                   reduction_axes.data(), input_id, output_id,
                                   flags));

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
    float sub_out = fp16_ieee_to_fp32_value(subgraph_output[i]);
    float op_out = fp16_ieee_to_fp32_value(operator_output[i]);
    ASSERT_NEAR(sub_out, op_out, std::abs(0.05f * std::min(sub_out, op_out)));
  }
}

TEST_P(MeanTestF32, matches_operator_api) {
  bool keep_dims = GetParam();
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  uint32_t flags = keep_dims ? XNN_FLAG_KEEP_DIMS : 0;
  // Call operator API.
  const xnn_status status = xnn_create_mean_nd_f32(flags, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);

  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  ASSERT_EQ(xnn_status_success,
    xnn_reshape_mean_nd_f32(op,
      reduction_axes.size(), reduction_axes.data(),
      input_shape.size(), input_shape.data(),
      /*threadpool=*/nullptr));

  ASSERT_EQ(xnn_status_success, xnn_setup_mean_nd_f32(op, input.data(), operator_output.data()));

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
  int output_num_dims = input_shape.size();
  if (!keep_dims) {
    output_num_dims -= reduction_axes.size();
  }
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_num_dims,
                              output_shape.data(), nullptr, /*external_id=*/1,
                              XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_static_mean(subgraph, reduction_axes.size(),
                                   reduction_axes.data(), input_id, output_id,
                                   flags));

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
  for (int i = 0; i < subgraph_output.size(); ++i) {
    ASSERT_NEAR(subgraph_output[i], operator_output[i], 2.5f * std::numeric_limits<float>::epsilon()) << " i " << i;
  }
}

TEST_F(MeanTestF32, reshape_output_keep_dims)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

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
      /*flags=*/XNN_FLAG_KEEP_DIMS));

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

  input_shape[0] += 2;
  input_shape[1] += 4;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input_id, input_shape.size(), input_shape.data()));
  const struct xnn_node* node = &subgraph->nodes[0];
  std::vector<size_t> unique_reduction_axes = reduction_axes;
  std::sort(unique_reduction_axes.begin(), unique_reduction_axes.end());
  auto end = std::unique(unique_reduction_axes.begin(), unique_reduction_axes.end());
  unique_reduction_axes.erase(end, unique_reduction_axes.end());
  // There are too many parameters which influence the workspace size so
  // knowing if reallocation is required or not is messy.
  node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  size_t current_axes = 0;
  for (size_t i = 0; i < output_shape->num_dims; ++i) {
    if (unique_reduction_axes[current_axes] == i) {
      ASSERT_EQ(output_shape->dim[i], 1);
      ++current_axes;
      if (current_axes == unique_reduction_axes.size()) {
        break;
      }
    } else {
      ASSERT_EQ(output_shape->dim[i], input_shape[i]);
    }
  }

  input_shape[0] -= 1;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input_id, input_shape.size(), input_shape.data()));
  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_success);
  current_axes = 0;
  for (size_t i = 0; i < output_shape->num_dims; ++i) {
    if (unique_reduction_axes[current_axes] == i) {
      ASSERT_EQ(output_shape->dim[i], 1);
      ++current_axes;
      if (current_axes == unique_reduction_axes.size()) {
        break;
      }
    } else {
      ASSERT_EQ(output_shape->dim[i], input_shape[i]);
    }
  }
}

TEST_F(MeanTestF32, reshape_output_no_keep_dims)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

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
  int output_num_dims = input_shape.size() - reduction_axes.size();
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_num_dims,
                              output_shape.data(), nullptr, /*external_id=*/1,
                              XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
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

  input_shape[0] += 2;
  input_shape[1] += 4;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input_id, input_shape.size(), input_shape.data()));
  const struct xnn_node* node = &subgraph->nodes[0];
  std::vector<size_t> unique_reduction_axes = reduction_axes;
  std::sort(unique_reduction_axes.begin(), unique_reduction_axes.end());
  auto end = std::unique(unique_reduction_axes.begin(), unique_reduction_axes.end());
  unique_reduction_axes.erase(end, unique_reduction_axes.end());
  // There are too many parameters which influence the workspace size so
  // knowing if reallocation is required or not is messy.
  node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  size_t current_axes = 0;
  size_t current_dim = 0;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (unique_reduction_axes[current_axes] == i) {
      ++current_axes;
      if (current_axes == unique_reduction_axes.size()) {
        break;
      }
    } else {
      ASSERT_EQ(output_shape->dim[current_dim], input_shape[i]);
      ++current_dim;
    }
  }

  input_shape[0] -= 1;
  ASSERT_EQ(xnn_status_success, xnn_reshape_external_value(runtime, input_id, input_shape.size(), input_shape.data()));
  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr), xnn_status_success);
  current_axes = 0;
  current_dim = 0;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (unique_reduction_axes[current_axes] == i) {
      ++current_axes;
      if (current_axes == unique_reduction_axes.size()) {
        break;
      }
    } else {
      ASSERT_EQ(output_shape->dim[current_dim], input_shape[i]);
      ++current_dim;
    }
  }
}

}  // namespace xnnpack
