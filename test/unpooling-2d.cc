// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>  // For std::generate, std::min.
#include <array>      // For std::array.
#include <cstddef>    // For size_t.
#include <cstdint>    // For uint32_t.
#include <memory>     // For std::unique_ptr.
#include <random>     // For std::uniform_real_distribution.
#include <vector>     // For std::vector.

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"

template <class T, class BiasType = T> class Unpooling2DTestBase : public ::testing::Test {
 protected:
  Unpooling2DTestBase() {
    input_size_dist = std::uniform_int_distribution<uint32_t>(10, 15);
    kernel_size_dist = std::uniform_int_distribution<uint32_t>(1, 5);
    stride_dist = std::uniform_int_distribution<uint32_t>(1, 3);
    f32dist = std::uniform_real_distribution<float>(0.1f, 1.0f);
    scale_dist = std::uniform_real_distribution<float>(1.0f, 5.0f);
    i32dist = std::uniform_int_distribution<int32_t>(-10000, 10000);
    u32dist = std::uniform_int_distribution<uint32_t>();

    batch_size = input_size_dist(rng);
    input_height = input_size_dist(rng);
    input_width = input_size_dist(rng);
    pooling_height = 2;
    pooling_width = 2;
    channels = input_size_dist(rng);
    output_height = xnn_compute_unpooling_output_dimension(input_height, padding_top + padding_bottom, pooling_height);
    output_width = xnn_compute_unpooling_output_dimension(input_width, padding_left + padding_right, pooling_width);

    index_dist = std::uniform_int_distribution<uint32_t>(0, pooling_height * pooling_width - 1);

    input_value_dims = {{batch_size, input_height, input_width, channels}};
    input_index_dims = {{batch_size, input_height, input_width, channels}};
    output_dims = {{batch_size, output_height, output_width, channels}};

    input = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + batch_size * input_height * input_width * channels);
    input_index = std::vector<T>(batch_size * input_height * input_width * channels);
    operator_output = std::vector<T>(batch_size * output_height * output_width * channels);
    subgraph_output = std::vector<T>(batch_size * output_height * output_width * channels);
  }

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<uint32_t> input_size_dist;
  std::uniform_int_distribution<uint32_t> kernel_size_dist;
  std::uniform_int_distribution<uint32_t> stride_dist;
  std::uniform_int_distribution<int32_t> i32dist;
  std::uniform_int_distribution<uint32_t> u32dist;
  std::uniform_int_distribution<uint32_t> index_dist;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_real_distribution<float> scale_dist;

  const uint32_t padding_top = 0;
  const uint32_t padding_right = 0;
  const uint32_t padding_bottom = 0;
  const uint32_t padding_left = 0;
  uint32_t batch_size;
  uint32_t input_height;
  uint32_t input_width;
  uint32_t kernel_height;
  uint32_t kernel_width;
  uint32_t pooling_height;
  uint32_t pooling_width;
  uint32_t channels;
  uint32_t output_height;
  uint32_t output_width;

  std::array<size_t, 4> input_value_dims;
  std::array<size_t, 4> input_index_dims;
  std::array<size_t, 4> output_dims;

  std::vector<T> input;
  std::vector<T> input_index;
  std::vector<T> operator_output;
  std::vector<T> subgraph_output;
};

using Unpooling2DTestX32 = Unpooling2DTestBase<uint32_t>;

TEST_F(Unpooling2DTestX32, define)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_value_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_value_dims.size(), input_value_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_value_id));
  ASSERT_NE(input_value_id, XNN_INVALID_NODE_ID);

  uint32_t input_index_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_index_dims.size(), input_index_dims.data(),
                          input_index.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &input_index_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success, xnn_define_unpooling_2d(
                          subgraph, padding_top, padding_right, padding_bottom, padding_left, pooling_height,
                          pooling_width, input_value_id, input_index_id, output_id,
                          /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_unpooling_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->params.pooling_2d.padding_top, padding_top);
  ASSERT_EQ(node->params.pooling_2d.padding_right, padding_right);
  ASSERT_EQ(node->params.pooling_2d.padding_bottom, padding_bottom);
  ASSERT_EQ(node->params.pooling_2d.padding_left, padding_left);
  ASSERT_EQ(node->params.pooling_2d.pooling_height, pooling_height);
  ASSERT_EQ(node->params.pooling_2d.pooling_width, pooling_width);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input_value_id);
  ASSERT_EQ(node->inputs[1], input_index_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(Unpooling2DTestX32, matches_operator_api)
{
  xnn_operator_t op = nullptr;

  std::generate(input.begin(), input.end(), [&]() { return u32dist(rng); });
  std::generate(input_index.begin(), input_index.end(), [&]() { return index_dist(rng); });
  std::generate(operator_output.begin(), operator_output.end(), [&]() { return u32dist(rng); });
  std::generate(subgraph_output.begin(), subgraph_output.end(), [&]() { return u32dist(rng); });

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  const xnn_status status = xnn_create_unpooling2d_nhwc_x32(
    padding_top, padding_right, padding_bottom, padding_left, pooling_height, pooling_width, channels, channels,
    channels, /*flags=*/0, &op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(op, xnn_delete_operator);

  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_unpooling2d_nhwc_x32(
                          op, batch_size, input_height, input_width, /*output_height_out=*/nullptr,
                          /*output_width_out=*/nullptr, /*threadpool=*/nullptr));
  ASSERT_EQ(
    xnn_status_success,
    xnn_setup_unpooling2d_nhwc_x32(
      op, input.data(), input_index.data(), operator_output.data()));

  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_value_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_value_dims.size(), input_value_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_value_id));
  ASSERT_NE(input_value_id, XNN_INVALID_NODE_ID);

  uint32_t input_index_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_index_dims.size(), input_index_dims.data(),
                          input_index.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &input_index_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success, xnn_define_unpooling_2d(
                          subgraph, padding_top, padding_right, padding_bottom, padding_left, pooling_height,
                          pooling_width, input_value_id, input_index_id, output_id,
                          /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_value_id, input.data()}, xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(Unpooling2DTestX32, reshape_output)
{
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(subgraph, xnn_delete_subgraph);

  uint32_t input_value_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_value_dims.size(), input_value_dims.data(), nullptr,
                          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_value_id));
  ASSERT_NE(input_value_id, XNN_INVALID_NODE_ID);

  uint32_t input_index_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, input_index_dims.size(), input_index_dims.data(),
                          input_index.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &input_index_id));

  uint32_t output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
    xnn_status_success, xnn_define_tensor_value(
                          subgraph, xnn_datatype_fp32, output_dims.size(), output_dims.data(), nullptr,
                          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(
    xnn_status_success, xnn_define_unpooling_2d(
                          subgraph, padding_top, padding_right, padding_bottom, padding_left, pooling_height,
                          pooling_width, input_value_id, input_index_id, output_id,
                          /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_unpooling_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->num_inputs, 2);
  ASSERT_EQ(node->inputs[0], input_value_id);
  ASSERT_EQ(node->inputs[1], input_index_id);
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

  input_value_dims[0] += 1;
  input_value_dims[1] += 1;
  input_value_dims[2] += 1;
  input_value_dims[3] += 1;

  input_index_dims[0] += 1;
  input_index_dims[1] += 1;
  input_index_dims[2] += 1;
  input_index_dims[3] += 1;

  ASSERT_EQ(
    xnn_status_success, xnn_reshape_external_value(runtime, 0, input_value_dims.size(), input_value_dims.data()));
  ASSERT_EQ(
    xnn_status_success, xnn_reshape_external_value(runtime, 1, input_index_dims.size(), input_index_dims.data()));

  ASSERT_EQ(
    node->reshape(&runtime->opdata[0], runtime->values, runtime->num_values, /*threadpool=*/nullptr),
    xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;

  const size_t expected_height =
    xnn_compute_unpooling_output_dimension(input_value_dims[1], padding_top + padding_bottom, pooling_height);
  const size_t expected_width =
    xnn_compute_unpooling_output_dimension(input_value_dims[2], padding_left + padding_right, pooling_width);

  ASSERT_EQ(output_shape->dim[0], input_value_dims[0]);
  ASSERT_EQ(output_shape->dim[1], expected_height);
  ASSERT_EQ(output_shape->dim[2], expected_width);
  ASSERT_EQ(output_shape->dim[3], input_value_dims[3]);
}
