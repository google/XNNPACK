// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph.h"
#include "subgraph-unary-tester.h"

using Reshape2dTestInt8 = UnaryTest<int8_t>;
using Reshape2dTestUint8 = UnaryTest<uint8_t>;
using Reshape2dTestF32 = UnaryTest<float>;

TEST_F(Reshape2dTestInt8, define) {
  const int32_t zero_point = i8dist(rng);
  const float scale = scale_dist(rng);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint8, zero_point, scale, dims.size(),
                dims.data(), nullptr, 0,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  std::vector<size_t> output_dims(2);
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint8, zero_point, scale,
                output_dims.size(), output_dims.data(), nullptr, 1,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_reshape_2d(subgraph, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_reshape_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qs8);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(Reshape2dTestUint8, define) {
  const int32_t zero_point = u8dist(rng);
  const float scale = scale_dist(rng);

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_quint8, zero_point, scale, dims.size(),
                dims.data(), nullptr, 0,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  std::vector<size_t> output_dims(2);
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_quint8, zero_point, scale,
                output_dims.size(), output_dims.data(), nullptr, 1,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_reshape_2d(subgraph, input_id, output_id, /*flags=*/0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_reshape_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_qu8);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(Reshape2dTestF32, define) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(0, /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, dims.size(),
                                    dims.data(), nullptr, XNN_INVALID_VALUE_ID,
                                    /*flags=*/0, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  uint32_t output_id = XNN_INVALID_NODE_ID;
  std::vector<size_t> output_dims(2);
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, output_dims.size(),
                              output_dims.data(), nullptr, XNN_INVALID_VALUE_ID,
                              /*flags=*/0, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_reshape_2d(subgraph, input_id, output_id, 0));

  ASSERT_EQ(subgraph->num_nodes, 1);
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->type, xnn_node_type_reshape_2d);
  ASSERT_EQ(node->compute_type, xnn_compute_type_fp32);
  ASSERT_EQ(node->num_inputs, 1);
  ASSERT_EQ(node->inputs[0], input_id);
  ASSERT_EQ(node->num_outputs, 1);
  ASSERT_EQ(node->outputs[0], output_id);
  ASSERT_EQ(node->flags, 0);
}

TEST_F(Reshape2dTestInt8, matches_operator_api) {
  const int32_t zero_point = i8dist(rng);
  const float scale = scale_dist(rng);
  std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), INT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), INT8_C(0xA5));

  std::vector<size_t> output_dims(2, 1);
  if (!dims.empty()) {
    for (size_t k = 0; k < dims.size() - 1; ++k) {
      output_dims[0] *= dims[k];
    }
    output_dims.back() = dims.back();
  }

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_copy_nc_x8(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  size_t batch_size = NumElements(dims);
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op, batch_size, 1, 1, 1,
                                                       /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_copy_nc_x8(op, input.data(), operator_output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_qint8, zero_point, scale, dims.size(),
                dims.data(), nullptr, /*external_id=*/0,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_quantized_tensor_value(
          subgraph, xnn_datatype_qint8, zero_point, scale, output_dims.size(),
          output_dims.data(), nullptr, /*external_id=*/1,
          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_reshape_2d(subgraph, input_id, output_id,
                                  /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(
      xnn_status_success,
      xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(Reshape2dTestUint8, matches_operator_api) {
  const int32_t zero_point = u8dist(rng);
  const float scale = scale_dist(rng);
  std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), UINT8_C(0xA5));
  std::fill(subgraph_output.begin(), subgraph_output.end(), UINT8_C(0xA5));

  std::vector<size_t> output_dims(2, 1);
  if (!dims.empty()) {
    for (int k = 0; k < dims.size() - 1; ++k) {
      output_dims[0] *= dims[k];
    }
    output_dims.back() = dims.back();
  }

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  const xnn_status status = xnn_create_copy_nc_x8(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  size_t batch_size = NumElements(dims);
  ASSERT_EQ(xnn_status_success, xnn_reshape_copy_nc_x8(op, batch_size, 1, 1, 1,
                                                       /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_copy_nc_x8(op, input.data(), operator_output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_quint8, zero_point, scale, dims.size(),
                dims.data(), nullptr, /*external_id=*/0,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_quantized_tensor_value(
          subgraph, xnn_datatype_quint8, zero_point, scale, output_dims.size(),
          output_dims.data(), nullptr, /*external_id=*/1,
          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  ASSERT_EQ(xnn_status_success,
            xnn_define_reshape_2d(subgraph, input_id, output_id,
                                  /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(
      xnn_status_success,
      xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(Reshape2dTestF32, matches_operator_api) {
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
  std::fill(operator_output.begin(), operator_output.end(), nanf(""));
  std::fill(subgraph_output.begin(), subgraph_output.end(), nanf(""));

  std::vector<size_t> output_dims(2, 1);
  if (!dims.empty()) {
    for (int k = 0; k < dims.size() - 1; ++k) {
      output_dims[0] *= dims[k];
    }
    output_dims.back() = dims.back();
  }

  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Call operator API.
  xnn_operator_t op = nullptr;
  xnn_status status = xnn_create_copy_nc_x32(/*flags=*/0, &op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);
  size_t batch_size = NumElements(dims);
  ASSERT_EQ(
      xnn_status_success,
      xnn_reshape_copy_nc_x32(op, batch_size, 1, 1, 1, /*threadpool=*/nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_copy_nc_x32(op, input.data(), operator_output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2,
                                                    /*flags=*/0, &subgraph));
  ASSERT_NE(nullptr, subgraph);
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, dims.size(),
                                    dims.data(), nullptr, /*external_id=*/0,
                                    XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);
  uint32_t output_id = XNN_INVALID_NODE_ID;

  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr,
                /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);
  ASSERT_EQ(xnn_status_success,
            xnn_define_reshape_2d(subgraph, input_id, output_id,
                                  /*flags=*/0));
  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(
      xnn_status_success,
      xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input.data()},
      xnn_external_value{output_id, subgraph_output.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  ASSERT_EQ(subgraph_output, operator_output);
}

TEST_F(Reshape2dTestF32, reshape_output) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  // Compute the initial input and output dimensions.
  std::vector<size_t> input_dims(4);
  std::generate(input_dims.begin(), input_dims.end(),
                [&] { return dim_dist(rng); });

  std::vector<size_t> output_dims(2, 1);
  for (int k = 0; k < input_dims.size() - 1; ++k) {
    output_dims[0] *= input_dims[k];
  }
  output_dims.back() = input_dims.back();

  // Allocate the input/output buffers.
  std::vector<float> input_data(NumElements(input_dims) +
                                XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> output_data(NumElements(output_dims));

  // Call subgraph API.
  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(/*external_value_ids=*/2,
                                                    /*flags=*/0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  input_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, input_dims.size(),
                input_dims.data(), nullptr, /*external_id=*/0,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_NE(input_id, XNN_INVALID_NODE_ID);

  output_id = XNN_INVALID_NODE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, output_dims.size(),
                output_dims.data(), nullptr, /*external_id=*/1,
                /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_NE(output_id, XNN_INVALID_NODE_ID);

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_define_reshape_2d(subgraph, input_id, output_id, /*flags=*/0));
  ASSERT_EQ(
      xnn_status_success,
      xnn_create_runtime_v3(subgraph, nullptr, nullptr, /*flags=*/0, &runtime));
  ASSERT_NE(nullptr, runtime);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  std::array<xnn_external_value, 2> external = {
      xnn_external_value{input_id, input_data.data()},
      xnn_external_value{output_id, output_data.data()}};
  ASSERT_EQ(xnn_status_success,
            xnn_setup_runtime(runtime, external.size(), external.data()));
  ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));

  // Change the input shape (make it large enough to trigger a reallocation).
  input_dims[0] += 2;
  input_dims[3] += 10;
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, input_id, input_dims.size(),
                                       input_dims.data()));
  const struct xnn_node* node = &subgraph->nodes[0];
  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values,
                          runtime->num_values, /*threadpool=*/nullptr),
            xnn_status_reallocation_required);
  const xnn_shape* output_shape = &runtime->values[node->outputs[0]].shape;
  EXPECT_EQ(output_shape->dim[0],
            input_dims[0] * input_dims[1] * input_dims[2]);
  EXPECT_EQ(output_shape->dim[1], input_dims[3]);

  // Change the input shape (make it a bit smaller again).
  input_dims[0] -= 1;
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, input_id, input_dims.size(),
                                       input_dims.data()));
  ASSERT_EQ(node->reshape(&runtime->opdata[0], runtime->values,
                          runtime->num_values, /*threadpool=*/nullptr),
            xnn_status_success);
  EXPECT_EQ(output_shape->dim[0],
            input_dims[0] * input_dims[1] * input_dims[2]);
  EXPECT_EQ(output_shape->dim[1], input_dims[3]);
}
