// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include <gtest/gtest.h>
#include "test/operators/dynamic-fully-connected-operator-tester.h"

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, unit_batch) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, unit_batch_with_qmin) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .qmin(128)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, unit_batch_with_qmax) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .qmax(128)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, unit_batch_with_input_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .input_stride(28)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, unit_batch_with_output_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .output_stride(29)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, unit_batch_transpose_weights) {
  DynamicFullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, unit_batch_without_bias) {
  DynamicFullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, small_batch) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, small_batch_with_qmin) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .qmin(128)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, small_batch_with_qmax) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .qmax(128)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, small_batch_with_input_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .input_stride(28)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, small_batch_with_output_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .output_stride(29)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, small_batch_transpose_weights) {
  DynamicFullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F16, small_batch_without_bias) {
  DynamicFullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF16();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, unit_batch) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, unit_batch_with_qmin) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .qmin(128)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, unit_batch_with_qmax) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .qmax(128)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, unit_batch_with_input_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .input_stride(28)
      .output_channels(19)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, unit_batch_with_output_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .output_stride(29)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, unit_batch_transpose_weights) {
  DynamicFullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(1)
      .input_channels(23)
      .output_channels(9)
      .iterations(1)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, unit_batch_without_bias) {
  DynamicFullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(1)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, small_batch) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, small_batch_with_qmin) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .qmin(128)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, small_batch_with_qmax) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .qmax(128)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, small_batch_with_input_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .input_stride(28)
      .output_channels(19)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, small_batch_with_output_stride) {
  DynamicFullyConnectedOperatorTester()
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .output_stride(29)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, small_batch_transpose_weights) {
  DynamicFullyConnectedOperatorTester()
      .transpose_weights(true)
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, small_batch_without_bias) {
  DynamicFullyConnectedOperatorTester()
      .has_bias(false)
      .batch_size(12)
      .input_channels(23)
      .output_channels(19)
      .iterations(3)
      .TestF32();
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, overflow_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_dynamic_fully_connected_nc_f32(
                -std::numeric_limits<float>::infinity(),
                +std::numeric_limits<float>::infinity(),
                /*flags=*/0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  size_t workspace_size = 0;
  ASSERT_EQ(xnn_status_out_of_memory,
            xnn_reshape_dynamic_fully_connected_nc_f32(
                op, /*batch_size=*/1, /*input_channels=*/10,
                /*output_channels=*/10, /*input_stride=*/SIZE_MAX / 2,
                /*output_stride=*/10, &workspace_size,
                /*threadpool=*/nullptr));
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, overflow_batch_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_dynamic_fully_connected_nc_f32(
                -std::numeric_limits<float>::infinity(),
                +std::numeric_limits<float>::infinity(),
                /*flags=*/0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  size_t workspace_size = 0;
  ASSERT_EQ(xnn_status_out_of_memory,
            xnn_reshape_dynamic_fully_connected_nc_f32(
                op, /*batch_size=*/SIZE_MAX / 50, /*input_channels=*/10,
                /*output_channels=*/10, /*input_stride=*/100,
                /*output_stride=*/100, &workspace_size,
                /*threadpool=*/nullptr));
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, overflow_n_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_dynamic_fully_connected_nc_f32(
                -std::numeric_limits<float>::infinity(),
                +std::numeric_limits<float>::infinity(),
                /*flags=*/0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  size_t workspace_size = 0;
  EXPECT_EQ(
      xnn_status_out_of_memory,
      xnn_reshape_dynamic_fully_connected_nc_f32(
          op, /*batch_size=*/1, /*input_channels=*/1,
          /*output_channels=*/(SIZE_MAX / 2) + 1, /*input_stride=*/1,
          /*output_stride=*/(SIZE_MAX / 2) + 1, &workspace_size,
          /*threadpool=*/nullptr));
}

TEST(DYNAMIC_FULLY_CONNECTED_NC_F32, overflow_k_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_dynamic_fully_connected_nc_f32(
                -std::numeric_limits<float>::infinity(),
                +std::numeric_limits<float>::infinity(),
                /*flags=*/0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  size_t workspace_size = 0;
  EXPECT_EQ(
      xnn_status_out_of_memory,
      xnn_reshape_dynamic_fully_connected_nc_f32(
          op, /*batch_size=*/1, /*input_channels=*/(SIZE_MAX / 2) + 1,
          /*output_channels=*/1, /*input_stride=*/(SIZE_MAX / 2) + 1,
          /*output_stride=*/1, &workspace_size, /*threadpool=*/nullptr));
}

