// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "dynamic-fully-connected-operator-tester.h"


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
