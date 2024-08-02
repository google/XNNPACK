// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "fully-connected-operator-tester.h"

TEST(FULLY_CONNECTED_NC_QS8, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_with_input_zero_point) {
  for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
    FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .input_zero_point(uint8_t(input_zero_point))
      .iterations(3)
      .TestQS8();
  }
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(1)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_with_input_zero_point) {
  for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
    FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .input_zero_point(uint8_t(input_zero_point))
      .iterations(3)
      .TestQS8QC8W();
  }
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_with_output_zero_point) {
  for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
    FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .output_zero_point(uint8_t(output_zero_point))
      .iterations(3)
      .TestQS8QC8W();
  }
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QS8_QC8W, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8QC8W();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_input_zero_point) {
  for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
    FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .input_zero_point(uint8_t(input_zero_point))
      .iterations(3)
      .TestQU8();
  }
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_kernel_zero_point) {
  for (int32_t kernel_zero_point = 1; kernel_zero_point <= 255; kernel_zero_point += 51) {
    FullyConnectedOperatorTester()
      .batch_size(1)
      .input_channels(22)
      .output_channels(19)
      .kernel_zero_point(uint8_t(kernel_zero_point))
      .iterations(3)
      .TestQU8();
  }
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}

#if !XNN_ARCH_WASM && XNN_ENABLE_JIT //  TODO(b/290880274)
TEST(FULLY_CONNECTED_NC_F32, unit_batch_with_jit) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_with_jit) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}
#endif  // !XNN_ARCH_WASM && XNN_ENABLE_JIT

TEST(FULLY_CONNECTED_NC_F32_QC4W, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .kernel_zero_point(0)
    .iterations(3)
    .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .kernel_zero_point(0)
    .iterations(3)
    .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .kernel_zero_point(0)
    .iterations(3)
    .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, unit_batch_with_kernel_zero_point) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(15)
    .iterations(3)
    .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .kernel_zero_point(0)
    .iterations(3)
    .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .kernel_zero_point(0)
    .iterations(3)
    .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .kernel_zero_point(0)
    .iterations(3)
    .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, small_batch_with_kernel_zero_point) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(15)
    .kernel_zero_point(0)
    .iterations(3)
    .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC4W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .kernel_zero_point(0)
    .iterations(3)
    .TestF32QC4W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_fp32_weights) {
  FullyConnectedOperatorTester()
    .weights_type(FullyConnectedOperatorTester::WeightsType::FP32)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_transpose_fp32_weights) {
  FullyConnectedOperatorTester()
    .weights_type(FullyConnectedOperatorTester::WeightsType::FP32)
    .transpose_weights(true)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .qmin(128)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .qmax(128)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .output_stride(29)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(20)  // legacy requires even number
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .qmin(128)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .qmax(128)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .input_stride(29)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .output_stride(29)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(12)
    .input_channels(22)
    .output_channels(20)  // legacy doesn't support odd nc
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .use_weights_cache(true)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC4W, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(20)    // legacy doesn't support odd nc
    .kernel_zero_point(0)
    .use_weights_cache(true)
    .iterations(3)
    .TestQD8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QB4W, bl) {
  for (size_t ic=32; ic<=256; ic*=2){
    for (size_t bs=32; bs<=ic; bs=bs*2) {
      FullyConnectedOperatorTester()
        .batch_size(12)
        .output_channels(18)
        .input_channels(ic)
        .block_size(bs)
        .kernel_zero_point(8)
        .iterations(3)
        .TestQD8F32QB4W();
    }
  }
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QB4W, bl_no_bias) {
  for (size_t ic=32; ic<=256; ic*=2){
    for (size_t bs=32; bs<=ic; bs=bs*2) {
      FullyConnectedOperatorTester()
        .has_bias(false)
        .batch_size(12)
        .output_channels(18)
        .input_channels(ic)
        .block_size(bs)
        .kernel_zero_point(8)
        .iterations(3)
        .TestQD8F32QB4W();
    }
  }
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(8)
    .output_channels(4)
    .iterations(1)
    .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(4)
    .input_channels(1)
    .output_channels(16)
    .iterations(1)
    .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(11)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F32_QC8W, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestQD8F32QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(8)
    .output_channels(4)
    .iterations(1)
    .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(4)
    .input_channels(1)
    .output_channels(16)
    .iterations(1)
    .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(11)
    .input_channels(22)
    .output_channels(19)
    .iterations(3)
    .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC8W, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestQD8F16QC8W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(8)
    .output_channels(4)
    .kernel_zero_point(0)
    .iterations(1)
    .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(20)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(4)
    .input_channels(2)
    .output_channels(16)
    .kernel_zero_point(0)
    .iterations(1)
    .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmin(128)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .qmax(128)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .output_stride(29)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(11)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .use_weights_cache(true)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QC4W, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(20)
    .use_weights_cache(true)
    .kernel_zero_point(0)
    .iterations(3)
    .TestQD8F16QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(8)
    .iterations(3)
    .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(8)
    .qmin(128)
    .iterations(3)
    .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(8)
    .qmax(128)
    .iterations(3)
    .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .input_stride(28)
    .output_channels(19)
    .kernel_zero_point(8)
    .iterations(3)
    .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(8)
    .output_stride(29)
    .iterations(3)
    .TestQP8F32QC4W();
}

// TODO(b/355416339): Re-enable once we can handle strides again
TEST(DISABLED_FULLY_CONNECTED_NC_QP8_F32_QC4W, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(20)  // legacy requires even number
    .kernel_zero_point(8)
    .iterations(3)
    .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(8)
    .iterations(3)
    .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(8)
    .iterations(3)
    .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(8)
    .qmin(128)
    .iterations(3)
    .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(8)
    .qmax(128)
    .iterations(3)
    .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .input_stride(29)
    .output_channels(19)
    .kernel_zero_point(8)
    .iterations(3)
    .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(8)
    .output_stride(29)
    .iterations(3)
    .TestQP8F32QC4W();
}

// TODO(b/355416339): Re-enable once we can handle strides again
TEST(DISABLED_FULLY_CONNECTED_NC_QP8_F32_QC4W, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(12)
    .input_channels(22)
    .output_channels(20)  // legacy doesn't support odd nc
    .kernel_zero_point(8)
    .iterations(3)
    .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(12)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(8)
    .iterations(3)
    .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QP8_F32_QC4W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(22)
    .output_channels(19)
    .kernel_zero_point(8)
    .use_weights_cache(true)
    .iterations(3)
    .TestQP8F32QC4W();
}

// TODO(b/355416339): Re-enable once we can handle strides again
TEST(DISABLED_FULLY_CONNECTED_NC_QP8_F32_QC4W, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(22)
    .output_channels(20)    // legacy doesn't support odd nc
    .kernel_zero_point(8)
    .use_weights_cache(true)
    .iterations(3)
    .TestQP8F32QC4W();
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QB4W, bl) {
  for (size_t ic=32; ic<=256; ic*=2){
    for (size_t bs=32; bs<=ic; bs=bs*2) {
      FullyConnectedOperatorTester()
        .batch_size(12)
        .output_channels(18)
        .input_channels(ic)
        .block_size(bs)
        .kernel_zero_point(8)
        .iterations(3)
        .TestQD8F16QB4W();
    }
  }
}

TEST(FULLY_CONNECTED_NC_QD8_F16_QB4W, bl_no_bias) {
  for (size_t ic=32; ic<=256; ic*=2){
    for (size_t bs=32; bs<=ic; bs=bs*2) {
      FullyConnectedOperatorTester()
        .has_bias(false)
        .batch_size(12)
        .output_channels(18)
        .input_channels(ic)
        .block_size(bs)
        .kernel_zero_point(8)
        .iterations(3)
        .TestQD8F16QB4W();
    }
  }
}
