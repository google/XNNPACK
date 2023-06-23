// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "fully-connected-operator-tester.h"


TEST(FULLY_CONNECTED_NC_QS8, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QS8, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestQS8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_QU8, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestQU8();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32();
}

#if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_JIT // TODO(b/287020333)
TEST(FULLY_CONNECTED_NC_F32, unit_batch_with_jit) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}

TEST(FULLY_CONNECTED_NC_F32, small_batch_with_jit) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .use_jit(true)
    .iterations(3)
    .TestF32();
}
#endif  // (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_JIT

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, DISABLED_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, DISABLED_small_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F32_QC8W, DISABLED_weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestF32QC8W();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .output_stride(29)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, unit_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_fp32_weights) {
  FullyConnectedOperatorTester()
    .weights_type(FullyConnectedOperatorTester::WeightsType::FP32)
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_with_qmin) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .qmin(128)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_with_qmax) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .qmax(128)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_with_input_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
    .input_stride(28)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_with_output_stride) {
  FullyConnectedOperatorTester()
    .batch_size(12)
    .input_channels(23)
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
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, small_batch_without_bias) {
  FullyConnectedOperatorTester()
    .has_bias(false)
    .batch_size(12)
    .input_channels(23)
    .output_channels(19)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, weights_cache_unit_batch) {
  FullyConnectedOperatorTester()
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}

TEST(FULLY_CONNECTED_NC_F16, weights_cache_unit_batch_transpose_weights) {
  FullyConnectedOperatorTester()
    .transpose_weights(true)
    .batch_size(1)
    .input_channels(23)
    .output_channels(19)
    .use_weights_cache(true)
    .iterations(3)
    .TestF16();
}
