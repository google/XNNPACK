// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>

#include <gtest/gtest.h>
#include "xnnpack/config.h"
#include "prelu-operator-tester.h"

#ifndef XNN_EXCLUDE_F16_TESTS
TEST(PRELU_NC_F16, unit_batch) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f16_prelu_config();
  if (prelu_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(1)
      .input_channels(input_channels)
      .iterations(3)
      .TestF16();
  }
}

TEST(PRELU_NC_F16, small_batch_with_broadcasted_slope) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f16_prelu_config();
  if (prelu_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(prelu_config->row_tile)
      .input_channels(input_channels)
      .slope_channels(1)
      .iterations(3)
      .TestF16();
  }
}

TEST(PRELU_NC_F16, small_batch) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f16_prelu_config();
  if (prelu_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(prelu_config->row_tile)
      .input_channels(input_channels)
      .iterations(3)
      .TestF16();
  }
}

TEST(PRELU_NC_F16, small_batch_with_x_stride) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f16_prelu_config();
  if (prelu_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(prelu_config->row_tile)
      .input_channels(input_channels)
      .x_stride(337)
      .iterations(3)
      .TestF16();
  }
}

TEST(PRELU_NC_F16, small_batch_with_y_stride) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f16_prelu_config();
  if (prelu_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(prelu_config->row_tile)
      .input_channels(input_channels)
      .y_stride(347)
      .iterations(3)
      .TestF16();
  }
}

TEST(PRELU_NC_F16, small_batch_with_x_stride_and_y_stride) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f16_prelu_config();
  if (prelu_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(prelu_config->row_tile)
      .input_channels(input_channels)
      .x_stride(337)
      .y_stride(347)
      .iterations(3)
      .TestF16();
  }
}

TEST(PRELU_NC_F16, large_batch) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f16_prelu_config();
  if (prelu_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(3 * prelu_config->row_tile + 1)
      .input_channels(input_channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(PRELU_NC_F16, large_batch_with_x_stride) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f16_prelu_config();
  if (prelu_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(3 * prelu_config->row_tile + 1)
      .input_channels(input_channels)
      .x_stride(337)
      .iterations(1)
      .TestF16();
  }
}

TEST(PRELU_NC_F16, large_batch_with_y_stride) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f16_prelu_config();
  if (prelu_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(3 * prelu_config->row_tile + 1)
      .input_channels(input_channels)
      .y_stride(347)
      .iterations(1)
      .TestF16();
  }
}

TEST(PRELU_NC_F16, large_batch_with_x_stride_and_y_stride) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f16_prelu_config();
  if (prelu_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(3 * prelu_config->row_tile + 1)
      .input_channels(input_channels)
      .x_stride(337)
      .y_stride(347)
      .iterations(1)
      .TestF16();
  }
}

TEST(PRELU_NC_F16, fp32_weights) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f16_prelu_config();
  if (prelu_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(3 * prelu_config->row_tile + 1)
      .input_channels(input_channels)
      .x_stride(337)
      .y_stride(347)
      .weights_type(PReLUOperatorTester::WeightsType::FP32)
      .iterations(1)
      .TestF16();
  }
}

TEST(PRELU_NC_F16, weights_cache_unit_batch) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f16_prelu_config();
  if (prelu_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(1)
      .input_channels(input_channels)
      .use_weights_cache(true)
      .iterations(3)
      .TestF16();
  }
}

TEST(PRELU_NC_F16, weights_cache_fp32_weights) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f16_prelu_config();
  if (prelu_config == nullptr) {
    GTEST_SKIP();  // F16 unsupported.
  }
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(3 * prelu_config->row_tile + 1)
      .input_channels(input_channels)
      .x_stride(345)
      .y_stride(347)
      .weights_type(PReLUOperatorTester::WeightsType::FP32)
      .use_weights_cache(true)
      .iterations(1)
      .TestF16();
  }
}
#endif  // XNN_EXCLUDE_F16_TESTS


TEST(PRELU_NC_F32, unit_batch) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f32_prelu_config();
  assert(prelu_config != nullptr);
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(1)
      .input_channels(input_channels)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, small_batch_with_broadcasted_slope) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f32_prelu_config();
  assert(prelu_config != nullptr);
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(prelu_config->row_tile)
      .input_channels(input_channels)
      .slope_channels(1)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, small_batch) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f32_prelu_config();
  assert(prelu_config != nullptr);
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(prelu_config->row_tile)
      .input_channels(input_channels)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, small_batch_with_x_stride) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f32_prelu_config();
  assert(prelu_config != nullptr);
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(prelu_config->row_tile)
      .input_channels(input_channels)
      .x_stride(337)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, small_batch_with_y_stride) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f32_prelu_config();
  assert(prelu_config != nullptr);
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(prelu_config->row_tile)
      .input_channels(input_channels)
      .y_stride(347)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, small_batch_with_x_stride_and_y_stride) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f32_prelu_config();
  assert(prelu_config != nullptr);
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(prelu_config->row_tile)
      .input_channels(input_channels)
      .x_stride(337)
      .y_stride(347)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, large_batch) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f32_prelu_config();
  assert(prelu_config != nullptr);
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(3 * prelu_config->row_tile + 1)
      .input_channels(input_channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, large_batch_with_x_stride) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f32_prelu_config();
  assert(prelu_config != nullptr);
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(3 * prelu_config->row_tile + 1)
      .input_channels(input_channels)
      .x_stride(337)
      .iterations(1)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, large_batch_with_y_stride) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f32_prelu_config();
  assert(prelu_config != nullptr);
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(3 * prelu_config->row_tile + 1)
      .input_channels(input_channels)
      .y_stride(347)
      .iterations(1)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, large_batch_with_x_stride_and_y_stride) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f32_prelu_config();
  assert(prelu_config != nullptr);
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(3 * prelu_config->row_tile + 1)
      .input_channels(input_channels)
      .x_stride(337)
      .y_stride(347)
      .iterations(1)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, weights_cache_unit_batch) {
  const struct xnn_prelu_config* prelu_config = xnn_init_f32_prelu_config();
  assert(prelu_config != nullptr);
  for (size_t input_channels = 1; input_channels < prelu_config->channel_tile * 10; input_channels += std::max<size_t>(1, prelu_config->channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(1)
      .input_channels(input_channels)
      .use_weights_cache(true)
      .iterations(3)
      .TestF32();
  }
}
