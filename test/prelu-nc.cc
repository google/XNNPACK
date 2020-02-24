// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/params.h>

#include "prelu-operator-tester.h"


TEST(PRELU_NC_F32, unit_batch) {
  for (size_t channels = 1; channels < xnn_params.f32.prelu.channel_tile * 10; channels += std::max<size_t>(1, xnn_params.f32.prelu.channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, small_batch) {
  for (size_t channels = 1; channels < xnn_params.f32.prelu.channel_tile * 10; channels += std::max<size_t>(1, xnn_params.f32.prelu.channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(xnn_params.f32.prelu.row_tile)
      .channels(channels)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, small_batch_with_x_stride) {
  for (size_t channels = 1; channels < xnn_params.f32.prelu.channel_tile * 10; channels += std::max<size_t>(1, xnn_params.f32.prelu.channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(xnn_params.f32.prelu.row_tile)
      .channels(channels)
      .x_stride(123)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, small_batch_with_y_stride) {
  for (size_t channels = 1; channels < xnn_params.f32.prelu.channel_tile * 10; channels += std::max<size_t>(1, xnn_params.f32.prelu.channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(xnn_params.f32.prelu.row_tile)
      .channels(channels)
      .y_stride(117)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, small_batch_with_x_stride_and_y_stride) {
  for (size_t channels = 1; channels < xnn_params.f32.prelu.channel_tile * 10; channels += std::max<size_t>(1, xnn_params.f32.prelu.channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(xnn_params.f32.prelu.row_tile)
      .channels(channels)
      .x_stride(123)
      .y_stride(117)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, large_batch) {
  for (size_t channels = 1; channels < xnn_params.f32.prelu.channel_tile * 10; channels += std::max<size_t>(1, xnn_params.f32.prelu.channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(3 * xnn_params.f32.prelu.row_tile + 1)
      .channels(channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, large_batch_with_x_stride) {
  for (size_t channels = 1; channels < xnn_params.f32.prelu.channel_tile * 10; channels += std::max<size_t>(1, xnn_params.f32.prelu.channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(3 * xnn_params.f32.prelu.row_tile + 1)
      .channels(channels)
      .x_stride(123)
      .iterations(1)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, large_batch_with_y_stride) {
  for (size_t channels = 1; channels < xnn_params.f32.prelu.channel_tile * 10; channels += std::max<size_t>(1, xnn_params.f32.prelu.channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(3 * xnn_params.f32.prelu.row_tile + 1)
      .channels(channels)
      .y_stride(117)
      .iterations(1)
      .TestF32();
  }
}

TEST(PRELU_NC_F32, large_batch_with_x_stride_and_y_stride) {
  for (size_t channels = 1; channels < xnn_params.f32.prelu.channel_tile * 10; channels += std::max<size_t>(1, xnn_params.f32.prelu.channel_tile - 1)) {
    PReLUOperatorTester()
      .batch_size(3 * xnn_params.f32.prelu.row_tile + 1)
      .channels(channels)
      .x_stride(123)
      .y_stride(117)
      .iterations(1)
      .TestF32();
  }
}
