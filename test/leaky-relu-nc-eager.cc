// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <vector>

#include <gtest/gtest.h>

#include "leaky-relu-operator-tester.h"

TEST(LEAKY_RELU_NC_F32, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    LeakyReLUOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestRunF32();
  }
}

TEST(LEAKY_RELU_NC_F32, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    LeakyReLUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestRunF32();
  }
}

TEST(LEAKY_RELU_NC_F32, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    LeakyReLUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestRunF32();
  }
}

TEST(LEAKY_RELU_NC_F32, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    LeakyReLUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestRunF32();
  }
}

TEST(LEAKY_RELU_NC_F32, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    LeakyReLUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestRunF32();
  }
}

TEST(LEAKY_RELU_NC_F32, small_batch_with_negative_slope) {
  for (size_t batch_size = 1; batch_size <= 3; batch_size += 2) {
    for (size_t channels = 1; channels < 100; channels += 15) {
      for (float negative_slope : std::vector<float>({-10.0f, -1.0f, -0.1f, 0.1f, 10.0f})) {
        LeakyReLUOperatorTester()
          .batch_size(3)
          .channels(channels)
          .negative_slope(negative_slope)
          .iterations(1)
          .TestRunF32();
      }
    }
  }
}
