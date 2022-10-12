// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "ceiling-operator-tester.h"


TEST(CEILING_NC_F32, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    CeilingOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestRunF32();
  }
}

TEST(CEILING_NC_F32, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    CeilingOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestRunF32();
  }
}

TEST(CEILING_NC_F32, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    CeilingOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestRunF32();
  }
}

TEST(CEILING_NC_F32, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    CeilingOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestRunF32();
  }
}

TEST(CEILING_NC_F32, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    CeilingOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .output_stride(117)
        .iterations(3)
        .TestRunF32();
  }
}
