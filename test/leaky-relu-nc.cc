// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "leaky-relu-operator-tester.h"


TEST(LEAKY_RELU_NC_Q8, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    LeakyReLUOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestQ8();
  }
}

TEST(LEAKY_RELU_NC_Q8, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    LeakyReLUOperatorTester()
      .batch_size(1)
      .channels(channels)
      .qmin(128)
      .iterations(3)
      .TestQ8();
  }
}

TEST(LEAKY_RELU_NC_Q8, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    LeakyReLUOperatorTester()
      .batch_size(1)
      .channels(channels)
      .qmax(128)
      .iterations(3)
      .TestQ8();
  }
}

TEST(LEAKY_RELU_NC_Q8, unit_batch_with_negative_slope) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float negative_slope = 1.0e-4f; negative_slope < 1.0f; negative_slope *= 3.14159265f) {
      LeakyReLUOperatorTester()
        .batch_size(1)
        .channels(channels)
        .negative_slope(negative_slope)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(LEAKY_RELU_NC_Q8, unit_batch_with_input_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float input_scale = 1.0e-2f; input_scale < 1.0e+2f; input_scale *= 3.14159265f) {
      LeakyReLUOperatorTester()
        .batch_size(1)
        .channels(channels)
        .input_scale(input_scale)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(LEAKY_RELU_NC_Q8, unit_batch_with_input_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      LeakyReLUOperatorTester()
        .batch_size(1)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(LEAKY_RELU_NC_Q8, unit_batch_with_output_scale) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (float output_scale = 1.0e-2f; output_scale < 1.0e+2f; output_scale *= 3.14159265f) {
      LeakyReLUOperatorTester()
        .batch_size(1)
        .channels(channels)
        .output_scale(output_scale)
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(LEAKY_RELU_NC_Q8, unit_batch_with_output_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t output_zero_point = 0; output_zero_point <= 255; output_zero_point += 51) {
      LeakyReLUOperatorTester()
        .batch_size(1)
        .channels(channels)
        .output_zero_point(uint8_t(output_zero_point))
        .iterations(1)
        .TestQ8();
    }
  }
}

TEST(LEAKY_RELU_NC_Q8, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    LeakyReLUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestQ8();
  }
}

TEST(LEAKY_RELU_NC_Q8, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    LeakyReLUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestQ8();
  }
}

TEST(LEAKY_RELU_NC_Q8, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    LeakyReLUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestQ8();
  }
}

TEST(LEAKY_RELU_NC_Q8, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    LeakyReLUOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestQ8();
  }
}
