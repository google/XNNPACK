// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "clamp-operator-tester.h"


TEST(CLAMP_NC_U8, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestU8();
  }
}

TEST(CLAMP_NC_U8, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      ClampOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmin(qmin)
        .qmax(255)
        .iterations(3)
        .TestU8();
    }
  }
}

TEST(CLAMP_NC_U8, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      ClampOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmin(0)
        .qmax(qmax)
        .iterations(3)
        .TestU8();
    }
  }
}

TEST(CLAMP_NC_U8, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestU8();
  }
}

TEST(CLAMP_NC_U8, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestU8();
  }
}

TEST(CLAMP_NC_U8, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestU8();
  }
}

TEST(CLAMP_NC_U8, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestU8();
  }
}

TEST(CLAMP_NC_F32, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestF32();
  }
}

TEST(CLAMP_NC_F32, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      ClampOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmin(qmin)
        .qmax(255)
        .iterations(3)
        .TestF32();
    }
  }
}

TEST(CLAMP_NC_F32, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      ClampOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmin(0)
        .qmax(qmax)
        .iterations(3)
        .TestF32();
    }
  }
}

TEST(CLAMP_NC_F32, unit_batch_with_relu) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
      .batch_size(1)
      .channels(channels)
      .relu_activation(true)
      .iterations(3)
      .TestF32();
  }
}
TEST(CLAMP_NC_F32, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestF32();
  }
}

TEST(CLAMP_NC_F32, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestF32();
  }
}

TEST(CLAMP_NC_F32, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestF32();
  }
}

TEST(CLAMP_NC_F32, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestF32();
  }
}
