// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/params.h>

#include "prelu-operator-tester.h"


TEST(PRELU_OP_F32, unit_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    PReLUOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_OP_F32, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    PReLUOperatorTester()
      .batch_size(1)
      .channels(channels)
      .qmin(128)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_OP_F32, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    PReLUOperatorTester()
      .batch_size(1)
      .channels(channels)
      .qmax(128)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_OP_F32, small_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    PReLUOperatorTester()
      .batch_size(xnn_params.f32.prelu.mr)
      .channels(channels)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_OP_F32, small_batch_with_x_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    PReLUOperatorTester()
      .batch_size(xnn_params.f32.prelu.mr)
      .channels(channels)
      .x_stride(123)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_OP_F32, small_batch_with_y_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    PReLUOperatorTester()
      .batch_size(xnn_params.f32.prelu.mr)
      .channels(channels)
      .y_stride(117)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_OP_F32, small_batch_with_x_stride_and_y_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    PReLUOperatorTester()
      .batch_size(xnn_params.f32.prelu.mr)
      .channels(channels)
      .x_stride(123)
      .y_stride(117)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_OP_F32, small_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    PReLUOperatorTester()
      .batch_size(xnn_params.f32.prelu.mr)
      .channels(channels)
      .qmin(128)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_OP_F32, small_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    PReLUOperatorTester()
      .batch_size(xnn_params.f32.prelu.mr)
      .channels(channels)
      .qmax(128)
      .iterations(3)
      .TestF32();
  }
}

TEST(PRELU_OP_F32, large_batch) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    PReLUOperatorTester()
      .batch_size(3 * xnn_params.f32.prelu.mr + 1)
      .channels(channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(PRELU_OP_F32, large_batch_with_x_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    PReLUOperatorTester()
      .batch_size(3 * xnn_params.f32.prelu.mr + 1)
      .channels(channels)
      .x_stride(123)
      .iterations(1)
      .TestF32();
  }
}

TEST(PRELU_OP_F32, large_batch_with_y_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    PReLUOperatorTester()
      .batch_size(3 * xnn_params.f32.prelu.mr + 1)
      .channels(channels)
      .y_stride(117)
      .iterations(1)
      .TestF32();
  }
}

TEST(PRELU_OP_F32, large_batch_with_x_stride_and_y_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    PReLUOperatorTester()
      .batch_size(3 * xnn_params.f32.prelu.mr + 1)
      .channels(channels)
      .x_stride(123)
      .y_stride(117)
      .iterations(1)
      .TestF32();
  }
}
