// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "global-average-pooling-spnchw-operator-tester.h"

#include <xnnpack/params.h>


TEST(GLOBAL_AVERAGE_POOLING_SPNCHW_OP_F32, single_channel) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  GlobalAveragePoolingSpNCHWOperatorTester()
    .height(29)
    .width(27)
    .channels(1)
    .TestF32();
}

TEST(GLOBAL_AVERAGE_POOLING_SPNCHW_OP_F32, varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t channels = 2; channels <= 16; channels += 3) {
    GlobalAveragePoolingSpNCHWOperatorTester()
      .height(29)
      .width(27)
      .channels(channels)
      .TestF32();
  }
}

TEST(GLOBAL_AVERAGE_POOLING_SPNCHW_OP_F32, varying_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t height = 27; height <= 33; height++) {
    GlobalAveragePoolingSpNCHWOperatorTester()
      .height(height)
      .width(27)
      .channels(19)
      .TestF32();
  }
}

TEST(GLOBAL_AVERAGE_POOLING_SPNCHW_OP_F32, varying_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  for (size_t width = 25; width <= 31; width++) {
    GlobalAveragePoolingSpNCHWOperatorTester()
      .height(29)
      .width(width)
      .channels(19)
      .TestF32();
  }
}

TEST(GLOBAL_AVERAGE_POOLING_SPNCHW_OP_F32, qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  GlobalAveragePoolingSpNCHWOperatorTester()
    .height(29)
    .width(27)
    .channels(19)
    .qmin(128)
    .TestF32();
}

TEST(GLOBAL_AVERAGE_POOLING_SPNCHW_OP_F32, qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize());
  GlobalAveragePoolingSpNCHWOperatorTester()
    .height(29)
    .width(27)
    .channels(19)
    .qmax(128)
    .TestF32();
}
