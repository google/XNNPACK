// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "global-average-pooling-operator-tester.h"


TEST(GLOBAL_AVERAGE_POOLING_NCW_F32, single_channel) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  GlobalAveragePoolingOperatorTester()
    .width(27)
    .channels(1)
    .TestNCWxF32();
}

TEST(GLOBAL_AVERAGE_POOLING_NCW_F32, varying_channels) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 2; channels <= 16; channels += 3) {
    GlobalAveragePoolingOperatorTester()
      .width(27)
      .channels(channels)
      .TestNCWxF32();
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NCW_F32, varying_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t width = 25; width <= 31; width++) {
    GlobalAveragePoolingOperatorTester()
      .width(width)
      .channels(19)
      .TestNCWxF32();
  }
}

TEST(GLOBAL_AVERAGE_POOLING_NCW_F32, qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  GlobalAveragePoolingOperatorTester()
    .width(27)
    .channels(19)
    .qmin(128)
    .TestNCWxF32();
}

TEST(GLOBAL_AVERAGE_POOLING_NCW_F32, qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  GlobalAveragePoolingOperatorTester()
    .width(27)
    .channels(19)
    .qmax(128)
    .TestNCWxF32();
}
