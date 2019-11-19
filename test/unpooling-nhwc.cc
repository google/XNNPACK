// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "unpooling-operator-tester.h"


TEST(UNPOOLING_NHWC_X32, unit_height_horizontal_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pooling_size : std::vector<size_t>{{2, 3, 5, 7}}) {
      UnpoolingOperatorTester()
        .batch_size(1)
        .input_height(1)
        .input_width(7)
        .pooling_height(1)
        .pooling_width(pooling_size)
        .channels(channels)
        .TestX32();
    }
  }
}

TEST(UNPOOLING_NHWC_X32, unit_height_horizontal_pool_with_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pooling_size : std::vector<size_t>{{3, 5, 7}}) {
      for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          UnpoolingOperatorTester()
            .batch_size(1)
            .input_height(1)
            .input_width(7)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(1)
            .pooling_width(pooling_size)
            .channels(channels)
            .TestX32();
        }
      }
    }
  }
}

TEST(UNPOOLING_NHWC_X32, unit_height_vertical_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pooling_size : std::vector<size_t>{{2, 3, 5, 7}}) {
      UnpoolingOperatorTester()
        .batch_size(1)
        .input_height(7)
        .input_width(1)
        .pooling_height(pooling_size)
        .pooling_width(1)
        .channels(channels)
        .TestX32();
    }
  }
}

TEST(UNPOOLING_NHWC_X32, unit_height_vertical_pool_with_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pooling_size : std::vector<size_t>{{3, 5, 7}}) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          UnpoolingOperatorTester()
            .batch_size(1)
            .input_height(7)
            .input_width(1)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pooling_size)
            .pooling_width(1)
            .channels(channels)
            .TestX32();
        }
      }
    }
  }
}

TEST(UNPOOLING_NHWC_X32, unit_height_square_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pooling_size : std::vector<size_t>{{2, 3, 5}}) {
      UnpoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(4)
        .pooling_height(pooling_size)
        .pooling_width(pooling_size)
        .channels(channels)
        .TestX32();
    }
  }
}

TEST(UNPOOLING_NHWC_X32, unit_height_3x3_pool_with_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
            UnpoolingOperatorTester()
              .batch_size(1)
              .input_height(2)
              .input_width(4)
              .padding_left(padding_left)
              .padding_top(padding_top)
              .padding_right(padding_right)
              .padding_bottom(padding_bottom)
              .pooling_height(3)
              .pooling_width(3)
              .channels(channels)
              .iterations(3)
              .TestX32();
          }
        }
      }
    }
  }
}

TEST(UNPOOLING_NHWC_X32, small_height_square_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pooling_size : std::vector<size_t>{{2, 3, 5}}) {
      UnpoolingOperatorTester()
        .batch_size(1)
        .input_height(4)
        .input_width(7)
        .pooling_height(pooling_size)
        .pooling_width(pooling_size)
        .channels(channels)
        .TestX32();
    }
  }
}

TEST(UNPOOLING_NHWC_X32, small_height_3x3_pool_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    UnpoolingOperatorTester()
      .batch_size(1)
      .input_height(4)
      .input_width(7)
      .pooling_height(3)
      .pooling_width(3)
      .input_pixel_stride(channels * 2 + 1)
      .channels(channels)
      .TestX32();
  }
}

TEST(UNPOOLING_NHWC_X32, small_height_3x3_pool_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    UnpoolingOperatorTester()
      .batch_size(1)
      .input_height(4)
      .input_width(7)
      .pooling_height(3)
      .pooling_width(3)
      .output_pixel_stride(channels * 2 + 3)
      .channels(channels)
      .TestX32();
  }
}

TEST(UNPOOLING_NHWC_X32, small_height_3x3_pool_with_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 50; channels += 15) {
    for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
            UnpoolingOperatorTester()
              .batch_size(1)
              .input_height(4)
              .input_width(7)
              .pooling_height(3)
              .pooling_width(3)
              .channels(channels)
              .iterations(1)
              .TestX32();
          }
        }
      }
    }
  }
}

TEST(UNPOOLING_NHWC_X32, small_height_and_batch_square_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pooling_size : std::vector<size_t>{{2, 3}}) {
      UnpoolingOperatorTester()
        .batch_size(3)
        .input_height(4)
        .input_width(7)
        .pooling_height(pooling_size)
        .pooling_width(pooling_size)
        .channels(channels)
        .TestX32();
    }
  }
}

TEST(UNPOOLING_NHWC_X32, setup_increasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  UnpoolingOperatorTester()
    .batch_size(3)
    .next_batch_size(5)
    .input_height(4)
    .input_width(4)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupX32();
}

TEST(UNPOOLING_NHWC_X32, setup_decreasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  UnpoolingOperatorTester()
    .batch_size(5)
    .next_batch_size(3)
    .input_height(4)
    .input_width(4)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupX32();
}

TEST(UNPOOLING_NHWC_X32, setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  UnpoolingOperatorTester()
    .batch_size(3)
    .input_height(4)
    .input_width(4)
    .next_input_height(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupX32();
  UnpoolingOperatorTester()
    .batch_size(3)
    .input_height(4)
    .input_width(4)
    .next_input_height(7)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupX32();
}

TEST(UNPOOLING_NHWC_X32, setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  UnpoolingOperatorTester()
    .batch_size(3)
    .input_height(4)
    .input_width(4)
    .next_input_width(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupX32();
  UnpoolingOperatorTester()
    .batch_size(3)
    .input_height(4)
    .input_width(4)
    .next_input_width(7)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupX32();
}

TEST(UNPOOLING_NHWC_X32, setup_swap_height_and_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  UnpoolingOperatorTester()
    .batch_size(3)
    .input_height(5)
    .input_width(4)
    .next_input_height(4)
    .next_input_width(5)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupX32();
}
