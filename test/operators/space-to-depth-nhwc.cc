// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "test/operators/space-to-depth-operator-tester.h"

TEST(SPACE_TO_DEPTH_NHWC_X8, one_output_pixel) {
  size_t block_size = 3;
  SpaceToDepthOperatorTester()
      .input_size(block_size, block_size)
      .block_size(block_size)
      .input_channels(17)
      .TestNHWCxX8();
}

TEST(SPACE_TO_DEPTH_NHWC_X8, one_column) {
  size_t block_size = 3;
  for (size_t input_height = 2; input_height <= 7; input_height++) {
    SpaceToDepthOperatorTester()
        .input_size(input_height * block_size, block_size)
        .block_size(block_size)
        .input_channels(17)
        .TestNHWCxX8();
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X8, one_row) {
  size_t block_size = 3;
  for (size_t input_width = 2; input_width <= 7; input_width++) {
    SpaceToDepthOperatorTester()
        .input_size(block_size, input_width * block_size)
        .block_size(block_size)
        .input_channels(17)
        .TestNHWCxX8();
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X8, varying_input_size) {
  size_t block_size = 3;
  for (size_t input_height = 1; input_height <= 5; input_height++) {
    for (size_t input_width = 1; input_width <= 5; input_width++) {
      SpaceToDepthOperatorTester()
          .input_size(input_height * block_size, input_width * block_size)
          .block_size(block_size)
          .input_channels(17)
          .TestNHWCxX8();
    }
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X8, varying_block_size) {
  for (uint32_t block_size = 2; block_size <= 5; block_size++) {
    SpaceToDepthOperatorTester()
        .input_size(7 * block_size, 5 * block_size)
        .block_size(block_size)
        .input_channels(17)
        .TestNHWCxX8();
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X8, varying_input_channels) {
  size_t block_size = 3;
  for (size_t input_channels = 1; input_channels <= 15; input_channels++) {
    SpaceToDepthOperatorTester()
        .input_size(7 * block_size, 5 * block_size)
        .block_size(block_size)
        .input_channels(input_channels)
        .TestNHWCxX8();
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X8, varying_batch_size) {
  size_t block_size = 3;
  for (size_t batch_size = 2; batch_size <= 3; batch_size++) {
    SpaceToDepthOperatorTester()
        .batch_size(batch_size)
        .input_size(7 * block_size, 5 * block_size)
        .block_size(block_size)
        .input_channels(17)
        .TestNHWCxX8();
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X16, one_output_pixel) {
  size_t block_size = 3;
  SpaceToDepthOperatorTester()
      .input_size(block_size, block_size)
      .block_size(block_size)
      .input_channels(17)
      .TestNHWCxX16();
}

TEST(SPACE_TO_DEPTH_NHWC_X16, one_column) {
  size_t block_size = 3;
  for (size_t input_height = 2; input_height <= 7; input_height++) {
    SpaceToDepthOperatorTester()
        .input_size(input_height * block_size, block_size)
        .block_size(block_size)
        .input_channels(17)
        .TestNHWCxX16();
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X16, one_row) {
  size_t block_size = 3;
  for (size_t input_width = 2; input_width <= 7; input_width++) {
    SpaceToDepthOperatorTester()
        .input_size(block_size, input_width * block_size)
        .block_size(block_size)
        .input_channels(17)
        .TestNHWCxX16();
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X16, varying_input_size) {
  size_t block_size = 3;
  for (size_t input_height = 1; input_height <= 5; input_height++) {
    for (size_t input_width = 1; input_width <= 5; input_width++) {
      SpaceToDepthOperatorTester()
          .input_size(input_height * block_size, input_width * block_size)
          .block_size(block_size)
          .input_channels(17)
          .TestNHWCxX16();
    }
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X16, varying_block_size) {
  for (uint32_t block_size = 2; block_size <= 5; block_size++) {
    SpaceToDepthOperatorTester()
        .input_size(7 * block_size, 5 * block_size)
        .block_size(block_size)
        .input_channels(17)
        .TestNHWCxX16();
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X16, varying_input_channels) {
  size_t block_size = 3;
  for (size_t input_channels = 1; input_channels <= 15; input_channels++) {
    SpaceToDepthOperatorTester()
        .input_size(7 * block_size, 5 * block_size)
        .block_size(block_size)
        .input_channels(input_channels)
        .TestNHWCxX16();
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X16, varying_batch_size) {
  size_t block_size = 3;
  for (size_t batch_size = 2; batch_size <= 3; batch_size++) {
    SpaceToDepthOperatorTester()
        .batch_size(batch_size)
        .input_size(7 * block_size, 5 * block_size)
        .block_size(block_size)
        .input_channels(17)
        .TestNHWCxX32();
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X32, one_output_pixel) {
  size_t block_size = 3;
  SpaceToDepthOperatorTester()
      .input_size(block_size, block_size)
      .block_size(block_size)
      .input_channels(17)
      .TestNHWCxX32();
}

TEST(SPACE_TO_DEPTH_NHWC_X32, one_column) {
  size_t block_size = 3;
  for (size_t input_height = 2; input_height <= 7; input_height++) {
    SpaceToDepthOperatorTester()
        .input_size(input_height * block_size, block_size)
        .block_size(block_size)
        .input_channels(17)
        .TestNHWCxX32();
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X32, one_row) {
  size_t block_size = 3;
  for (size_t input_width = 2; input_width <= 7; input_width++) {
    SpaceToDepthOperatorTester()
        .input_size(block_size, input_width * block_size)
        .block_size(block_size)
        .input_channels(17)
        .TestNHWCxX32();
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X32, varying_input_size) {
  size_t block_size = 3;
  for (size_t input_height = 1; input_height <= 5; input_height++) {
    for (size_t input_width = 1; input_width <= 5; input_width++) {
      SpaceToDepthOperatorTester()
          .input_size(input_height * block_size, input_width * block_size)
          .block_size(block_size)
          .input_channels(17)
          .TestNHWCxX32();
    }
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X32, varying_block_size) {
  for (uint32_t block_size = 2; block_size <= 5; block_size++) {
    SpaceToDepthOperatorTester()
        .input_size(7 * block_size, 5 * block_size)
        .block_size(block_size)
        .input_channels(17)
        .TestNHWCxX32();
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X32, varying_input_channels) {
  size_t block_size = 3;
  for (size_t input_channels = 1; input_channels <= 15; input_channels++) {
    SpaceToDepthOperatorTester()
        .input_size(7 * block_size, 5 * block_size)
        .block_size(block_size)
        .input_channels(input_channels)
        .TestNHWCxX32();
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X32, varying_batch_size) {
  size_t block_size = 3;
  for (size_t batch_size = 2; batch_size <= 3; batch_size++) {
    SpaceToDepthOperatorTester()
        .batch_size(batch_size)
        .input_size(7 * block_size, 5 * block_size)
        .block_size(block_size)
        .input_channels(17)
        .TestNHWCxX32();
  }
}

TEST(SPACE_TO_DEPTH_NHWC_X32, overflow_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  xnn_operator_t space_to_depth_op = nullptr;

  const uint32_t block_size = 2;
  ASSERT_EQ(xnn_status_success,
            xnn_create_space_to_depth_nhwc_x32(
                block_size, 0, &space_to_depth_op));
  ASSERT_NE(nullptr, space_to_depth_op);

  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      space_to_depth_op, xnn_delete_operator);

  // Normal control case: must succeed.
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_space_to_depth_nhwc_x32(
                space_to_depth_op, /*batch_size=*/1, /*input_height=*/4,
                /*input_width=*/4, /*input_channels=*/2,
                /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                /*output_channels_out=*/nullptr, /*threadpool=*/nullptr));

  // Overflow case:
  // input_channels = 2, so output_channels = 2 * 2 * 2 = 8.
  // input_height = 2 (divisible by block_size 2).
  // input_width = (SIZE_MAX / 2) + 2 (even number).
  // input_row_stride = input_width * input_channels = ((SIZE_MAX / 2) + 2) * 2
  // which overflows size_t.
  const size_t large_width = (SIZE_MAX / 2) + 2;
  const size_t height = 2;
  const size_t channels = 2;

  ASSERT_EQ(xnn_status_invalid_parameter,
            xnn_reshape_space_to_depth_nhwc_x32(
                space_to_depth_op, /*batch_size=*/1, height, large_width,
                channels, /*output_height_out=*/nullptr,
                /*output_width_out=*/nullptr, /*output_channels_out=*/nullptr,
                /*threadpool=*/nullptr));
}
