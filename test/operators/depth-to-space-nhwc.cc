// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include "test/operators/depth-to-space-operator-tester.h"

TEST(DEPTH_TO_SPACE_NHWC_X8, one_pixel) {
  DepthToSpaceOperatorTester()
      .input_size(1, 1)
      .block_size(3)
      .output_channels(17)
      .TestNHWCxX8();
}

TEST(DEPTH_TO_SPACE_NHWC_X8, one_column) {
  for (size_t input_height = 2; input_height <= 7; input_height++) {
    DepthToSpaceOperatorTester()
        .input_size(input_height, 1)
        .block_size(3)
        .output_channels(17)
        .TestNHWCxX8();
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X8, one_row) {
  for (size_t input_width = 2; input_width <= 7; input_width++) {
    DepthToSpaceOperatorTester()
        .input_size(1, input_width)
        .block_size(3)
        .output_channels(17)
        .TestNHWCxX8();
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X8, varying_input_size) {
  for (size_t input_height = 1; input_height <= 5; input_height++) {
    for (size_t input_width = 1; input_width <= 5; input_width++) {
      DepthToSpaceOperatorTester()
          .input_size(input_height, input_width)
          .block_size(3)
          .output_channels(17)
          .TestNHWCxX8();
    }
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X8, varying_block_size) {
  for (uint32_t block_size = 2; block_size <= 5; block_size++) {
    DepthToSpaceOperatorTester()
        .input_size(7, 5)
        .block_size(block_size)
        .output_channels(17)
        .TestNHWCxX8();
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X8, varying_output_channels) {
  for (size_t output_channels = 1; output_channels <= 15; output_channels++) {
    DepthToSpaceOperatorTester()
        .input_size(7, 5)
        .block_size(3)
        .output_channels(output_channels)
        .TestNHWCxX8();
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X8, varying_batch_size) {
  for (size_t batch_size = 2; batch_size <= 3; batch_size++) {
    DepthToSpaceOperatorTester()
        .batch_size(batch_size)
        .input_size(7, 5)
        .block_size(3)
        .output_channels(17)
        .TestNHWCxX8();
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X16, one_pixel) {
  DepthToSpaceOperatorTester()
      .input_size(1, 1)
      .block_size(3)
      .output_channels(17)
      .TestNHWCxX16();
}

TEST(DEPTH_TO_SPACE_NHWC_X16, one_column) {
  for (size_t input_height = 2; input_height <= 7; input_height++) {
    DepthToSpaceOperatorTester()
        .input_size(input_height, 1)
        .block_size(3)
        .output_channels(17)
        .TestNHWCxX16();
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X16, one_row) {
  for (size_t input_width = 2; input_width <= 7; input_width++) {
    DepthToSpaceOperatorTester()
        .input_size(1, input_width)
        .block_size(3)
        .output_channels(17)
        .TestNHWCxX16();
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X16, varying_input_size) {
  for (size_t input_height = 1; input_height <= 5; input_height++) {
    for (size_t input_width = 1; input_width <= 5; input_width++) {
      DepthToSpaceOperatorTester()
          .input_size(input_height, input_width)
          .block_size(3)
          .output_channels(17)
          .TestNHWCxX16();
    }
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X16, varying_block_size) {
  for (uint32_t block_size = 2; block_size <= 5; block_size++) {
    DepthToSpaceOperatorTester()
        .input_size(7, 5)
        .block_size(block_size)
        .output_channels(17)
        .TestNHWCxX16();
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X16, varying_output_channels) {
  for (size_t output_channels = 1; output_channels <= 15; output_channels++) {
    DepthToSpaceOperatorTester()
        .input_size(7, 5)
        .block_size(3)
        .output_channels(output_channels)
        .TestNHWCxX16();
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X16, varying_batch_size) {
  for (size_t batch_size = 2; batch_size <= 3; batch_size++) {
    DepthToSpaceOperatorTester()
        .batch_size(batch_size)
        .input_size(7, 5)
        .block_size(3)
        .output_channels(17)
        .TestNHWCxX16();
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X32, one_pixel) {
  DepthToSpaceOperatorTester()
      .input_size(1, 1)
      .block_size(3)
      .output_channels(17)
      .TestNHWCxX32();
}

TEST(DEPTH_TO_SPACE_NHWC_X32, one_column) {
  for (size_t input_height = 2; input_height <= 7; input_height++) {
    DepthToSpaceOperatorTester()
        .input_size(input_height, 1)
        .block_size(3)
        .output_channels(17)
        .TestNHWCxX32();
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X32, one_row) {
  for (size_t input_width = 2; input_width <= 7; input_width++) {
    DepthToSpaceOperatorTester()
        .input_size(1, input_width)
        .block_size(3)
        .output_channels(17)
        .TestNHWCxX32();
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X32, varying_input_size) {
  for (size_t input_height = 1; input_height <= 5; input_height++) {
    for (size_t input_width = 1; input_width <= 5; input_width++) {
      DepthToSpaceOperatorTester()
          .input_size(input_height, input_width)
          .block_size(3)
          .output_channels(17)
          .TestNHWCxX32();
    }
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X32, varying_block_size) {
  for (uint32_t block_size = 2; block_size <= 5; block_size++) {
    DepthToSpaceOperatorTester()
        .input_size(7, 5)
        .block_size(block_size)
        .output_channels(17)
        .TestNHWCxX32();
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X32, varying_output_channels) {
  for (size_t output_channels = 1; output_channels <= 15; output_channels++) {
    DepthToSpaceOperatorTester()
        .input_size(7, 5)
        .block_size(3)
        .output_channels(output_channels)
        .TestNHWCxX32();
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X32, varying_batch_size) {
  for (size_t batch_size = 2; batch_size <= 3; batch_size++) {
    DepthToSpaceOperatorTester()
        .batch_size(batch_size)
        .input_size(7, 5)
        .block_size(3)
        .output_channels(17)
        .TestNHWCxX32();
  }
}

TEST(DEPTH_TO_SPACE_NHWC_X32, zero_batch_populates_output_dimensions) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t depth_to_space_op = nullptr;
  const uint32_t block_size = 2;
  ASSERT_EQ(xnn_status_success,
            xnn_create_depth_to_space_nhwc_x32(
                block_size, 0, &depth_to_space_op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      depth_to_space_op, xnn_delete_operator);

  size_t out_h = 0, out_w = 0, out_c = 0;
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_depth_to_space_nhwc_x32(
                depth_to_space_op, /*batch_size=*/0, /*input_height=*/2,
                /*input_width=*/3, /*input_channels=*/12, &out_h, &out_w,
                &out_c, /*threadpool=*/nullptr));
  EXPECT_EQ(out_h, 4);
  EXPECT_EQ(out_w, 6);
  EXPECT_EQ(out_c, 3);
}

