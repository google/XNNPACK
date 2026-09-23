// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "test/operators/depth-to-space-operator-tester.h"

TEST(DEPTH_TO_SPACE_NCHW2NHWC_X16, one_pixel) {
  DepthToSpaceOperatorTester()
      .input_size(1, 1)
      .block_size(3)
      .output_channels(17)
      .TestNCHW2NHWCxX16();
}

TEST(DEPTH_TO_SPACE_NCHW2NHWC_X16, one_column) {
  for (size_t input_height = 1; input_height <= 7; input_height++) {
    DepthToSpaceOperatorTester()
        .input_size(input_height, 1)
        .block_size(3)
        .output_channels(17)
        .TestNCHW2NHWCxX16();
  }
}

TEST(DEPTH_TO_SPACE_NCHW2NHWC_X16, one_row) {
  for (size_t input_width = 1; input_width <= 7; input_width++) {
    DepthToSpaceOperatorTester()
        .input_size(1, input_width)
        .block_size(3)
        .output_channels(17)
        .TestNCHW2NHWCxX16();
  }
}

TEST(DEPTH_TO_SPACE_NCHW2NHWC_X16, varying_input_size) {
  for (size_t input_height = 1; input_height <= 5; input_height++) {
    for (size_t input_width = 1; input_width <= 5; input_width++) {
      DepthToSpaceOperatorTester()
          .input_size(input_height, input_width)
          .block_size(3)
          .output_channels(17)
          .TestNCHW2NHWCxX16();
    }
  }
}

TEST(DEPTH_TO_SPACE_NCHW2NHWC_X16, varying_block_size) {
  for (uint32_t block_size = 2; block_size <= 5; block_size++) {
    DepthToSpaceOperatorTester()
        .input_size(7, 5)
        .block_size(block_size)
        .output_channels(17)
        .TestNCHW2NHWCxX16();
  }
}

TEST(DEPTH_TO_SPACE_NCHW2NHWC_X16, varying_output_channels) {
  for (size_t output_channels = 1; output_channels <= 15; output_channels++) {
    DepthToSpaceOperatorTester()
        .input_size(7, 5)
        .block_size(3)
        .output_channels(output_channels)
        .TestNCHW2NHWCxX16();
  }
}

TEST(DEPTH_TO_SPACE_NCHW2NHWC_X16, varying_batch_size) {
  for (size_t batch_size = 2; batch_size <= 3; batch_size++) {
    DepthToSpaceOperatorTester()
        .batch_size(batch_size)
        .input_size(7, 5)
        .block_size(3)
        .output_channels(17)
        .TestNCHW2NHWCxX16();
  }
}

TEST(DEPTH_TO_SPACE_NCHW2NHWC_X32, one_pixel) {
  DepthToSpaceOperatorTester()
      .input_size(1, 1)
      .block_size(3)
      .output_channels(17)
      .TestNCHW2NHWCxX32();
}

TEST(DEPTH_TO_SPACE_NCHW2NHWC_X32, one_column) {
  for (size_t input_height = 1; input_height <= 7; input_height++) {
    DepthToSpaceOperatorTester()
        .input_size(input_height, 1)
        .block_size(3)
        .output_channels(17)
        .TestNCHW2NHWCxX32();
  }
}

TEST(DEPTH_TO_SPACE_NCHW2NHWC_X32, one_row) {
  for (size_t input_width = 1; input_width <= 7; input_width++) {
    DepthToSpaceOperatorTester()
        .input_size(1, input_width)
        .block_size(3)
        .output_channels(17)
        .TestNCHW2NHWCxX32();
  }
}

TEST(DEPTH_TO_SPACE_NCHW2NHWC_X32, varying_input_size) {
  for (size_t input_height = 1; input_height <= 5; input_height++) {
    for (size_t input_width = 1; input_width <= 5; input_width++) {
      DepthToSpaceOperatorTester()
          .input_size(input_height, input_width)
          .block_size(3)
          .output_channels(17)
          .TestNCHW2NHWCxX32();
    }
  }
}

TEST(DEPTH_TO_SPACE_NCHW2NHWC_X32, varying_block_size) {
  for (uint32_t block_size = 2; block_size <= 5; block_size++) {
    DepthToSpaceOperatorTester()
        .input_size(7, 5)
        .block_size(block_size)
        .output_channels(17)
        .TestNCHW2NHWCxX32();
  }
}

TEST(DEPTH_TO_SPACE_NCHW2NHWC_X32, varying_output_channels) {
  for (size_t output_channels = 1; output_channels <= 15; output_channels++) {
    DepthToSpaceOperatorTester()
        .input_size(7, 5)
        .block_size(3)
        .output_channels(output_channels)
        .TestNCHW2NHWCxX32();
  }
}

TEST(DEPTH_TO_SPACE_NCHW2NHWC_X32, varying_batch_size) {
  for (size_t batch_size = 2; batch_size <= 3; batch_size++) {
    DepthToSpaceOperatorTester()
        .batch_size(batch_size)
        .input_size(7, 5)
        .block_size(3)
        .output_channels(17)
        .TestNCHW2NHWCxX32();
  }
}

TEST(DEPTH_TO_SPACE_NCHW2NHWC_X32, overflow_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  xnn_operator_t depth_to_space_op = nullptr;

  const uint32_t block_size = 2;
  ASSERT_EQ(xnn_status_success,
            xnn_create_depth_to_space_nchw2nhwc_x32(
                block_size, 0, &depth_to_space_op));
  ASSERT_NE(nullptr, depth_to_space_op);

  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      depth_to_space_op, xnn_delete_operator);

  // Normal control case: must succeed.
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_depth_to_space_nchw2nhwc_x32(
                depth_to_space_op, /*batch_size=*/1, /*input_height=*/4,
                /*input_width=*/4, /*input_channels=*/4,
                /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                /*output_channels_out=*/nullptr, /*threadpool=*/nullptr));

  // Overflow case: input_height * output_stride_1 overflows size_t.
  // input_width = (SIZE_MAX / 4) + 1, output_channels = 1.
  // output_stride[2] = ((SIZE_MAX / 4) + 1) * 2 = (SIZE_MAX / 2) + 2.
  // output_stride[1] = 2 * ((SIZE_MAX / 2) + 2) overflows size_t.
  const size_t large_width = (SIZE_MAX / 4) + 1;
  const size_t channels = 4; // block_size^2 * output_channels (2^2 * 1)

  ASSERT_EQ(xnn_status_invalid_parameter,
            xnn_reshape_depth_to_space_nchw2nhwc_x32(
                depth_to_space_op, /*batch_size=*/1, /*input_height=*/1,
                large_width, channels, /*output_height_out=*/nullptr,
                /*output_width_out=*/nullptr, /*output_channels_out=*/nullptr,
                /*threadpool=*/nullptr));
}
