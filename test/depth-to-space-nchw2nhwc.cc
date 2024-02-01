// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "depth-to-space-operator-tester.h"

#include <gtest/gtest.h>


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
