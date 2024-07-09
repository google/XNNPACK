// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include "space-to-depth-operator-tester.h"

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
