// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "depth-to-space-operator-tester.h"

#include <gtest/gtest.h>

TEST(RESIZE_DEPTH_TO_SPACE_NCHW2NHWC_X32, one_column) {
    for (size_t input_height = 1; input_height <= 3; input_height++) {
      for (size_t block_size = 2; block_size <= 5; block_size++) {
        for (size_t output_channels = 1; output_channels <= 7; output_channels++) {
          DepthToSpaceOperatorTester()
            .input_size(input_height, 1)
            .block_size(block_size)
            .output_channels(output_channels)
            .iterations(3)
            .TestNCHW2NHWCxF32();
        }
      }
    }
}

TEST(RESIZE_DEPTH_TO_SPACE_NCHW2NHWC_X32, one_row) {
    for (size_t input_width = 1; input_width <= 3; input_width++) {
      for (size_t block_size = 2; block_size <= 5; block_size++) {
        for (size_t output_channels = 1; output_channels <= 3; output_channels++) {
          DepthToSpaceOperatorTester()
            .input_size(1, input_width)
            .block_size(block_size)
            .output_channels(output_channels)
            .iterations(3)
            .TestNCHW2NHWCxF32();
        }
      }
    }
}

TEST(RESIZE_DEPTH_TO_SPACE_NCHW2NHWC_X32, varying_input_size) {
    for (size_t input_height = 1; input_height <= 7; input_height++) {
       for (size_t input_width = 1; input_width <= 7; input_width++) {
         for (size_t block_size = 2; block_size <= 5; block_size++) {
           for (size_t output_channels = 1; output_channels <= 3; output_channels++) {
             DepthToSpaceOperatorTester()
               .input_size(input_height, input_width)
               .block_size(block_size)
               .output_channels(output_channels)
               .iterations(3)
               .TestNCHW2NHWCxF32();
           }
         }
       }
    }
}

TEST(RESIZE_DEPTH_TO_SPACE_NCHW2NHWC_X32, varying_batch_size) {
  for (size_t batch_size = 2; batch_size <= 3; batch_size++) {
    for (size_t input_size = 2; input_size <= 6; input_size += 2) {
      for (size_t block_size = 2; block_size <= 6; block_size += 2) {
        for (size_t output_channels = 1; output_channels <= 3; output_channels++) {
          DepthToSpaceOperatorTester()
            .batch_size(batch_size)
            .input_size(input_size, input_size)
            .block_size(block_size)
            .output_channels(output_channels)
            .iterations(3)
            .TestNCHW2NHWCxF32();
        }
      }
    }
  }
}
