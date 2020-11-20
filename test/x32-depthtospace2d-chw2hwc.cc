// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/depthtospace.h>
#include "depthtospace-microkernel-tester.h"


TEST(X32_DEPTHTOSPACE_CHW2HWC__SCALAR, channels_eq_1) {
  DepthToSpaceMicrokernelTester()
    .output_channels(1)
    .Test(xnn_x32_depthtospace2d_chw2hwc_ukernel__scalar);
}

TEST(X32_DEPTHTOSPACE_CHW2HWC__SCALAR, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    DepthToSpaceMicrokernelTester()
      .output_channels(channels)
      .Test(xnn_x32_depthtospace2d_chw2hwc_ukernel__scalar);
  }
}

TEST(X32_DEPTHTOSPACE_CHW2HWC__SCALAR, non_unit_size) {
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .Test(xnn_x32_depthtospace2d_chw2hwc_ukernel__scalar);
      }
    }
  }
}

TEST(X32_DEPTHTOSPACE_CHW2HWC__SCALAR, non_unit_size_block_size_3) {
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .block_size(3)
          .Test(xnn_x32_depthtospace2d_chw2hwc_ukernel__scalar);
      }
    }
  }
}
