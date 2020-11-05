// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x32-depthtospace-chw2hwc.yaml
//   Generator: tools/generate-depthtospace-chw2hwc-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/depthtospace.h>
#include "depth-to-space-microkernel-tester.h"


TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C1, smoke) {
  DepthToSpaceMicrokernelTester()
    .output_channels(1)
    .input_height(1)
    .input_width(1)
    .block_size(2)
    .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c1);
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C1, block_size_gt_2) {
  for (size_t block_size = 2; block_size <= 8; block_size++) {
    DepthToSpaceMicrokernelTester()
      .output_channels(1)
      .input_height(1)
      .input_width(1)
      .block_size(block_size)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c1);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C1, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    DepthToSpaceMicrokernelTester()
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c1);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C1, non_unit_size) {
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c1);
      }
    }
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C1, non_unit_size_block_size_3) {
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .block_size(3)
          .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c1);
      }
    }
  }
}


TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2, smoke) {
  DepthToSpaceMicrokernelTester()
    .output_channels(1)
    .input_height(1)
    .input_width(1)
    .block_size(2)
    .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2);
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2, block_size_gt_2) {
  for (size_t block_size = 2; block_size <= 8; block_size++) {
    DepthToSpaceMicrokernelTester()
      .output_channels(1)
      .input_height(1)
      .input_width(1)
      .block_size(block_size)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    DepthToSpaceMicrokernelTester()
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2, non_unit_size) {
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2);
      }
    }
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2, non_unit_size_block_size_3) {
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .block_size(3)
          .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2);
      }
    }
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2, channels_lt_2) {
  for (size_t channels = 1; channels < 2; channels++) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2, channels_div_2) {
  for (size_t channels = 2; channels <= 4 * 2; channels += 2) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2, channels_gt_2) {
  for (size_t channels = 2 + 1; channels < 2 * 2; channels++) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2);
  }
}


TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4, smoke) {
  DepthToSpaceMicrokernelTester()
    .output_channels(1)
    .input_height(1)
    .input_width(1)
    .block_size(2)
    .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4);
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4, block_size_gt_2) {
  for (size_t block_size = 2; block_size <= 8; block_size++) {
    DepthToSpaceMicrokernelTester()
      .output_channels(1)
      .input_height(1)
      .input_width(1)
      .block_size(block_size)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    DepthToSpaceMicrokernelTester()
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4, non_unit_size) {
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4);
      }
    }
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4, non_unit_size_block_size_3) {
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .block_size(3)
          .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4);
      }
    }
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4, channels_lt_4) {
  for (size_t channels = 1; channels < 4; channels++) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4, channels_div_4) {
  for (size_t channels = 4; channels <= 4 * 4; channels += 4) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4, channels_gt_4) {
  for (size_t channels = 4 + 1; channels < 2 * 4; channels++) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4);
  }
}


TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C1_IB2, smoke) {
  DepthToSpaceMicrokernelTester()
    .output_channels(1)
    .input_height(1)
    .input_width(1)
    .block_size(2)
    .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c1_ib2);
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C1_IB2, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    DepthToSpaceMicrokernelTester()
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c1_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C1_IB2, non_unit_size) {
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c1_ib2);
      }
    }
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C1_IB2, non_unit_size_block_size_3) {
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .block_size(3)
          .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c1_ib2);
      }
    }
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C1_IB2, block_size_lt_2) {
  for (size_t block_size = 2; block_size < 2; block_size++) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .block_size(block_size)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c1_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C1_IB2, block_size_div_2) {
  for (size_t block_size = 2; block_size <= 3 * 2; block_size += 2) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .block_size(block_size)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c1_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C1_IB2, block_size_gt_2) {
  for (size_t block_size = 2 + 1; block_size < 2 * 2; block_size++) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .block_size(block_size)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c1_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2_IB2, smoke) {
  DepthToSpaceMicrokernelTester()
    .output_channels(1)
    .input_height(1)
    .input_width(1)
    .block_size(2)
    .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2_ib2);
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2_IB2, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    DepthToSpaceMicrokernelTester()
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2_IB2, non_unit_size) {
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2_ib2);
      }
    }
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2_IB2, non_unit_size_block_size_3) {
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .block_size(3)
          .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2_ib2);
      }
    }
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2_IB2, channels_lt_2) {
  for (size_t channels = 1; channels < 2; channels++) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2_IB2, channels_div_2) {
  for (size_t channels = 2; channels <= 4 * 2; channels += 2) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2_IB2, channels_gt_2) {
  for (size_t channels = 2 + 1; channels < 2 * 2; channels++) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2_IB2, block_size_lt_2) {
  for (size_t block_size = 2; block_size < 2; block_size++) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .block_size(block_size)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2_IB2, block_size_div_2) {
  for (size_t block_size = 2; block_size <= 3 * 2; block_size += 2) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .block_size(block_size)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C2_IB2, block_size_gt_2) {
  for (size_t block_size = 2 + 1; block_size < 2 * 2; block_size++) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .block_size(block_size)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c2_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4_IB2, smoke) {
  DepthToSpaceMicrokernelTester()
    .output_channels(1)
    .input_height(1)
    .input_width(1)
    .block_size(2)
    .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4_ib2);
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4_IB2, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    DepthToSpaceMicrokernelTester()
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4_IB2, non_unit_size) {
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4_ib2);
      }
    }
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4_IB2, non_unit_size_block_size_3) {
  for (size_t height = 2; height < 5; height++) {
    for (size_t width = 2; width < 5; width++) {
      for (size_t channels = 1; channels < 16; channels += 3) {
        DepthToSpaceMicrokernelTester()
          .output_channels(channels)
          .input_height(height)
          .input_width(width)
          .block_size(3)
          .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4_ib2);
      }
    }
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4_IB2, channels_lt_4) {
  for (size_t channels = 1; channels < 4; channels++) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4_IB2, channels_div_4) {
  for (size_t channels = 4; channels <= 4 * 4; channels += 4) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4_IB2, channels_gt_4) {
  for (size_t channels = 4 + 1; channels < 2 * 4; channels++) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .output_channels(channels)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4_IB2, block_size_lt_2) {
  for (size_t block_size = 2; block_size < 2; block_size++) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .block_size(block_size)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4_IB2, block_size_div_2) {
  for (size_t block_size = 2; block_size <= 3 * 2; block_size += 2) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .block_size(block_size)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4_ib2);
  }
}

TEST(X32_DEPTH_TO_SPACE_CHW2HWC__SCALAR_C4_IB2, block_size_gt_2) {
  for (size_t block_size = 2 + 1; block_size < 2 * 2; block_size++) {
    DepthToSpaceMicrokernelTester()
      .input_height(3)
      .input_width(3)
      .block_size(block_size)
      .Test(xnn_x32_depth_to_space_chw2hwc_ukernel__scalar_c4_ib2);
  }
}