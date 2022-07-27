// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/s16-rmaxabs.yaml
//   Generator: tools/generate-rmaxabs-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/rmaxabs.h>
#include "rmaxabs-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_RMAXABS__NEON_X8, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    RMaxAbsMicrokernelTester()
      .channels(8)
      .Test(xnn_s16_rmaxabs_ukernel__neon_x8);
  }

  TEST(S16_RMAXABS__NEON_X8, channels_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 80; channels += 8) {
      RMaxAbsMicrokernelTester()
        .channels(channels)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x8);
    }
  }

  TEST(S16_RMAXABS__NEON_X8, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      RMaxAbsMicrokernelTester()
        .channels(channels)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x8);
    }
  }

  TEST(S16_RMAXABS__NEON_X8, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      RMaxAbsMicrokernelTester()
        .channels(channels)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_RMAXABS__NEON_X16, channels_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RMaxAbsMicrokernelTester()
      .channels(16)
      .Test(xnn_s16_rmaxabs_ukernel__neon_x16);
  }

  TEST(S16_RMAXABS__NEON_X16, channels_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 160; channels += 16) {
      RMaxAbsMicrokernelTester()
        .channels(channels)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x16);
    }
  }

  TEST(S16_RMAXABS__NEON_X16, channels_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      RMaxAbsMicrokernelTester()
        .channels(channels)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x16);
    }
  }

  TEST(S16_RMAXABS__NEON_X16, channels_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      RMaxAbsMicrokernelTester()
        .channels(channels)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_RMAXABS__NEON_X24, channels_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    RMaxAbsMicrokernelTester()
      .channels(24)
      .Test(xnn_s16_rmaxabs_ukernel__neon_x24);
  }

  TEST(S16_RMAXABS__NEON_X24, channels_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 48; channels < 240; channels += 24) {
      RMaxAbsMicrokernelTester()
        .channels(channels)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x24);
    }
  }

  TEST(S16_RMAXABS__NEON_X24, channels_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 24; channels++) {
      RMaxAbsMicrokernelTester()
        .channels(channels)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x24);
    }
  }

  TEST(S16_RMAXABS__NEON_X24, channels_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 25; channels < 48; channels++) {
      RMaxAbsMicrokernelTester()
        .channels(channels)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_RMAXABS__NEON_X32, channels_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    RMaxAbsMicrokernelTester()
      .channels(32)
      .Test(xnn_s16_rmaxabs_ukernel__neon_x32);
  }

  TEST(S16_RMAXABS__NEON_X32, channels_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 320; channels += 32) {
      RMaxAbsMicrokernelTester()
        .channels(channels)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x32);
    }
  }

  TEST(S16_RMAXABS__NEON_X32, channels_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      RMaxAbsMicrokernelTester()
        .channels(channels)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x32);
    }
  }

  TEST(S16_RMAXABS__NEON_X32, channels_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      RMaxAbsMicrokernelTester()
        .channels(channels)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


TEST(S16_RMAXABS__SCALAR_X1, channels_eq_1) {
  RMaxAbsMicrokernelTester()
    .channels(1)
    .Test(xnn_s16_rmaxabs_ukernel__scalar_x1);
}

TEST(S16_RMAXABS__SCALAR_X1, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    RMaxAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x1);
  }
}


TEST(S16_RMAXABS__SCALAR_X2, channels_eq_2) {
  RMaxAbsMicrokernelTester()
    .channels(2)
    .Test(xnn_s16_rmaxabs_ukernel__scalar_x2);
}

TEST(S16_RMAXABS__SCALAR_X2, channels_div_2) {
  for (size_t channels = 4; channels < 20; channels += 2) {
    RMaxAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x2);
  }
}

TEST(S16_RMAXABS__SCALAR_X2, channels_lt_2) {
  for (size_t channels = 1; channels < 2; channels++) {
    RMaxAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x2);
  }
}

TEST(S16_RMAXABS__SCALAR_X2, channels_gt_2) {
  for (size_t channels = 3; channels < 4; channels++) {
    RMaxAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x2);
  }
}


TEST(S16_RMAXABS__SCALAR_X3, channels_eq_3) {
  RMaxAbsMicrokernelTester()
    .channels(3)
    .Test(xnn_s16_rmaxabs_ukernel__scalar_x3);
}

TEST(S16_RMAXABS__SCALAR_X3, channels_div_3) {
  for (size_t channels = 6; channels < 30; channels += 3) {
    RMaxAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x3);
  }
}

TEST(S16_RMAXABS__SCALAR_X3, channels_lt_3) {
  for (size_t channels = 1; channels < 3; channels++) {
    RMaxAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x3);
  }
}

TEST(S16_RMAXABS__SCALAR_X3, channels_gt_3) {
  for (size_t channels = 4; channels < 6; channels++) {
    RMaxAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x3);
  }
}


TEST(S16_RMAXABS__SCALAR_X4, channels_eq_4) {
  RMaxAbsMicrokernelTester()
    .channels(4)
    .Test(xnn_s16_rmaxabs_ukernel__scalar_x4);
}

TEST(S16_RMAXABS__SCALAR_X4, channels_div_4) {
  for (size_t channels = 8; channels < 40; channels += 4) {
    RMaxAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x4);
  }
}

TEST(S16_RMAXABS__SCALAR_X4, channels_lt_4) {
  for (size_t channels = 1; channels < 4; channels++) {
    RMaxAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x4);
  }
}

TEST(S16_RMAXABS__SCALAR_X4, channels_gt_4) {
  for (size_t channels = 5; channels < 8; channels++) {
    RMaxAbsMicrokernelTester()
      .channels(channels)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x4);
  }
}
