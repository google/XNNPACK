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
  TEST(S16_RMAXABS__NEON_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    RMaxAbsMicrokernelTester()
      .batch(8)
      .Test(xnn_s16_rmaxabs_ukernel__neon_x8);
  }

  TEST(S16_RMAXABS__NEON_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 16; batch < 80; batch += 8) {
      RMaxAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x8);
    }
  }

  TEST(S16_RMAXABS__NEON_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 8; batch++) {
      RMaxAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x8);
    }
  }

  TEST(S16_RMAXABS__NEON_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 9; batch < 16; batch++) {
      RMaxAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_RMAXABS__NEON_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RMaxAbsMicrokernelTester()
      .batch(16)
      .Test(xnn_s16_rmaxabs_ukernel__neon_x16);
  }

  TEST(S16_RMAXABS__NEON_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 32; batch < 160; batch += 16) {
      RMaxAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x16);
    }
  }

  TEST(S16_RMAXABS__NEON_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 16; batch++) {
      RMaxAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x16);
    }
  }

  TEST(S16_RMAXABS__NEON_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 17; batch < 32; batch++) {
      RMaxAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_RMAXABS__NEON_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    RMaxAbsMicrokernelTester()
      .batch(24)
      .Test(xnn_s16_rmaxabs_ukernel__neon_x24);
  }

  TEST(S16_RMAXABS__NEON_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 48; batch < 240; batch += 24) {
      RMaxAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x24);
    }
  }

  TEST(S16_RMAXABS__NEON_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 24; batch++) {
      RMaxAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x24);
    }
  }

  TEST(S16_RMAXABS__NEON_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 25; batch < 48; batch++) {
      RMaxAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_RMAXABS__NEON_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    RMaxAbsMicrokernelTester()
      .batch(32)
      .Test(xnn_s16_rmaxabs_ukernel__neon_x32);
  }

  TEST(S16_RMAXABS__NEON_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 64; batch < 320; batch += 32) {
      RMaxAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x32);
    }
  }

  TEST(S16_RMAXABS__NEON_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 32; batch++) {
      RMaxAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x32);
    }
  }

  TEST(S16_RMAXABS__NEON_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 33; batch < 64; batch++) {
      RMaxAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_s16_rmaxabs_ukernel__neon_x32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


TEST(S16_RMAXABS__SCALAR_X1, batch_eq_1) {
  RMaxAbsMicrokernelTester()
    .batch(1)
    .Test(xnn_s16_rmaxabs_ukernel__scalar_x1);
}

TEST(S16_RMAXABS__SCALAR_X1, batch_gt_1) {
  for (size_t batch = 2; batch < 10; batch++) {
    RMaxAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x1);
  }
}


TEST(S16_RMAXABS__SCALAR_X2, batch_eq_2) {
  RMaxAbsMicrokernelTester()
    .batch(2)
    .Test(xnn_s16_rmaxabs_ukernel__scalar_x2);
}

TEST(S16_RMAXABS__SCALAR_X2, batch_div_2) {
  for (size_t batch = 4; batch < 20; batch += 2) {
    RMaxAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x2);
  }
}

TEST(S16_RMAXABS__SCALAR_X2, batch_lt_2) {
  for (size_t batch = 1; batch < 2; batch++) {
    RMaxAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x2);
  }
}

TEST(S16_RMAXABS__SCALAR_X2, batch_gt_2) {
  for (size_t batch = 3; batch < 4; batch++) {
    RMaxAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x2);
  }
}


TEST(S16_RMAXABS__SCALAR_X3, batch_eq_3) {
  RMaxAbsMicrokernelTester()
    .batch(3)
    .Test(xnn_s16_rmaxabs_ukernel__scalar_x3);
}

TEST(S16_RMAXABS__SCALAR_X3, batch_div_3) {
  for (size_t batch = 6; batch < 30; batch += 3) {
    RMaxAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x3);
  }
}

TEST(S16_RMAXABS__SCALAR_X3, batch_lt_3) {
  for (size_t batch = 1; batch < 3; batch++) {
    RMaxAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x3);
  }
}

TEST(S16_RMAXABS__SCALAR_X3, batch_gt_3) {
  for (size_t batch = 4; batch < 6; batch++) {
    RMaxAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x3);
  }
}


TEST(S16_RMAXABS__SCALAR_X4, batch_eq_4) {
  RMaxAbsMicrokernelTester()
    .batch(4)
    .Test(xnn_s16_rmaxabs_ukernel__scalar_x4);
}

TEST(S16_RMAXABS__SCALAR_X4, batch_div_4) {
  for (size_t batch = 8; batch < 40; batch += 4) {
    RMaxAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x4);
  }
}

TEST(S16_RMAXABS__SCALAR_X4, batch_lt_4) {
  for (size_t batch = 1; batch < 4; batch++) {
    RMaxAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x4);
  }
}

TEST(S16_RMAXABS__SCALAR_X4, batch_gt_4) {
  for (size_t batch = 5; batch < 8; batch++) {
    RMaxAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_s16_rmaxabs_ukernel__scalar_x4);
  }
}
