// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/cs16-vsquareabs.yaml
//   Generator: tools/generate-vsquareabs-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/vsquareabs.h"
#include "vsquareabs-microkernel-tester.h"


TEST(CS16_VSQUAREABS__SCALAR_X1, batch_eq_1) {
  VSquareAbsMicrokernelTester()
    .batch(1)
    .Test(xnn_cs16_vsquareabs_ukernel__scalar_x1);
}

TEST(CS16_VSQUAREABS__SCALAR_X1, batch_gt_1) {
  for (size_t batch = 2; batch < 10; batch++) {
    VSquareAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x1);
  }
}


TEST(CS16_VSQUAREABS__SCALAR_X2, batch_eq_2) {
  VSquareAbsMicrokernelTester()
    .batch(2)
    .Test(xnn_cs16_vsquareabs_ukernel__scalar_x2);
}

TEST(CS16_VSQUAREABS__SCALAR_X2, batch_div_2) {
  for (size_t batch = 4; batch < 20; batch += 2) {
    VSquareAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x2);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X2, batch_lt_2) {
  for (size_t batch = 1; batch < 2; batch++) {
    VSquareAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x2);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X2, batch_gt_2) {
  for (size_t batch = 3; batch < 4; batch++) {
    VSquareAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x2);
  }
}


TEST(CS16_VSQUAREABS__SCALAR_X3, batch_eq_3) {
  VSquareAbsMicrokernelTester()
    .batch(3)
    .Test(xnn_cs16_vsquareabs_ukernel__scalar_x3);
}

TEST(CS16_VSQUAREABS__SCALAR_X3, batch_div_3) {
  for (size_t batch = 6; batch < 30; batch += 3) {
    VSquareAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x3);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X3, batch_lt_3) {
  for (size_t batch = 1; batch < 3; batch++) {
    VSquareAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x3);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X3, batch_gt_3) {
  for (size_t batch = 4; batch < 6; batch++) {
    VSquareAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x3);
  }
}


TEST(CS16_VSQUAREABS__SCALAR_X4, batch_eq_4) {
  VSquareAbsMicrokernelTester()
    .batch(4)
    .Test(xnn_cs16_vsquareabs_ukernel__scalar_x4);
}

TEST(CS16_VSQUAREABS__SCALAR_X4, batch_div_4) {
  for (size_t batch = 8; batch < 40; batch += 4) {
    VSquareAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x4);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X4, batch_lt_4) {
  for (size_t batch = 1; batch < 4; batch++) {
    VSquareAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x4);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X4, batch_gt_4) {
  for (size_t batch = 5; batch < 8; batch++) {
    VSquareAbsMicrokernelTester()
      .batch(batch)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x4);
  }
}


#if XNN_ARCH_HEXAGON
  TEST(CS16_VSQUAREABS__HEXAGON_X2, batch_eq_2) {
    VSquareAbsMicrokernelTester()
      .batch(2)
      .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x2);
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X2, batch_div_2) {
    for (size_t batch = 4; batch < 20; batch += 2) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x2);
    }
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X2, batch_lt_2) {
    for (size_t batch = 1; batch < 2; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x2);
    }
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X2, batch_gt_2) {
    for (size_t batch = 3; batch < 4; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x2);
    }
  }
#endif  // XNN_ARCH_HEXAGON


#if XNN_ARCH_HEXAGON
  TEST(CS16_VSQUAREABS__HEXAGON_X4, batch_eq_4) {
    VSquareAbsMicrokernelTester()
      .batch(4)
      .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x4);
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X4, batch_div_4) {
    for (size_t batch = 8; batch < 40; batch += 4) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x4);
    }
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X4, batch_lt_4) {
    for (size_t batch = 1; batch < 4; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x4);
    }
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X4, batch_gt_4) {
    for (size_t batch = 5; batch < 8; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x4);
    }
  }
#endif  // XNN_ARCH_HEXAGON


#if XNN_ARCH_HEXAGON
  TEST(CS16_VSQUAREABS__HEXAGON_X6, batch_eq_6) {
    VSquareAbsMicrokernelTester()
      .batch(6)
      .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x6);
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X6, batch_div_6) {
    for (size_t batch = 12; batch < 60; batch += 6) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x6);
    }
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X6, batch_lt_6) {
    for (size_t batch = 1; batch < 6; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x6);
    }
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X6, batch_gt_6) {
    for (size_t batch = 7; batch < 12; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x6);
    }
  }
#endif  // XNN_ARCH_HEXAGON


#if XNN_ARCH_HEXAGON
  TEST(CS16_VSQUAREABS__HEXAGON_X8, batch_eq_8) {
    VSquareAbsMicrokernelTester()
      .batch(8)
      .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x8);
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X8, batch_div_8) {
    for (size_t batch = 16; batch < 80; batch += 8) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x8);
    }
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X8, batch_lt_8) {
    for (size_t batch = 1; batch < 8; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x8);
    }
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X8, batch_gt_8) {
    for (size_t batch = 9; batch < 16; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x8);
    }
  }
#endif  // XNN_ARCH_HEXAGON


#if XNN_ARCH_HEXAGON
  TEST(CS16_VSQUAREABS__HEXAGON_X10, batch_eq_10) {
    VSquareAbsMicrokernelTester()
      .batch(10)
      .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x10);
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X10, batch_div_10) {
    for (size_t batch = 20; batch < 100; batch += 10) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x10);
    }
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X10, batch_lt_10) {
    for (size_t batch = 1; batch < 10; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x10);
    }
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X10, batch_gt_10) {
    for (size_t batch = 11; batch < 20; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x10);
    }
  }
#endif  // XNN_ARCH_HEXAGON


#if XNN_ARCH_HEXAGON
  TEST(CS16_VSQUAREABS__HEXAGON_X12, batch_eq_12) {
    VSquareAbsMicrokernelTester()
      .batch(12)
      .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x12);
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X12, batch_div_12) {
    for (size_t batch = 24; batch < 120; batch += 12) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x12);
    }
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X12, batch_lt_12) {
    for (size_t batch = 1; batch < 12; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x12);
    }
  }

  TEST(CS16_VSQUAREABS__HEXAGON_X12, batch_gt_12) {
    for (size_t batch = 13; batch < 24; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__hexagon_x12);
    }
  }
#endif  // XNN_ARCH_HEXAGON


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VSquareAbsMicrokernelTester()
      .batch(4)
      .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x4);
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 8; batch < 40; batch += 4) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x4);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 4; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x4);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 5; batch < 8; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VSquareAbsMicrokernelTester()
      .batch(8)
      .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x8);
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 16; batch < 80; batch += 8) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x8);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 8; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x8);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 9; batch < 16; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VSquareAbsMicrokernelTester()
      .batch(12)
      .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x12);
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 24; batch < 120; batch += 12) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x12);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 12; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x12);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 13; batch < 24; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VSquareAbsMicrokernelTester()
      .batch(16)
      .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x16);
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 32; batch < 160; batch += 16) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x16);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 16; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x16);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 17; batch < 32; batch++) {
      VSquareAbsMicrokernelTester()
        .batch(batch)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
