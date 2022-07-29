// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/cs16-vsquareabs.yaml
//   Generator: tools/generate-vsquareabs-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vsquareabs.h>
#include "vsquareabs-microkernel-tester.h"


TEST(CS16_VSQUAREABS__SCALAR_X1, batch_eq_1) {
  VSquareAbsMicrokernelTester()
    .batch_size(1)
    .Test(xnn_cs16_vsquareabs_ukernel__scalar_x1);
}

TEST(CS16_VSQUAREABS__SCALAR_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VSquareAbsMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x1);
  }
}


TEST(CS16_VSQUAREABS__SCALAR_X2, batch_eq_2) {
  VSquareAbsMicrokernelTester()
    .batch_size(2)
    .Test(xnn_cs16_vsquareabs_ukernel__scalar_x2);
}

TEST(CS16_VSQUAREABS__SCALAR_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VSquareAbsMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x2);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VSquareAbsMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x2);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VSquareAbsMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x2);
  }
}


TEST(CS16_VSQUAREABS__SCALAR_X3, batch_eq_3) {
  VSquareAbsMicrokernelTester()
    .batch_size(3)
    .Test(xnn_cs16_vsquareabs_ukernel__scalar_x3);
}

TEST(CS16_VSQUAREABS__SCALAR_X3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    VSquareAbsMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x3);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    VSquareAbsMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x3);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    VSquareAbsMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x3);
  }
}


TEST(CS16_VSQUAREABS__SCALAR_X4, batch_eq_4) {
  VSquareAbsMicrokernelTester()
    .batch_size(4)
    .Test(xnn_cs16_vsquareabs_ukernel__scalar_x4);
}

TEST(CS16_VSQUAREABS__SCALAR_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VSquareAbsMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x4);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VSquareAbsMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x4);
  }
}

TEST(CS16_VSQUAREABS__SCALAR_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VSquareAbsMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_cs16_vsquareabs_ukernel__scalar_x4);
  }
}


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VSquareAbsMicrokernelTester()
      .batch_size(4)
      .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x4);
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VSquareAbsMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x4);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VSquareAbsMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x4);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VSquareAbsMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VSquareAbsMicrokernelTester()
      .batch_size(8)
      .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x8);
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VSquareAbsMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x8);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VSquareAbsMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x8);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VSquareAbsMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VSquareAbsMicrokernelTester()
      .batch_size(12)
      .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x12);
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VSquareAbsMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x12);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VSquareAbsMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x12);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VSquareAbsMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VSquareAbsMicrokernelTester()
      .batch_size(16)
      .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x16);
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VSquareAbsMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x16);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VSquareAbsMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x16);
    }
  }

  TEST(CS16_VSQUAREABS__NEON_MLAL_LD128_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VSquareAbsMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
