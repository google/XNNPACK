// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/i16-vlshift.yaml
//   Generator: tools/generate-vlshift-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vlshift.h>
#include "vlshift-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(I16_VLSHIFT__NEON_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VLShiftMicrokernelTester()
      .batch(8)
      .Test(xnn_i16_vlshift_ukernel__neon_x8);
  }

  TEST(I16_VLSHIFT__NEON_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 16; batch < 80; batch += 8) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .Test(xnn_i16_vlshift_ukernel__neon_x8);
    }
  }

  TEST(I16_VLSHIFT__NEON_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 8; batch++) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .Test(xnn_i16_vlshift_ukernel__neon_x8);
    }
  }

  TEST(I16_VLSHIFT__NEON_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 9; batch < 16; batch++) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .Test(xnn_i16_vlshift_ukernel__neon_x8);
    }
  }

  TEST(I16_VLSHIFT__NEON_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch <= 40; batch += 7) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .inplace(true)
        .iterations(1)
        .Test(xnn_i16_vlshift_ukernel__neon_x8);
    }
  }

  TEST(I16_VLSHIFT__NEON_X8, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 16; shift++) {
      VLShiftMicrokernelTester()
        .batch(8)
        .shift(shift)
        .Test(xnn_i16_vlshift_ukernel__neon_x8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(I16_VLSHIFT__NEON_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VLShiftMicrokernelTester()
      .batch(16)
      .Test(xnn_i16_vlshift_ukernel__neon_x16);
  }

  TEST(I16_VLSHIFT__NEON_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 32; batch < 160; batch += 16) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .Test(xnn_i16_vlshift_ukernel__neon_x16);
    }
  }

  TEST(I16_VLSHIFT__NEON_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 16; batch++) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .Test(xnn_i16_vlshift_ukernel__neon_x16);
    }
  }

  TEST(I16_VLSHIFT__NEON_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 17; batch < 32; batch++) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .Test(xnn_i16_vlshift_ukernel__neon_x16);
    }
  }

  TEST(I16_VLSHIFT__NEON_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch <= 80; batch += 15) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .inplace(true)
        .iterations(1)
        .Test(xnn_i16_vlshift_ukernel__neon_x16);
    }
  }

  TEST(I16_VLSHIFT__NEON_X16, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 16; shift++) {
      VLShiftMicrokernelTester()
        .batch(16)
        .shift(shift)
        .Test(xnn_i16_vlshift_ukernel__neon_x16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(I16_VLSHIFT__NEON_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VLShiftMicrokernelTester()
      .batch(24)
      .Test(xnn_i16_vlshift_ukernel__neon_x24);
  }

  TEST(I16_VLSHIFT__NEON_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 48; batch < 240; batch += 24) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .Test(xnn_i16_vlshift_ukernel__neon_x24);
    }
  }

  TEST(I16_VLSHIFT__NEON_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 24; batch++) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .Test(xnn_i16_vlshift_ukernel__neon_x24);
    }
  }

  TEST(I16_VLSHIFT__NEON_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 25; batch < 48; batch++) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .Test(xnn_i16_vlshift_ukernel__neon_x24);
    }
  }

  TEST(I16_VLSHIFT__NEON_X24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch <= 120; batch += 23) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .inplace(true)
        .iterations(1)
        .Test(xnn_i16_vlshift_ukernel__neon_x24);
    }
  }

  TEST(I16_VLSHIFT__NEON_X24, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 16; shift++) {
      VLShiftMicrokernelTester()
        .batch(24)
        .shift(shift)
        .Test(xnn_i16_vlshift_ukernel__neon_x24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(I16_VLSHIFT__NEON_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    VLShiftMicrokernelTester()
      .batch(32)
      .Test(xnn_i16_vlshift_ukernel__neon_x32);
  }

  TEST(I16_VLSHIFT__NEON_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 64; batch < 320; batch += 32) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .Test(xnn_i16_vlshift_ukernel__neon_x32);
    }
  }

  TEST(I16_VLSHIFT__NEON_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 32; batch++) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .Test(xnn_i16_vlshift_ukernel__neon_x32);
    }
  }

  TEST(I16_VLSHIFT__NEON_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 33; batch < 64; batch++) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .Test(xnn_i16_vlshift_ukernel__neon_x32);
    }
  }

  TEST(I16_VLSHIFT__NEON_X32, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch <= 160; batch += 31) {
      VLShiftMicrokernelTester()
        .batch(batch)
        .inplace(true)
        .iterations(1)
        .Test(xnn_i16_vlshift_ukernel__neon_x32);
    }
  }

  TEST(I16_VLSHIFT__NEON_X32, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 16; shift++) {
      VLShiftMicrokernelTester()
        .batch(32)
        .shift(shift)
        .Test(xnn_i16_vlshift_ukernel__neon_x32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


TEST(I16_VLSHIFT__SCALAR_X1, batch_eq_1) {
  VLShiftMicrokernelTester()
    .batch(1)
    .Test(xnn_i16_vlshift_ukernel__scalar_x1);
}

TEST(I16_VLSHIFT__SCALAR_X1, batch_gt_1) {
  for (size_t batch = 2; batch < 10; batch++) {
    VLShiftMicrokernelTester()
      .batch(batch)
      .Test(xnn_i16_vlshift_ukernel__scalar_x1);
  }
}

TEST(I16_VLSHIFT__SCALAR_X1, inplace) {
  for (size_t batch = 1; batch <= 5; batch += 1) {
    VLShiftMicrokernelTester()
      .batch(batch)
      .inplace(true)
      .iterations(1)
      .Test(xnn_i16_vlshift_ukernel__scalar_x1);
  }
}

TEST(I16_VLSHIFT__SCALAR_X1, shift) {
  for (uint32_t shift = 0; shift < 16; shift++) {
    VLShiftMicrokernelTester()
      .batch(1)
      .shift(shift)
      .Test(xnn_i16_vlshift_ukernel__scalar_x1);
  }
}


TEST(I16_VLSHIFT__SCALAR_X2, batch_eq_2) {
  VLShiftMicrokernelTester()
    .batch(2)
    .Test(xnn_i16_vlshift_ukernel__scalar_x2);
}

TEST(I16_VLSHIFT__SCALAR_X2, batch_div_2) {
  for (size_t batch = 4; batch < 20; batch += 2) {
    VLShiftMicrokernelTester()
      .batch(batch)
      .Test(xnn_i16_vlshift_ukernel__scalar_x2);
  }
}

TEST(I16_VLSHIFT__SCALAR_X2, batch_lt_2) {
  for (size_t batch = 1; batch < 2; batch++) {
    VLShiftMicrokernelTester()
      .batch(batch)
      .Test(xnn_i16_vlshift_ukernel__scalar_x2);
  }
}

TEST(I16_VLSHIFT__SCALAR_X2, batch_gt_2) {
  for (size_t batch = 3; batch < 4; batch++) {
    VLShiftMicrokernelTester()
      .batch(batch)
      .Test(xnn_i16_vlshift_ukernel__scalar_x2);
  }
}

TEST(I16_VLSHIFT__SCALAR_X2, inplace) {
  for (size_t batch = 1; batch <= 10; batch += 1) {
    VLShiftMicrokernelTester()
      .batch(batch)
      .inplace(true)
      .iterations(1)
      .Test(xnn_i16_vlshift_ukernel__scalar_x2);
  }
}

TEST(I16_VLSHIFT__SCALAR_X2, shift) {
  for (uint32_t shift = 0; shift < 16; shift++) {
    VLShiftMicrokernelTester()
      .batch(2)
      .shift(shift)
      .Test(xnn_i16_vlshift_ukernel__scalar_x2);
  }
}


TEST(I16_VLSHIFT__SCALAR_X3, batch_eq_3) {
  VLShiftMicrokernelTester()
    .batch(3)
    .Test(xnn_i16_vlshift_ukernel__scalar_x3);
}

TEST(I16_VLSHIFT__SCALAR_X3, batch_div_3) {
  for (size_t batch = 6; batch < 30; batch += 3) {
    VLShiftMicrokernelTester()
      .batch(batch)
      .Test(xnn_i16_vlshift_ukernel__scalar_x3);
  }
}

TEST(I16_VLSHIFT__SCALAR_X3, batch_lt_3) {
  for (size_t batch = 1; batch < 3; batch++) {
    VLShiftMicrokernelTester()
      .batch(batch)
      .Test(xnn_i16_vlshift_ukernel__scalar_x3);
  }
}

TEST(I16_VLSHIFT__SCALAR_X3, batch_gt_3) {
  for (size_t batch = 4; batch < 6; batch++) {
    VLShiftMicrokernelTester()
      .batch(batch)
      .Test(xnn_i16_vlshift_ukernel__scalar_x3);
  }
}

TEST(I16_VLSHIFT__SCALAR_X3, inplace) {
  for (size_t batch = 1; batch <= 15; batch += 2) {
    VLShiftMicrokernelTester()
      .batch(batch)
      .inplace(true)
      .iterations(1)
      .Test(xnn_i16_vlshift_ukernel__scalar_x3);
  }
}

TEST(I16_VLSHIFT__SCALAR_X3, shift) {
  for (uint32_t shift = 0; shift < 16; shift++) {
    VLShiftMicrokernelTester()
      .batch(3)
      .shift(shift)
      .Test(xnn_i16_vlshift_ukernel__scalar_x3);
  }
}


TEST(I16_VLSHIFT__SCALAR_X4, batch_eq_4) {
  VLShiftMicrokernelTester()
    .batch(4)
    .Test(xnn_i16_vlshift_ukernel__scalar_x4);
}

TEST(I16_VLSHIFT__SCALAR_X4, batch_div_4) {
  for (size_t batch = 8; batch < 40; batch += 4) {
    VLShiftMicrokernelTester()
      .batch(batch)
      .Test(xnn_i16_vlshift_ukernel__scalar_x4);
  }
}

TEST(I16_VLSHIFT__SCALAR_X4, batch_lt_4) {
  for (size_t batch = 1; batch < 4; batch++) {
    VLShiftMicrokernelTester()
      .batch(batch)
      .Test(xnn_i16_vlshift_ukernel__scalar_x4);
  }
}

TEST(I16_VLSHIFT__SCALAR_X4, batch_gt_4) {
  for (size_t batch = 5; batch < 8; batch++) {
    VLShiftMicrokernelTester()
      .batch(batch)
      .Test(xnn_i16_vlshift_ukernel__scalar_x4);
  }
}

TEST(I16_VLSHIFT__SCALAR_X4, inplace) {
  for (size_t batch = 1; batch <= 20; batch += 3) {
    VLShiftMicrokernelTester()
      .batch(batch)
      .inplace(true)
      .iterations(1)
      .Test(xnn_i16_vlshift_ukernel__scalar_x4);
  }
}

TEST(I16_VLSHIFT__SCALAR_X4, shift) {
  for (uint32_t shift = 0; shift < 16; shift++) {
    VLShiftMicrokernelTester()
      .batch(4)
      .shift(shift)
      .Test(xnn_i16_vlshift_ukernel__scalar_x4);
  }
}
