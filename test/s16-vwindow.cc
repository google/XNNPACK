// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/s16-vwindow.yaml
//   Generator: tools/generate-vwindow-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vwindow.h>
#include "vwindow-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_VWINDOW__NEON_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VWindowMicrokernelTester()
      .rows(8)
      .batch(8)
      .Test(xnn_s16_vwindow_ukernel__neon_x8);
  }

  TEST(S16_VWINDOW__NEON_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 16; batch < 80; batch += 8) {
      VWindowMicrokernelTester()
        .rows(8)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__neon_x8);
    }
  }

  TEST(S16_VWINDOW__NEON_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 8; batch++) {
      VWindowMicrokernelTester()
        .rows(8)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__neon_x8);
    }
  }

  TEST(S16_VWINDOW__NEON_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 9; batch < 16; batch++) {
      VWindowMicrokernelTester()
        .rows(8)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__neon_x8);
    }
  }

  TEST(S16_VWINDOW__NEON_X8, rows_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 8; rows++) {
      for (size_t batch = 1; batch <= 40; batch += 7) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .Test(xnn_s16_vwindow_ukernel__neon_x8);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X8, rows_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 16; rows <= 32; rows += 8) {
      for (size_t batch = 1; batch <= 40; batch += 7) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .Test(xnn_s16_vwindow_ukernel__neon_x8);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X8, rows_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 9; rows < 16; rows++) {
      for (size_t batch = 1; batch <= 40; batch += 7) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .Test(xnn_s16_vwindow_ukernel__neon_x8);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 24; rows += 7) {
      for (size_t batch = 1; batch <= 40; batch += 7) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_vwindow_ukernel__neon_x8);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X8, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      VWindowMicrokernelTester()
        .rows(8)
        .batch(8)
        .shift(shift)
        .Test(xnn_s16_vwindow_ukernel__neon_x8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_VWINDOW__NEON_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VWindowMicrokernelTester()
      .rows(16)
      .batch(16)
      .Test(xnn_s16_vwindow_ukernel__neon_x16);
  }

  TEST(S16_VWINDOW__NEON_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 32; batch < 160; batch += 16) {
      VWindowMicrokernelTester()
        .rows(16)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__neon_x16);
    }
  }

  TEST(S16_VWINDOW__NEON_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 16; batch++) {
      VWindowMicrokernelTester()
        .rows(16)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__neon_x16);
    }
  }

  TEST(S16_VWINDOW__NEON_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 17; batch < 32; batch++) {
      VWindowMicrokernelTester()
        .rows(16)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__neon_x16);
    }
  }

  TEST(S16_VWINDOW__NEON_X16, rows_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 16; rows++) {
      for (size_t batch = 1; batch <= 80; batch += 15) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .Test(xnn_s16_vwindow_ukernel__neon_x16);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X16, rows_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 32; rows <= 64; rows += 16) {
      for (size_t batch = 1; batch <= 80; batch += 15) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .Test(xnn_s16_vwindow_ukernel__neon_x16);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X16, rows_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 17; rows < 32; rows++) {
      for (size_t batch = 1; batch <= 80; batch += 15) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .Test(xnn_s16_vwindow_ukernel__neon_x16);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 48; rows += 15) {
      for (size_t batch = 1; batch <= 80; batch += 15) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_vwindow_ukernel__neon_x16);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X16, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      VWindowMicrokernelTester()
        .rows(16)
        .batch(16)
        .shift(shift)
        .Test(xnn_s16_vwindow_ukernel__neon_x16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_VWINDOW__NEON_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VWindowMicrokernelTester()
      .rows(24)
      .batch(24)
      .Test(xnn_s16_vwindow_ukernel__neon_x24);
  }

  TEST(S16_VWINDOW__NEON_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 48; batch < 240; batch += 24) {
      VWindowMicrokernelTester()
        .rows(24)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__neon_x24);
    }
  }

  TEST(S16_VWINDOW__NEON_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 24; batch++) {
      VWindowMicrokernelTester()
        .rows(24)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__neon_x24);
    }
  }

  TEST(S16_VWINDOW__NEON_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 25; batch < 48; batch++) {
      VWindowMicrokernelTester()
        .rows(24)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__neon_x24);
    }
  }

  TEST(S16_VWINDOW__NEON_X24, rows_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 24; rows++) {
      for (size_t batch = 1; batch <= 120; batch += 23) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .Test(xnn_s16_vwindow_ukernel__neon_x24);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X24, rows_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 48; rows <= 96; rows += 24) {
      for (size_t batch = 1; batch <= 120; batch += 23) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .Test(xnn_s16_vwindow_ukernel__neon_x24);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X24, rows_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 25; rows < 48; rows++) {
      for (size_t batch = 1; batch <= 120; batch += 23) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .Test(xnn_s16_vwindow_ukernel__neon_x24);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 72; rows += 23) {
      for (size_t batch = 1; batch <= 120; batch += 23) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_vwindow_ukernel__neon_x24);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X24, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      VWindowMicrokernelTester()
        .rows(24)
        .batch(24)
        .shift(shift)
        .Test(xnn_s16_vwindow_ukernel__neon_x24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_VWINDOW__NEON_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    VWindowMicrokernelTester()
      .rows(32)
      .batch(32)
      .Test(xnn_s16_vwindow_ukernel__neon_x32);
  }

  TEST(S16_VWINDOW__NEON_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 64; batch < 320; batch += 32) {
      VWindowMicrokernelTester()
        .rows(32)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__neon_x32);
    }
  }

  TEST(S16_VWINDOW__NEON_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 32; batch++) {
      VWindowMicrokernelTester()
        .rows(32)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__neon_x32);
    }
  }

  TEST(S16_VWINDOW__NEON_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 33; batch < 64; batch++) {
      VWindowMicrokernelTester()
        .rows(32)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__neon_x32);
    }
  }

  TEST(S16_VWINDOW__NEON_X32, rows_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 32; rows++) {
      for (size_t batch = 1; batch <= 160; batch += 31) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .Test(xnn_s16_vwindow_ukernel__neon_x32);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X32, rows_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 64; rows <= 128; rows += 32) {
      for (size_t batch = 1; batch <= 160; batch += 31) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .Test(xnn_s16_vwindow_ukernel__neon_x32);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X32, rows_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 33; rows < 64; rows++) {
      for (size_t batch = 1; batch <= 160; batch += 31) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .Test(xnn_s16_vwindow_ukernel__neon_x32);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X32, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 96; rows += 31) {
      for (size_t batch = 1; batch <= 160; batch += 31) {
        VWindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_vwindow_ukernel__neon_x32);
      }
    }
  }

  TEST(S16_VWINDOW__NEON_X32, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      VWindowMicrokernelTester()
        .rows(32)
        .batch(32)
        .shift(shift)
        .Test(xnn_s16_vwindow_ukernel__neon_x32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


TEST(S16_VWINDOW__SCALAR_X1, batch_eq_1) {
  VWindowMicrokernelTester()
    .rows(1)
    .batch(1)
    .Test(xnn_s16_vwindow_ukernel__scalar_x1);
}

TEST(S16_VWINDOW__SCALAR_X1, batch_gt_1) {
  for (size_t batch = 2; batch < 10; batch++) {
    VWindowMicrokernelTester()
      .rows(1)
      .batch(batch)
      .Test(xnn_s16_vwindow_ukernel__scalar_x1);
  }
}

TEST(S16_VWINDOW__SCALAR_X1, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t batch = 1; batch <= 5; batch += 1) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__scalar_x1);
    }
  }
}

TEST(S16_VWINDOW__SCALAR_X1, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t batch = 1; batch <= 5; batch += 1) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_vwindow_ukernel__scalar_x1);
    }
  }
}

TEST(S16_VWINDOW__SCALAR_X1, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    VWindowMicrokernelTester()
      .rows(1)
      .batch(1)
      .shift(shift)
      .Test(xnn_s16_vwindow_ukernel__scalar_x1);
  }
}


TEST(S16_VWINDOW__SCALAR_X2, batch_eq_2) {
  VWindowMicrokernelTester()
    .rows(2)
    .batch(2)
    .Test(xnn_s16_vwindow_ukernel__scalar_x2);
}

TEST(S16_VWINDOW__SCALAR_X2, batch_div_2) {
  for (size_t batch = 4; batch < 20; batch += 2) {
    VWindowMicrokernelTester()
      .rows(2)
      .batch(batch)
      .Test(xnn_s16_vwindow_ukernel__scalar_x2);
  }
}

TEST(S16_VWINDOW__SCALAR_X2, batch_lt_2) {
  for (size_t batch = 1; batch < 2; batch++) {
    VWindowMicrokernelTester()
      .rows(2)
      .batch(batch)
      .Test(xnn_s16_vwindow_ukernel__scalar_x2);
  }
}

TEST(S16_VWINDOW__SCALAR_X2, batch_gt_2) {
  for (size_t batch = 3; batch < 4; batch++) {
    VWindowMicrokernelTester()
      .rows(2)
      .batch(batch)
      .Test(xnn_s16_vwindow_ukernel__scalar_x2);
  }
}

TEST(S16_VWINDOW__SCALAR_X2, rows_lt_2) {
  for (size_t rows = 1; rows < 2; rows++) {
    for (size_t batch = 1; batch <= 10; batch += 1) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__scalar_x2);
    }
  }
}

TEST(S16_VWINDOW__SCALAR_X2, rows_div_2) {
  for (size_t rows = 4; rows <= 8; rows += 2) {
    for (size_t batch = 1; batch <= 10; batch += 1) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__scalar_x2);
    }
  }
}

TEST(S16_VWINDOW__SCALAR_X2, rows_gt_2) {
  for (size_t rows = 3; rows < 4; rows++) {
    for (size_t batch = 1; batch <= 10; batch += 1) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__scalar_x2);
    }
  }
}

TEST(S16_VWINDOW__SCALAR_X2, inplace) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t batch = 1; batch <= 10; batch += 1) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_vwindow_ukernel__scalar_x2);
    }
  }
}

TEST(S16_VWINDOW__SCALAR_X2, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    VWindowMicrokernelTester()
      .rows(2)
      .batch(2)
      .shift(shift)
      .Test(xnn_s16_vwindow_ukernel__scalar_x2);
  }
}


TEST(S16_VWINDOW__SCALAR_X3, batch_eq_3) {
  VWindowMicrokernelTester()
    .rows(3)
    .batch(3)
    .Test(xnn_s16_vwindow_ukernel__scalar_x3);
}

TEST(S16_VWINDOW__SCALAR_X3, batch_div_3) {
  for (size_t batch = 6; batch < 30; batch += 3) {
    VWindowMicrokernelTester()
      .rows(3)
      .batch(batch)
      .Test(xnn_s16_vwindow_ukernel__scalar_x3);
  }
}

TEST(S16_VWINDOW__SCALAR_X3, batch_lt_3) {
  for (size_t batch = 1; batch < 3; batch++) {
    VWindowMicrokernelTester()
      .rows(3)
      .batch(batch)
      .Test(xnn_s16_vwindow_ukernel__scalar_x3);
  }
}

TEST(S16_VWINDOW__SCALAR_X3, batch_gt_3) {
  for (size_t batch = 4; batch < 6; batch++) {
    VWindowMicrokernelTester()
      .rows(3)
      .batch(batch)
      .Test(xnn_s16_vwindow_ukernel__scalar_x3);
  }
}

TEST(S16_VWINDOW__SCALAR_X3, rows_lt_3) {
  for (size_t rows = 1; rows < 3; rows++) {
    for (size_t batch = 1; batch <= 15; batch += 2) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__scalar_x3);
    }
  }
}

TEST(S16_VWINDOW__SCALAR_X3, rows_div_3) {
  for (size_t rows = 6; rows <= 12; rows += 3) {
    for (size_t batch = 1; batch <= 15; batch += 2) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__scalar_x3);
    }
  }
}

TEST(S16_VWINDOW__SCALAR_X3, rows_gt_3) {
  for (size_t rows = 4; rows < 6; rows++) {
    for (size_t batch = 1; batch <= 15; batch += 2) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__scalar_x3);
    }
  }
}

TEST(S16_VWINDOW__SCALAR_X3, inplace) {
  for (size_t rows = 1; rows <= 9; rows += 2) {
    for (size_t batch = 1; batch <= 15; batch += 2) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_vwindow_ukernel__scalar_x3);
    }
  }
}

TEST(S16_VWINDOW__SCALAR_X3, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    VWindowMicrokernelTester()
      .rows(3)
      .batch(3)
      .shift(shift)
      .Test(xnn_s16_vwindow_ukernel__scalar_x3);
  }
}


TEST(S16_VWINDOW__SCALAR_X4, batch_eq_4) {
  VWindowMicrokernelTester()
    .rows(4)
    .batch(4)
    .Test(xnn_s16_vwindow_ukernel__scalar_x4);
}

TEST(S16_VWINDOW__SCALAR_X4, batch_div_4) {
  for (size_t batch = 8; batch < 40; batch += 4) {
    VWindowMicrokernelTester()
      .rows(4)
      .batch(batch)
      .Test(xnn_s16_vwindow_ukernel__scalar_x4);
  }
}

TEST(S16_VWINDOW__SCALAR_X4, batch_lt_4) {
  for (size_t batch = 1; batch < 4; batch++) {
    VWindowMicrokernelTester()
      .rows(4)
      .batch(batch)
      .Test(xnn_s16_vwindow_ukernel__scalar_x4);
  }
}

TEST(S16_VWINDOW__SCALAR_X4, batch_gt_4) {
  for (size_t batch = 5; batch < 8; batch++) {
    VWindowMicrokernelTester()
      .rows(4)
      .batch(batch)
      .Test(xnn_s16_vwindow_ukernel__scalar_x4);
  }
}

TEST(S16_VWINDOW__SCALAR_X4, rows_lt_4) {
  for (size_t rows = 1; rows < 4; rows++) {
    for (size_t batch = 1; batch <= 20; batch += 3) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__scalar_x4);
    }
  }
}

TEST(S16_VWINDOW__SCALAR_X4, rows_div_4) {
  for (size_t rows = 8; rows <= 16; rows += 4) {
    for (size_t batch = 1; batch <= 20; batch += 3) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__scalar_x4);
    }
  }
}

TEST(S16_VWINDOW__SCALAR_X4, rows_gt_4) {
  for (size_t rows = 5; rows < 8; rows++) {
    for (size_t batch = 1; batch <= 20; batch += 3) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .Test(xnn_s16_vwindow_ukernel__scalar_x4);
    }
  }
}

TEST(S16_VWINDOW__SCALAR_X4, inplace) {
  for (size_t rows = 1; rows <= 12; rows += 3) {
    for (size_t batch = 1; batch <= 20; batch += 3) {
      VWindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_vwindow_ukernel__scalar_x4);
    }
  }
}

TEST(S16_VWINDOW__SCALAR_X4, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    VWindowMicrokernelTester()
      .rows(4)
      .batch(4)
      .shift(shift)
      .Test(xnn_s16_vwindow_ukernel__scalar_x4);
  }
}
