// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/s16-window.yaml
//   Generator: tools/generate-window-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/window.h>
#include "window-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT12__NEON_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .batch(8)
      .shift(12)
      .Test(xnn_s16_window_shift12_ukernel__neon_x8);
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 16; batch < 80; batch += 8) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_x8);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 8; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_x8);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 9; batch < 16; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_x8);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X8, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t batch = 1; batch <= 40; batch += 7) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(12)
          .Test(xnn_s16_window_shift12_ukernel__neon_x8);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t batch = 1; batch <= 40; batch += 7) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(12)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift12_ukernel__neon_x8);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT12__NEON_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .batch(16)
      .shift(12)
      .Test(xnn_s16_window_shift12_ukernel__neon_x16);
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 32; batch < 160; batch += 16) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_x16);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 16; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_x16);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 17; batch < 32; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_x16);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X16, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t batch = 1; batch <= 80; batch += 15) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(12)
          .Test(xnn_s16_window_shift12_ukernel__neon_x16);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t batch = 1; batch <= 80; batch += 15) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(12)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift12_ukernel__neon_x16);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT12__NEON_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .batch(24)
      .shift(12)
      .Test(xnn_s16_window_shift12_ukernel__neon_x24);
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 48; batch < 240; batch += 24) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_x24);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 24; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_x24);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 25; batch < 48; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_x24);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X24, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t batch = 1; batch <= 120; batch += 23) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(12)
          .Test(xnn_s16_window_shift12_ukernel__neon_x24);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t batch = 1; batch <= 120; batch += 23) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(12)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift12_ukernel__neon_x24);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT12__NEON_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .batch(32)
      .shift(12)
      .Test(xnn_s16_window_shift12_ukernel__neon_x32);
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 64; batch < 320; batch += 32) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_x32);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 32; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_x32);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 33; batch < 64; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_x32);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X32, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t batch = 1; batch <= 160; batch += 31) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(12)
          .Test(xnn_s16_window_shift12_ukernel__neon_x32);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_X32, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t batch = 1; batch <= 160; batch += 31) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(12)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift12_ukernel__neon_x32);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT15__NEON_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .batch(8)
      .shift(15)
      .Test(xnn_s16_window_shift15_ukernel__neon_x8);
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 16; batch < 80; batch += 8) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_x8);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 8; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_x8);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 9; batch < 16; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_x8);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X8, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t batch = 1; batch <= 40; batch += 7) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(15)
          .Test(xnn_s16_window_shift15_ukernel__neon_x8);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t batch = 1; batch <= 40; batch += 7) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(15)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift15_ukernel__neon_x8);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT15__NEON_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .batch(16)
      .shift(15)
      .Test(xnn_s16_window_shift15_ukernel__neon_x16);
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 32; batch < 160; batch += 16) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_x16);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 16; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_x16);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 17; batch < 32; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_x16);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X16, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t batch = 1; batch <= 80; batch += 15) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(15)
          .Test(xnn_s16_window_shift15_ukernel__neon_x16);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t batch = 1; batch <= 80; batch += 15) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(15)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift15_ukernel__neon_x16);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT15__NEON_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .batch(24)
      .shift(15)
      .Test(xnn_s16_window_shift15_ukernel__neon_x24);
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 48; batch < 240; batch += 24) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_x24);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 24; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_x24);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 25; batch < 48; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_x24);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X24, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t batch = 1; batch <= 120; batch += 23) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(15)
          .Test(xnn_s16_window_shift15_ukernel__neon_x24);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t batch = 1; batch <= 120; batch += 23) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(15)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift15_ukernel__neon_x24);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT15__NEON_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .batch(32)
      .shift(15)
      .Test(xnn_s16_window_shift15_ukernel__neon_x32);
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 64; batch < 320; batch += 32) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_x32);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 32; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_x32);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 33; batch < 64; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_x32);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X32, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t batch = 1; batch <= 160; batch += 31) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(15)
          .Test(xnn_s16_window_shift15_ukernel__neon_x32);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_X32, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t batch = 1; batch <= 160; batch += 31) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(15)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift15_ukernel__neon_x32);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW__NEON_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .batch(8)
      .shift(0)
      .Test(xnn_s16_window_ukernel__neon_x8);
  }

  TEST(S16_WINDOW__NEON_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 16; batch < 80; batch += 8) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_x8);
    }
  }

  TEST(S16_WINDOW__NEON_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 8; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_x8);
    }
  }

  TEST(S16_WINDOW__NEON_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 9; batch < 16; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_x8);
    }
  }

  TEST(S16_WINDOW__NEON_X8, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t batch = 1; batch <= 40; batch += 7) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(0)
          .Test(xnn_s16_window_ukernel__neon_x8);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t batch = 1; batch <= 40; batch += 7) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(0)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_ukernel__neon_x8);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X8, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      WindowMicrokernelTester()
        .rows(1)
        .batch(8)
        .shift(shift)
        .Test(xnn_s16_window_ukernel__neon_x8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW__NEON_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .batch(16)
      .shift(0)
      .Test(xnn_s16_window_ukernel__neon_x16);
  }

  TEST(S16_WINDOW__NEON_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 32; batch < 160; batch += 16) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_x16);
    }
  }

  TEST(S16_WINDOW__NEON_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 16; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_x16);
    }
  }

  TEST(S16_WINDOW__NEON_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 17; batch < 32; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_x16);
    }
  }

  TEST(S16_WINDOW__NEON_X16, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t batch = 1; batch <= 80; batch += 15) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(0)
          .Test(xnn_s16_window_ukernel__neon_x16);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t batch = 1; batch <= 80; batch += 15) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(0)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_ukernel__neon_x16);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X16, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      WindowMicrokernelTester()
        .rows(1)
        .batch(16)
        .shift(shift)
        .Test(xnn_s16_window_ukernel__neon_x16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW__NEON_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .batch(24)
      .shift(0)
      .Test(xnn_s16_window_ukernel__neon_x24);
  }

  TEST(S16_WINDOW__NEON_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 48; batch < 240; batch += 24) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_x24);
    }
  }

  TEST(S16_WINDOW__NEON_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 24; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_x24);
    }
  }

  TEST(S16_WINDOW__NEON_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 25; batch < 48; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_x24);
    }
  }

  TEST(S16_WINDOW__NEON_X24, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t batch = 1; batch <= 120; batch += 23) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(0)
          .Test(xnn_s16_window_ukernel__neon_x24);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t batch = 1; batch <= 120; batch += 23) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(0)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_ukernel__neon_x24);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X24, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      WindowMicrokernelTester()
        .rows(1)
        .batch(24)
        .shift(shift)
        .Test(xnn_s16_window_ukernel__neon_x24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW__NEON_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .batch(32)
      .shift(0)
      .Test(xnn_s16_window_ukernel__neon_x32);
  }

  TEST(S16_WINDOW__NEON_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 64; batch < 320; batch += 32) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_x32);
    }
  }

  TEST(S16_WINDOW__NEON_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 32; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_x32);
    }
  }

  TEST(S16_WINDOW__NEON_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 33; batch < 64; batch++) {
      WindowMicrokernelTester()
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_x32);
    }
  }

  TEST(S16_WINDOW__NEON_X32, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t batch = 1; batch <= 160; batch += 31) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(0)
          .Test(xnn_s16_window_ukernel__neon_x32);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X32, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t batch = 1; batch <= 160; batch += 31) {
        WindowMicrokernelTester()
          .rows(rows)
          .batch(batch)
          .shift(0)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_ukernel__neon_x32);
      }
    }
  }

  TEST(S16_WINDOW__NEON_X32, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      WindowMicrokernelTester()
        .rows(1)
        .batch(32)
        .shift(shift)
        .Test(xnn_s16_window_ukernel__neon_x32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


TEST(S16_WINDOW__SCALAR_X1, batch_eq_1) {
  WindowMicrokernelTester()
    .rows(1)
    .batch(1)
    .shift(0)
    .Test(xnn_s16_window_ukernel__scalar_x1);
}

TEST(S16_WINDOW__SCALAR_X1, batch_gt_1) {
  for (size_t batch = 2; batch < 10; batch++) {
    WindowMicrokernelTester()
      .batch(batch)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_x1);
  }
}

TEST(S16_WINDOW__SCALAR_X1, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t batch = 1; batch <= 5; batch += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__scalar_x1);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X1, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t batch = 1; batch <= 5; batch += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .shift(0)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_window_ukernel__scalar_x1);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X1, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    WindowMicrokernelTester()
      .rows(1)
      .batch(1)
      .shift(shift)
      .Test(xnn_s16_window_ukernel__scalar_x1);
  }
}


TEST(S16_WINDOW__SCALAR_X2, batch_eq_2) {
  WindowMicrokernelTester()
    .rows(1)
    .batch(2)
    .shift(0)
    .Test(xnn_s16_window_ukernel__scalar_x2);
}

TEST(S16_WINDOW__SCALAR_X2, batch_div_2) {
  for (size_t batch = 4; batch < 20; batch += 2) {
    WindowMicrokernelTester()
      .batch(batch)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_x2);
  }
}

TEST(S16_WINDOW__SCALAR_X2, batch_lt_2) {
  for (size_t batch = 1; batch < 2; batch++) {
    WindowMicrokernelTester()
      .batch(batch)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_x2);
  }
}

TEST(S16_WINDOW__SCALAR_X2, batch_gt_2) {
  for (size_t batch = 3; batch < 4; batch++) {
    WindowMicrokernelTester()
      .batch(batch)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_x2);
  }
}

TEST(S16_WINDOW__SCALAR_X2, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t batch = 1; batch <= 10; batch += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__scalar_x2);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X2, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t batch = 1; batch <= 10; batch += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .shift(0)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_window_ukernel__scalar_x2);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X2, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    WindowMicrokernelTester()
      .rows(1)
      .batch(2)
      .shift(shift)
      .Test(xnn_s16_window_ukernel__scalar_x2);
  }
}


TEST(S16_WINDOW__SCALAR_X3, batch_eq_3) {
  WindowMicrokernelTester()
    .rows(1)
    .batch(3)
    .shift(0)
    .Test(xnn_s16_window_ukernel__scalar_x3);
}

TEST(S16_WINDOW__SCALAR_X3, batch_div_3) {
  for (size_t batch = 6; batch < 30; batch += 3) {
    WindowMicrokernelTester()
      .batch(batch)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_x3);
  }
}

TEST(S16_WINDOW__SCALAR_X3, batch_lt_3) {
  for (size_t batch = 1; batch < 3; batch++) {
    WindowMicrokernelTester()
      .batch(batch)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_x3);
  }
}

TEST(S16_WINDOW__SCALAR_X3, batch_gt_3) {
  for (size_t batch = 4; batch < 6; batch++) {
    WindowMicrokernelTester()
      .batch(batch)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_x3);
  }
}

TEST(S16_WINDOW__SCALAR_X3, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t batch = 1; batch <= 15; batch += 2) {
      WindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__scalar_x3);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X3, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t batch = 1; batch <= 15; batch += 2) {
      WindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .shift(0)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_window_ukernel__scalar_x3);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X3, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    WindowMicrokernelTester()
      .rows(1)
      .batch(3)
      .shift(shift)
      .Test(xnn_s16_window_ukernel__scalar_x3);
  }
}


TEST(S16_WINDOW__SCALAR_X4, batch_eq_4) {
  WindowMicrokernelTester()
    .rows(1)
    .batch(4)
    .shift(0)
    .Test(xnn_s16_window_ukernel__scalar_x4);
}

TEST(S16_WINDOW__SCALAR_X4, batch_div_4) {
  for (size_t batch = 8; batch < 40; batch += 4) {
    WindowMicrokernelTester()
      .batch(batch)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_x4);
  }
}

TEST(S16_WINDOW__SCALAR_X4, batch_lt_4) {
  for (size_t batch = 1; batch < 4; batch++) {
    WindowMicrokernelTester()
      .batch(batch)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_x4);
  }
}

TEST(S16_WINDOW__SCALAR_X4, batch_gt_4) {
  for (size_t batch = 5; batch < 8; batch++) {
    WindowMicrokernelTester()
      .batch(batch)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_x4);
  }
}

TEST(S16_WINDOW__SCALAR_X4, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t batch = 1; batch <= 20; batch += 3) {
      WindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .shift(0)
        .Test(xnn_s16_window_ukernel__scalar_x4);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X4, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t batch = 1; batch <= 20; batch += 3) {
      WindowMicrokernelTester()
        .rows(rows)
        .batch(batch)
        .shift(0)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_window_ukernel__scalar_x4);
    }
  }
}

TEST(S16_WINDOW__SCALAR_X4, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    WindowMicrokernelTester()
      .rows(1)
      .batch(4)
      .shift(shift)
      .Test(xnn_s16_window_ukernel__scalar_x4);
  }
}
