// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/s16-window.yaml
//   Generator: tools/generate-window-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/window.h"
#include "window-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT12__NEON_U8, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(8)
      .shift(12)
      .Test(xnn_s16_window_shift12_ukernel__neon_u8);
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U8, channels_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 80; channels += 8) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_u8);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U8, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_u8);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U8, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_u8);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U8, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(12)
          .Test(xnn_s16_window_shift12_ukernel__neon_u8);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(12)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift12_ukernel__neon_u8);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT12__NEON_U16, channels_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(16)
      .shift(12)
      .Test(xnn_s16_window_shift12_ukernel__neon_u16);
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U16, channels_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 160; channels += 16) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_u16);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U16, channels_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_u16);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U16, channels_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_u16);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U16, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(12)
          .Test(xnn_s16_window_shift12_ukernel__neon_u16);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(12)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift12_ukernel__neon_u16);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT12__NEON_U24, channels_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(24)
      .shift(12)
      .Test(xnn_s16_window_shift12_ukernel__neon_u24);
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U24, channels_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 48; channels < 240; channels += 24) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_u24);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U24, channels_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 24; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_u24);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U24, channels_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 25; channels < 48; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_u24);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U24, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 120; channels += 23) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(12)
          .Test(xnn_s16_window_shift12_ukernel__neon_u24);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 120; channels += 23) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(12)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift12_ukernel__neon_u24);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT12__NEON_U32, channels_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(32)
      .shift(12)
      .Test(xnn_s16_window_shift12_ukernel__neon_u32);
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U32, channels_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 320; channels += 32) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_u32);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U32, channels_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_u32);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U32, channels_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(12)
        .Test(xnn_s16_window_shift12_ukernel__neon_u32);
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U32, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 160; channels += 31) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(12)
          .Test(xnn_s16_window_shift12_ukernel__neon_u32);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT12__NEON_U32, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 160; channels += 31) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(12)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift12_ukernel__neon_u32);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT15__NEON_U8, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(8)
      .shift(15)
      .Test(xnn_s16_window_shift15_ukernel__neon_u8);
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U8, channels_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 80; channels += 8) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_u8);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U8, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_u8);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U8, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_u8);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U8, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(15)
          .Test(xnn_s16_window_shift15_ukernel__neon_u8);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(15)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift15_ukernel__neon_u8);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT15__NEON_U16, channels_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(16)
      .shift(15)
      .Test(xnn_s16_window_shift15_ukernel__neon_u16);
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U16, channels_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 160; channels += 16) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_u16);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U16, channels_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_u16);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U16, channels_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_u16);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U16, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(15)
          .Test(xnn_s16_window_shift15_ukernel__neon_u16);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(15)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift15_ukernel__neon_u16);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT15__NEON_U24, channels_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(24)
      .shift(15)
      .Test(xnn_s16_window_shift15_ukernel__neon_u24);
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U24, channels_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 48; channels < 240; channels += 24) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_u24);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U24, channels_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 24; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_u24);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U24, channels_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 25; channels < 48; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_u24);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U24, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 120; channels += 23) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(15)
          .Test(xnn_s16_window_shift15_ukernel__neon_u24);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 120; channels += 23) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(15)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift15_ukernel__neon_u24);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW_SHIFT15__NEON_U32, channels_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(32)
      .shift(15)
      .Test(xnn_s16_window_shift15_ukernel__neon_u32);
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U32, channels_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 320; channels += 32) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_u32);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U32, channels_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_u32);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U32, channels_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(15)
        .Test(xnn_s16_window_shift15_ukernel__neon_u32);
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U32, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 160; channels += 31) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(15)
          .Test(xnn_s16_window_shift15_ukernel__neon_u32);
      }
    }
  }

  TEST(S16_WINDOW_SHIFT15__NEON_U32, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 160; channels += 31) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(15)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_shift15_ukernel__neon_u32);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW__NEON_U8, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(8)
      .shift(0)
      .Test(xnn_s16_window_ukernel__neon_u8);
  }

  TEST(S16_WINDOW__NEON_U8, channels_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 80; channels += 8) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_u8);
    }
  }

  TEST(S16_WINDOW__NEON_U8, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_u8);
    }
  }

  TEST(S16_WINDOW__NEON_U8, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_u8);
    }
  }

  TEST(S16_WINDOW__NEON_U8, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(0)
          .Test(xnn_s16_window_ukernel__neon_u8);
      }
    }
  }

  TEST(S16_WINDOW__NEON_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(0)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_ukernel__neon_u8);
      }
    }
  }

  TEST(S16_WINDOW__NEON_U8, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(8)
        .shift(shift)
        .Test(xnn_s16_window_ukernel__neon_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW__NEON_U16, channels_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(16)
      .shift(0)
      .Test(xnn_s16_window_ukernel__neon_u16);
  }

  TEST(S16_WINDOW__NEON_U16, channels_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 160; channels += 16) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_u16);
    }
  }

  TEST(S16_WINDOW__NEON_U16, channels_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_u16);
    }
  }

  TEST(S16_WINDOW__NEON_U16, channels_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_u16);
    }
  }

  TEST(S16_WINDOW__NEON_U16, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(0)
          .Test(xnn_s16_window_ukernel__neon_u16);
      }
    }
  }

  TEST(S16_WINDOW__NEON_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(0)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_ukernel__neon_u16);
      }
    }
  }

  TEST(S16_WINDOW__NEON_U16, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(16)
        .shift(shift)
        .Test(xnn_s16_window_ukernel__neon_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW__NEON_U24, channels_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(24)
      .shift(0)
      .Test(xnn_s16_window_ukernel__neon_u24);
  }

  TEST(S16_WINDOW__NEON_U24, channels_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 48; channels < 240; channels += 24) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_u24);
    }
  }

  TEST(S16_WINDOW__NEON_U24, channels_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 24; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_u24);
    }
  }

  TEST(S16_WINDOW__NEON_U24, channels_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 25; channels < 48; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_u24);
    }
  }

  TEST(S16_WINDOW__NEON_U24, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 120; channels += 23) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(0)
          .Test(xnn_s16_window_ukernel__neon_u24);
      }
    }
  }

  TEST(S16_WINDOW__NEON_U24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 120; channels += 23) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(0)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_ukernel__neon_u24);
      }
    }
  }

  TEST(S16_WINDOW__NEON_U24, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(24)
        .shift(shift)
        .Test(xnn_s16_window_ukernel__neon_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S16_WINDOW__NEON_U32, channels_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    WindowMicrokernelTester()
      .rows(1)
      .channels(32)
      .shift(0)
      .Test(xnn_s16_window_ukernel__neon_u32);
  }

  TEST(S16_WINDOW__NEON_U32, channels_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 64; channels < 320; channels += 32) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_u32);
    }
  }

  TEST(S16_WINDOW__NEON_U32, channels_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 32; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_u32);
    }
  }

  TEST(S16_WINDOW__NEON_U32, channels_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 33; channels < 64; channels++) {
      WindowMicrokernelTester()
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__neon_u32);
    }
  }

  TEST(S16_WINDOW__NEON_U32, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 160; channels += 31) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(0)
          .Test(xnn_s16_window_ukernel__neon_u32);
      }
    }
  }

  TEST(S16_WINDOW__NEON_U32, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 160; channels += 31) {
        WindowMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .shift(0)
          .inplace(true)
          .iterations(1)
          .Test(xnn_s16_window_ukernel__neon_u32);
      }
    }
  }

  TEST(S16_WINDOW__NEON_U32, shift) {
    TEST_REQUIRES_ARM_NEON;
    for (uint32_t shift = 0; shift < 32; shift++) {
      WindowMicrokernelTester()
        .rows(1)
        .channels(32)
        .shift(shift)
        .Test(xnn_s16_window_ukernel__neon_u32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


TEST(S16_WINDOW__SCALAR_U1, channels_eq_1) {
  WindowMicrokernelTester()
    .rows(1)
    .channels(1)
    .shift(0)
    .Test(xnn_s16_window_ukernel__scalar_u1);
}

TEST(S16_WINDOW__SCALAR_U1, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    WindowMicrokernelTester()
      .channels(channels)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_u1);
  }
}

TEST(S16_WINDOW__SCALAR_U1, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__scalar_u1);
    }
  }
}

TEST(S16_WINDOW__SCALAR_U1, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .shift(0)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_window_ukernel__scalar_u1);
    }
  }
}

TEST(S16_WINDOW__SCALAR_U1, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(1)
      .shift(shift)
      .Test(xnn_s16_window_ukernel__scalar_u1);
  }
}

TEST(S16_WINDOW__SCALAR_U2, channels_eq_2) {
  WindowMicrokernelTester()
    .rows(1)
    .channels(2)
    .shift(0)
    .Test(xnn_s16_window_ukernel__scalar_u2);
}

TEST(S16_WINDOW__SCALAR_U2, channels_div_2) {
  for (size_t channels = 4; channels < 20; channels += 2) {
    WindowMicrokernelTester()
      .channels(channels)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_u2);
  }
}

TEST(S16_WINDOW__SCALAR_U2, channels_lt_2) {
  for (size_t channels = 1; channels < 2; channels++) {
    WindowMicrokernelTester()
      .channels(channels)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_u2);
  }
}

TEST(S16_WINDOW__SCALAR_U2, channels_gt_2) {
  for (size_t channels = 3; channels < 4; channels++) {
    WindowMicrokernelTester()
      .channels(channels)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_u2);
  }
}

TEST(S16_WINDOW__SCALAR_U2, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__scalar_u2);
    }
  }
}

TEST(S16_WINDOW__SCALAR_U2, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t channels = 1; channels <= 10; channels += 1) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .shift(0)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_window_ukernel__scalar_u2);
    }
  }
}

TEST(S16_WINDOW__SCALAR_U2, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(2)
      .shift(shift)
      .Test(xnn_s16_window_ukernel__scalar_u2);
  }
}

TEST(S16_WINDOW__SCALAR_U3, channels_eq_3) {
  WindowMicrokernelTester()
    .rows(1)
    .channels(3)
    .shift(0)
    .Test(xnn_s16_window_ukernel__scalar_u3);
}

TEST(S16_WINDOW__SCALAR_U3, channels_div_3) {
  for (size_t channels = 6; channels < 30; channels += 3) {
    WindowMicrokernelTester()
      .channels(channels)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_u3);
  }
}

TEST(S16_WINDOW__SCALAR_U3, channels_lt_3) {
  for (size_t channels = 1; channels < 3; channels++) {
    WindowMicrokernelTester()
      .channels(channels)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_u3);
  }
}

TEST(S16_WINDOW__SCALAR_U3, channels_gt_3) {
  for (size_t channels = 4; channels < 6; channels++) {
    WindowMicrokernelTester()
      .channels(channels)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_u3);
  }
}

TEST(S16_WINDOW__SCALAR_U3, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 15; channels += 2) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__scalar_u3);
    }
  }
}

TEST(S16_WINDOW__SCALAR_U3, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t channels = 1; channels <= 15; channels += 2) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .shift(0)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_window_ukernel__scalar_u3);
    }
  }
}

TEST(S16_WINDOW__SCALAR_U3, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(3)
      .shift(shift)
      .Test(xnn_s16_window_ukernel__scalar_u3);
  }
}

TEST(S16_WINDOW__SCALAR_U4, channels_eq_4) {
  WindowMicrokernelTester()
    .rows(1)
    .channels(4)
    .shift(0)
    .Test(xnn_s16_window_ukernel__scalar_u4);
}

TEST(S16_WINDOW__SCALAR_U4, channels_div_4) {
  for (size_t channels = 8; channels < 40; channels += 4) {
    WindowMicrokernelTester()
      .channels(channels)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_u4);
  }
}

TEST(S16_WINDOW__SCALAR_U4, channels_lt_4) {
  for (size_t channels = 1; channels < 4; channels++) {
    WindowMicrokernelTester()
      .channels(channels)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_u4);
  }
}

TEST(S16_WINDOW__SCALAR_U4, channels_gt_4) {
  for (size_t channels = 5; channels < 8; channels++) {
    WindowMicrokernelTester()
      .channels(channels)
      .shift(0)
      .Test(xnn_s16_window_ukernel__scalar_u4);
  }
}

TEST(S16_WINDOW__SCALAR_U4, rows_gt_1) {
  for (size_t rows = 2; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .shift(0)
        .Test(xnn_s16_window_ukernel__scalar_u4);
    }
  }
}

TEST(S16_WINDOW__SCALAR_U4, inplace) {
  for (size_t rows = 1; rows <= 3; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      WindowMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .shift(0)
        .inplace(true)
        .iterations(1)
        .Test(xnn_s16_window_ukernel__scalar_u4);
    }
  }
}

TEST(S16_WINDOW__SCALAR_U4, shift) {
  for (uint32_t shift = 0; shift < 32; shift++) {
    WindowMicrokernelTester()
      .rows(1)
      .channels(4)
      .shift(shift)
      .Test(xnn_s16_window_ukernel__scalar_u4);
  }
}