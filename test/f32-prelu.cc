// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-prelu.yaml
//   Generator: tools/generate-prelu-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/prelu.h"
#include "prelu-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PRELU__NEON_1X4, channels_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PReLUMicrokernelTester()
      .rows(1)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__neon_1x4);
  }

  TEST(F32_PRELU__NEON_1X4, channels_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_1x4);
    }
  }

  TEST(F32_PRELU__NEON_1X4, channels_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_1x4);
    }
  }

  TEST(F32_PRELU__NEON_1X4, channels_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_1x4);
    }
  }

  TEST(F32_PRELU__NEON_1X4, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_1x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_1X4, input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_1x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_1X4, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_1x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_1X4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_1x4);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PRELU__NEON_1X8, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PReLUMicrokernelTester()
      .rows(1)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__neon_1x8);
  }

  TEST(F32_PRELU__NEON_1X8, channels_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_1x8);
    }
  }

  TEST(F32_PRELU__NEON_1X8, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_1x8);
    }
  }

  TEST(F32_PRELU__NEON_1X8, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_1x8);
    }
  }

  TEST(F32_PRELU__NEON_1X8, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_1x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_1X8, input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_1x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_1X8, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_1x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_1X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_1x8);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PRELU__NEON_1X16, channels_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    PReLUMicrokernelTester()
      .rows(1)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__neon_1x16);
  }

  TEST(F32_PRELU__NEON_1X16, channels_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_1x16);
    }
  }

  TEST(F32_PRELU__NEON_1X16, channels_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_1x16);
    }
  }

  TEST(F32_PRELU__NEON_1X16, channels_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_1x16);
    }
  }

  TEST(F32_PRELU__NEON_1X16, rows_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_1x16);
      }
    }
  }

  TEST(F32_PRELU__NEON_1X16, input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_1x16);
      }
    }
  }

  TEST(F32_PRELU__NEON_1X16, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_1x16);
      }
    }
  }

  TEST(F32_PRELU__NEON_1X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_1x16);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PRELU__NEON_2X4, channels_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__neon_2x4);
  }

  TEST(F32_PRELU__NEON_2X4, channels_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_2x4);
    }
  }

  TEST(F32_PRELU__NEON_2X4, channels_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_2x4);
    }
  }

  TEST(F32_PRELU__NEON_2X4, channels_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_2x4);
    }
  }

  TEST(F32_PRELU__NEON_2X4, rows_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_2x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X4, rows_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_2x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X4, rows_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_2x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X4, input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X4, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x4);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PRELU__NEON_2X8, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__neon_2x8);
  }

  TEST(F32_PRELU__NEON_2X8, channels_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_2x8);
    }
  }

  TEST(F32_PRELU__NEON_2X8, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_2x8);
    }
  }

  TEST(F32_PRELU__NEON_2X8, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_2x8);
    }
  }

  TEST(F32_PRELU__NEON_2X8, rows_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_2x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X8, rows_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_2x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X8, rows_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_2x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X8, input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X8, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x8);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PRELU__NEON_2X16, channels_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__neon_2x16);
  }

  TEST(F32_PRELU__NEON_2X16, channels_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_2x16);
    }
  }

  TEST(F32_PRELU__NEON_2X16, channels_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_2x16);
    }
  }

  TEST(F32_PRELU__NEON_2X16, channels_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_2x16);
    }
  }

  TEST(F32_PRELU__NEON_2X16, rows_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_2x16);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X16, rows_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_2x16);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X16, rows_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_2x16);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X16, input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x16);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X16, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x16);
      }
    }
  }

  TEST(F32_PRELU__NEON_2X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_2x16);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PRELU__NEON_4X4, channels_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    PReLUMicrokernelTester()
      .rows(4)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__neon_4x4);
  }

  TEST(F32_PRELU__NEON_4X4, channels_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_4x4);
    }
  }

  TEST(F32_PRELU__NEON_4X4, channels_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_4x4);
    }
  }

  TEST(F32_PRELU__NEON_4X4, channels_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_4x4);
    }
  }

  TEST(F32_PRELU__NEON_4X4, rows_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_4x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_4X4, rows_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 8; rows <= 16; rows += 4) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_4x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_4X4, rows_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 5; rows < 8; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_4x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_4X4, input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_4x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_4X4, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_4x4);
      }
    }
  }

  TEST(F32_PRELU__NEON_4X4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_4x4);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PRELU__NEON_4X8, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    PReLUMicrokernelTester()
      .rows(4)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__neon_4x8);
  }

  TEST(F32_PRELU__NEON_4X8, channels_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_4x8);
    }
  }

  TEST(F32_PRELU__NEON_4X8, channels_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_4x8);
    }
  }

  TEST(F32_PRELU__NEON_4X8, channels_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_4x8);
    }
  }

  TEST(F32_PRELU__NEON_4X8, rows_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_4x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_4X8, rows_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 8; rows <= 16; rows += 4) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_4x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_4X8, rows_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 5; rows < 8; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_4x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_4X8, input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_4x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_4X8, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_4x8);
      }
    }
  }

  TEST(F32_PRELU__NEON_4X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_4x8);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PRELU__NEON_4X16, channels_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    PReLUMicrokernelTester()
      .rows(4)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__neon_4x16);
  }

  TEST(F32_PRELU__NEON_4X16, channels_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_4x16);
    }
  }

  TEST(F32_PRELU__NEON_4X16, channels_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_4x16);
    }
  }

  TEST(F32_PRELU__NEON_4X16, channels_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__neon_4x16);
    }
  }

  TEST(F32_PRELU__NEON_4X16, rows_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_4x16);
      }
    }
  }

  TEST(F32_PRELU__NEON_4X16, rows_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 8; rows <= 16; rows += 4) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_4x16);
      }
    }
  }

  TEST(F32_PRELU__NEON_4X16, rows_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 5; rows < 8; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__neon_4x16);
      }
    }
  }

  TEST(F32_PRELU__NEON_4X16, input_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_4x16);
      }
    }
  }

  TEST(F32_PRELU__NEON_4X16, output_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_4x16);
      }
    }
  }

  TEST(F32_PRELU__NEON_4X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__neon_4x16);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PRELU__SSE_2X4, channels_eq_4) {
    TEST_REQUIRES_X86_SSE;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__sse_2x4);
  }

  TEST(F32_PRELU__SSE_2X4, channels_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse_2x4);
    }
  }

  TEST(F32_PRELU__SSE_2X4, channels_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse_2x4);
    }
  }

  TEST(F32_PRELU__SSE_2X4, channels_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse_2x4);
    }
  }

  TEST(F32_PRELU__SSE_2X4, rows_lt_2) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE_2X4, rows_div_2) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE_2X4, rows_gt_2) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE_2X4, input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE_2X4, output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE_2X4, inplace) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse_2x4);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PRELU__SSE_2X8, channels_eq_8) {
    TEST_REQUIRES_X86_SSE;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__sse_2x8);
  }

  TEST(F32_PRELU__SSE_2X8, channels_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse_2x8);
    }
  }

  TEST(F32_PRELU__SSE_2X8, channels_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse_2x8);
    }
  }

  TEST(F32_PRELU__SSE_2X8, channels_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse_2x8);
    }
  }

  TEST(F32_PRELU__SSE_2X8, rows_lt_2) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE_2X8, rows_div_2) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE_2X8, rows_gt_2) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE_2X8, input_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE_2X8, output_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE_2X8, inplace) {
    TEST_REQUIRES_X86_SSE;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse_2x8);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PRELU__SSE2_2X4, channels_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__sse2_2x4);
  }

  TEST(F32_PRELU__SSE2_2X4, channels_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse2_2x4);
    }
  }

  TEST(F32_PRELU__SSE2_2X4, channels_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse2_2x4);
    }
  }

  TEST(F32_PRELU__SSE2_2X4, channels_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse2_2x4);
    }
  }

  TEST(F32_PRELU__SSE2_2X4, rows_lt_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse2_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X4, rows_div_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse2_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X4, rows_gt_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse2_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X4, input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X4, output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x4);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PRELU__SSE2_2X8, channels_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__sse2_2x8);
  }

  TEST(F32_PRELU__SSE2_2X8, channels_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse2_2x8);
    }
  }

  TEST(F32_PRELU__SSE2_2X8, channels_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse2_2x8);
    }
  }

  TEST(F32_PRELU__SSE2_2X8, channels_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse2_2x8);
    }
  }

  TEST(F32_PRELU__SSE2_2X8, rows_lt_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse2_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X8, rows_div_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse2_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X8, rows_gt_2) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse2_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X8, input_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X8, output_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE2_2X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse2_2x8);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PRELU__SSE41_2X4, channels_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__sse41_2x4);
  }

  TEST(F32_PRELU__SSE41_2X4, channels_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse41_2x4);
    }
  }

  TEST(F32_PRELU__SSE41_2X4, channels_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse41_2x4);
    }
  }

  TEST(F32_PRELU__SSE41_2X4, channels_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse41_2x4);
    }
  }

  TEST(F32_PRELU__SSE41_2X4, rows_lt_2) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse41_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X4, rows_div_2) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse41_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X4, rows_gt_2) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse41_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X4, input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X4, output_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x4);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x4);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PRELU__SSE41_2X8, channels_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__sse41_2x8);
  }

  TEST(F32_PRELU__SSE41_2X8, channels_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse41_2x8);
    }
  }

  TEST(F32_PRELU__SSE41_2X8, channels_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse41_2x8);
    }
  }

  TEST(F32_PRELU__SSE41_2X8, channels_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__sse41_2x8);
    }
  }

  TEST(F32_PRELU__SSE41_2X8, rows_lt_2) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse41_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X8, rows_div_2) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse41_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X8, rows_gt_2) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__sse41_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X8, input_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X8, output_stride) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x8);
      }
    }
  }

  TEST(F32_PRELU__SSE41_2X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__sse41_2x8);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PRELU__AVX_2X8, channels_eq_8) {
    TEST_REQUIRES_X86_AVX;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__avx_2x8);
  }

  TEST(F32_PRELU__AVX_2X8, channels_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__avx_2x8);
    }
  }

  TEST(F32_PRELU__AVX_2X8, channels_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__avx_2x8);
    }
  }

  TEST(F32_PRELU__AVX_2X8, channels_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__avx_2x8);
    }
  }

  TEST(F32_PRELU__AVX_2X8, rows_lt_2) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__avx_2x8);
      }
    }
  }

  TEST(F32_PRELU__AVX_2X8, rows_div_2) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__avx_2x8);
      }
    }
  }

  TEST(F32_PRELU__AVX_2X8, rows_gt_2) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__avx_2x8);
      }
    }
  }

  TEST(F32_PRELU__AVX_2X8, input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__avx_2x8);
      }
    }
  }

  TEST(F32_PRELU__AVX_2X8, output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__avx_2x8);
      }
    }
  }

  TEST(F32_PRELU__AVX_2X8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__avx_2x8);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PRELU__AVX_2X16, channels_eq_16) {
    TEST_REQUIRES_X86_AVX;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__avx_2x16);
  }

  TEST(F32_PRELU__AVX_2X16, channels_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__avx_2x16);
    }
  }

  TEST(F32_PRELU__AVX_2X16, channels_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__avx_2x16);
    }
  }

  TEST(F32_PRELU__AVX_2X16, channels_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__avx_2x16);
    }
  }

  TEST(F32_PRELU__AVX_2X16, rows_lt_2) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__avx_2x16);
      }
    }
  }

  TEST(F32_PRELU__AVX_2X16, rows_div_2) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__avx_2x16);
      }
    }
  }

  TEST(F32_PRELU__AVX_2X16, rows_gt_2) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__avx_2x16);
      }
    }
  }

  TEST(F32_PRELU__AVX_2X16, input_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__avx_2x16);
      }
    }
  }

  TEST(F32_PRELU__AVX_2X16, output_stride) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__avx_2x16);
      }
    }
  }

  TEST(F32_PRELU__AVX_2X16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__avx_2x16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PRELU__AVX512F_2X16, channels_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__avx512f_2x16);
  }

  TEST(F32_PRELU__AVX512F_2X16, channels_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__avx512f_2x16);
    }
  }

  TEST(F32_PRELU__AVX512F_2X16, channels_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__avx512f_2x16);
    }
  }

  TEST(F32_PRELU__AVX512F_2X16, channels_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__avx512f_2x16);
    }
  }

  TEST(F32_PRELU__AVX512F_2X16, rows_lt_2) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__avx512f_2x16);
      }
    }
  }

  TEST(F32_PRELU__AVX512F_2X16, rows_div_2) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__avx512f_2x16);
      }
    }
  }

  TEST(F32_PRELU__AVX512F_2X16, rows_gt_2) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__avx512f_2x16);
      }
    }
  }

  TEST(F32_PRELU__AVX512F_2X16, input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__avx512f_2x16);
      }
    }
  }

  TEST(F32_PRELU__AVX512F_2X16, output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__avx512f_2x16);
      }
    }
  }

  TEST(F32_PRELU__AVX512F_2X16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__avx512f_2x16);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PRELU__AVX512F_2X32, channels_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    PReLUMicrokernelTester()
      .rows(2)
      .channels(32)
      .Test(xnn_f32_prelu_ukernel__avx512f_2x32);
  }

  TEST(F32_PRELU__AVX512F_2X32, channels_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 64; channels < 320; channels += 32) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__avx512f_2x32);
    }
  }

  TEST(F32_PRELU__AVX512F_2X32, channels_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 1; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__avx512f_2x32);
    }
  }

  TEST(F32_PRELU__AVX512F_2X32, channels_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t channels = 33; channels < 64; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__avx512f_2x32);
    }
  }

  TEST(F32_PRELU__AVX512F_2X32, rows_lt_2) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 160; channels += 31) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__avx512f_2x32);
      }
    }
  }

  TEST(F32_PRELU__AVX512F_2X32, rows_div_2) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 160; channels += 31) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__avx512f_2x32);
      }
    }
  }

  TEST(F32_PRELU__AVX512F_2X32, rows_gt_2) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 160; channels += 31) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__avx512f_2x32);
      }
    }
  }

  TEST(F32_PRELU__AVX512F_2X32, input_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 160; channels += 31) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(163)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__avx512f_2x32);
      }
    }
  }

  TEST(F32_PRELU__AVX512F_2X32, output_stride) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 160; channels += 31) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(163)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__avx512f_2x32);
      }
    }
  }

  TEST(F32_PRELU__AVX512F_2X32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 160; channels += 31) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__avx512f_2x32);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X4, channels_eq_4) {
    PReLUMicrokernelTester()
      .rows(1)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x4);
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X4, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X4, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X4, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X4, rows_gt_1) {
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X4, input_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X4, output_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X4, inplace) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x4);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X8, channels_eq_8) {
    PReLUMicrokernelTester()
      .rows(1)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x8);
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X8, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X8, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X8, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X8, rows_gt_1) {
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X8, input_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X8, output_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X8, inplace) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x8);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X16, channels_eq_16) {
    PReLUMicrokernelTester()
      .rows(1)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x16);
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X16, channels_div_16) {
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X16, channels_lt_16) {
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X16, channels_gt_16) {
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X16, rows_gt_1) {
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X16, input_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X16, output_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_1X16, inplace) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_1x16);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X4, channels_eq_4) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x4);
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X4, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X4, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X4, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X4, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X4, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X4, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X4, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X4, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X4, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x4);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X8, channels_eq_8) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x8);
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X8, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X8, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X8, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X8, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X8, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X8, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X8, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X8, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X8, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x8);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X16, channels_eq_16) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x16);
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X16, channels_div_16) {
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X16, channels_lt_16) {
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X16, channels_gt_16) {
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X16, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X16, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X16, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X16, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X16, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_2X16, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_2x16);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X4, channels_eq_4) {
    PReLUMicrokernelTester()
      .rows(4)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x4);
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X4, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X4, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X4, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X4, rows_lt_4) {
    for (size_t rows = 1; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X4, rows_div_4) {
    for (size_t rows = 8; rows <= 16; rows += 4) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X4, rows_gt_4) {
    for (size_t rows = 5; rows < 8; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X4, input_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X4, output_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X4, inplace) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x4);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X8, channels_eq_8) {
    PReLUMicrokernelTester()
      .rows(4)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x8);
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X8, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X8, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X8, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X8, rows_lt_4) {
    for (size_t rows = 1; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X8, rows_div_4) {
    for (size_t rows = 8; rows <= 16; rows += 4) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X8, rows_gt_4) {
    for (size_t rows = 5; rows < 8; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X8, input_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X8, output_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X8, inplace) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x8);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X16, channels_eq_16) {
    PReLUMicrokernelTester()
      .rows(4)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x16);
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X16, channels_div_16) {
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X16, channels_lt_16) {
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X16, channels_gt_16) {
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X16, rows_lt_4) {
    for (size_t rows = 1; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X16, rows_div_4) {
    for (size_t rows = 8; rows <= 16; rows += 4) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X16, rows_gt_4) {
    for (size_t rows = 5; rows < 8; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X16, input_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X16, output_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_IMINMAX_4X16, inplace) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_iminmax_4x16);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X4, channels_eq_4) {
    PReLUMicrokernelTester()
      .rows(1)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x4);
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X4, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X4, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X4, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X4, rows_gt_1) {
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X4, input_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X4, output_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X4, inplace) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x4);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X8, channels_eq_8) {
    PReLUMicrokernelTester()
      .rows(1)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x8);
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X8, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X8, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X8, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X8, rows_gt_1) {
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X8, input_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X8, output_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X8, inplace) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x8);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X16, channels_eq_16) {
    PReLUMicrokernelTester()
      .rows(1)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x16);
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X16, channels_div_16) {
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X16, channels_lt_16) {
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X16, channels_gt_16) {
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X16, rows_gt_1) {
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X16, input_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X16, output_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_1X16, inplace) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_1x16);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X4, channels_eq_4) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x4);
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X4, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X4, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X4, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X4, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X4, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X4, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X4, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X4, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X4, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x4);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X8, channels_eq_8) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x8);
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X8, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X8, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X8, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X8, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X8, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X8, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X8, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X8, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X8, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x8);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X16, channels_eq_16) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x16);
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X16, channels_div_16) {
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X16, channels_lt_16) {
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X16, channels_gt_16) {
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X16, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X16, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X16, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X16, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X16, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_2X16, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_2x16);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X4, channels_eq_4) {
    PReLUMicrokernelTester()
      .rows(4)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x4);
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X4, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X4, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X4, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x4);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X4, rows_lt_4) {
    for (size_t rows = 1; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X4, rows_div_4) {
    for (size_t rows = 8; rows <= 16; rows += 4) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X4, rows_gt_4) {
    for (size_t rows = 5; rows < 8; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X4, input_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X4, output_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X4, inplace) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x4);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X8, channels_eq_8) {
    PReLUMicrokernelTester()
      .rows(4)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x8);
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X8, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X8, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X8, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x8);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X8, rows_lt_4) {
    for (size_t rows = 1; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X8, rows_div_4) {
    for (size_t rows = 8; rows <= 16; rows += 4) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X8, rows_gt_4) {
    for (size_t rows = 5; rows < 8; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X8, input_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X8, output_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X8, inplace) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x8);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X16, channels_eq_16) {
    PReLUMicrokernelTester()
      .rows(4)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x16);
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X16, channels_div_16) {
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X16, channels_lt_16) {
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X16, channels_gt_16) {
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x16);
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X16, rows_lt_4) {
    for (size_t rows = 1; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X16, rows_div_4) {
    for (size_t rows = 8; rows <= 16; rows += 4) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X16, rows_gt_4) {
    for (size_t rows = 5; rows < 8; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X16, input_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X16, output_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMSIMD_LANESELECT_4X16, inplace) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmsimd_laneselect_4x16);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X4, channels_eq_4) {
    PReLUMicrokernelTester()
      .rows(1)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x4);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X4, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X4, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X4, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X4, rows_gt_1) {
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X4, input_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X4, output_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X4, inplace) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x4);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X8, channels_eq_8) {
    PReLUMicrokernelTester()
      .rows(1)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x8);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X8, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X8, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X8, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X8, rows_gt_1) {
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X8, input_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X8, output_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X8, inplace) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x8);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X16, channels_eq_16) {
    PReLUMicrokernelTester()
      .rows(1)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x16);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X16, channels_div_16) {
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X16, channels_lt_16) {
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X16, channels_gt_16) {
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X16, rows_gt_1) {
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X16, input_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X16, output_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_1X16, inplace) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_1x16);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X4, channels_eq_4) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x4);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X4, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X4, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X4, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X4, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X4, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X4, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X4, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X4, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X4, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x4);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X8, channels_eq_8) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x8);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X8, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X8, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X8, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X8, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X8, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X8, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X8, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X8, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X8, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x8);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X16, channels_eq_16) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x16);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X16, channels_div_16) {
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X16, channels_lt_16) {
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X16, channels_gt_16) {
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X16, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X16, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X16, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X16, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X16, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_2X16, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_2x16);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X4, channels_eq_4) {
    PReLUMicrokernelTester()
      .rows(4)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x4);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X4, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X4, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X4, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X4, rows_lt_4) {
    for (size_t rows = 1; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X4, rows_div_4) {
    for (size_t rows = 8; rows <= 16; rows += 4) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X4, rows_gt_4) {
    for (size_t rows = 5; rows < 8; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X4, input_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X4, output_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X4, inplace) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x4);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X8, channels_eq_8) {
    PReLUMicrokernelTester()
      .rows(4)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x8);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X8, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X8, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X8, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X8, rows_lt_4) {
    for (size_t rows = 1; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X8, rows_div_4) {
    for (size_t rows = 8; rows <= 16; rows += 4) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X8, rows_gt_4) {
    for (size_t rows = 5; rows < 8; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X8, input_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X8, output_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X8, inplace) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x8);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X16, channels_eq_16) {
    PReLUMicrokernelTester()
      .rows(4)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x16);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X16, channels_div_16) {
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X16, channels_lt_16) {
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X16, channels_gt_16) {
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X16, rows_lt_4) {
    for (size_t rows = 1; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X16, rows_div_4) {
    for (size_t rows = 8; rows <= 16; rows += 4) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X16, rows_gt_4) {
    for (size_t rows = 5; rows < 8; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X16, input_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X16, output_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_IMINMAX_4X16, inplace) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_iminmax_4x16);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X4, channels_eq_4) {
    PReLUMicrokernelTester()
      .rows(1)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x4);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X4, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X4, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X4, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X4, rows_gt_1) {
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X4, input_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X4, output_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X4, inplace) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x4);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X8, channels_eq_8) {
    PReLUMicrokernelTester()
      .rows(1)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x8);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X8, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X8, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X8, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X8, rows_gt_1) {
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X8, input_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X8, output_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X8, inplace) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x8);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X16, channels_eq_16) {
    PReLUMicrokernelTester()
      .rows(1)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x16);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X16, channels_div_16) {
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X16, channels_lt_16) {
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X16, channels_gt_16) {
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(1)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X16, rows_gt_1) {
    for (size_t rows = 2; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X16, input_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X16, output_stride) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_1X16, inplace) {
    for (size_t rows = 1; rows <= 3; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_1x16);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X4, channels_eq_4) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x4);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X4, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X4, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X4, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X4, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X4, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X4, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X4, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X4, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X4, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x4);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X8, channels_eq_8) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x8);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X8, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X8, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X8, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X8, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X8, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X8, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X8, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X8, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X8, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x8);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X16, channels_eq_16) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x16);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X16, channels_div_16) {
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X16, channels_lt_16) {
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X16, channels_gt_16) {
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X16, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X16, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X16, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X16, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X16, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_2X16, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_2x16);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X4, channels_eq_4) {
    PReLUMicrokernelTester()
      .rows(4)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x4);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X4, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X4, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X4, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x4);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X4, rows_lt_4) {
    for (size_t rows = 1; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X4, rows_div_4) {
    for (size_t rows = 8; rows <= 16; rows += 4) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X4, rows_gt_4) {
    for (size_t rows = 5; rows < 8; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X4, input_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X4, output_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x4);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X4, inplace) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x4);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X8, channels_eq_8) {
    PReLUMicrokernelTester()
      .rows(4)
      .channels(8)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x8);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X8, channels_div_8) {
    for (size_t channels = 16; channels < 80; channels += 8) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X8, channels_lt_8) {
    for (size_t channels = 1; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X8, channels_gt_8) {
    for (size_t channels = 9; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x8);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X8, rows_lt_4) {
    for (size_t rows = 1; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X8, rows_div_4) {
    for (size_t rows = 8; rows <= 16; rows += 4) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X8, rows_gt_4) {
    for (size_t rows = 5; rows < 8; rows++) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X8, input_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X8, output_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(43)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x8);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X8, inplace) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 40; channels += 7) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x8);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X16, channels_eq_16) {
    PReLUMicrokernelTester()
      .rows(4)
      .channels(16)
      .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x16);
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X16, channels_div_16) {
    for (size_t channels = 32; channels < 160; channels += 16) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X16, channels_lt_16) {
    for (size_t channels = 1; channels < 16; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X16, channels_gt_16) {
    for (size_t channels = 17; channels < 32; channels++) {
      PReLUMicrokernelTester()
        .rows(4)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x16);
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X16, rows_lt_4) {
    for (size_t rows = 1; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X16, rows_div_4) {
    for (size_t rows = 8; rows <= 16; rows += 4) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X16, rows_gt_4) {
    for (size_t rows = 5; rows < 8; rows++) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X16, input_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X16, output_stride) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(83)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x16);
      }
    }
  }

  TEST(F32_PRELU__WASMRELAXEDSIMD_LANESELECT_4X16, inplace) {
    for (size_t rows = 1; rows <= 12; rows += 3) {
      for (size_t channels = 1; channels <= 80; channels += 15) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasmrelaxedsimd_laneselect_4x16);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASM_2X1, channels_eq_1) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(1)
      .Test(xnn_f32_prelu_ukernel__wasm_2x1);
  }

  TEST(F32_PRELU__WASM_2X1, channels_gt_1) {
    for (size_t channels = 2; channels < 10; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasm_2x1);
    }
  }

  TEST(F32_PRELU__WASM_2X1, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasm_2x1);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X1, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasm_2x1);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X1, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasm_2x1);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X1, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(7)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x1);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X1, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(7)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x1);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X1, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 5; channels += 1) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x1);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_PRELU__WASM_2X4, channels_eq_4) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(4)
      .Test(xnn_f32_prelu_ukernel__wasm_2x4);
  }

  TEST(F32_PRELU__WASM_2X4, channels_div_4) {
    for (size_t channels = 8; channels < 40; channels += 4) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasm_2x4);
    }
  }

  TEST(F32_PRELU__WASM_2X4, channels_lt_4) {
    for (size_t channels = 1; channels < 4; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasm_2x4);
    }
  }

  TEST(F32_PRELU__WASM_2X4, channels_gt_4) {
    for (size_t channels = 5; channels < 8; channels++) {
      PReLUMicrokernelTester()
        .rows(2)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__wasm_2x4);
    }
  }

  TEST(F32_PRELU__WASM_2X4, rows_lt_2) {
    for (size_t rows = 1; rows < 2; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasm_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X4, rows_div_2) {
    for (size_t rows = 4; rows <= 8; rows += 2) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasm_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X4, rows_gt_2) {
    for (size_t rows = 3; rows < 4; rows++) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .Test(xnn_f32_prelu_ukernel__wasm_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X4, input_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .input_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X4, output_stride) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .output_stride(23)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x4);
      }
    }
  }

  TEST(F32_PRELU__WASM_2X4, inplace) {
    for (size_t rows = 1; rows <= 6; rows += 1) {
      for (size_t channels = 1; channels <= 20; channels += 3) {
        PReLUMicrokernelTester()
          .rows(rows)
          .channels(channels)
          .inplace(true)
          .iterations(1)
          .Test(xnn_f32_prelu_ukernel__wasm_2x4);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_PRELU__SCALAR_2X1, channels_eq_1) {
  PReLUMicrokernelTester()
    .rows(2)
    .channels(1)
    .Test(xnn_f32_prelu_ukernel__scalar_2x1);
}

TEST(F32_PRELU__SCALAR_2X1, channels_gt_1) {
  for (size_t channels = 2; channels < 10; channels++) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(channels)
      .Test(xnn_f32_prelu_ukernel__scalar_2x1);
  }
}

TEST(F32_PRELU__SCALAR_2X1, rows_lt_2) {
  for (size_t rows = 1; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__scalar_2x1);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X1, rows_div_2) {
  for (size_t rows = 4; rows <= 8; rows += 2) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__scalar_2x1);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X1, rows_gt_2) {
  for (size_t rows = 3; rows < 4; rows++) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__scalar_2x1);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X1, input_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(7)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x1);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X1, output_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .output_stride(7)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x1);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X1, inplace) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 5; channels += 1) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .inplace(true)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x1);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X4, channels_eq_4) {
  PReLUMicrokernelTester()
    .rows(2)
    .channels(4)
    .Test(xnn_f32_prelu_ukernel__scalar_2x4);
}

TEST(F32_PRELU__SCALAR_2X4, channels_div_4) {
  for (size_t channels = 8; channels < 40; channels += 4) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(channels)
      .Test(xnn_f32_prelu_ukernel__scalar_2x4);
  }
}

TEST(F32_PRELU__SCALAR_2X4, channels_lt_4) {
  for (size_t channels = 1; channels < 4; channels++) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(channels)
      .Test(xnn_f32_prelu_ukernel__scalar_2x4);
  }
}

TEST(F32_PRELU__SCALAR_2X4, channels_gt_4) {
  for (size_t channels = 5; channels < 8; channels++) {
    PReLUMicrokernelTester()
      .rows(2)
      .channels(channels)
      .Test(xnn_f32_prelu_ukernel__scalar_2x4);
  }
}

TEST(F32_PRELU__SCALAR_2X4, rows_lt_2) {
  for (size_t rows = 1; rows < 2; rows++) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__scalar_2x4);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X4, rows_div_2) {
  for (size_t rows = 4; rows <= 8; rows += 2) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__scalar_2x4);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X4, rows_gt_2) {
  for (size_t rows = 3; rows < 4; rows++) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .Test(xnn_f32_prelu_ukernel__scalar_2x4);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X4, input_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .input_stride(23)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x4);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X4, output_stride) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .output_stride(23)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x4);
    }
  }
}

TEST(F32_PRELU__SCALAR_2X4, inplace) {
  for (size_t rows = 1; rows <= 6; rows += 1) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      PReLUMicrokernelTester()
        .rows(rows)
        .channels(channels)
        .inplace(true)
        .iterations(1)
        .Test(xnn_f32_prelu_ukernel__scalar_2x4);
    }
  }
}