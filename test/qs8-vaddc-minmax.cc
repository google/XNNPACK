// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-vaddc-minmax.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vadd.h>
#include "vaddc-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VADDC_MINMAX__NEON_LD64_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VAddCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x8);
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X8, a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X8, b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X8, y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X8, a_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X8, b_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X8, y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X8, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X8, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VADDC_MINMAX__NEON_LD64_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VAddCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x16);
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X16, a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X16, b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X16, y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X16, a_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X16, b_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X16, y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X16, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X16, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VADDC_MINMAX__NEON_LD64_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VAddCMicrokernelTester()
      .batch_size(24)
      .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x24);
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X24, a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X24, b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X24, y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X24, a_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X24, b_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X24, y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X24, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X24, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_VADDC_MINMAX__NEON_LD64_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    VAddCMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x32);
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X32, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X32, a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X32, b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X32, y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X32, a_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X32, b_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X32, y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X32, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__NEON_LD64_X32, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__neon_ld64_x32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VAddCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8);
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X8, a_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X8, b_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X8, y_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X8, a_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X8, b_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X8, y_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X8, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X8, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VAddCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x16);
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X16, a_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X16, b_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X16, y_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X16, a_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X16, b_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X16, y_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X16, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X16, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    VAddCMicrokernelTester()
      .batch_size(24)
      .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x24);
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X24, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X24, a_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X24, b_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X24, y_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X24, a_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X24, b_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X24, y_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X24, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X24, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE2;
    VAddCMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x32);
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X32, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X32, a_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X32, b_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X32, y_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X32, a_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X32, b_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X32, y_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X32, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE2_MUL16_LD64_X32, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse2_mul16_ld64_x32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VAddCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8);
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X8, a_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X8, b_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X8, y_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X8, a_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X8, b_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X8, y_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X8, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X8, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VAddCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x16);
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X16, a_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X16, b_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X16, y_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X16, a_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X16, b_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X16, y_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X16, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X16, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VAddCMicrokernelTester()
      .batch_size(24)
      .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x24);
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X24, a_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X24, b_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X24, y_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X24, a_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X24, b_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X24, y_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X24, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X24, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE41;
    VAddCMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x32);
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X32, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X32, a_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X32, b_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X32, y_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X32, a_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X32, b_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X32, y_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X32, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL16_LD64_X32, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul16_ld64_x32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VAddCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x8);
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X8, a_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X8, b_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X8, y_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X8, a_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X8, b_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X8, y_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X8, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X8, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VAddCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x16);
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X16, a_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X16, b_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X16, y_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X16, a_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X16, b_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X16, y_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X16, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X16, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VAddCMicrokernelTester()
      .batch_size(24)
      .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x24);
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X24, a_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X24, b_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X24, y_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X24, a_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X24, b_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X24, y_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X24, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X24, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE41;
    VAddCMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x32);
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X32, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X32, a_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X32, b_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X32, y_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X32, a_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X32, b_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X32, y_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X32, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__SSE41_MUL32_LD32_X32, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__sse41_mul32_ld32_x32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X8, batch_eq_8) {
    TEST_REQUIRES_X86_XOP;
    VAddCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8);
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X8, batch_div_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X8, batch_lt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X8, batch_gt_8) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X8, inplace) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X8, a_zero_point) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X8, b_zero_point) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X8, y_zero_point) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X8, a_scale) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X8, b_scale) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X8, y_scale) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X8, qmin) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X8, qmax) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X16, batch_eq_16) {
    TEST_REQUIRES_X86_XOP;
    VAddCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x16);
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X16, batch_div_16) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X16, batch_lt_16) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X16, batch_gt_16) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X16, inplace) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X16, a_zero_point) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X16, b_zero_point) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X16, y_zero_point) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X16, a_scale) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X16, b_scale) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X16, y_scale) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X16, qmin) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X16, qmax) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X24, batch_eq_24) {
    TEST_REQUIRES_X86_XOP;
    VAddCMicrokernelTester()
      .batch_size(24)
      .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x24);
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X24, batch_div_24) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X24, batch_lt_24) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X24, batch_gt_24) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X24, inplace) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X24, a_zero_point) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X24, b_zero_point) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X24, y_zero_point) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X24, a_scale) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X24, b_scale) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X24, y_scale) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X24, qmin) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X24, qmax) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X32, batch_eq_32) {
    TEST_REQUIRES_X86_XOP;
    VAddCMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x32);
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X32, batch_div_32) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X32, batch_lt_32) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X32, batch_gt_32) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X32, inplace) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X32, a_zero_point) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X32, b_zero_point) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X32, y_zero_point) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X32, a_scale) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X32, b_scale) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X32, y_scale) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X32, qmin) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__XOP_MUL32_LD32_X32, qmax) {
    TEST_REQUIRES_X86_XOP;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__xop_mul32_ld32_x32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VAddCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x8);
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X8, a_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X8, b_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X8, y_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X8, a_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X8, b_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X8, y_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X8, qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X8, qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VAddCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16);
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X16, a_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X16, b_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X16, y_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X16, a_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X16, b_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X16, y_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X16, qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X16, qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VAddCMicrokernelTester()
      .batch_size(24)
      .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x24);
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X24, a_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X24, b_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X24, y_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X24, a_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X24, b_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X24, y_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X24, qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X24, qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VAddCMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x32);
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X32, a_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X32, b_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X32, y_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X32, a_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X32, b_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X32, y_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X32, qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__AVX2_MUL32_LD64_X32, qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__avx2_mul32_ld64_x32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD
  TEST(QS8_VADDC_MINMAX__WASMSIMD_X8, batch_eq_8) {
    VAddCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x8);
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X8, a_zero_point) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X8, b_zero_point) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X8, y_zero_point) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X8, a_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X8, b_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X8, y_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x8);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X8, qmin) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x8);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X8, qmax) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x8);
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(QS8_VADDC_MINMAX__WASMSIMD_X16, batch_eq_16) {
    VAddCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x16);
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X16, a_zero_point) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X16, b_zero_point) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X16, y_zero_point) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X16, a_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X16, b_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X16, y_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x16);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X16, qmin) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x16);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X16, qmax) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x16);
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(QS8_VADDC_MINMAX__WASMSIMD_X24, batch_eq_24) {
    VAddCMicrokernelTester()
      .batch_size(24)
      .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x24);
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X24, a_zero_point) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X24, b_zero_point) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X24, y_zero_point) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X24, a_scale) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X24, b_scale) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X24, y_scale) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x24);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X24, qmin) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x24);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X24, qmax) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x24);
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(QS8_VADDC_MINMAX__WASMSIMD_X32, batch_eq_32) {
    VAddCMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x32);
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X32, inplace) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X32, a_zero_point) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X32, b_zero_point) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X32, y_zero_point) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X32, a_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X32, b_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X32, y_scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VAddCMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x32);
      }
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X32, qmin) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x32);
    }
  }

  TEST(QS8_VADDC_MINMAX__WASMSIMD_X32, qmax) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VAddCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs8_vaddc_minmax_ukernel__wasmsimd_x32);
    }
  }
#endif  // XNN_ARCH_WASMSIMD
