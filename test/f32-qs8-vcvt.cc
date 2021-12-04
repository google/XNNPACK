// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-qs8-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vcvt.h>
#include "vcvt-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QS8_VCVT__NEON_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(8)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__neon_x8, xnn_init_f32_qs8_cvt_neon_params);
  }

  TEST(F32_QS8_VCVT__NEON_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x8, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x8, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x8, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X8, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x8, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X8, zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neon_x8, xnn_init_f32_qs8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEON_X8, saturation) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x8, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X8, overflow) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x8, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X8, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neon_x8, xnn_init_f32_qs8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEON_X8, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__neon_x8, xnn_init_f32_qs8_cvt_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QS8_VCVT__NEON_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(16)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__neon_x16, xnn_init_f32_qs8_cvt_neon_params);
  }

  TEST(F32_QS8_VCVT__NEON_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x16, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x16, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x16, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X16, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x16, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X16, zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neon_x16, xnn_init_f32_qs8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEON_X16, saturation) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x16, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X16, overflow) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x16, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X16, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neon_x16, xnn_init_f32_qs8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEON_X16, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__neon_x16, xnn_init_f32_qs8_cvt_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QS8_VCVT__NEON_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(24)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__neon_x24, xnn_init_f32_qs8_cvt_neon_params);
  }

  TEST(F32_QS8_VCVT__NEON_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x24, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x24, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x24, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X24, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x24, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X24, zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neon_x24, xnn_init_f32_qs8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEON_X24, saturation) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x24, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X24, overflow) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x24, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X24, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neon_x24, xnn_init_f32_qs8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEON_X24, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__neon_x24, xnn_init_f32_qs8_cvt_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QS8_VCVT__NEON_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(32)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__neon_x32, xnn_init_f32_qs8_cvt_neon_params);
  }

  TEST(F32_QS8_VCVT__NEON_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x32, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x32, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x32, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X32, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x32, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X32, zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neon_x32, xnn_init_f32_qs8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEON_X32, saturation) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x32, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X32, overflow) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neon_x32, xnn_init_f32_qs8_cvt_neon_params);
    }
  }

  TEST(F32_QS8_VCVT__NEON_X32, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neon_x32, xnn_init_f32_qs8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEON_X32, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__neon_x32, xnn_init_f32_qs8_cvt_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QS8_VCVT__NEONV8_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    VCvtMicrokernelTester()
      .batch_size(8)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x8, xnn_init_f32_qs8_cvt_neonv8_params);
  }

  TEST(F32_QS8_VCVT__NEONV8_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x8, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x8, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x8, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X8, scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x8, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X8, zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x8, xnn_init_f32_qs8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X8, saturation) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x8, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X8, overflow) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x8, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X8, qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x8, xnn_init_f32_qs8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X8, qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x8, xnn_init_f32_qs8_cvt_neonv8_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QS8_VCVT__NEONV8_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    VCvtMicrokernelTester()
      .batch_size(16)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x16, xnn_init_f32_qs8_cvt_neonv8_params);
  }

  TEST(F32_QS8_VCVT__NEONV8_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x16, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x16, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x16, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X16, scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x16, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X16, zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x16, xnn_init_f32_qs8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X16, saturation) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x16, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X16, overflow) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x16, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X16, qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x16, xnn_init_f32_qs8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X16, qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x16, xnn_init_f32_qs8_cvt_neonv8_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QS8_VCVT__NEONV8_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_V8;
    VCvtMicrokernelTester()
      .batch_size(24)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x24, xnn_init_f32_qs8_cvt_neonv8_params);
  }

  TEST(F32_QS8_VCVT__NEONV8_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x24, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x24, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x24, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X24, scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x24, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X24, zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x24, xnn_init_f32_qs8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X24, saturation) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x24, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X24, overflow) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x24, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X24, qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x24, xnn_init_f32_qs8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X24, qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x24, xnn_init_f32_qs8_cvt_neonv8_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QS8_VCVT__NEONV8_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_V8;
    VCvtMicrokernelTester()
      .batch_size(32)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x32, xnn_init_f32_qs8_cvt_neonv8_params);
  }

  TEST(F32_QS8_VCVT__NEONV8_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x32, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x32, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x32, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X32, scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x32, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X32, zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x32, xnn_init_f32_qs8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X32, saturation) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x32, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X32, overflow) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x32, xnn_init_f32_qs8_cvt_neonv8_params);
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X32, qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x32, xnn_init_f32_qs8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__NEONV8_X32, qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__neonv8_x32, xnn_init_f32_qs8_cvt_neonv8_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QS8_VCVT__SSE2_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(8)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x8, xnn_init_f32_qs8_cvt_sse2_params);
  }

  TEST(F32_QS8_VCVT__SSE2_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x8, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x8, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x8, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X8, scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x8, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X8, zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x8, xnn_init_f32_qs8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X8, saturation) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x8, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X8, overflow) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x8, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X8, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x8, xnn_init_f32_qs8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X8, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x8, xnn_init_f32_qs8_cvt_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QS8_VCVT__SSE2_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(16)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x16, xnn_init_f32_qs8_cvt_sse2_params);
  }

  TEST(F32_QS8_VCVT__SSE2_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x16, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x16, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x16, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X16, scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x16, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X16, zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x16, xnn_init_f32_qs8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X16, saturation) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x16, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X16, overflow) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x16, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X16, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x16, xnn_init_f32_qs8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X16, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x16, xnn_init_f32_qs8_cvt_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QS8_VCVT__SSE2_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(24)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x24, xnn_init_f32_qs8_cvt_sse2_params);
  }

  TEST(F32_QS8_VCVT__SSE2_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x24, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x24, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x24, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X24, scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x24, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X24, zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x24, xnn_init_f32_qs8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X24, saturation) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x24, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X24, overflow) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x24, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X24, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x24, xnn_init_f32_qs8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X24, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x24, xnn_init_f32_qs8_cvt_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QS8_VCVT__SSE2_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(32)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x32, xnn_init_f32_qs8_cvt_sse2_params);
  }

  TEST(F32_QS8_VCVT__SSE2_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x32, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x32, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x32, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X32, scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x32, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X32, zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x32, xnn_init_f32_qs8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X32, saturation) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x32, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X32, overflow) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x32, xnn_init_f32_qs8_cvt_sse2_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X32, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x32, xnn_init_f32_qs8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE2_X32, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__sse2_x32, xnn_init_f32_qs8_cvt_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QS8_VCVT__SSE41_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(8)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x8, xnn_init_f32_qs8_cvt_sse4_params);
  }

  TEST(F32_QS8_VCVT__SSE41_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x8, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x8, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x8, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X8, scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x8, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X8, zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x8, xnn_init_f32_qs8_cvt_sse4_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X8, saturation) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x8, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X8, overflow) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x8, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X8, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x8, xnn_init_f32_qs8_cvt_sse4_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X8, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x8, xnn_init_f32_qs8_cvt_sse4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QS8_VCVT__SSE41_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(16)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x16, xnn_init_f32_qs8_cvt_sse4_params);
  }

  TEST(F32_QS8_VCVT__SSE41_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x16, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x16, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x16, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X16, scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x16, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X16, zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x16, xnn_init_f32_qs8_cvt_sse4_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X16, saturation) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x16, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X16, overflow) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x16, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X16, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x16, xnn_init_f32_qs8_cvt_sse4_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X16, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x16, xnn_init_f32_qs8_cvt_sse4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QS8_VCVT__SSE41_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(24)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x24, xnn_init_f32_qs8_cvt_sse4_params);
  }

  TEST(F32_QS8_VCVT__SSE41_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x24, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x24, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x24, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X24, scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x24, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X24, zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x24, xnn_init_f32_qs8_cvt_sse4_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X24, saturation) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x24, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X24, overflow) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x24, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X24, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x24, xnn_init_f32_qs8_cvt_sse4_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X24, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x24, xnn_init_f32_qs8_cvt_sse4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QS8_VCVT__SSE41_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(32)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x32, xnn_init_f32_qs8_cvt_sse4_params);
  }

  TEST(F32_QS8_VCVT__SSE41_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x32, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x32, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x32, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X32, scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x32, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X32, zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x32, xnn_init_f32_qs8_cvt_sse4_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X32, saturation) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x32, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X32, overflow) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x32, xnn_init_f32_qs8_cvt_sse4_params);
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X32, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x32, xnn_init_f32_qs8_cvt_sse4_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__SSE41_X32, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__sse41_x32, xnn_init_f32_qs8_cvt_sse4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD
  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X8, batch_eq_8) {
    VCvtMicrokernelTester()
      .batch_size(8)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x8, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x8, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x8, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x8, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X8, scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x8, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X8, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x8, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X8, saturation) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x8, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X8, overflow) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x8, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X8, qmin) {
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x8, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X8, qmax) {
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x8, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X16, batch_eq_16) {
    VCvtMicrokernelTester()
      .batch_size(16)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x16, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x16, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x16, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x16, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X16, scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x16, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X16, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x16, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X16, saturation) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x16, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X16, overflow) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x16, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X16, qmin) {
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x16, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X16, qmax) {
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x16, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X24, batch_eq_24) {
    VCvtMicrokernelTester()
      .batch_size(24)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x24, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x24, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x24, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x24, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X24, scale) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x24, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X24, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x24, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X24, saturation) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x24, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X24, overflow) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x24, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X24, qmin) {
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x24, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X24, qmax) {
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x24, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X32, batch_eq_32) {
    VCvtMicrokernelTester()
      .batch_size(32)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x32, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x32, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x32, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x32, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X32, scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x32, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X32, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x32, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X32, saturation) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x32, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X32, overflow) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x32, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X32, qmin) {
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x32, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_CVT_X32, qmax) {
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_cvt_x32, xnn_init_f32_qs8_cvt_wasmsimd_cvt_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X8, batch_eq_8) {
    VCvtMicrokernelTester()
      .batch_size(8)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x8, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x8, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x8, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x8, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X8, scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x8, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X8, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x8, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X8, saturation) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x8, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X8, overflow) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x8, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X8, qmin) {
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x8, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X8, qmax) {
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x8, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X16, batch_eq_16) {
    VCvtMicrokernelTester()
      .batch_size(16)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x16, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x16, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x16, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x16, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X16, scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x16, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X16, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x16, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X16, saturation) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x16, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X16, overflow) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x16, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X16, qmin) {
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x16, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X16, qmax) {
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x16, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X24, batch_eq_24) {
    VCvtMicrokernelTester()
      .batch_size(24)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x24, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x24, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x24, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x24, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X24, scale) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x24, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X24, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x24, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X24, saturation) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x24, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X24, overflow) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x24, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X24, qmin) {
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x24, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X24, qmax) {
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x24, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X32, batch_eq_32) {
    VCvtMicrokernelTester()
      .batch_size(32)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x32, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x32, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x32, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x32, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X32, scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x32, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X32, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x32, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X32, saturation) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x32, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X32, overflow) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x32, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X32, qmin) {
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x32, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASMSIMD_MAGIC_X32, qmax) {
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__wasmsimd_magic_x32, xnn_init_f32_qs8_cvt_wasmsimd_magic_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X1, batch_eq_1) {
    VCvtMicrokernelTester()
      .batch_size(1)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X1, batch_gt_1) {
    for (size_t batch_size = 2; batch_size < 10; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X1, scale) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X1, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X1, saturation) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X1, overflow) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X1, qmin) {
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X1, qmax) {
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X2, batch_eq_2) {
    VCvtMicrokernelTester()
      .batch_size(2)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X2, batch_div_2) {
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X2, batch_lt_2) {
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X2, batch_gt_2) {
    for (size_t batch_size = 3; batch_size < 4; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X2, scale) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X2, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X2, saturation) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X2, overflow) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X2, qmin) {
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X2, qmax) {
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X3, batch_eq_3) {
    VCvtMicrokernelTester()
      .batch_size(3)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X3, batch_div_3) {
    for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X3, batch_lt_3) {
    for (size_t batch_size = 1; batch_size < 3; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X3, batch_gt_3) {
    for (size_t batch_size = 4; batch_size < 6; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X3, scale) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X3, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X3, saturation) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X3, overflow) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X3, qmin) {
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X3, qmax) {
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD
  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X4, batch_eq_4) {
    VCvtMicrokernelTester()
      .batch_size(4)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X4, scale) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X4, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X4, saturation) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X4, overflow) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X4, qmin) {
    for (int16_t qmin = -128; qmin < 127; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
      }
    }
  }

  TEST(F32_QS8_VCVT__WASM_MAGIC_FMINMAX_X4, qmax) {
    for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qs8_vcvt_ukernel__wasm_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD


TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X1, batch_eq_1) {
  VCvtMicrokernelTester()
    .batch_size(1)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X1, scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X1, zero_point) {
  for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X1, saturation) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X1, overflow) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X1, qmin) {
  for (int16_t qmin = -128; qmin < 127; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(qmin)
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X1, qmax) {
  for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }
}


TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X2, batch_eq_2) {
  VCvtMicrokernelTester()
    .batch_size(2)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X2, scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X2, zero_point) {
  for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X2, saturation) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X2, overflow) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X2, qmin) {
  for (int16_t qmin = -128; qmin < 127; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(qmin)
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X2, qmax) {
  for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }
}


TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X3, batch_eq_3) {
  VCvtMicrokernelTester()
    .batch_size(3)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X3, scale) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X3, zero_point) {
  for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X3, saturation) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X3, overflow) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X3, qmin) {
  for (int16_t qmin = -128; qmin < 127; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(qmin)
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X3, qmax) {
  for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }
}


TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X4, batch_eq_4) {
  VCvtMicrokernelTester()
    .batch_size(4)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X4, scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X4, zero_point) {
  for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X4, saturation) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X4, overflow) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X4, qmin) {
  for (int16_t qmin = -128; qmin < 127; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(qmin)
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_FMINMAX_X4, qmax) {
  for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_fminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_fminmax_params);
    }
  }
}


TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X1, batch_eq_1) {
  VCvtMicrokernelTester()
    .batch_size(1)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X1, scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X1, zero_point) {
  for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X1, saturation) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X1, overflow) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X1, qmin) {
  for (int16_t qmin = -128; qmin < 127; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(qmin)
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X1, qmax) {
  for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x1, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
    }
  }
}


TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X2, batch_eq_2) {
  VCvtMicrokernelTester()
    .batch_size(2)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X2, scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X2, zero_point) {
  for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X2, saturation) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X2, overflow) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X2, qmin) {
  for (int16_t qmin = -128; qmin < 127; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(qmin)
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X2, qmax) {
  for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x2, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
    }
  }
}


TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X3, batch_eq_3) {
  VCvtMicrokernelTester()
    .batch_size(3)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X3, scale) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X3, zero_point) {
  for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X3, saturation) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X3, overflow) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X3, qmin) {
  for (int16_t qmin = -128; qmin < 127; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(qmin)
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X3, qmax) {
  for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x3, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
    }
  }
}


TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X4, batch_eq_4) {
  VCvtMicrokernelTester()
    .batch_size(4)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X4, scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X4, zero_point) {
  for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .zero_point(zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X4, saturation) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X4, overflow) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X4, qmin) {
  for (int16_t qmin = -128; qmin < 127; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(qmin)
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
    }
  }
}

TEST(F32_QS8_VCVT__SCALAR_MAGIC_IMINMAX_X4, qmax) {
  for (int16_t qmax = -127; qmax <= 127; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qs8_vcvt_ukernel__scalar_magic_iminmax_x4, xnn_init_f32_qs8_cvt_scalar_magic_iminmax_params);
    }
  }
}
