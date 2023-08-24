// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-qu8-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <limits>

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vcvt.h>
#include "vcvt-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QU8_VCVT__NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(8)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__neon_u8, xnn_init_f32_qu8_cvt_neon_params);
  }

  TEST(F32_QU8_VCVT__NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u8, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u8, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u8, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U8, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u8, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U8, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neon_u8, xnn_init_f32_qu8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEON_U8, saturation) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u8, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U8, overflow) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u8, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U8, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neon_u8, xnn_init_f32_qu8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEON_U8, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__neon_u8, xnn_init_f32_qu8_cvt_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QU8_VCVT__NEON_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(16)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__neon_u16, xnn_init_f32_qu8_cvt_neon_params);
  }

  TEST(F32_QU8_VCVT__NEON_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u16, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u16, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u16, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U16, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u16, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U16, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neon_u16, xnn_init_f32_qu8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEON_U16, saturation) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u16, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U16, overflow) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u16, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U16, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neon_u16, xnn_init_f32_qu8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEON_U16, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__neon_u16, xnn_init_f32_qu8_cvt_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QU8_VCVT__NEON_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(24)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__neon_u24, xnn_init_f32_qu8_cvt_neon_params);
  }

  TEST(F32_QU8_VCVT__NEON_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u24, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u24, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u24, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U24, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u24, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U24, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neon_u24, xnn_init_f32_qu8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEON_U24, saturation) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u24, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U24, overflow) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u24, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U24, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neon_u24, xnn_init_f32_qu8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEON_U24, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__neon_u24, xnn_init_f32_qu8_cvt_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QU8_VCVT__NEON_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(32)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__neon_u32, xnn_init_f32_qu8_cvt_neon_params);
  }

  TEST(F32_QU8_VCVT__NEON_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u32, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u32, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u32, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U32, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u32, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U32, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neon_u32, xnn_init_f32_qu8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEON_U32, saturation) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u32, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U32, overflow) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neon_u32, xnn_init_f32_qu8_cvt_neon_params);
    }
  }

  TEST(F32_QU8_VCVT__NEON_U32, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neon_u32, xnn_init_f32_qu8_cvt_neon_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEON_U32, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__neon_u32, xnn_init_f32_qu8_cvt_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QU8_VCVT__NEONV8_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    VCvtMicrokernelTester()
      .batch_size(8)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u8, xnn_init_f32_qu8_cvt_neonv8_params);
  }

  TEST(F32_QU8_VCVT__NEONV8_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u8, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u8, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u8, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U8, scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u8, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U8, output_zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u8, xnn_init_f32_qu8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U8, saturation) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u8, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U8, overflow) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u8, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U8, qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u8, xnn_init_f32_qu8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U8, qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u8, xnn_init_f32_qu8_cvt_neonv8_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QU8_VCVT__NEONV8_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    VCvtMicrokernelTester()
      .batch_size(16)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u16, xnn_init_f32_qu8_cvt_neonv8_params);
  }

  TEST(F32_QU8_VCVT__NEONV8_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u16, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u16, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u16, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U16, scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u16, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U16, output_zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u16, xnn_init_f32_qu8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U16, saturation) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u16, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U16, overflow) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u16, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U16, qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u16, xnn_init_f32_qu8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U16, qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u16, xnn_init_f32_qu8_cvt_neonv8_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QU8_VCVT__NEONV8_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_V8;
    VCvtMicrokernelTester()
      .batch_size(24)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u24, xnn_init_f32_qu8_cvt_neonv8_params);
  }

  TEST(F32_QU8_VCVT__NEONV8_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u24, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u24, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u24, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U24, scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u24, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U24, output_zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u24, xnn_init_f32_qu8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U24, saturation) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u24, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U24, overflow) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u24, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U24, qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u24, xnn_init_f32_qu8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U24, qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u24, xnn_init_f32_qu8_cvt_neonv8_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_QU8_VCVT__NEONV8_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_V8;
    VCvtMicrokernelTester()
      .batch_size(32)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u32, xnn_init_f32_qu8_cvt_neonv8_params);
  }

  TEST(F32_QU8_VCVT__NEONV8_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u32, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u32, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u32, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U32, scale) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u32, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U32, output_zero_point) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u32, xnn_init_f32_qu8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U32, saturation) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u32, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U32, overflow) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u32, xnn_init_f32_qu8_cvt_neonv8_params);
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U32, qmin) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u32, xnn_init_f32_qu8_cvt_neonv8_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__NEONV8_U32, qmax) {
    TEST_REQUIRES_ARM_NEON_V8;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__neonv8_u32, xnn_init_f32_qu8_cvt_neonv8_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__SSE2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(8)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u8, xnn_init_f32_qu8_cvt_sse2_params);
  }

  TEST(F32_QU8_VCVT__SSE2_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u8, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u8, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u8, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U8, scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u8, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U8, output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u8, xnn_init_f32_qu8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U8, saturation) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u8, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U8, overflow) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u8, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U8, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u8, xnn_init_f32_qu8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U8, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u8, xnn_init_f32_qu8_cvt_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__SSE2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(16)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u16, xnn_init_f32_qu8_cvt_sse2_params);
  }

  TEST(F32_QU8_VCVT__SSE2_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u16, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u16, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u16, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U16, scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u16, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U16, output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u16, xnn_init_f32_qu8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U16, saturation) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u16, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U16, overflow) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u16, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U16, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u16, xnn_init_f32_qu8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U16, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u16, xnn_init_f32_qu8_cvt_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__SSE2_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(24)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u24, xnn_init_f32_qu8_cvt_sse2_params);
  }

  TEST(F32_QU8_VCVT__SSE2_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u24, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u24, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u24, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U24, scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u24, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U24, output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u24, xnn_init_f32_qu8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U24, saturation) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u24, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U24, overflow) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u24, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U24, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u24, xnn_init_f32_qu8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U24, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u24, xnn_init_f32_qu8_cvt_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__SSE2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(32)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u32, xnn_init_f32_qu8_cvt_sse2_params);
  }

  TEST(F32_QU8_VCVT__SSE2_U32, batch_div_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u32, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u32, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u32, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U32, scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u32, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U32, output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u32, xnn_init_f32_qu8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U32, saturation) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u32, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U32, overflow) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u32, xnn_init_f32_qu8_cvt_sse2_params);
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U32, qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u32, xnn_init_f32_qu8_cvt_sse2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__SSE2_U32, qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__sse2_u32, xnn_init_f32_qu8_cvt_sse2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__AVX_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VCvtMicrokernelTester()
      .batch_size(8)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__avx_u8, xnn_init_f32_qu8_cvt_avx_params);
  }

  TEST(F32_QU8_VCVT__AVX_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u8, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u8, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u8, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U8, scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u8, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U8, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx_u8, xnn_init_f32_qu8_cvt_avx_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX_U8, saturation) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u8, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U8, overflow) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u8, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U8, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx_u8, xnn_init_f32_qu8_cvt_avx_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX_U8, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__avx_u8, xnn_init_f32_qu8_cvt_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__AVX_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VCvtMicrokernelTester()
      .batch_size(16)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__avx_u16, xnn_init_f32_qu8_cvt_avx_params);
  }

  TEST(F32_QU8_VCVT__AVX_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u16, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u16, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u16, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U16, scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u16, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U16, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx_u16, xnn_init_f32_qu8_cvt_avx_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX_U16, saturation) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u16, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U16, overflow) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u16, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U16, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx_u16, xnn_init_f32_qu8_cvt_avx_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX_U16, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__avx_u16, xnn_init_f32_qu8_cvt_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__AVX_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VCvtMicrokernelTester()
      .batch_size(24)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__avx_u24, xnn_init_f32_qu8_cvt_avx_params);
  }

  TEST(F32_QU8_VCVT__AVX_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u24, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u24, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u24, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U24, scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u24, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U24, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx_u24, xnn_init_f32_qu8_cvt_avx_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX_U24, saturation) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u24, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U24, overflow) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u24, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U24, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx_u24, xnn_init_f32_qu8_cvt_avx_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX_U24, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__avx_u24, xnn_init_f32_qu8_cvt_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__AVX_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VCvtMicrokernelTester()
      .batch_size(32)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__avx_u32, xnn_init_f32_qu8_cvt_avx_params);
  }

  TEST(F32_QU8_VCVT__AVX_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u32, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u32, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u32, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U32, scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u32, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U32, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx_u32, xnn_init_f32_qu8_cvt_avx_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX_U32, saturation) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u32, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U32, overflow) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx_u32, xnn_init_f32_qu8_cvt_avx_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX_U32, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx_u32, xnn_init_f32_qu8_cvt_avx_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX_U32, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__avx_u32, xnn_init_f32_qu8_cvt_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__AVX2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VCvtMicrokernelTester()
      .batch_size(16)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u16, xnn_init_f32_qu8_cvt_avx2_params);
  }

  TEST(F32_QU8_VCVT__AVX2_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u16, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u16, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u16, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U16, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u16, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U16, output_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u16, xnn_init_f32_qu8_cvt_avx2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U16, saturation) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u16, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U16, overflow) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u16, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U16, qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u16, xnn_init_f32_qu8_cvt_avx2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U16, qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u16, xnn_init_f32_qu8_cvt_avx2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__AVX2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VCvtMicrokernelTester()
      .batch_size(32)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u32, xnn_init_f32_qu8_cvt_avx2_params);
  }

  TEST(F32_QU8_VCVT__AVX2_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u32, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u32, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u32, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U32, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u32, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U32, output_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u32, xnn_init_f32_qu8_cvt_avx2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U32, saturation) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u32, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U32, overflow) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u32, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U32, qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u32, xnn_init_f32_qu8_cvt_avx2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U32, qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u32, xnn_init_f32_qu8_cvt_avx2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__AVX2_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VCvtMicrokernelTester()
      .batch_size(48)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u48, xnn_init_f32_qu8_cvt_avx2_params);
  }

  TEST(F32_QU8_VCVT__AVX2_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u48, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u48, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u48, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U48, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u48, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U48, output_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u48, xnn_init_f32_qu8_cvt_avx2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U48, saturation) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u48, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U48, overflow) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u48, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U48, qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u48, xnn_init_f32_qu8_cvt_avx2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U48, qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u48, xnn_init_f32_qu8_cvt_avx2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__AVX2_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VCvtMicrokernelTester()
      .batch_size(64)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u64, xnn_init_f32_qu8_cvt_avx2_params);
  }

  TEST(F32_QU8_VCVT__AVX2_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u64, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u64, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u64, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U64, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u64, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U64, output_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u64, xnn_init_f32_qu8_cvt_avx2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U64, saturation) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u64, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U64, overflow) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u64, xnn_init_f32_qu8_cvt_avx2_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U64, qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u64, xnn_init_f32_qu8_cvt_avx2_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX2_U64, qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__avx2_u64, xnn_init_f32_qu8_cvt_avx2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__AVX512SKX_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VCvtMicrokernelTester()
      .batch_size(32)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u32, xnn_init_f32_qu8_cvt_avx512_params);
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u32, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u32, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u32, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U32, scale) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u32, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U32, output_zero_point) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u32, xnn_init_f32_qu8_cvt_avx512_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U32, saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u32, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U32, overflow) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u32, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U32, qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u32, xnn_init_f32_qu8_cvt_avx512_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U32, qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u32, xnn_init_f32_qu8_cvt_avx512_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__AVX512SKX_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VCvtMicrokernelTester()
      .batch_size(64)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u64, xnn_init_f32_qu8_cvt_avx512_params);
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u64, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u64, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u64, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U64, scale) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u64, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U64, output_zero_point) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u64, xnn_init_f32_qu8_cvt_avx512_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U64, saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u64, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U64, overflow) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u64, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U64, qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u64, xnn_init_f32_qu8_cvt_avx512_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U64, qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u64, xnn_init_f32_qu8_cvt_avx512_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__AVX512SKX_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VCvtMicrokernelTester()
      .batch_size(96)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u96, xnn_init_f32_qu8_cvt_avx512_params);
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u96, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u96, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 97; batch_size < 192; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u96, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U96, scale) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u96, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U96, output_zero_point) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u96, xnn_init_f32_qu8_cvt_avx512_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U96, saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u96, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U96, overflow) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u96, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U96, qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u96, xnn_init_f32_qu8_cvt_avx512_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U96, qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u96, xnn_init_f32_qu8_cvt_avx512_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_QU8_VCVT__AVX512SKX_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VCvtMicrokernelTester()
      .batch_size(128)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u128, xnn_init_f32_qu8_cvt_avx512_params);
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u128, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u128, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 129; batch_size < 256; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u128, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U128, scale) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u128, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U128, output_zero_point) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u128, xnn_init_f32_qu8_cvt_avx512_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U128, saturation) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u128, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U128, overflow) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u128, xnn_init_f32_qu8_cvt_avx512_params);
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U128, qmin) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u128, xnn_init_f32_qu8_cvt_avx512_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__AVX512SKX_U128, qmax) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__avx512skx_u128, xnn_init_f32_qu8_cvt_avx512_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U8, batch_eq_8) {
    VCvtMicrokernelTester()
      .batch_size(8)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u8, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u8, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u8, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u8, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U8, scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u8, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U8, output_zero_point) {
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u8, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U8, saturation) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u8, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U8, overflow) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u8, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U8, qmin) {
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u8, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U8, qmax) {
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u8, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U16, batch_eq_16) {
    VCvtMicrokernelTester()
      .batch_size(16)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u16, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u16, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u16, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u16, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U16, scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u16, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U16, output_zero_point) {
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u16, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U16, saturation) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u16, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U16, overflow) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u16, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U16, qmin) {
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u16, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U16, qmax) {
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u16, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U24, batch_eq_24) {
    VCvtMicrokernelTester()
      .batch_size(24)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u24, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u24, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u24, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u24, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U24, scale) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u24, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U24, output_zero_point) {
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u24, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U24, saturation) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u24, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U24, overflow) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u24, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U24, qmin) {
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u24, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U24, qmax) {
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u24, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U32, batch_eq_32) {
    VCvtMicrokernelTester()
      .batch_size(32)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u32, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u32, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u32, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u32, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U32, scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u32, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U32, output_zero_point) {
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u32, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U32, saturation) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u32, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U32, overflow) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u32, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U32, qmin) {
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u32, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_CVT_U32, qmax) {
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_u32, xnn_init_f32_qu8_cvt_wasmsimd_cvt_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U8, batch_eq_8) {
    VCvtMicrokernelTester()
      .batch_size(8)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u8, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u8, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u8, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u8, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U8, scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u8, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U8, output_zero_point) {
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u8, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U8, saturation) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u8, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U8, overflow) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u8, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U8, qmin) {
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u8, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U8, qmax) {
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u8, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U16, batch_eq_16) {
    VCvtMicrokernelTester()
      .batch_size(16)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u16, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u16, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u16, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u16, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U16, scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u16, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U16, output_zero_point) {
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u16, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U16, saturation) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u16, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U16, overflow) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u16, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U16, qmin) {
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u16, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U16, qmax) {
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u16, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U24, batch_eq_24) {
    VCvtMicrokernelTester()
      .batch_size(24)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u24, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u24, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u24, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u24, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U24, scale) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u24, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U24, output_zero_point) {
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u24, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U24, saturation) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u24, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U24, overflow) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u24, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U24, qmin) {
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u24, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U24, qmax) {
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u24, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U32, batch_eq_32) {
    VCvtMicrokernelTester()
      .batch_size(32)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u32, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u32, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u32, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u32, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U32, scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u32, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U32, output_zero_point) {
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u32, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U32, saturation) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u32, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U32, overflow) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u32, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U32, qmin) {
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u32, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASMSIMD_MAGIC_U32, qmax) {
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_u32, xnn_init_f32_qu8_cvt_wasmsimd_magic_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QU8_VCVT__WASM_FMAGIC_U1, batch_eq_1) {
    VCvtMicrokernelTester()
      .batch_size(1)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U1, batch_gt_1) {
    for (size_t batch_size = 2; batch_size < 10; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U1, scale) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U1, output_zero_point) {
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U1, saturation) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U1, overflow) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U1, qmin) {
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U1, qmax) {
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QU8_VCVT__WASM_FMAGIC_U2, batch_eq_2) {
    VCvtMicrokernelTester()
      .batch_size(2)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U2, batch_div_2) {
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U2, batch_lt_2) {
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U2, batch_gt_2) {
    for (size_t batch_size = 3; batch_size < 4; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U2, scale) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U2, output_zero_point) {
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U2, saturation) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U2, overflow) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U2, qmin) {
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U2, qmax) {
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QU8_VCVT__WASM_FMAGIC_U3, batch_eq_3) {
    VCvtMicrokernelTester()
      .batch_size(3)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U3, batch_div_3) {
    for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U3, batch_lt_3) {
    for (size_t batch_size = 1; batch_size < 3; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U3, batch_gt_3) {
    for (size_t batch_size = 4; batch_size < 6; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U3, scale) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U3, output_zero_point) {
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U3, saturation) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U3, overflow) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U3, qmin) {
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U3, qmax) {
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_QU8_VCVT__WASM_FMAGIC_U4, batch_eq_4) {
    VCvtMicrokernelTester()
      .batch_size(4)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U4, scale) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .output_zero_point(100)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U4, output_zero_point) {
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U4, saturation) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U4, overflow) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(4294967296.0f)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U4, qmin) {
    for (int16_t qmin = 0; qmin < 255; qmin += 51) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
      }
    }
  }

  TEST(F32_QU8_VCVT__WASM_FMAGIC_U4, qmax) {
    for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .scale(500)
          .output_zero_point(128)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .Test(xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U1, batch_eq_1) {
  VCvtMicrokernelTester()
    .batch_size(1)
    .qmin(std::numeric_limits<uint8_t>::min())
    .qmax(std::numeric_limits<uint8_t>::max())
    .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U1, scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .output_zero_point(100)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U1, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U1, saturation) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .output_zero_point(128)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U1, overflow) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U1, qmin) {
  for (int16_t qmin = 0; qmin < 255; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(qmin)
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U1, qmax) {
  for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u1, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U2, batch_eq_2) {
  VCvtMicrokernelTester()
    .batch_size(2)
    .qmin(std::numeric_limits<uint8_t>::min())
    .qmax(std::numeric_limits<uint8_t>::max())
    .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U2, scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .output_zero_point(100)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U2, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U2, saturation) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .output_zero_point(128)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U2, overflow) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U2, qmin) {
  for (int16_t qmin = 0; qmin < 255; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(qmin)
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U2, qmax) {
  for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u2, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U3, batch_eq_3) {
  VCvtMicrokernelTester()
    .batch_size(3)
    .qmin(std::numeric_limits<uint8_t>::min())
    .qmax(std::numeric_limits<uint8_t>::max())
    .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U3, scale) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .output_zero_point(100)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U3, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U3, saturation) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .output_zero_point(128)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U3, overflow) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U3, qmin) {
  for (int16_t qmin = 0; qmin < 255; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(qmin)
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U3, qmax) {
  for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u3, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U4, batch_eq_4) {
  VCvtMicrokernelTester()
    .batch_size(4)
    .qmin(std::numeric_limits<uint8_t>::min())
    .qmax(std::numeric_limits<uint8_t>::max())
    .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U4, scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .output_zero_point(100)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U4, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U4, saturation) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .output_zero_point(128)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U4, overflow) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U4, qmin) {
  for (int16_t qmin = 0; qmin < 255; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(qmin)
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_FMAGIC_U4, qmax) {
  for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u4, xnn_init_f32_qu8_cvt_scalar_fmagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U1, batch_eq_1) {
  VCvtMicrokernelTester()
    .batch_size(1)
    .qmin(std::numeric_limits<uint8_t>::min())
    .qmax(std::numeric_limits<uint8_t>::max())
    .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u1, xnn_init_f32_qu8_cvt_scalar_imagic_params);
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u1, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U1, scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .output_zero_point(100)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u1, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U1, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u1, xnn_init_f32_qu8_cvt_scalar_imagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U1, saturation) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .output_zero_point(128)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u1, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U1, overflow) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u1, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U1, qmin) {
  for (int16_t qmin = 0; qmin < 255; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(qmin)
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u1, xnn_init_f32_qu8_cvt_scalar_imagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U1, qmax) {
  for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u1, xnn_init_f32_qu8_cvt_scalar_imagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U2, batch_eq_2) {
  VCvtMicrokernelTester()
    .batch_size(2)
    .qmin(std::numeric_limits<uint8_t>::min())
    .qmax(std::numeric_limits<uint8_t>::max())
    .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f32_qu8_cvt_scalar_imagic_params);
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U2, scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .output_zero_point(100)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U2, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f32_qu8_cvt_scalar_imagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U2, saturation) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .output_zero_point(128)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U2, overflow) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U2, qmin) {
  for (int16_t qmin = 0; qmin < 255; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(qmin)
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f32_qu8_cvt_scalar_imagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U2, qmax) {
  for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u2, xnn_init_f32_qu8_cvt_scalar_imagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U3, batch_eq_3) {
  VCvtMicrokernelTester()
    .batch_size(3)
    .qmin(std::numeric_limits<uint8_t>::min())
    .qmax(std::numeric_limits<uint8_t>::max())
    .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f32_qu8_cvt_scalar_imagic_params);
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U3, scale) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .output_zero_point(100)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U3, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f32_qu8_cvt_scalar_imagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U3, saturation) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .output_zero_point(128)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U3, overflow) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U3, qmin) {
  for (int16_t qmin = 0; qmin < 255; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(qmin)
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f32_qu8_cvt_scalar_imagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U3, qmax) {
  for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u3, xnn_init_f32_qu8_cvt_scalar_imagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U4, batch_eq_4) {
  VCvtMicrokernelTester()
    .batch_size(4)
    .qmin(std::numeric_limits<uint8_t>::min())
    .qmax(std::numeric_limits<uint8_t>::max())
    .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f32_qu8_cvt_scalar_imagic_params);
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U4, scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .output_zero_point(100)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U4, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f32_qu8_cvt_scalar_imagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U4, saturation) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .output_zero_point(128)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U4, overflow) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f32_qu8_cvt_scalar_imagic_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U4, qmin) {
  for (int16_t qmin = 0; qmin < 255; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(qmin)
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f32_qu8_cvt_scalar_imagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_IMAGIC_U4, qmax) {
  for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_imagic_u4, xnn_init_f32_qu8_cvt_scalar_imagic_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U1, batch_eq_1) {
  VCvtMicrokernelTester()
    .batch_size(1)
    .qmin(std::numeric_limits<uint8_t>::min())
    .qmax(std::numeric_limits<uint8_t>::max())
    .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u1, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u1, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U1, scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .output_zero_point(100)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u1, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U1, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u1, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U1, saturation) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .output_zero_point(128)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u1, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U1, overflow) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u1, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U1, qmin) {
  for (int16_t qmin = 0; qmin < 255; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(qmin)
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u1, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U1, qmax) {
  for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u1, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U2, batch_eq_2) {
  VCvtMicrokernelTester()
    .batch_size(2)
    .qmin(std::numeric_limits<uint8_t>::min())
    .qmax(std::numeric_limits<uint8_t>::max())
    .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u2, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u2, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u2, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u2, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U2, scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .output_zero_point(100)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u2, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U2, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u2, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U2, saturation) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .output_zero_point(128)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u2, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U2, overflow) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u2, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U2, qmin) {
  for (int16_t qmin = 0; qmin < 255; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(qmin)
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u2, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U2, qmax) {
  for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u2, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U3, batch_eq_3) {
  VCvtMicrokernelTester()
    .batch_size(3)
    .qmin(std::numeric_limits<uint8_t>::min())
    .qmax(std::numeric_limits<uint8_t>::max())
    .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u3, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u3, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u3, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u3, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U3, scale) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .output_zero_point(100)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u3, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U3, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u3, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U3, saturation) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .output_zero_point(128)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u3, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U3, overflow) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u3, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U3, qmin) {
  for (int16_t qmin = 0; qmin < 255; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(qmin)
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u3, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U3, qmax) {
  for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u3, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U4, batch_eq_4) {
  VCvtMicrokernelTester()
    .batch_size(4)
    .qmin(std::numeric_limits<uint8_t>::min())
    .qmax(std::numeric_limits<uint8_t>::max())
    .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U4, scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .output_zero_point(100)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U4, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U4, saturation) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(500)
      .output_zero_point(128)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U4, overflow) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(4294967296.0f)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U4, qmin) {
  for (int16_t qmin = 0; qmin < 255; qmin += 51) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(qmin)
        .qmax(std::numeric_limits<uint8_t>::max())
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
    }
  }
}

TEST(F32_QU8_VCVT__SCALAR_LRINTF_U4, qmax) {
  for (int16_t qmax = 1; qmax <= 255; qmax += 51) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(500)
        .output_zero_point(128)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(qmax)
        .Test(xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_u4, xnn_init_f32_qu8_cvt_scalar_lrintf_params);
    }
  }
}