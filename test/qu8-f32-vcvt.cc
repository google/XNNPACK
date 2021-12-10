// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qu8-f32-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vcvt.h>
#include "vcvt-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_F32_VCVT__NEON_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qu8_f32_vcvt_ukernel__neon_x8, xnn_init_qu8_f32_cvt_neon_params);
  }

  TEST(QU8_F32_VCVT__NEON_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x8, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x8, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x8, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X8, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x8, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X8, zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .Test(xnn_qu8_f32_vcvt_ukernel__neon_x8, xnn_init_qu8_f32_cvt_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_F32_VCVT__NEON_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qu8_f32_vcvt_ukernel__neon_x16, xnn_init_qu8_f32_cvt_neon_params);
  }

  TEST(QU8_F32_VCVT__NEON_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x16, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x16, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x16, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X16, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x16, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X16, zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .Test(xnn_qu8_f32_vcvt_ukernel__neon_x16, xnn_init_qu8_f32_cvt_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_F32_VCVT__NEON_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_qu8_f32_vcvt_ukernel__neon_x24, xnn_init_qu8_f32_cvt_neon_params);
  }

  TEST(QU8_F32_VCVT__NEON_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x24, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x24, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x24, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X24, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x24, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X24, zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .Test(xnn_qu8_f32_vcvt_ukernel__neon_x24, xnn_init_qu8_f32_cvt_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QU8_F32_VCVT__NEON_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qu8_f32_vcvt_ukernel__neon_x32, xnn_init_qu8_f32_cvt_neon_params);
  }

  TEST(QU8_F32_VCVT__NEON_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x32, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x32, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x32, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X32, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qu8_f32_vcvt_ukernel__neon_x32, xnn_init_qu8_f32_cvt_neon_params);
    }
  }

  TEST(QU8_F32_VCVT__NEON_X32, zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .Test(xnn_qu8_f32_vcvt_ukernel__neon_x32, xnn_init_qu8_f32_cvt_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_F32_VCVT__SSE41_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x8, xnn_init_qu8_f32_cvt_sse4_params);
  }

  TEST(QU8_F32_VCVT__SSE41_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x8, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x8, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x8, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X8, scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x8, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X8, zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x8, xnn_init_qu8_f32_cvt_sse4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_F32_VCVT__SSE41_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x16, xnn_init_qu8_f32_cvt_sse4_params);
  }

  TEST(QU8_F32_VCVT__SSE41_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x16, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x16, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x16, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X16, scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x16, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X16, zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x16, xnn_init_qu8_f32_cvt_sse4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_F32_VCVT__SSE41_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x24, xnn_init_qu8_f32_cvt_sse4_params);
  }

  TEST(QU8_F32_VCVT__SSE41_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x24, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x24, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x24, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X24, scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x24, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X24, zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x24, xnn_init_qu8_f32_cvt_sse4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QU8_F32_VCVT__SSE41_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x32, xnn_init_qu8_f32_cvt_sse4_params);
  }

  TEST(QU8_F32_VCVT__SSE41_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x32, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x32, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x32, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X32, scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x32, xnn_init_qu8_f32_cvt_sse4_params);
    }
  }

  TEST(QU8_F32_VCVT__SSE41_X32, zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .Test(xnn_qu8_f32_vcvt_ukernel__sse41_x32, xnn_init_qu8_f32_cvt_sse4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD
  TEST(QU8_F32_VCVT__WASMSIMD_X8, batch_eq_8) {
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x8, xnn_init_qu8_f32_cvt_wasmsimd_params);
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x8, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x8, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x8, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X8, scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x8, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X8, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x8, xnn_init_qu8_f32_cvt_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(QU8_F32_VCVT__WASMSIMD_X16, batch_eq_16) {
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x16, xnn_init_qu8_f32_cvt_wasmsimd_params);
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x16, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x16, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x16, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X16, scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x16, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X16, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x16, xnn_init_qu8_f32_cvt_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(QU8_F32_VCVT__WASMSIMD_X24, batch_eq_24) {
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x24, xnn_init_qu8_f32_cvt_wasmsimd_params);
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x24, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x24, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x24, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X24, scale) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x24, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X24, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x24, xnn_init_qu8_f32_cvt_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(QU8_F32_VCVT__WASMSIMD_X32, batch_eq_32) {
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x32, xnn_init_qu8_f32_cvt_wasmsimd_params);
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x32, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x32, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x32, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X32, scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x32, xnn_init_qu8_f32_cvt_wasmsimd_params);
    }
  }

  TEST(QU8_F32_VCVT__WASMSIMD_X32, zero_point) {
    for (int16_t zero_point = 0; zero_point < 5; zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .zero_point(zero_point)
          .Test(xnn_qu8_f32_vcvt_ukernel__wasmsimd_x32, xnn_init_qu8_f32_cvt_wasmsimd_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD
