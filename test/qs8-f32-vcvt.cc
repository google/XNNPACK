// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-f32-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <limits>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/vcvt.h"
#include "vcvt-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_F32_VCVT__NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 8;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__neon_u8, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 8;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 8;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U8, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U8, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__neon_u8, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_F32_VCVT__NEON_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__neon_u16, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__NEON_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U16, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U16, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__neon_u16, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_F32_VCVT__NEON_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 24;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__neon_u24, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__NEON_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 24;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 24;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U24, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U24, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__neon_u24, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS8_F32_VCVT__NEON_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 32;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__neon_u32, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__NEON_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 32;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 32;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U32, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__neon_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__NEON_U32, input_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__neon_u32, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__SSE2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 8;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u8, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__SSE2_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 8;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 8;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U8, scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U8, input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u8, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__SSE2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u16, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__SSE2_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U16, scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U16, input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u16, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__SSE2_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 24;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u24, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__SSE2_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 24;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 24;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U24, scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U24, input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u24, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__SSE2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 32;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u32, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__SSE2_U32, batch_div_32) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 32;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 32;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U32, scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE2_U32, input_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__sse2_u32, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__SSE41_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 8;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u8, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__SSE41_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 8;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 8;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U8, scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U8, input_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u8, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__SSE41_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u16, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__SSE41_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U16, scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U16, input_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u16, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__SSE41_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 24;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u24, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__SSE41_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 24;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 24;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U24, scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U24, input_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u24, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__SSE41_U32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 32;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u32, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__SSE41_U32, batch_div_32) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 32;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 32;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U32, scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__SSE41_U32, input_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__sse41_u32, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__AVX_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 8;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__avx_u8, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__AVX_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 8;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 8;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U8, scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U8, input_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__avx_u8, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__AVX_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__avx_u16, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__AVX_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U16, scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U16, input_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__avx_u16, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__AVX_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 24;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__avx_u24, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__AVX_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 24;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 24;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U24, scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U24, input_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__avx_u24, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__AVX_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 32;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__avx_u32, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__AVX_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 32;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 32;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U32, scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX_U32, input_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__avx_u32, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__AVX2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_tile = 8;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u8, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__AVX2_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_tile = 8;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_tile = 8;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U8, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U8, input_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u8, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__AVX2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u16, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__AVX2_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U16, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U16, input_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u16, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__AVX2_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_tile = 24;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u24, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__AVX2_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_tile = 24;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_tile = 24;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U24, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U24, input_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u24, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__AVX2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_tile = 32;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u32, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__AVX2_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_tile = 32;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_tile = 32;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U32, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX2_U32, input_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__avx2_u32, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__AVX512SKX_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u16, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U16, scale) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U16, input_zero_point) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u16, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__AVX512SKX_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 32;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u32, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 32;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 32;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U32, scale) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U32, input_zero_point) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u32, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__AVX512SKX_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 48;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u48, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 48;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u48, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 48;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u48, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u48, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U48, scale) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u48, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U48, input_zero_point) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u48, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F32_VCVT__AVX512SKX_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 64;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u64, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 64;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u64, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 64;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u64, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u64, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U64, scale) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u64, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__AVX512SKX_U64, input_zero_point) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__avx512skx_u64, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_F32_VCVT__WASMSIMD_U8, batch_eq_8) {
    const size_t batch_tile = 8;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u8, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U8, batch_div_8) {
    const size_t batch_tile = 8;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U8, batch_lt_8) {
    const size_t batch_tile = 8;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U8, scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u8, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U8, input_zero_point) {
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u8, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_F32_VCVT__WASMSIMD_U16, batch_eq_16) {
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u16, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U16, batch_div_16) {
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U16, batch_lt_16) {
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U16, scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u16, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U16, input_zero_point) {
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u16, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_F32_VCVT__WASMSIMD_U24, batch_eq_24) {
    const size_t batch_tile = 24;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u24, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U24, batch_div_24) {
    const size_t batch_tile = 24;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U24, batch_lt_24) {
    const size_t batch_tile = 24;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U24, scale) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u24, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U24, input_zero_point) {
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u24, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS8_F32_VCVT__WASMSIMD_U32, batch_eq_32) {
    const size_t batch_tile = 32;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u32, xnn_init_qs8_f32_cvt_scalar_params);
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U32, batch_div_32) {
    const size_t batch_tile = 32;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U32, batch_lt_32) {
    const size_t batch_tile = 32;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U32, scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u32, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }

  TEST(QS8_F32_VCVT__WASMSIMD_U32, input_zero_point) {
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f32_vcvt_ukernel__wasmsimd_u32, xnn_init_qs8_f32_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(QS8_F32_VCVT__SCALAR_U1, batch_eq_1) {
  const size_t batch_tile = 1;
  VCvtMicrokernelTester()
    .batch_size(batch_tile)
    .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u1, xnn_init_qs8_f32_cvt_scalar_params);
}

TEST(QS8_F32_VCVT__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u1, xnn_init_qs8_f32_cvt_scalar_params);
  }
}

TEST(QS8_F32_VCVT__SCALAR_U1, scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u1, xnn_init_qs8_f32_cvt_scalar_params);
  }
}

TEST(QS8_F32_VCVT__SCALAR_U1, input_zero_point) {
  for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u1, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }
}


TEST(QS8_F32_VCVT__SCALAR_U2, batch_eq_2) {
  const size_t batch_tile = 2;
  VCvtMicrokernelTester()
    .batch_size(batch_tile)
    .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u2, xnn_init_qs8_f32_cvt_scalar_params);
}

TEST(QS8_F32_VCVT__SCALAR_U2, batch_div_2) {
  const size_t batch_tile = 2;
  for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u2, xnn_init_qs8_f32_cvt_scalar_params);
  }
}

TEST(QS8_F32_VCVT__SCALAR_U2, batch_lt_2) {
  const size_t batch_tile = 2;
  for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u2, xnn_init_qs8_f32_cvt_scalar_params);
  }
}

TEST(QS8_F32_VCVT__SCALAR_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u2, xnn_init_qs8_f32_cvt_scalar_params);
  }
}

TEST(QS8_F32_VCVT__SCALAR_U2, scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u2, xnn_init_qs8_f32_cvt_scalar_params);
  }
}

TEST(QS8_F32_VCVT__SCALAR_U2, input_zero_point) {
  for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u2, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }
}


TEST(QS8_F32_VCVT__SCALAR_U3, batch_eq_3) {
  const size_t batch_tile = 3;
  VCvtMicrokernelTester()
    .batch_size(batch_tile)
    .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u3, xnn_init_qs8_f32_cvt_scalar_params);
}

TEST(QS8_F32_VCVT__SCALAR_U3, batch_div_3) {
  const size_t batch_tile = 3;
  for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u3, xnn_init_qs8_f32_cvt_scalar_params);
  }
}

TEST(QS8_F32_VCVT__SCALAR_U3, batch_lt_3) {
  const size_t batch_tile = 3;
  for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u3, xnn_init_qs8_f32_cvt_scalar_params);
  }
}

TEST(QS8_F32_VCVT__SCALAR_U3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u3, xnn_init_qs8_f32_cvt_scalar_params);
  }
}

TEST(QS8_F32_VCVT__SCALAR_U3, scale) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u3, xnn_init_qs8_f32_cvt_scalar_params);
  }
}

TEST(QS8_F32_VCVT__SCALAR_U3, input_zero_point) {
  for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u3, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }
}


TEST(QS8_F32_VCVT__SCALAR_U4, batch_eq_4) {
  const size_t batch_tile = 4;
  VCvtMicrokernelTester()
    .batch_size(batch_tile)
    .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u4, xnn_init_qs8_f32_cvt_scalar_params);
}

TEST(QS8_F32_VCVT__SCALAR_U4, batch_div_4) {
  const size_t batch_tile = 4;
  for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u4, xnn_init_qs8_f32_cvt_scalar_params);
  }
}

TEST(QS8_F32_VCVT__SCALAR_U4, batch_lt_4) {
  const size_t batch_tile = 4;
  for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u4, xnn_init_qs8_f32_cvt_scalar_params);
  }
}

TEST(QS8_F32_VCVT__SCALAR_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u4, xnn_init_qs8_f32_cvt_scalar_params);
  }
}

TEST(QS8_F32_VCVT__SCALAR_U4, scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u4, xnn_init_qs8_f32_cvt_scalar_params);
  }
}

TEST(QS8_F32_VCVT__SCALAR_U4, input_zero_point) {
  for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(input_zero_point)
        .Test(xnn_qs8_f32_vcvt_ukernel__scalar_u4, xnn_init_qs8_f32_cvt_scalar_params);
    }
  }
}
