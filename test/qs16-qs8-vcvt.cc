// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs16-qs8-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <limits>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/vcvt.h"
#include "vcvt-microkernel-tester.h"


#if XNN_ARCH_ARM
  TEST(QS16_QS8_VCVT__ASM_AARCH32_NEON_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(16)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__asm_aarch32_neon_u16, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__ASM_AARCH32_NEON_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__asm_aarch32_neon_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__ASM_AARCH32_NEON_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__asm_aarch32_neon_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__ASM_AARCH32_NEON_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__asm_aarch32_neon_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__ASM_AARCH32_NEON_U16, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__asm_aarch32_neon_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__ASM_AARCH32_NEON_U16, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__asm_aarch32_neon_u16, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS16_QS8_VCVT__NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(8)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u8, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__NEON_U8, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__NEON_U8, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u8, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS16_QS8_VCVT__NEON_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(16)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u16, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__NEON_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__NEON_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__NEON_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__NEON_U16, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__NEON_U16, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u16, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS16_QS8_VCVT__NEON_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(32)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u32, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__NEON_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u32, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__NEON_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u32, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__NEON_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u32, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__NEON_U32, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u32, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__NEON_U32, output_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__neon_u32, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_QS8_VCVT__SSE2_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(4)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u4, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__SSE2_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE2_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE2_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE2_U4, scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE2_U4, output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u4, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_QS8_VCVT__SSE2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(8)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u8, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__SSE2_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE2_U8, scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE2_U8, output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u8, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_QS8_VCVT__SSE2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(16)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u16, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__SSE2_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE2_U16, scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE2_U16, output_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__sse2_u16, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_QS8_VCVT__SSSE3_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSSE3;
    VCvtMicrokernelTester()
      .batch_size(4)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u4, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__SSSE3_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSSE3_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSSE3_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSSE3_U4, scale) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSSE3_U4, output_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u4, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_QS8_VCVT__SSSE3_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSSE3;
    VCvtMicrokernelTester()
      .batch_size(8)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u8, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__SSSE3_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSSE3_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSSE3_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSSE3_U8, scale) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSSE3_U8, output_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u8, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_QS8_VCVT__SSSE3_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSSE3;
    VCvtMicrokernelTester()
      .batch_size(16)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u16, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__SSSE3_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSSE3_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSSE3_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSSE3_U16, scale) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSSE3_U16, output_zero_point) {
    TEST_REQUIRES_X86_SSSE3;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__ssse3_u16, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_QS8_VCVT__SSE41_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(4)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u4, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__SSE41_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE41_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE41_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE41_U4, scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE41_U4, output_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u4, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_QS8_VCVT__SSE41_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(8)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u8, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__SSE41_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE41_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE41_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE41_U8, scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE41_U8, output_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u8, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_QS8_VCVT__SSE41_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(16)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u16, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__SSE41_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE41_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE41_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE41_U16, scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__SSE41_U16, output_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__sse41_u16, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_QS8_VCVT__AVX_U4, batch_eq_4) {
    TEST_REQUIRES_X86_AVX;
    VCvtMicrokernelTester()
      .batch_size(4)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u4, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__AVX_U4, batch_div_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__AVX_U4, batch_lt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__AVX_U4, batch_gt_4) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__AVX_U4, scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__AVX_U4, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u4, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_QS8_VCVT__AVX_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VCvtMicrokernelTester()
      .batch_size(8)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u8, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__AVX_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__AVX_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__AVX_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__AVX_U8, scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__AVX_U8, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u8, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_QS8_VCVT__AVX_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VCvtMicrokernelTester()
      .batch_size(16)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u16, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__AVX_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__AVX_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__AVX_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__AVX_U16, scale) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__AVX_U16, output_zero_point) {
    TEST_REQUIRES_X86_AVX;
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__avx_u16, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS16_QS8_VCVT__WASMSIMD_U8, batch_eq_8) {
    VCvtMicrokernelTester()
      .batch_size(8)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u8, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__WASMSIMD_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__WASMSIMD_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__WASMSIMD_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__WASMSIMD_U8, scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u8, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__WASMSIMD_U8, output_zero_point) {
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u8, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS16_QS8_VCVT__WASMSIMD_U16, batch_eq_16) {
    VCvtMicrokernelTester()
      .batch_size(16)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u16, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__WASMSIMD_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__WASMSIMD_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__WASMSIMD_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__WASMSIMD_U16, scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u16, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__WASMSIMD_U16, output_zero_point) {
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u16, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS16_QS8_VCVT__WASMSIMD_U32, batch_eq_32) {
    VCvtMicrokernelTester()
      .batch_size(32)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u32, xnn_init_qs16_qs8_cvt_scalar_params);
  }

  TEST(QS16_QS8_VCVT__WASMSIMD_U32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u32, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__WASMSIMD_U32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u32, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__WASMSIMD_U32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u32, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__WASMSIMD_U32, scale) {
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .input_zero_point(0)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u32, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }

  TEST(QS16_QS8_VCVT__WASMSIMD_U32, output_zero_point) {
    for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(0)
          .output_zero_point(output_zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .Test(xnn_qs16_qs8_vcvt_ukernel__wasmsimd_u32, xnn_init_qs16_qs8_cvt_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(QS16_QS8_VCVT__SCALAR_U1, batch_eq_1) {
  VCvtMicrokernelTester()
    .batch_size(1)
    .input_zero_point(0)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u1, xnn_init_qs16_qs8_cvt_scalar_params);
}

TEST(QS16_QS8_VCVT__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u1, xnn_init_qs16_qs8_cvt_scalar_params);
  }
}

TEST(QS16_QS8_VCVT__SCALAR_U1, scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u1, xnn_init_qs16_qs8_cvt_scalar_params);
  }
}

TEST(QS16_QS8_VCVT__SCALAR_U1, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u1, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }
}


TEST(QS16_QS8_VCVT__SCALAR_U2, batch_eq_2) {
  VCvtMicrokernelTester()
    .batch_size(2)
    .input_zero_point(0)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u2, xnn_init_qs16_qs8_cvt_scalar_params);
}

TEST(QS16_QS8_VCVT__SCALAR_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u2, xnn_init_qs16_qs8_cvt_scalar_params);
  }
}

TEST(QS16_QS8_VCVT__SCALAR_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u2, xnn_init_qs16_qs8_cvt_scalar_params);
  }
}

TEST(QS16_QS8_VCVT__SCALAR_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u2, xnn_init_qs16_qs8_cvt_scalar_params);
  }
}

TEST(QS16_QS8_VCVT__SCALAR_U2, scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u2, xnn_init_qs16_qs8_cvt_scalar_params);
  }
}

TEST(QS16_QS8_VCVT__SCALAR_U2, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u2, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }
}


TEST(QS16_QS8_VCVT__SCALAR_U4, batch_eq_4) {
  VCvtMicrokernelTester()
    .batch_size(4)
    .input_zero_point(0)
    .qmin(std::numeric_limits<int8_t>::min())
    .qmax(std::numeric_limits<int8_t>::max())
    .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u4, xnn_init_qs16_qs8_cvt_scalar_params);
}

TEST(QS16_QS8_VCVT__SCALAR_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u4, xnn_init_qs16_qs8_cvt_scalar_params);
  }
}

TEST(QS16_QS8_VCVT__SCALAR_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u4, xnn_init_qs16_qs8_cvt_scalar_params);
  }
}

TEST(QS16_QS8_VCVT__SCALAR_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u4, xnn_init_qs16_qs8_cvt_scalar_params);
  }
}

TEST(QS16_QS8_VCVT__SCALAR_U4, scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .scale(50)
      .input_zero_point(0)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u4, xnn_init_qs16_qs8_cvt_scalar_params);
  }
}

TEST(QS16_QS8_VCVT__SCALAR_U4, output_zero_point) {
  for (int16_t output_zero_point = 0; output_zero_point < 5; output_zero_point += 2) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .output_zero_point(output_zero_point)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .Test(xnn_qs16_qs8_vcvt_ukernel__scalar_u4, xnn_init_qs16_qs8_cvt_scalar_params);
    }
  }
}
