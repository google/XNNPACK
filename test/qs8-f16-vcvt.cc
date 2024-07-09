// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs8-f16-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <limits>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/vcvt.h"
#include "vcvt-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_F16_VCVT__NEONFP16ARITH_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u8, xnn_init_qs8_f16_cvt_neonfp16arith_params);
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u8, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u8, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u8, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U8, scale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u8, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U8, input_zero_point) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u8, xnn_init_qs8_f16_cvt_neonfp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_F16_VCVT__NEONFP16ARITH_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u16, xnn_init_qs8_f16_cvt_neonfp16arith_params);
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u16, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u16, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u16, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U16, scale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u16, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U16, input_zero_point) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u16, xnn_init_qs8_f16_cvt_neonfp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_F16_VCVT__NEONFP16ARITH_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u24, xnn_init_qs8_f16_cvt_neonfp16arith_params);
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u24, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u24, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u24, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U24, scale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u24, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U24, input_zero_point) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u24, xnn_init_qs8_f16_cvt_neonfp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(QS8_F16_VCVT__NEONFP16ARITH_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u32, xnn_init_qs8_f16_cvt_neonfp16arith_params);
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u32, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u32, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u32, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U32, scale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u32, xnn_init_qs8_f16_cvt_neonfp16arith_params);
    }
  }

  TEST(QS8_F16_VCVT__NEONFP16ARITH_U32, input_zero_point) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f16_vcvt_ukernel__neonfp16arith_u32, xnn_init_qs8_f16_cvt_neonfp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F16_VCVT__AVX2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u16, xnn_init_qs8_f16_cvt_avx_params);
  }

  TEST(QS8_F16_VCVT__AVX2_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u16, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u16, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u16, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U16, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u16, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U16, input_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u16, xnn_init_qs8_f16_cvt_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F16_VCVT__AVX2_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u24, xnn_init_qs8_f16_cvt_avx_params);
  }

  TEST(QS8_F16_VCVT__AVX2_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u24, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u24, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u24, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U24, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u24, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U24, input_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u24, xnn_init_qs8_f16_cvt_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F16_VCVT__AVX2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u32, xnn_init_qs8_f16_cvt_avx_params);
  }

  TEST(QS8_F16_VCVT__AVX2_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u32, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u32, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u32, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U32, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u32, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U32, input_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u32, xnn_init_qs8_f16_cvt_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS8_F16_VCVT__AVX2_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VCvtMicrokernelTester()
      .batch_size(64)
      .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u64, xnn_init_qs8_f16_cvt_avx_params);
  }

  TEST(QS8_F16_VCVT__AVX2_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u64, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u64, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u64, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U64, scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .scale(50)
        .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u64, xnn_init_qs8_f16_cvt_avx_params);
    }
  }

  TEST(QS8_F16_VCVT__AVX2_U64, input_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (int16_t input_zero_point = 0; input_zero_point < 5; input_zero_point += 2) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VCvtMicrokernelTester()
          .batch_size(batch_size)
          .input_zero_point(input_zero_point)
          .Test(xnn_qs8_f16_vcvt_ukernel__avx2_u64, xnn_init_qs8_f16_cvt_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
