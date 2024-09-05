// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/s32-f32-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <limits>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/vcvt.h"
#include "vcvt-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S32_F32_VCVT__NEON_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(4)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__neon_u4, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__NEON_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__neon_u4, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__NEON_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__neon_u4, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__NEON_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__neon_u4, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S32_F32_VCVT__NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(8)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__neon_u8, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__neon_u8, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__neon_u8, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__neon_u8, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S32_F32_VCVT__NEON_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(12)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__neon_u12, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__NEON_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__neon_u12, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__NEON_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__neon_u12, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__NEON_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__neon_u12, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(S32_F32_VCVT__NEON_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(16)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__neon_u16, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__NEON_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__neon_u16, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__NEON_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__neon_u16, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__NEON_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__neon_u16, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_F32_VCVT__AVX2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VCvtMicrokernelTester()
      .batch_size(8)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__avx2_u8, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__AVX2_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx2_u8, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx2_u8, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx2_u8, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_F32_VCVT__AVX2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VCvtMicrokernelTester()
      .batch_size(16)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__avx2_u16, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__AVX2_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx2_u16, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx2_u16, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx2_u16, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_F32_VCVT__AVX2_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VCvtMicrokernelTester()
      .batch_size(24)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__avx2_u24, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__AVX2_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx2_u24, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX2_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx2_u24, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX2_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx2_u24, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_F32_VCVT__AVX2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VCvtMicrokernelTester()
      .batch_size(32)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__avx2_u32, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__AVX2_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx2_u32, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx2_u32, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx2_u32, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_F32_VCVT__AVX512F_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VCvtMicrokernelTester()
      .batch_size(16)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u16, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__AVX512F_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u16, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX512F_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u16, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX512F_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u16, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_F32_VCVT__AVX512F_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VCvtMicrokernelTester()
      .batch_size(32)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u32, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__AVX512F_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u32, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX512F_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u32, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX512F_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u32, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_F32_VCVT__AVX512F_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VCvtMicrokernelTester()
      .batch_size(48)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u48, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__AVX512F_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u48, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX512F_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u48, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX512F_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u48, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(S32_F32_VCVT__AVX512F_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VCvtMicrokernelTester()
      .batch_size(64)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u64, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__AVX512F_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u64, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX512F_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u64, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__AVX512F_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__avx512f_u64, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(S32_F32_VCVT__WASMSIMD_U4, batch_eq_4) {
    VCvtMicrokernelTester()
      .batch_size(4)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u4, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__WASMSIMD_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u4, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__WASMSIMD_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u4, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__WASMSIMD_U4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u4, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(S32_F32_VCVT__WASMSIMD_U8, batch_eq_8) {
    VCvtMicrokernelTester()
      .batch_size(8)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u8, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__WASMSIMD_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u8, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__WASMSIMD_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u8, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__WASMSIMD_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u8, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(S32_F32_VCVT__WASMSIMD_U12, batch_eq_12) {
    VCvtMicrokernelTester()
      .batch_size(12)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u12, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__WASMSIMD_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u12, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__WASMSIMD_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u12, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__WASMSIMD_U12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u12, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(S32_F32_VCVT__WASMSIMD_U16, batch_eq_16) {
    VCvtMicrokernelTester()
      .batch_size(16)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u16, xnn_init_s32_f32_cvt_scalar_params);
  }

  TEST(S32_F32_VCVT__WASMSIMD_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u16, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__WASMSIMD_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u16, xnn_init_s32_f32_cvt_scalar_params);
    }
  }

  TEST(S32_F32_VCVT__WASMSIMD_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .input_zero_point(0)
        .Test(xnn_s32_f32_vcvt_ukernel__wasmsimd_u16, xnn_init_s32_f32_cvt_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(S32_F32_VCVT__SCALAR_U1, batch_eq_1) {
  VCvtMicrokernelTester()
    .batch_size(1)
    .input_zero_point(0)
    .Test(xnn_s32_f32_vcvt_ukernel__scalar_u1, xnn_init_s32_f32_cvt_scalar_params);
}

TEST(S32_F32_VCVT__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__scalar_u1, xnn_init_s32_f32_cvt_scalar_params);
  }
}


TEST(S32_F32_VCVT__SCALAR_U2, batch_eq_2) {
  VCvtMicrokernelTester()
    .batch_size(2)
    .input_zero_point(0)
    .Test(xnn_s32_f32_vcvt_ukernel__scalar_u2, xnn_init_s32_f32_cvt_scalar_params);
}

TEST(S32_F32_VCVT__SCALAR_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__scalar_u2, xnn_init_s32_f32_cvt_scalar_params);
  }
}

TEST(S32_F32_VCVT__SCALAR_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__scalar_u2, xnn_init_s32_f32_cvt_scalar_params);
  }
}

TEST(S32_F32_VCVT__SCALAR_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__scalar_u2, xnn_init_s32_f32_cvt_scalar_params);
  }
}


TEST(S32_F32_VCVT__SCALAR_U3, batch_eq_3) {
  VCvtMicrokernelTester()
    .batch_size(3)
    .input_zero_point(0)
    .Test(xnn_s32_f32_vcvt_ukernel__scalar_u3, xnn_init_s32_f32_cvt_scalar_params);
}

TEST(S32_F32_VCVT__SCALAR_U3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__scalar_u3, xnn_init_s32_f32_cvt_scalar_params);
  }
}

TEST(S32_F32_VCVT__SCALAR_U3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__scalar_u3, xnn_init_s32_f32_cvt_scalar_params);
  }
}

TEST(S32_F32_VCVT__SCALAR_U3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__scalar_u3, xnn_init_s32_f32_cvt_scalar_params);
  }
}


TEST(S32_F32_VCVT__SCALAR_U4, batch_eq_4) {
  VCvtMicrokernelTester()
    .batch_size(4)
    .input_zero_point(0)
    .Test(xnn_s32_f32_vcvt_ukernel__scalar_u4, xnn_init_s32_f32_cvt_scalar_params);
}

TEST(S32_F32_VCVT__SCALAR_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__scalar_u4, xnn_init_s32_f32_cvt_scalar_params);
  }
}

TEST(S32_F32_VCVT__SCALAR_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__scalar_u4, xnn_init_s32_f32_cvt_scalar_params);
  }
}

TEST(S32_F32_VCVT__SCALAR_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .input_zero_point(0)
      .Test(xnn_s32_f32_vcvt_ukernel__scalar_u4, xnn_init_s32_f32_cvt_scalar_params);
  }
}
