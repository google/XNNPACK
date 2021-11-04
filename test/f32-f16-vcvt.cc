// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-f16-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vcvt.h>
#include "vcvt-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_F16_VCVT__NEONFP16_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_x8);
  }

  TEST(F32_F16_VCVT__NEONFP16_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_x8);
    }
  }

  TEST(F32_F16_VCVT__NEONFP16_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_x8);
    }
  }

  TEST(F32_F16_VCVT__NEONFP16_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_x8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_F16_VCVT__NEONFP16_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_x16);
  }

  TEST(F32_F16_VCVT__NEONFP16_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_x16);
    }
  }

  TEST(F32_F16_VCVT__NEONFP16_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_x16);
    }
  }

  TEST(F32_F16_VCVT__NEONFP16_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_x16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__F16C_X8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_f16_vcvt_ukernel__f16c_x8);
  }

  TEST(F32_F16_VCVT__F16C_X8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__f16c_x8);
    }
  }

  TEST(F32_F16_VCVT__F16C_X8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__f16c_x8);
    }
  }

  TEST(F32_F16_VCVT__F16C_X8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__f16c_x8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__F16C_X16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_f16_vcvt_ukernel__f16c_x16);
  }

  TEST(F32_F16_VCVT__F16C_X16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__f16c_x16);
    }
  }

  TEST(F32_F16_VCVT__F16C_X16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__f16c_x16);
    }
  }

  TEST(F32_F16_VCVT__F16C_X16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__f16c_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__AVX512SKX_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_x16);
  }

  TEST(F32_F16_VCVT__AVX512SKX_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_x16);
    }
  }

  TEST(F32_F16_VCVT__AVX512SKX_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_x16);
    }
  }

  TEST(F32_F16_VCVT__AVX512SKX_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__AVX512SKX_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_x32);
  }

  TEST(F32_F16_VCVT__AVX512SKX_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_x32);
    }
  }

  TEST(F32_F16_VCVT__AVX512SKX_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_x32);
    }
  }

  TEST(F32_F16_VCVT__AVX512SKX_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_x32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

  TEST(F32_F16_VCVT__SCALAR_FLOAT_X1, batch_eq_1) {
    VCvtMicrokernelTester().batch_size(1).Test(
        xnn_f32_f16_vcvt_ukernel__scalar_float_x1);
  }

  TEST(F32_F16_VCVT__SCALAR_FLOAT_X1, batch_gt_1) {
    for (size_t batch_size = 2; batch_size < 10; batch_size++) {
      VCvtMicrokernelTester()
          .batch_size(batch_size)
          .Test(xnn_f32_f16_vcvt_ukernel__scalar_float_x1);
    }
  }

  TEST(F32_F16_VCVT__SCALAR_FLOAT_X2, batch_eq_2) {
    VCvtMicrokernelTester().batch_size(2).Test(
        xnn_f32_f16_vcvt_ukernel__scalar_float_x2);
  }

  TEST(F32_F16_VCVT__SCALAR_FLOAT_X2, batch_div_2) {
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VCvtMicrokernelTester()
          .batch_size(batch_size)
          .Test(xnn_f32_f16_vcvt_ukernel__scalar_float_x2);
    }
  }

  TEST(F32_F16_VCVT__SCALAR_FLOAT_X2, batch_lt_2) {
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VCvtMicrokernelTester()
          .batch_size(batch_size)
          .Test(xnn_f32_f16_vcvt_ukernel__scalar_float_x2);
    }
  }

  TEST(F32_F16_VCVT__SCALAR_FLOAT_X2, batch_gt_2) {
    for (size_t batch_size = 3; batch_size < 4; batch_size++) {
      VCvtMicrokernelTester()
          .batch_size(batch_size)
          .Test(xnn_f32_f16_vcvt_ukernel__scalar_float_x2);
    }
  }

  TEST(F32_F16_VCVT__SCALAR_FLOAT_X3, batch_eq_3) {
    VCvtMicrokernelTester().batch_size(3).Test(
        xnn_f32_f16_vcvt_ukernel__scalar_float_x3);
  }

  TEST(F32_F16_VCVT__SCALAR_FLOAT_X3, batch_div_3) {
    for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
      VCvtMicrokernelTester()
          .batch_size(batch_size)
          .Test(xnn_f32_f16_vcvt_ukernel__scalar_float_x3);
    }
  }

  TEST(F32_F16_VCVT__SCALAR_FLOAT_X3, batch_lt_3) {
    for (size_t batch_size = 1; batch_size < 3; batch_size++) {
      VCvtMicrokernelTester()
          .batch_size(batch_size)
          .Test(xnn_f32_f16_vcvt_ukernel__scalar_float_x3);
    }
  }

  TEST(F32_F16_VCVT__SCALAR_FLOAT_X3, batch_gt_3) {
    for (size_t batch_size = 4; batch_size < 6; batch_size++) {
      VCvtMicrokernelTester()
          .batch_size(batch_size)
          .Test(xnn_f32_f16_vcvt_ukernel__scalar_float_x3);
    }
  }

  TEST(F32_F16_VCVT__SCALAR_FLOAT_X4, batch_eq_4) {
    VCvtMicrokernelTester().batch_size(4).Test(
        xnn_f32_f16_vcvt_ukernel__scalar_float_x4);
  }

  TEST(F32_F16_VCVT__SCALAR_FLOAT_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VCvtMicrokernelTester()
          .batch_size(batch_size)
          .Test(xnn_f32_f16_vcvt_ukernel__scalar_float_x4);
    }
  }

  TEST(F32_F16_VCVT__SCALAR_FLOAT_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VCvtMicrokernelTester()
          .batch_size(batch_size)
          .Test(xnn_f32_f16_vcvt_ukernel__scalar_float_x4);
    }
  }

  TEST(F32_F16_VCVT__SCALAR_FLOAT_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
          .batch_size(batch_size)
          .Test(xnn_f32_f16_vcvt_ukernel__scalar_float_x4);
    }
  }