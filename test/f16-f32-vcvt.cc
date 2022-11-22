// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-f32-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vcvt.h>
#include "vcvt-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32_VCVT__NEON_INT16_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x8, xnn_init_f16_f32_cvt_neon_params);
  }

  TEST(F16_F32_VCVT__NEON_INT16_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x8, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT16_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x8, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT16_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x8, xnn_init_f16_f32_cvt_neon_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32_VCVT__NEON_INT16_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x16, xnn_init_f16_f32_cvt_neon_params);
  }

  TEST(F16_F32_VCVT__NEON_INT16_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x16, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT16_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x16, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT16_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x16, xnn_init_f16_f32_cvt_neon_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32_VCVT__NEON_INT16_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x24, xnn_init_f16_f32_cvt_neon_params);
  }

  TEST(F16_F32_VCVT__NEON_INT16_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x24, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT16_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x24, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT16_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x24, xnn_init_f16_f32_cvt_neon_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32_VCVT__NEON_INT16_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x32, xnn_init_f16_f32_cvt_neon_params);
  }

  TEST(F16_F32_VCVT__NEON_INT16_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x32, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT16_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x32, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT16_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int16_x32, xnn_init_f16_f32_cvt_neon_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32_VCVT__NEON_INT32_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x8, xnn_init_f16_f32_cvt_neon_params);
  }

  TEST(F16_F32_VCVT__NEON_INT32_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x8, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT32_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x8, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT32_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x8, xnn_init_f16_f32_cvt_neon_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32_VCVT__NEON_INT32_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x16, xnn_init_f16_f32_cvt_neon_params);
  }

  TEST(F16_F32_VCVT__NEON_INT32_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x16, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT32_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x16, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT32_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x16, xnn_init_f16_f32_cvt_neon_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32_VCVT__NEON_INT32_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x24, xnn_init_f16_f32_cvt_neon_params);
  }

  TEST(F16_F32_VCVT__NEON_INT32_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x24, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT32_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x24, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT32_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x24, xnn_init_f16_f32_cvt_neon_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32_VCVT__NEON_INT32_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x32, xnn_init_f16_f32_cvt_neon_params);
  }

  TEST(F16_F32_VCVT__NEON_INT32_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x32, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT32_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x32, xnn_init_f16_f32_cvt_neon_params);
    }
  }

  TEST(F16_F32_VCVT__NEON_INT32_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neon_int32_x32, xnn_init_f16_f32_cvt_neon_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32_VCVT__NEONFP16_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32_vcvt_ukernel__neonfp16_x8);
  }

  TEST(F16_F32_VCVT__NEONFP16_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neonfp16_x8);
    }
  }

  TEST(F16_F32_VCVT__NEONFP16_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neonfp16_x8);
    }
  }

  TEST(F16_F32_VCVT__NEONFP16_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neonfp16_x8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32_VCVT__NEONFP16_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32_vcvt_ukernel__neonfp16_x16);
  }

  TEST(F16_F32_VCVT__NEONFP16_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neonfp16_x16);
    }
  }

  TEST(F16_F32_VCVT__NEONFP16_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neonfp16_x16);
    }
  }

  TEST(F16_F32_VCVT__NEONFP16_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__neonfp16_x16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE2_INT16_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x8, xnn_init_f16_f32_cvt_sse_int16_params);
  }

  TEST(F16_F32_VCVT__SSE2_INT16_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x8, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT16_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x8, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT16_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x8, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE2_INT16_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x16, xnn_init_f16_f32_cvt_sse_int16_params);
  }

  TEST(F16_F32_VCVT__SSE2_INT16_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x16, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT16_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x16, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT16_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x16, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE2_INT16_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x24, xnn_init_f16_f32_cvt_sse_int16_params);
  }

  TEST(F16_F32_VCVT__SSE2_INT16_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x24, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT16_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x24, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT16_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x24, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE2_INT16_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x32, xnn_init_f16_f32_cvt_sse_int16_params);
  }

  TEST(F16_F32_VCVT__SSE2_INT16_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x32, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT16_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x32, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT16_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int16_x32, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE2_INT32_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x8, xnn_init_f16_f32_cvt_sse_int32_params);
  }

  TEST(F16_F32_VCVT__SSE2_INT32_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x8, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT32_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x8, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT32_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x8, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE2_INT32_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x16, xnn_init_f16_f32_cvt_sse_int32_params);
  }

  TEST(F16_F32_VCVT__SSE2_INT32_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x16, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT32_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x16, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT32_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x16, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE2_INT32_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x24, xnn_init_f16_f32_cvt_sse_int32_params);
  }

  TEST(F16_F32_VCVT__SSE2_INT32_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x24, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT32_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x24, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT32_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x24, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE2_INT32_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE2;
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x32, xnn_init_f16_f32_cvt_sse_int32_params);
  }

  TEST(F16_F32_VCVT__SSE2_INT32_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x32, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT32_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x32, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE2_INT32_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse2_int32_x32, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE41_INT16_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x8, xnn_init_f16_f32_cvt_sse_int16_params);
  }

  TEST(F16_F32_VCVT__SSE41_INT16_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x8, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT16_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x8, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT16_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x8, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE41_INT16_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x16, xnn_init_f16_f32_cvt_sse_int16_params);
  }

  TEST(F16_F32_VCVT__SSE41_INT16_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x16, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT16_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x16, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT16_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x16, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE41_INT16_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x24, xnn_init_f16_f32_cvt_sse_int16_params);
  }

  TEST(F16_F32_VCVT__SSE41_INT16_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x24, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT16_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x24, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT16_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x24, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE41_INT16_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x32, xnn_init_f16_f32_cvt_sse_int16_params);
  }

  TEST(F16_F32_VCVT__SSE41_INT16_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x32, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT16_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x32, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT16_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int16_x32, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE41_INT32_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x8, xnn_init_f16_f32_cvt_sse_int32_params);
  }

  TEST(F16_F32_VCVT__SSE41_INT32_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x8, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT32_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x8, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT32_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x8, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE41_INT32_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x16, xnn_init_f16_f32_cvt_sse_int32_params);
  }

  TEST(F16_F32_VCVT__SSE41_INT32_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x16, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT32_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x16, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT32_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x16, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE41_INT32_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x24, xnn_init_f16_f32_cvt_sse_int32_params);
  }

  TEST(F16_F32_VCVT__SSE41_INT32_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x24, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT32_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x24, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT32_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x24, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__SSE41_INT32_X32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE41;
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x32, xnn_init_f16_f32_cvt_sse_int32_params);
  }

  TEST(F16_F32_VCVT__SSE41_INT32_X32, batch_div_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x32, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT32_X32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x32, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__SSE41_INT32_X32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__sse41_int32_x32, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__AVX_INT16_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x8, xnn_init_f16_f32_cvt_sse_int16_params);
  }

  TEST(F16_F32_VCVT__AVX_INT16_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x8, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT16_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x8, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT16_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x8, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__AVX_INT16_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x16, xnn_init_f16_f32_cvt_sse_int16_params);
  }

  TEST(F16_F32_VCVT__AVX_INT16_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x16, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT16_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x16, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT16_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x16, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__AVX_INT16_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x24, xnn_init_f16_f32_cvt_sse_int16_params);
  }

  TEST(F16_F32_VCVT__AVX_INT16_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x24, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT16_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x24, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT16_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x24, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__AVX_INT16_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x32, xnn_init_f16_f32_cvt_sse_int16_params);
  }

  TEST(F16_F32_VCVT__AVX_INT16_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x32, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT16_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x32, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT16_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int16_x32, xnn_init_f16_f32_cvt_sse_int16_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__AVX_INT32_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x8, xnn_init_f16_f32_cvt_sse_int32_params);
  }

  TEST(F16_F32_VCVT__AVX_INT32_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x8, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT32_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x8, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT32_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x8, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__AVX_INT32_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x16, xnn_init_f16_f32_cvt_sse_int32_params);
  }

  TEST(F16_F32_VCVT__AVX_INT32_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x16, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT32_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x16, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT32_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x16, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__AVX_INT32_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x24, xnn_init_f16_f32_cvt_sse_int32_params);
  }

  TEST(F16_F32_VCVT__AVX_INT32_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x24, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT32_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x24, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT32_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x24, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__AVX_INT32_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x32, xnn_init_f16_f32_cvt_sse_int32_params);
  }

  TEST(F16_F32_VCVT__AVX_INT32_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x32, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT32_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x32, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }

  TEST(F16_F32_VCVT__AVX_INT32_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx_int32_x32, xnn_init_f16_f32_cvt_sse_int32_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__F16C_X8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32_vcvt_ukernel__f16c_x8);
  }

  TEST(F16_F32_VCVT__F16C_X8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__f16c_x8);
    }
  }

  TEST(F16_F32_VCVT__F16C_X8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__f16c_x8);
    }
  }

  TEST(F16_F32_VCVT__F16C_X8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__f16c_x8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__F16C_X16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32_vcvt_ukernel__f16c_x16);
  }

  TEST(F16_F32_VCVT__F16C_X16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__f16c_x16);
    }
  }

  TEST(F16_F32_VCVT__F16C_X16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__f16c_x16);
    }
  }

  TEST(F16_F32_VCVT__F16C_X16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__f16c_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__AVX512SKX_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32_vcvt_ukernel__avx512skx_x16);
  }

  TEST(F16_F32_VCVT__AVX512SKX_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx512skx_x16);
    }
  }

  TEST(F16_F32_VCVT__AVX512SKX_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx512skx_x16);
    }
  }

  TEST(F16_F32_VCVT__AVX512SKX_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx512skx_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32_VCVT__AVX512SKX_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32_vcvt_ukernel__avx512skx_x32);
  }

  TEST(F16_F32_VCVT__AVX512SKX_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx512skx_x32);
    }
  }

  TEST(F16_F32_VCVT__AVX512SKX_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx512skx_x32);
    }
  }

  TEST(F16_F32_VCVT__AVX512SKX_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__avx512skx_x32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMSIMD_INT16_X8, batch_eq_8) {
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x8, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT16_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x8, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT16_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x8, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT16_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x8, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMSIMD_INT16_X16, batch_eq_16) {
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x16, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT16_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x16, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT16_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x16, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT16_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x16, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMSIMD_INT16_X24, batch_eq_24) {
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x24, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT16_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x24, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT16_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x24, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT16_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x24, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMSIMD_INT16_X32, batch_eq_32) {
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x32, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT16_X32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x32, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT16_X32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x32, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT16_X32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x32, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMSIMD_INT32_X8, batch_eq_8) {
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x8, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT32_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x8, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT32_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x8, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT32_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x8, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMSIMD_INT32_X16, batch_eq_16) {
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x16, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT32_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x16, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT32_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x16, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT32_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x16, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMSIMD_INT32_X24, batch_eq_24) {
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x24, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT32_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x24, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT32_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x24, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT32_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x24, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMSIMD_INT32_X32, batch_eq_32) {
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x32, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT32_X32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x32, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT32_X32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x32, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMSIMD_INT32_X32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x32, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X8, batch_eq_8) {
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x8, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x8, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x8, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x8, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X16, batch_eq_16) {
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x16, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x16, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x16, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x16, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X24, batch_eq_24) {
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x24, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x24, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x24, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x24, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X32, batch_eq_32) {
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x32, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x32, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x32, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT16_X32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x32, xnn_init_f16_f32_cvt_wasmsimd_int16_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X8, batch_eq_8) {
    VCvtMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x8, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x8, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x8, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x8, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X16, batch_eq_16) {
    VCvtMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x16, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x16, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x16, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x16, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X24, batch_eq_24) {
    VCvtMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x24, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x24, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x24, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x24, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X32, batch_eq_32) {
    VCvtMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x32, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X32, batch_div_32) {
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x32, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X32, batch_lt_32) {
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x32, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }

  TEST(F16_F32_VCVT__WASMRELAXEDSIMD_INT32_X32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x32, xnn_init_f16_f32_cvt_wasmsimd_int32_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


TEST(F16_F32_VCVT__SCALAR_X1, batch_eq_1) {
  VCvtMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f16_f32_vcvt_ukernel__scalar_x1, xnn_init_f16_f32_cvt_scalar_params);
}

TEST(F16_F32_VCVT__SCALAR_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_f32_vcvt_ukernel__scalar_x1, xnn_init_f16_f32_cvt_scalar_params);
  }
}


TEST(F16_F32_VCVT__SCALAR_X2, batch_eq_2) {
  VCvtMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f16_f32_vcvt_ukernel__scalar_x2, xnn_init_f16_f32_cvt_scalar_params);
}

TEST(F16_F32_VCVT__SCALAR_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_f32_vcvt_ukernel__scalar_x2, xnn_init_f16_f32_cvt_scalar_params);
  }
}

TEST(F16_F32_VCVT__SCALAR_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_f32_vcvt_ukernel__scalar_x2, xnn_init_f16_f32_cvt_scalar_params);
  }
}

TEST(F16_F32_VCVT__SCALAR_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_f32_vcvt_ukernel__scalar_x2, xnn_init_f16_f32_cvt_scalar_params);
  }
}


TEST(F16_F32_VCVT__SCALAR_X3, batch_eq_3) {
  VCvtMicrokernelTester()
    .batch_size(3)
    .Test(xnn_f16_f32_vcvt_ukernel__scalar_x3, xnn_init_f16_f32_cvt_scalar_params);
}

TEST(F16_F32_VCVT__SCALAR_X3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_f32_vcvt_ukernel__scalar_x3, xnn_init_f16_f32_cvt_scalar_params);
  }
}

TEST(F16_F32_VCVT__SCALAR_X3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_f32_vcvt_ukernel__scalar_x3, xnn_init_f16_f32_cvt_scalar_params);
  }
}

TEST(F16_F32_VCVT__SCALAR_X3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_f32_vcvt_ukernel__scalar_x3, xnn_init_f16_f32_cvt_scalar_params);
  }
}


TEST(F16_F32_VCVT__SCALAR_X4, batch_eq_4) {
  VCvtMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f16_f32_vcvt_ukernel__scalar_x4, xnn_init_f16_f32_cvt_scalar_params);
}

TEST(F16_F32_VCVT__SCALAR_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_f32_vcvt_ukernel__scalar_x4, xnn_init_f16_f32_cvt_scalar_params);
  }
}

TEST(F16_F32_VCVT__SCALAR_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_f32_vcvt_ukernel__scalar_x4, xnn_init_f16_f32_cvt_scalar_params);
  }
}

TEST(F16_F32_VCVT__SCALAR_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_f32_vcvt_ukernel__scalar_x4, xnn_init_f16_f32_cvt_scalar_params);
  }
}
