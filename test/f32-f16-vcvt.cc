// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-f16-vcvt.yaml
//   Generator: tools/generate-vcvt-test.py


#include <limits>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/vcvt.h"
#include "vcvt-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_F16_VCVT__NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 8;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__neon_u8);
  }

  TEST(F32_F16_VCVT__NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 8;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neon_u8);
    }
  }

  TEST(F32_F16_VCVT__NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 8;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neon_u8);
    }
  }

  TEST(F32_F16_VCVT__NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neon_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_F16_VCVT__NEON_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__neon_u16);
  }

  TEST(F32_F16_VCVT__NEON_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neon_u16);
    }
  }

  TEST(F32_F16_VCVT__NEON_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neon_u16);
    }
  }

  TEST(F32_F16_VCVT__NEON_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neon_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_F16_VCVT__NEON_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 24;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__neon_u24);
  }

  TEST(F32_F16_VCVT__NEON_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 24;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neon_u24);
    }
  }

  TEST(F32_F16_VCVT__NEON_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 24;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neon_u24);
    }
  }

  TEST(F32_F16_VCVT__NEON_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neon_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_F16_VCVT__NEON_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 32;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__neon_u32);
  }

  TEST(F32_F16_VCVT__NEON_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 32;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neon_u32);
    }
  }

  TEST(F32_F16_VCVT__NEON_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_tile = 32;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neon_u32);
    }
  }

  TEST(F32_F16_VCVT__NEON_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neon_u32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_F16_VCVT__NEONFP16_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    const size_t batch_tile = 8;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_u8);
  }

  TEST(F32_F16_VCVT__NEONFP16_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    const size_t batch_tile = 8;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_u8);
    }
  }

  TEST(F32_F16_VCVT__NEONFP16_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    const size_t batch_tile = 8;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_u8);
    }
  }

  TEST(F32_F16_VCVT__NEONFP16_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_F16_VCVT__NEONFP16_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_u16);
  }

  TEST(F32_F16_VCVT__NEONFP16_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_u16);
    }
  }

  TEST(F32_F16_VCVT__NEONFP16_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_u16);
    }
  }

  TEST(F32_F16_VCVT__NEONFP16_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__neonfp16_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__SSE2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 8;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__sse2_u8);
  }

  TEST(F32_F16_VCVT__SSE2_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 8;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse2_u8);
    }
  }

  TEST(F32_F16_VCVT__SSE2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 8;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse2_u8);
    }
  }

  TEST(F32_F16_VCVT__SSE2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse2_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__SSE2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__sse2_u16);
  }

  TEST(F32_F16_VCVT__SSE2_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse2_u16);
    }
  }

  TEST(F32_F16_VCVT__SSE2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse2_u16);
    }
  }

  TEST(F32_F16_VCVT__SSE2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse2_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__SSE2_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 24;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__sse2_u24);
  }

  TEST(F32_F16_VCVT__SSE2_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 24;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse2_u24);
    }
  }

  TEST(F32_F16_VCVT__SSE2_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 24;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse2_u24);
    }
  }

  TEST(F32_F16_VCVT__SSE2_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse2_u24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__SSE2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 32;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__sse2_u32);
  }

  TEST(F32_F16_VCVT__SSE2_U32, batch_div_32) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 32;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse2_u32);
    }
  }

  TEST(F32_F16_VCVT__SSE2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_tile = 32;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse2_u32);
    }
  }

  TEST(F32_F16_VCVT__SSE2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse2_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__SSE41_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 8;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__sse41_u8);
  }

  TEST(F32_F16_VCVT__SSE41_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 8;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse41_u8);
    }
  }

  TEST(F32_F16_VCVT__SSE41_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 8;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse41_u8);
    }
  }

  TEST(F32_F16_VCVT__SSE41_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse41_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__SSE41_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__sse41_u16);
  }

  TEST(F32_F16_VCVT__SSE41_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse41_u16);
    }
  }

  TEST(F32_F16_VCVT__SSE41_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse41_u16);
    }
  }

  TEST(F32_F16_VCVT__SSE41_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse41_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__SSE41_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 24;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__sse41_u24);
  }

  TEST(F32_F16_VCVT__SSE41_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 24;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse41_u24);
    }
  }

  TEST(F32_F16_VCVT__SSE41_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 24;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse41_u24);
    }
  }

  TEST(F32_F16_VCVT__SSE41_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse41_u24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__SSE41_U32, batch_eq_32) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 32;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__sse41_u32);
  }

  TEST(F32_F16_VCVT__SSE41_U32, batch_div_32) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 32;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse41_u32);
    }
  }

  TEST(F32_F16_VCVT__SSE41_U32, batch_lt_32) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_tile = 32;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse41_u32);
    }
  }

  TEST(F32_F16_VCVT__SSE41_U32, batch_gt_32) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__sse41_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__AVX_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 8;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__avx_u8);
  }

  TEST(F32_F16_VCVT__AVX_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 8;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx_u8);
    }
  }

  TEST(F32_F16_VCVT__AVX_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 8;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx_u8);
    }
  }

  TEST(F32_F16_VCVT__AVX_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__AVX_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__avx_u16);
  }

  TEST(F32_F16_VCVT__AVX_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx_u16);
    }
  }

  TEST(F32_F16_VCVT__AVX_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx_u16);
    }
  }

  TEST(F32_F16_VCVT__AVX_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__AVX_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 24;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__avx_u24);
  }

  TEST(F32_F16_VCVT__AVX_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 24;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx_u24);
    }
  }

  TEST(F32_F16_VCVT__AVX_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 24;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx_u24);
    }
  }

  TEST(F32_F16_VCVT__AVX_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx_u24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__AVX_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 32;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__avx_u32);
  }

  TEST(F32_F16_VCVT__AVX_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 32;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx_u32);
    }
  }

  TEST(F32_F16_VCVT__AVX_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_tile = 32;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx_u32);
    }
  }

  TEST(F32_F16_VCVT__AVX_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__F16C_U8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_tile = 8;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__f16c_u8);
  }

  TEST(F32_F16_VCVT__F16C_U8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_tile = 8;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__f16c_u8);
    }
  }

  TEST(F32_F16_VCVT__F16C_U8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_tile = 8;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__f16c_u8);
    }
  }

  TEST(F32_F16_VCVT__F16C_U8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__f16c_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__F16C_U16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__f16c_u16);
  }

  TEST(F32_F16_VCVT__F16C_U16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__f16c_u16);
    }
  }

  TEST(F32_F16_VCVT__F16C_U16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__f16c_u16);
    }
  }

  TEST(F32_F16_VCVT__F16C_U16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__f16c_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__AVX512SKX_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_u16);
  }

  TEST(F32_F16_VCVT__AVX512SKX_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_u16);
    }
  }

  TEST(F32_F16_VCVT__AVX512SKX_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_u16);
    }
  }

  TEST(F32_F16_VCVT__AVX512SKX_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_F16_VCVT__AVX512SKX_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 32;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_u32);
  }

  TEST(F32_F16_VCVT__AVX512SKX_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 32;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_u32);
    }
  }

  TEST(F32_F16_VCVT__AVX512SKX_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_tile = 32;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_u32);
    }
  }

  TEST(F32_F16_VCVT__AVX512SKX_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__avx512skx_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_F16_VCVT__WASMSIMD_U8, batch_eq_8) {
    const size_t batch_tile = 8;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u8);
  }

  TEST(F32_F16_VCVT__WASMSIMD_U8, batch_div_8) {
    const size_t batch_tile = 8;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_F16_VCVT__WASMSIMD_U8, batch_lt_8) {
    const size_t batch_tile = 8;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_F16_VCVT__WASMSIMD_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u8);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_F16_VCVT__WASMSIMD_U16, batch_eq_16) {
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u16);
  }

  TEST(F32_F16_VCVT__WASMSIMD_U16, batch_div_16) {
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u16);
    }
  }

  TEST(F32_F16_VCVT__WASMSIMD_U16, batch_lt_16) {
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u16);
    }
  }

  TEST(F32_F16_VCVT__WASMSIMD_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u16);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_F16_VCVT__WASMSIMD_U24, batch_eq_24) {
    const size_t batch_tile = 24;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u24);
  }

  TEST(F32_F16_VCVT__WASMSIMD_U24, batch_div_24) {
    const size_t batch_tile = 24;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u24);
    }
  }

  TEST(F32_F16_VCVT__WASMSIMD_U24, batch_lt_24) {
    const size_t batch_tile = 24;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u24);
    }
  }

  TEST(F32_F16_VCVT__WASMSIMD_U24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u24);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_F16_VCVT__WASMSIMD_U32, batch_eq_32) {
    const size_t batch_tile = 32;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u32);
  }

  TEST(F32_F16_VCVT__WASMSIMD_U32, batch_div_32) {
    const size_t batch_tile = 32;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u32);
    }
  }

  TEST(F32_F16_VCVT__WASMSIMD_U32, batch_lt_32) {
    const size_t batch_tile = 32;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u32);
    }
  }

  TEST(F32_F16_VCVT__WASMSIMD_U32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmsimd_u32);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U8, batch_eq_8) {
    const size_t batch_tile = 8;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u8);
  }

  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U8, batch_div_8) {
    const size_t batch_tile = 8;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u8);
    }
  }

  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U8, batch_lt_8) {
    const size_t batch_tile = 8;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u8);
    }
  }

  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u8);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U16, batch_eq_16) {
    const size_t batch_tile = 16;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u16);
  }

  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U16, batch_div_16) {
    const size_t batch_tile = 16;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u16);
    }
  }

  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U16, batch_lt_16) {
    const size_t batch_tile = 16;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u16);
    }
  }

  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u16);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U24, batch_eq_24) {
    const size_t batch_tile = 24;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u24);
  }

  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U24, batch_div_24) {
    const size_t batch_tile = 24;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u24);
    }
  }

  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U24, batch_lt_24) {
    const size_t batch_tile = 24;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u24);
    }
  }

  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u24);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U32, batch_eq_32) {
    const size_t batch_tile = 32;
    VCvtMicrokernelTester()
      .batch_size(batch_tile)
      .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u32);
  }

  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U32, batch_div_32) {
    const size_t batch_tile = 32;
    for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u32);
    }
  }

  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U32, batch_lt_32) {
    const size_t batch_tile = 32;
    for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u32);
    }
  }

  TEST(F32_F16_VCVT__WASMRELAXEDSIMD_U32, batch_gt_32) {
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCvtMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u32);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_F16_VCVT__SCALAR_BITCAST_U1, batch_eq_1) {
  const size_t batch_tile = 1;
  VCvtMicrokernelTester()
    .batch_size(batch_tile)
    .Test(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u1);
}

TEST(F32_F16_VCVT__SCALAR_BITCAST_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u1);
  }
}


TEST(F32_F16_VCVT__SCALAR_BITCAST_U2, batch_eq_2) {
  const size_t batch_tile = 2;
  VCvtMicrokernelTester()
    .batch_size(batch_tile)
    .Test(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u2);
}

TEST(F32_F16_VCVT__SCALAR_BITCAST_U2, batch_div_2) {
  const size_t batch_tile = 2;
  for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u2);
  }
}

TEST(F32_F16_VCVT__SCALAR_BITCAST_U2, batch_lt_2) {
  const size_t batch_tile = 2;
  for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u2);
  }
}

TEST(F32_F16_VCVT__SCALAR_BITCAST_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u2);
  }
}


TEST(F32_F16_VCVT__SCALAR_BITCAST_U3, batch_eq_3) {
  const size_t batch_tile = 3;
  VCvtMicrokernelTester()
    .batch_size(batch_tile)
    .Test(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u3);
}

TEST(F32_F16_VCVT__SCALAR_BITCAST_U3, batch_div_3) {
  const size_t batch_tile = 3;
  for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u3);
  }
}

TEST(F32_F16_VCVT__SCALAR_BITCAST_U3, batch_lt_3) {
  const size_t batch_tile = 3;
  for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u3);
  }
}

TEST(F32_F16_VCVT__SCALAR_BITCAST_U3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u3);
  }
}


TEST(F32_F16_VCVT__SCALAR_BITCAST_U4, batch_eq_4) {
  const size_t batch_tile = 4;
  VCvtMicrokernelTester()
    .batch_size(batch_tile)
    .Test(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u4);
}

TEST(F32_F16_VCVT__SCALAR_BITCAST_U4, batch_div_4) {
  const size_t batch_tile = 4;
  for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u4);
  }
}

TEST(F32_F16_VCVT__SCALAR_BITCAST_U4, batch_lt_4) {
  const size_t batch_tile = 4;
  for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u4);
  }
}

TEST(F32_F16_VCVT__SCALAR_BITCAST_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u4);
  }
}


TEST(F32_F16_VCVT__SCALAR_FABSF_U1, batch_eq_1) {
  const size_t batch_tile = 1;
  VCvtMicrokernelTester()
    .batch_size(batch_tile)
    .Test(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u1);
}

TEST(F32_F16_VCVT__SCALAR_FABSF_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u1);
  }
}


TEST(F32_F16_VCVT__SCALAR_FABSF_U2, batch_eq_2) {
  const size_t batch_tile = 2;
  VCvtMicrokernelTester()
    .batch_size(batch_tile)
    .Test(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u2);
}

TEST(F32_F16_VCVT__SCALAR_FABSF_U2, batch_div_2) {
  const size_t batch_tile = 2;
  for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u2);
  }
}

TEST(F32_F16_VCVT__SCALAR_FABSF_U2, batch_lt_2) {
  const size_t batch_tile = 2;
  for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u2);
  }
}

TEST(F32_F16_VCVT__SCALAR_FABSF_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u2);
  }
}


TEST(F32_F16_VCVT__SCALAR_FABSF_U3, batch_eq_3) {
  const size_t batch_tile = 3;
  VCvtMicrokernelTester()
    .batch_size(batch_tile)
    .Test(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u3);
}

TEST(F32_F16_VCVT__SCALAR_FABSF_U3, batch_div_3) {
  const size_t batch_tile = 3;
  for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u3);
  }
}

TEST(F32_F16_VCVT__SCALAR_FABSF_U3, batch_lt_3) {
  const size_t batch_tile = 3;
  for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u3);
  }
}

TEST(F32_F16_VCVT__SCALAR_FABSF_U3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u3);
  }
}


TEST(F32_F16_VCVT__SCALAR_FABSF_U4, batch_eq_4) {
  const size_t batch_tile = 4;
  VCvtMicrokernelTester()
    .batch_size(batch_tile)
    .Test(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u4);
}

TEST(F32_F16_VCVT__SCALAR_FABSF_U4, batch_div_4) {
  const size_t batch_tile = 4;
  for (size_t batch_size = batch_tile*2; batch_size < batch_tile*10; batch_size += batch_tile) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u4);
  }
}

TEST(F32_F16_VCVT__SCALAR_FABSF_U4, batch_lt_4) {
  const size_t batch_tile = 4;
  for (size_t batch_size = 1; batch_size < batch_tile; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u4);
  }
}

TEST(F32_F16_VCVT__SCALAR_FABSF_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VCvtMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u4);
  }
}
