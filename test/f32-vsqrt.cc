// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vsqrt.yaml
//   Generator: tools/generate-vunary-test.py


#include <vector>

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/vunary.h>

#include "vunary-microkernel-tester.h"


#if XNN_ARCH_ARM64
  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4);
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u8);
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u8);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u4);
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u4);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u4);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u4);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u8);
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u8);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u8);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u8);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u12);
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u12);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u12);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u12);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u16);
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u16);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u16);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u16);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u20);
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u20);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u20);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u20);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u24);
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u24);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u24);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u24);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U28, batch_eq_28) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(28)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u28);
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U28, batch_div_28) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 56; batch_size < 280; batch_size += 28) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u28);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U28, batch_lt_28) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 28; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u28);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U28, batch_gt_28) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 28 + 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u28);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U28, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 140; batch_size += 27) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u28);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u32);
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u32);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u32);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u32);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U32, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U36, batch_eq_36) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(36)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u36);
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U36, batch_div_36) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 72; batch_size < 360; batch_size += 36) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u36);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U36, batch_lt_36) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 36; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u36);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U36, batch_gt_36) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 36 + 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u36);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U36, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 180; batch_size += 35) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u36);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u40);
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u40);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u40);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u40);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR1RSQRTS1FMA1ADJ_U40, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_u40);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u4);
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u4);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u4);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u4);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u8);
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u8);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u8);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u8);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u12);
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u12);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u12);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u12);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u16);
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u16);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u16);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u16);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u20);
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u20);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u20);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u20);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u24);
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u24);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u24);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u24);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U28, batch_eq_28) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(28)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u28);
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U28, batch_div_28) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 56; batch_size < 280; batch_size += 28) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u28);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U28, batch_lt_28) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 28; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u28);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U28, batch_gt_28) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 28 + 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u28);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U28, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 140; batch_size += 27) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u28);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u32);
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u32);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u32);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u32);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U32, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U36, batch_eq_36) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(36)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u36);
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U36, batch_div_36) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 72; batch_size < 360; batch_size += 36) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u36);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U36, batch_lt_36) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 36; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u36);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U36, batch_gt_36) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 36 + 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u36);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U36, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 180; batch_size += 35) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u36);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u40);
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u40);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u40);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u40);
    }
  }

  TEST(F32_VSQRT__NEONFMA_NR2FMA1ADJ_U40, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_u40);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VSQRT__RVV_SQRT_U1V, batch_eq_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(1 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u1v);
  }

  TEST(F32_VSQRT__RVV_SQRT_U1V, batch_gt_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1; batch_size < 10 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u1v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U1V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size <= 5 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u1v);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VSQRT__RVV_SQRT_U2V, batch_eq_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u2v);
  }

  TEST(F32_VSQRT__RVV_SQRT_U2V, batch_div_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 4 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size < 20 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 2 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u2v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U2V, batch_lt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size < 2 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u2v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U2V, batch_gt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 2 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1; batch_size < 4 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u2v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U2V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size <= 10 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u2v);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VSQRT__RVV_SQRT_U4V, batch_eq_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(4 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u4v);
  }

  TEST(F32_VSQRT__RVV_SQRT_U4V, batch_div_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 8 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size < 40 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 4 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u4v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U4V, batch_lt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size < 4 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u4v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U4V, batch_gt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 4 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1; batch_size < 8 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u4v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U4V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size <= 20 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u4v);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VSQRT__RVV_SQRT_U8V, batch_eq_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(8 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u8v);
  }

  TEST(F32_VSQRT__RVV_SQRT_U8V, batch_div_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 16 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size < 80 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 8 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u8v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U8V, batch_lt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size < 8 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u8v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U8V, batch_gt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 8 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1; batch_size < 16 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u8v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U8V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size <= 40 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u8v);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__SSE_SQRT_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u4);
  }

  TEST(F32_VSQRT__SSE_SQRT_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U4, inplace) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__SSE_SQRT_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u8);
  }

  TEST(F32_VSQRT__SSE_SQRT_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U8, inplace) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__AVX_SQRT_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u8, xnn_init_f32_sqrt_avx_params);
  }

  TEST(F32_VSQRT__AVX_SQRT_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u8, xnn_init_f32_sqrt_avx_params);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u8, xnn_init_f32_sqrt_avx_params);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u8, xnn_init_f32_sqrt_avx_params);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u8, xnn_init_f32_sqrt_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__AVX_SQRT_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u16, xnn_init_f32_sqrt_avx_params);
  }

  TEST(F32_VSQRT__AVX_SQRT_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u16, xnn_init_f32_sqrt_avx_params);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u16, xnn_init_f32_sqrt_avx_params);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u16, xnn_init_f32_sqrt_avx_params);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u16, xnn_init_f32_sqrt_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u8, xnn_init_f32_sqrt_fma_params);
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u8, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u8, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u8, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u8, xnn_init_f32_sqrt_fma_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u16, xnn_init_f32_sqrt_fma_params);
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u16, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u16, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u16, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u16, xnn_init_f32_sqrt_fma_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u24, xnn_init_f32_sqrt_fma_params);
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u24, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u24, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u24, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u24, xnn_init_f32_sqrt_fma_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u32, xnn_init_f32_sqrt_fma_params);
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u32, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u32, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u32, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u32, xnn_init_f32_sqrt_fma_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u40, xnn_init_f32_sqrt_fma_params);
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u40, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u40, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u40, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u40, xnn_init_f32_sqrt_fma_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u48, xnn_init_f32_sqrt_fma_params);
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u48, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u48, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u48, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u48, xnn_init_f32_sqrt_fma_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u56, xnn_init_f32_sqrt_fma_params);
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u56, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u56, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u56, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u56, xnn_init_f32_sqrt_fma_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u64, xnn_init_f32_sqrt_fma_params);
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u64, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u64, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u64, xnn_init_f32_sqrt_fma_params);
    }
  }

  TEST(F32_VSQRT__FMA3_NR1FMA1ADJ_U64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_u64, xnn_init_f32_sqrt_fma_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u16, xnn_init_f32_sqrt_avx512_params);
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u16, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u16, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u16, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u16, xnn_init_f32_sqrt_avx512_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u32, xnn_init_f32_sqrt_avx512_params);
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u32, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u32, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u32, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u32, xnn_init_f32_sqrt_avx512_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u48, xnn_init_f32_sqrt_avx512_params);
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u48, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u48, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u48, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u48, xnn_init_f32_sqrt_avx512_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u64, xnn_init_f32_sqrt_avx512_params);
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u64, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u64, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u64, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u64, xnn_init_f32_sqrt_avx512_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u80, xnn_init_f32_sqrt_avx512_params);
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u80, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u80, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u80, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u80, xnn_init_f32_sqrt_avx512_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u96, xnn_init_f32_sqrt_avx512_params);
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u96, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u96, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u96, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u96, xnn_init_f32_sqrt_avx512_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u112, xnn_init_f32_sqrt_avx512_params);
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u112, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u112, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u112, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u112, xnn_init_f32_sqrt_avx512_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u128, xnn_init_f32_sqrt_avx512_params);
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u128, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u128, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u128, xnn_init_f32_sqrt_avx512_params);
    }
  }

  TEST(F32_VSQRT__AVX512F_NR1FMA1ADJ_U128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_u128, xnn_init_f32_sqrt_avx512_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSQRT__WASMSIMD_SQRT_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u4);
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u4);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSQRT__WASMSIMD_SQRT_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8);
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_VSQRT__SCALAR_SQRT_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u1);
}

TEST(F32_VSQRT__SCALAR_SQRT_U1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u1);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u1);
  }
}


TEST(F32_VSQRT__SCALAR_SQRT_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u2);
}

TEST(F32_VSQRT__SCALAR_SQRT_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u2);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u2);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u2);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u2);
  }
}


TEST(F32_VSQRT__SCALAR_SQRT_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u4);
}

TEST(F32_VSQRT__SCALAR_SQRT_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u4);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u4);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u4);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u4);
  }
}
