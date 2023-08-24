// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vsigmoid.yaml
//   Generator: tools/generate-vunary-test.py


#include <vector>

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/vunary.h>

#include "vunary-microkernel-tester.h"


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u4, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u4, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u4, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u4, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u4, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u8, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u8, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u8, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u8, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u8, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u12, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u12, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u12, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u12, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u12, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u16, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u16, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u16, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u16, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u16, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u20, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u20, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u20, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u20, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U20, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u20, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u24, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u24, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u24, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u24, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u24, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u4, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u4, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u4, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u4, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u4, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u8, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u8, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u8, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u8, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u8, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u12, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u12, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u12, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u12, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u12, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u16, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u16, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u16, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u16, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u16, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u20, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u20, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u20, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u20, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U20, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u20, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u24, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u24, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u24, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u24, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u24, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u4, xnn_init_f32_sigmoid_neon_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u4, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u4, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u4, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u4, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u8, xnn_init_f32_sigmoid_neon_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u8, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u8, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u8, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u8, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u12, xnn_init_f32_sigmoid_neon_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u12, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u12, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u12, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u12, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u16, xnn_init_f32_sigmoid_neon_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u16, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u16, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u16, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u16, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u20, xnn_init_f32_sigmoid_neon_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u20, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u20, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u20, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U20, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u20, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u24, xnn_init_f32_sigmoid_neon_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u24, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u24, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u24, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u24, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U20, batch_div_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U20, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U24, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U20, batch_div_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U20, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U24, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U40, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U56, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U72, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U40, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U56, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U64, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U72, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U80, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U12, batch_gt_12) {
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U16, batch_gt_16) {
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U20, batch_gt_20) {
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U24, batch_gt_24) {
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U12, batch_gt_12) {
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U16, batch_gt_16) {
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U20, batch_gt_20) {
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U24, batch_gt_24) {
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U4, batch_div_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U4, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U8, batch_div_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U8, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U12, batch_div_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U12, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U16, batch_div_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U16, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U20, batch_div_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U20, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U24, batch_div_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U24, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U4, batch_div_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U4, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U8, batch_div_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U8, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U12, batch_div_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U12, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U16, batch_div_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U16, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U20, batch_div_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U20, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U24, batch_div_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U24, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U12, batch_gt_12) {
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U16, batch_gt_16) {
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U20, batch_gt_20) {
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U24, batch_gt_24) {
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U12, batch_gt_12) {
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U16, batch_gt_16) {
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U20, batch_gt_20) {
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U24, batch_gt_24) {
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U12, batch_gt_12) {
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U16, batch_gt_16) {
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U20, batch_gt_20) {
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U24, batch_gt_24) {
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U12, batch_gt_12) {
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U16, batch_gt_16) {
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U20, batch_gt_20) {
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U24, batch_gt_24) {
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u1, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u1, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u1, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u4, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u1, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u1, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u1, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u2, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u2, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u2, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u2, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u2, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u4, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u4, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u4, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u4, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u4, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u1, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u1, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u1, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u2, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u2, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u2, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u2, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u2, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u4, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u4, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u4, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u4, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u4, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}
