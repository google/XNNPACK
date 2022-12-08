// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vsigmoid.yaml
//   Generator: tools/generate-vunary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vunary.h>
#include "vunary-microkernel-tester.h"


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x4, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x4, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x4, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x4, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x4, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x8, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x8, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x8, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x8, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x8, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x12, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x12, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x12, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x12, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x12, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x16, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x16, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x16, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x16, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x16, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x20, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x20, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x20, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x20, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X20, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x20, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x24, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x24, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x24, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x24, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x24, xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x4, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x4, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x4, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x4, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x4, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x8, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x8, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x8, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x8, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x8, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x12, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x12, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x12, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x12, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x12, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x16, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x16, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x16, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x16, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x16, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x20, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x20, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x20, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x20, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X20, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x20, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x24, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x24, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x24, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x24, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x24, xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x4, xnn_init_f32_sigmoid_neon_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x4, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x4, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x4, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x4, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x8, xnn_init_f32_sigmoid_neon_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x8, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x8, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x8, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x8, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x12, xnn_init_f32_sigmoid_neon_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x12, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x12, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x12, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x12, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x16, xnn_init_f32_sigmoid_neon_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x16, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x16, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x16, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x16, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x20, xnn_init_f32_sigmoid_neon_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x20, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x20, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x20, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X20, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x20, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x24, xnn_init_f32_sigmoid_neon_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x24, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x24, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x24, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_X24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x24, xnn_init_f32_sigmoid_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x4, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x8, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x12, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x16, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x20, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x24, xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x4, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x8, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x12, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x16, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x20, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x24, xnn_init_f32_sigmoid_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X20, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_X24, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X20, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_X24, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_X24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x4, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x8, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x12, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x16, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x20, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_X24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x24, xnn_init_f32_sigmoid_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X40, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X56, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 73; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X72, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x8, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x16, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x24, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x32, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X40, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x40, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x48, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X56, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x56, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X64, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x64, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 73; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X72, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x72, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_X80, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x80, xnn_init_f32_sigmoid_avx_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 73; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 73; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x8, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x16, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x24, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x32, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x40, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x48, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x56, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x64, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 73; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x72, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x80, xnn_init_f32_sigmoid_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 97; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 113; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 129; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_X128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x16, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x32, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x48, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x64, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x80, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 97; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x96, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 113; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x112, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 129; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_X128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x128, xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 97; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 113; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 129; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_X128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x16, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x32, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x48, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x64, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x80, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 97; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x96, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 113; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x112, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 129; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_X128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x128, xnn_init_f32_sigmoid_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 97; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 113; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 129; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_X128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x16, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x32, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x48, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x64, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x80, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 97; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x96, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 113; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x112, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 129; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_X128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x128, xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X20, batch_gt_20) {
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_X24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X20, batch_gt_20) {
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_X24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X20, batch_gt_20) {
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_X24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X20, batch_gt_20) {
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_X24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X20, batch_gt_20) {
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_X24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x4, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x8, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x12, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x16, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X20, batch_gt_20) {
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x20, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_X24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x24, xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_X1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x1, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x1, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x1, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_X2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x2, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x2, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x2, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x2, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x2, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_X4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x4, xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_X1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x1, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x1, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x1, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_X2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x2, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x2, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x2, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x2, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x2, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_X4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x4, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x4, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x4, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x4, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x4, xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_X1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x1, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x1, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x1, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_X2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x2, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x2, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x2, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x2, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x2, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_X4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x4, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x4, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x4, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x4, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x4, xnn_init_f32_sigmoid_scalar_rr2_p5_params);
  }
}
