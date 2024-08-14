// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vsigmoid.yaml
//   Generator: tools/generate-vunary-test.py


#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"
#include "next_prime.h"
#include "vunary-microkernel-tester.h"


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u4);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u4);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u8);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u8);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u12);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u12);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u16);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u16);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u20);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u20);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u24);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT64_P2_DIV_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_u24);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u4);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u4);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u4);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u4);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u4);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u8);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u8);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u8);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u8);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u8);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u12);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u12);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u12);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u12);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u12);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u16);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u16);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u20);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u20);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u20);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u20);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u20);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u24);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u24);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u24);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u24);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_LUT2048_P1_DIV_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_u24);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u4);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u4);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u8);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u8);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u12);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u12);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u16);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u16);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u20);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u20);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u24);
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__AARCH64_NEONFMA_RR1_P5_DIV_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_u24);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u4);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u8);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u12);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u16);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u20);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U20, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u24);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_U24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u4);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u8);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u12);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u16);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u20);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U20, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u24);
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_U24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u4);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u8);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u12);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u16);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u20);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U20, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u24);
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEON_RR2_P5_NR2RECPS_U24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u4);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u8);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u12);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u16);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u20);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u24);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u4);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u8);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u12);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u16);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u20);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u24);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u4);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u8);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u12);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u16);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u20);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u24);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u4);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u8);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u12);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u16);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u20);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u24);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u4);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u8);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u12);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u16);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u20);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u24);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u4);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u8);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u12);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u16);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u20);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u24);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u4);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u8);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u12);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u16);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u20);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u24);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u4);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u8);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u12);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u16);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u20);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u24);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2FMA_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u4);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u4);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u8);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u8);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u12);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u12);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u16);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u16);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u20);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u20);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u24);
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u24);
    }
  }

  TEST(F32_VSIGMOID__NEONFMA_RR1_P5_NR2RECPS_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u24);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u4);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u8);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u12);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u12);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u16);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u20);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U20, batch_div_20) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U20, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u20);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u24);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_LUT64_P2_DIV_U24, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_u24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u4);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u8);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u12);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u12);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u16);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u20);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U20, batch_div_20) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U20, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u20);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u24);
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__SSE2_RR2_P5_DIV_U24, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_u24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u4);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u8);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u12);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u12);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u16);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u20);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u20);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u24);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_LUT64_P2_DIV_U24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_u24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u4);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u8);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u12);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u12);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u16);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u20);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u20);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u24);
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__SSE41_RR2_P5_DIV_U24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_u24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u8);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u16);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u24);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U24, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u32);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u40);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u40);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u40);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u40);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U40, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u40);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u48);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u56);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u56);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u56);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u56);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U56, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u56);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u64);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u72);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u72);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u72);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u72);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U72, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u72);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u80);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_u80);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u8);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u8);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u8);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u8);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u16);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u24);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u24);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u24);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u24);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U24, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u32);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U32, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u40);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u40);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u40);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u40);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U40, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u40);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u48);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U48, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u56);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u56);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u56);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u56);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U56, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u56);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u64);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U64, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u72);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u72);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u72);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u72);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U72, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u72);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u80);
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX_RR2_P5_NR2_U80, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_u80);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u8);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u16);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u24);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u32);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u40);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u40);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u40);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u40);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u40);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u48);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u56);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u56);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u56);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u56);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u56);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u64);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u72);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u72);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u72);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u72);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u72);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u80);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_u80);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u8);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u8);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u8);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u8);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u16);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u24);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u24);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u24);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u24);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u32);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u40);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u40);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u40);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u40);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u40);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u48);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u56);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u56);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u56);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u56);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u56);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u64);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u72);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u72);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u72);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u72);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u72);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u80);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR1FMA_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_u80);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u8);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u8);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u8);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u8);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u16);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u24);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u24);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u24);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u24);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u32);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u40);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u40);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u40);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u40);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 40;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u40);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u48);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u56);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u56);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u56);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u56);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 56;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u56);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u64);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u72);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u72);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u72);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u72);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 72;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u72);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u80);
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX2_RR1_P5_NR2FMA_U80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_u80);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u16);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u32);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u48);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u64);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u80);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u80);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u96);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u96);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u112);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u112);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u128);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_DIV_U128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_u128);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u16);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u32);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u48);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u64);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u80);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u80);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u96);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u96);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u112);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u112);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u128);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_LUT16_P3_PERM_SCALEF_NR1FMA_U128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_u128);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u16);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u32);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u48);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u64);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u80);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u80);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u96);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u96);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u112);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u112);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u128);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_DIV_U128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_u128);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u16);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u32);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u48);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u64);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u80);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u80);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u96);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u96);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u112);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u112);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u128);
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR1_P5_SCALEF_NR1FMA_U128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_u128);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u16);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u32);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u48);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u64);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u80);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u80);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u96);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u96);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u112);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u112);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u128);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_DIV_U128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_u128);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u16);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u16);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u32);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u32);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u48);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u48);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u64);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u64);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u80);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u80);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 80;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u80);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u96);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u96);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 96;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u96);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u112);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u112);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 112;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u112);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u128);
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u128);
    }
  }

  TEST(F32_VSIGMOID__AVX512F_RR2_LUT32_P2_PERM2_SCALEF_NR1FMA_U128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 128;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_u128);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u4);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U4, batch_div_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U4, batch_lt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U4, batch_gt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U4, inplace) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u4);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u8);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U8, batch_div_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U8, batch_lt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U8, batch_gt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U8, inplace) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u8);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u12);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U12, batch_div_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U12, batch_lt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U12, batch_gt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U12, inplace) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u12);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u16);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U16, batch_div_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U16, batch_lt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U16, batch_gt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U16, inplace) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u16);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u20);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U20, batch_div_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U20, batch_lt_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U20, batch_gt_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U20, inplace) {
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u20);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u24);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U24, batch_div_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U24, batch_lt_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U24, batch_gt_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_LUT64_P2_DIV_U24, inplace) {
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_u24);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u4);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U4, batch_div_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U4, batch_lt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U4, batch_gt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U4, inplace) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u4);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u8);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U8, batch_div_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U8, batch_lt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U8, batch_gt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U8, inplace) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u8);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u12);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U12, batch_div_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U12, batch_lt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U12, batch_gt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U12, inplace) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u12);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u16);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U16, batch_div_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U16, batch_lt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U16, batch_gt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U16, inplace) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u16);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u20);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U20, batch_div_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U20, batch_lt_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U20, batch_gt_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U20, inplace) {
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u20);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u24);
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U24, batch_div_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U24, batch_lt_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U24, batch_gt_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMSIMD_RR2_P5_DIV_U24, inplace) {
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_u24);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u4);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U4, batch_div_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U4, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u4);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u8);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U8, batch_div_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U8, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u8);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u12);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U12, batch_div_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U12, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u12);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u16);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U16, batch_div_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U16, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u16);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u20);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U20, batch_div_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U20, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u20);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u24);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U24, batch_div_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_FMA_RR2_P5_DIV_U24, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_fma_rr2_p5_div_u24);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u4);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U4, batch_div_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U4, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u4);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u8);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U8, batch_div_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U8, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u8);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u12);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U12, batch_div_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U12, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u12);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u16);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U16, batch_div_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U16, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u16);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U20, batch_eq_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u20);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U20, batch_div_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U20, batch_lt_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U20, batch_gt_20) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U20, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u20);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u24);
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U24, batch_div_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMBLENDVPS_RR2_P5_DIV_U24, inplace) {
    TEST_REQUIRES_WASM_BLENDVPS;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmblendvps_rr2_p5_div_u24);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u4);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U4, batch_div_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U4, batch_lt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U4, batch_gt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U4, inplace) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u4);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u8);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U8, batch_div_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U8, batch_lt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U8, batch_gt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U8, inplace) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u8);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u12);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U12, batch_div_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U12, batch_lt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U12, batch_gt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U12, inplace) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u12);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u16);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U16, batch_div_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U16, batch_lt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U16, batch_gt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U16, inplace) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u16);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u20);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U20, batch_div_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U20, batch_lt_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U20, batch_gt_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U20, inplace) {
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u20);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u24);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U24, batch_div_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U24, batch_lt_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U24, batch_gt_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_LUT64_P2_DIV_U24, inplace) {
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_u24);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u4);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U4, batch_div_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U4, batch_lt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U4, batch_gt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U4, inplace) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u4);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u8);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U8, batch_div_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U8, batch_lt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U8, batch_gt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U8, inplace) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u8);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u12);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U12, batch_div_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U12, batch_lt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U12, batch_gt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U12, inplace) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u12);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u16);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U16, batch_div_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U16, batch_lt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U16, batch_gt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U16, inplace) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u16);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u20);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U20, batch_div_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U20, batch_lt_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U20, batch_gt_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U20, inplace) {
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u20);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u24);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U24, batch_div_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U24, batch_lt_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U24, batch_gt_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_FMA_RR2_P5_DIV_U24, inplace) {
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_u24);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u4);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U4, batch_div_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U4, batch_lt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U4, batch_gt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U4, inplace) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u4);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u8);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U8, batch_div_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U8, batch_lt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U8, batch_gt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U8, inplace) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u8);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u12);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U12, batch_div_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U12, batch_lt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U12, batch_gt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U12, inplace) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u12);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u16);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U16, batch_div_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U16, batch_lt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U16, batch_gt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U16, inplace) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u16);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u20);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U20, batch_div_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U20, batch_lt_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U20, batch_gt_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U20, inplace) {
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u20);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u24);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U24, batch_div_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U24, batch_lt_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U24, batch_gt_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_LUT64_P2_DIV_U24, inplace) {
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_u24);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u4);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U4, batch_div_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U4, batch_lt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U4, batch_gt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u4);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U4, inplace) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u4);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u8);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U8, batch_div_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U8, batch_lt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U8, batch_gt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u8);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U8, inplace) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u8);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u12);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U12, batch_div_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U12, batch_lt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U12, batch_gt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u12);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U12, inplace) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u12);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u16);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U16, batch_div_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U16, batch_lt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U16, batch_gt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u16);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U16, inplace) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u16);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u20);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U20, batch_div_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U20, batch_lt_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U20, batch_gt_20) {
    const size_t batch_step = 20;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u20);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U20, inplace) {
    const size_t batch_step = 20;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u20);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u24);
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U24, batch_div_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U24, batch_lt_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U24, batch_gt_24) {
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u24);
    }
  }

  TEST(F32_VSIGMOID__WASMRELAXEDSIMD_RR2_P5_DIV_U24, inplace) {
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_u24);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u1);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U1, batch_gt_1) {
  const size_t batch_step = 1;
  for (size_t batch_size = batch_step + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u1);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U1, inplace) {
  const size_t batch_step = 1;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u1);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U2, batch_div_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U2, batch_lt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U2, batch_gt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U2, inplace) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u2);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u4);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U4, batch_div_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u4);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U4, batch_lt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u4);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U4, batch_gt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u4);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT64_P2_DIV_U4, inplace) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_u4);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u1);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U1, batch_gt_1) {
  const size_t batch_step = 1;
  for (size_t batch_size = batch_step + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u1);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U1, inplace) {
  const size_t batch_step = 1;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u1);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u2);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U2, batch_div_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u2);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U2, batch_lt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u2);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U2, batch_gt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u2);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U2, inplace) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u2);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u4);
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U4, batch_div_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u4);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U4, batch_lt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u4);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U4, batch_gt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u4);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_LUT2048_P1_DIV_U4, inplace) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_u4);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u1);
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U1, batch_gt_1) {
  const size_t batch_step = 1;
  for (size_t batch_size = batch_step + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u1);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U1, inplace) {
  const size_t batch_step = 1;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u1);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u2);
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U2, batch_div_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u2);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U2, batch_lt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u2);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U2, batch_gt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u2);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U2, inplace) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u2);
  }
}


TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u4);
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U4, batch_div_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u4);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U4, batch_lt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u4);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U4, batch_gt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u4);
  }
}

TEST(F32_VSIGMOID__SCALAR_RR2_P5_DIV_U4, inplace) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_u4);
  }
}
