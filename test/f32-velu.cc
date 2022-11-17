// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-velu.yaml
//   Generator: tools/generate-vunary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vunary.h>
#include "vunary-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEON_RR2_LUT16_P3_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x4, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x4, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x4, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x4, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x4, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X4, prescale) {
    TEST_REQUIRES_ARM_NEON;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x4, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X4, alpha) {
    TEST_REQUIRES_ARM_NEON;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x4, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X4, beta) {
    TEST_REQUIRES_ARM_NEON;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x4, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEON_RR2_LUT16_P3_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x8, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x8, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x8, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x8, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x8, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X8, prescale) {
    TEST_REQUIRES_ARM_NEON;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x8, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X8, alpha) {
    TEST_REQUIRES_ARM_NEON;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x8, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X8, beta) {
    TEST_REQUIRES_ARM_NEON;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x8, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEON_RR2_LUT16_P3_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x12, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x12, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x12, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x12, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x12, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X12, prescale) {
    TEST_REQUIRES_ARM_NEON;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x12, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X12, alpha) {
    TEST_REQUIRES_ARM_NEON;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x12, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X12, beta) {
    TEST_REQUIRES_ARM_NEON;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x12, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEON_RR2_LUT16_P3_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x16, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x16, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x16, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x16, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x16, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X16, prescale) {
    TEST_REQUIRES_ARM_NEON;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x16, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X16, alpha) {
    TEST_REQUIRES_ARM_NEON;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x16, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X16, beta) {
    TEST_REQUIRES_ARM_NEON;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x16, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEON_RR2_LUT16_P3_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x20, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x20, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x20, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x20, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X20, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x20, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X20, prescale) {
    TEST_REQUIRES_ARM_NEON;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x20, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X20, alpha) {
    TEST_REQUIRES_ARM_NEON;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x20, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X20, beta) {
    TEST_REQUIRES_ARM_NEON;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x20, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEON_RR2_LUT16_P3_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x24, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x24, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x24, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x24, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x24, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X24, prescale) {
    TEST_REQUIRES_ARM_NEON;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x24, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X24, alpha) {
    TEST_REQUIRES_ARM_NEON;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x24, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_LUT16_P3_X24, beta) {
    TEST_REQUIRES_ARM_NEON;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x24, xnn_init_f32_elu_neon_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEON_RR2_P6_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x4, xnn_init_f32_elu_neon_rr2_p6_params);
  }

  TEST(F32_VELU__NEON_RR2_P6_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x4, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x4, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x4, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x4, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X4, prescale) {
    TEST_REQUIRES_ARM_NEON;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x4, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X4, alpha) {
    TEST_REQUIRES_ARM_NEON;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x4, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X4, beta) {
    TEST_REQUIRES_ARM_NEON;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x4, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEON_RR2_P6_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x8, xnn_init_f32_elu_neon_rr2_p6_params);
  }

  TEST(F32_VELU__NEON_RR2_P6_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x8, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x8, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x8, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x8, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X8, prescale) {
    TEST_REQUIRES_ARM_NEON;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x8, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X8, alpha) {
    TEST_REQUIRES_ARM_NEON;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x8, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X8, beta) {
    TEST_REQUIRES_ARM_NEON;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x8, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEON_RR2_P6_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x12, xnn_init_f32_elu_neon_rr2_p6_params);
  }

  TEST(F32_VELU__NEON_RR2_P6_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x12, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x12, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x12, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x12, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X12, prescale) {
    TEST_REQUIRES_ARM_NEON;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x12, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X12, alpha) {
    TEST_REQUIRES_ARM_NEON;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x12, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X12, beta) {
    TEST_REQUIRES_ARM_NEON;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x12, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEON_RR2_P6_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x16, xnn_init_f32_elu_neon_rr2_p6_params);
  }

  TEST(F32_VELU__NEON_RR2_P6_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x16, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x16, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x16, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x16, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X16, prescale) {
    TEST_REQUIRES_ARM_NEON;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x16, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X16, alpha) {
    TEST_REQUIRES_ARM_NEON;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x16, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X16, beta) {
    TEST_REQUIRES_ARM_NEON;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x16, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEON_RR2_P6_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x20, xnn_init_f32_elu_neon_rr2_p6_params);
  }

  TEST(F32_VELU__NEON_RR2_P6_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x20, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x20, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x20, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X20, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x20, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X20, prescale) {
    TEST_REQUIRES_ARM_NEON;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x20, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X20, alpha) {
    TEST_REQUIRES_ARM_NEON;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x20, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X20, beta) {
    TEST_REQUIRES_ARM_NEON;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x20, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEON_RR2_P6_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x24, xnn_init_f32_elu_neon_rr2_p6_params);
  }

  TEST(F32_VELU__NEON_RR2_P6_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x24, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x24, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x24, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x24, xnn_init_f32_elu_neon_rr2_p6_params);
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X24, prescale) {
    TEST_REQUIRES_ARM_NEON;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x24, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X24, alpha) {
    TEST_REQUIRES_ARM_NEON;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x24, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEON_RR2_P6_X24, beta) {
    TEST_REQUIRES_ARM_NEON;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neon_rr2_p6_x24, xnn_init_f32_elu_neon_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x4, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x4, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x4, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x4, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x4, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X4, prescale) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x4, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X4, alpha) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x4, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X4, beta) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x4, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x8, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x8, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x8, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x8, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x8, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X8, prescale) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x8, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X8, alpha) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x8, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X8, beta) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x8, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x12, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x12, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x12, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x12, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x12, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X12, prescale) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x12, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X12, alpha) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x12, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X12, beta) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x12, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x16, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x16, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x16, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x16, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x16, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X16, prescale) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x16, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X16, alpha) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x16, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X16, beta) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x16, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x20, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x20, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x20, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x20, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x20, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X20, prescale) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x20, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X20, alpha) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x20, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X20, beta) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x20, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x24, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x24, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x24, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x24, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x24, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X24, prescale) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x24, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X24, alpha) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x24, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_LUT16_P3_X24, beta) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x24, xnn_init_f32_elu_neonfma_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEONFMA_RR1_P6_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x4, xnn_init_f32_elu_neonfma_rr1_p6_params);
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x4, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x4, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x4, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x4, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X4, prescale) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x4, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X4, alpha) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x4, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X4, beta) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x4, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEONFMA_RR1_P6_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x8, xnn_init_f32_elu_neonfma_rr1_p6_params);
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x8, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x8, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x8, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x8, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X8, prescale) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x8, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X8, alpha) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x8, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X8, beta) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x8, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEONFMA_RR1_P6_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x12, xnn_init_f32_elu_neonfma_rr1_p6_params);
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x12, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x12, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x12, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x12, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X12, prescale) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x12, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X12, alpha) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x12, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X12, beta) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x12, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEONFMA_RR1_P6_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x16, xnn_init_f32_elu_neonfma_rr1_p6_params);
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x16, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x16, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x16, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x16, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X16, prescale) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x16, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X16, alpha) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x16, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X16, beta) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x16, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEONFMA_RR1_P6_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x20, xnn_init_f32_elu_neonfma_rr1_p6_params);
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x20, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x20, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x20, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x20, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X20, prescale) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x20, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X20, alpha) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x20, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X20, beta) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x20, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VELU__NEONFMA_RR1_P6_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x24, xnn_init_f32_elu_neonfma_rr1_p6_params);
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x24, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x24, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x24, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x24, xnn_init_f32_elu_neonfma_rr1_p6_params);
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X24, prescale) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x24, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X24, alpha) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x24, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__NEONFMA_RR1_P6_X24, beta) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__neonfma_rr1_p6_x24, xnn_init_f32_elu_neonfma_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X4, prescale) {
    TEST_REQUIRES_X86_SSE2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X4, alpha) {
    TEST_REQUIRES_X86_SSE2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X4, beta) {
    TEST_REQUIRES_X86_SSE2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X8, prescale) {
    TEST_REQUIRES_X86_SSE2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X8, alpha) {
    TEST_REQUIRES_X86_SSE2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X8, beta) {
    TEST_REQUIRES_X86_SSE2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X12, prescale) {
    TEST_REQUIRES_X86_SSE2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X12, alpha) {
    TEST_REQUIRES_X86_SSE2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X12, beta) {
    TEST_REQUIRES_X86_SSE2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X16, prescale) {
    TEST_REQUIRES_X86_SSE2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X16, alpha) {
    TEST_REQUIRES_X86_SSE2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X16, beta) {
    TEST_REQUIRES_X86_SSE2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X20, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X20, prescale) {
    TEST_REQUIRES_X86_SSE2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X20, alpha) {
    TEST_REQUIRES_X86_SSE2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X20, beta) {
    TEST_REQUIRES_X86_SSE2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X24, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X24, prescale) {
    TEST_REQUIRES_X86_SSE2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X24, alpha) {
    TEST_REQUIRES_X86_SSE2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_LUT16_P3_X24, beta) {
    TEST_REQUIRES_X86_SSE2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE2_RR2_P6_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
  }

  TEST(F32_VELU__SSE2_RR2_P6_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X4, prescale) {
    TEST_REQUIRES_X86_SSE2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X4, alpha) {
    TEST_REQUIRES_X86_SSE2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X4, beta) {
    TEST_REQUIRES_X86_SSE2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE2_RR2_P6_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
  }

  TEST(F32_VELU__SSE2_RR2_P6_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X8, prescale) {
    TEST_REQUIRES_X86_SSE2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X8, alpha) {
    TEST_REQUIRES_X86_SSE2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X8, beta) {
    TEST_REQUIRES_X86_SSE2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE2_RR2_P6_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
  }

  TEST(F32_VELU__SSE2_RR2_P6_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X12, prescale) {
    TEST_REQUIRES_X86_SSE2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X12, alpha) {
    TEST_REQUIRES_X86_SSE2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X12, beta) {
    TEST_REQUIRES_X86_SSE2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE2_RR2_P6_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
  }

  TEST(F32_VELU__SSE2_RR2_P6_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X16, prescale) {
    TEST_REQUIRES_X86_SSE2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X16, alpha) {
    TEST_REQUIRES_X86_SSE2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X16, beta) {
    TEST_REQUIRES_X86_SSE2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE2_RR2_P6_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
  }

  TEST(F32_VELU__SSE2_RR2_P6_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X20, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X20, prescale) {
    TEST_REQUIRES_X86_SSE2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X20, alpha) {
    TEST_REQUIRES_X86_SSE2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X20, beta) {
    TEST_REQUIRES_X86_SSE2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE2_RR2_P6_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
  }

  TEST(F32_VELU__SSE2_RR2_P6_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X24, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X24, prescale) {
    TEST_REQUIRES_X86_SSE2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X24, alpha) {
    TEST_REQUIRES_X86_SSE2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE2_RR2_P6_X24, beta) {
    TEST_REQUIRES_X86_SSE2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse2_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X4, prescale) {
    TEST_REQUIRES_X86_SSE41;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X4, alpha) {
    TEST_REQUIRES_X86_SSE41;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X4, beta) {
    TEST_REQUIRES_X86_SSE41;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x4, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X8, prescale) {
    TEST_REQUIRES_X86_SSE41;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X8, alpha) {
    TEST_REQUIRES_X86_SSE41;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X8, beta) {
    TEST_REQUIRES_X86_SSE41;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x8, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X12, prescale) {
    TEST_REQUIRES_X86_SSE41;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X12, alpha) {
    TEST_REQUIRES_X86_SSE41;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X12, beta) {
    TEST_REQUIRES_X86_SSE41;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x12, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X16, prescale) {
    TEST_REQUIRES_X86_SSE41;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X16, alpha) {
    TEST_REQUIRES_X86_SSE41;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X16, beta) {
    TEST_REQUIRES_X86_SSE41;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x16, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X20, prescale) {
    TEST_REQUIRES_X86_SSE41;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X20, alpha) {
    TEST_REQUIRES_X86_SSE41;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X20, beta) {
    TEST_REQUIRES_X86_SSE41;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x20, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X24, prescale) {
    TEST_REQUIRES_X86_SSE41;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X24, alpha) {
    TEST_REQUIRES_X86_SSE41;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_LUT16_P3_X24, beta) {
    TEST_REQUIRES_X86_SSE41;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x24, xnn_init_f32_elu_sse2_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE41_RR2_P6_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
  }

  TEST(F32_VELU__SSE41_RR2_P6_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X4, prescale) {
    TEST_REQUIRES_X86_SSE41;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X4, alpha) {
    TEST_REQUIRES_X86_SSE41;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X4, beta) {
    TEST_REQUIRES_X86_SSE41;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x4, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE41_RR2_P6_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
  }

  TEST(F32_VELU__SSE41_RR2_P6_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X8, prescale) {
    TEST_REQUIRES_X86_SSE41;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X8, alpha) {
    TEST_REQUIRES_X86_SSE41;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X8, beta) {
    TEST_REQUIRES_X86_SSE41;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x8, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE41_RR2_P6_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
  }

  TEST(F32_VELU__SSE41_RR2_P6_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X12, prescale) {
    TEST_REQUIRES_X86_SSE41;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X12, alpha) {
    TEST_REQUIRES_X86_SSE41;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X12, beta) {
    TEST_REQUIRES_X86_SSE41;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x12, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE41_RR2_P6_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
  }

  TEST(F32_VELU__SSE41_RR2_P6_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X16, prescale) {
    TEST_REQUIRES_X86_SSE41;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X16, alpha) {
    TEST_REQUIRES_X86_SSE41;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X16, beta) {
    TEST_REQUIRES_X86_SSE41;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x16, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE41_RR2_P6_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
  }

  TEST(F32_VELU__SSE41_RR2_P6_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X20, prescale) {
    TEST_REQUIRES_X86_SSE41;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X20, alpha) {
    TEST_REQUIRES_X86_SSE41;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X20, beta) {
    TEST_REQUIRES_X86_SSE41;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x20, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__SSE41_RR2_P6_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
  }

  TEST(F32_VELU__SSE41_RR2_P6_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X24, prescale) {
    TEST_REQUIRES_X86_SSE41;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X24, alpha) {
    TEST_REQUIRES_X86_SSE41;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__SSE41_RR2_P6_X24, beta) {
    TEST_REQUIRES_X86_SSE41;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__sse41_rr2_p6_x24, xnn_init_f32_elu_sse2_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x8, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x8, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x8, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x8, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x8, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X8, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x8, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X8, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x8, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X8, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x8, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x16, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x16, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x16, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x16, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x16, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X16, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x16, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X16, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x16, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X16, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x16, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x24, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x24, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x24, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x24, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x24, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X24, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x24, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X24, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x24, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X24, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x24, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x32, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x32, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x32, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x32, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x32, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X32, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x32, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X32, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x32, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X32, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x32, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x40, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x40, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x40, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x40, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X40, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x40, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X40, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x40, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X40, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x40, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X40, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x40, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x48, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x48, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x48, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x48, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x48, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X48, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x48, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X48, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x48, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT4_P4_PERM_X48, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x48, xnn_init_f32_elu_avx_rr2_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_LUT16_P3_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x8, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x8, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x8, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x8, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x8, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X8, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x8, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X8, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x8, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X8, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x8, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_LUT16_P3_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x16, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x16, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x16, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x16, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x16, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X16, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x16, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X16, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x16, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X16, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x16, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_LUT16_P3_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x24, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x24, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x24, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x24, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x24, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X24, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x24, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X24, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x24, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X24, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x24, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_LUT16_P3_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x32, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x32, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x32, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x32, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x32, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X32, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x32, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X32, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x32, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X32, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x32, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_LUT16_P3_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x40, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x40, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x40, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x40, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X40, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x40, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X40, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x40, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X40, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x40, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X40, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x40, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_LUT16_P3_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x48, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x48, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x48, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x48, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x48, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X48, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x48, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X48, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x48, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_LUT16_P3_X48, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x48, xnn_init_f32_elu_avx_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_P6_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x8, xnn_init_f32_elu_avx_rr2_p6_params);
  }

  TEST(F32_VELU__AVX_RR2_P6_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x8, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x8, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x8, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x8, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X8, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x8, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X8, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x8, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X8, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x8, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_P6_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x16, xnn_init_f32_elu_avx_rr2_p6_params);
  }

  TEST(F32_VELU__AVX_RR2_P6_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x16, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x16, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x16, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x16, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X16, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x16, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X16, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x16, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X16, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x16, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_P6_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x24, xnn_init_f32_elu_avx_rr2_p6_params);
  }

  TEST(F32_VELU__AVX_RR2_P6_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x24, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x24, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x24, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x24, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X24, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x24, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X24, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x24, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X24, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x24, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_P6_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x32, xnn_init_f32_elu_avx_rr2_p6_params);
  }

  TEST(F32_VELU__AVX_RR2_P6_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x32, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x32, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x32, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x32, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X32, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x32, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X32, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x32, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X32, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x32, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_P6_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x40, xnn_init_f32_elu_avx_rr2_p6_params);
  }

  TEST(F32_VELU__AVX_RR2_P6_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x40, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x40, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x40, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X40, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x40, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X40, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x40, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X40, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x40, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X40, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x40, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX_RR2_P6_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x48, xnn_init_f32_elu_avx_rr2_p6_params);
  }

  TEST(F32_VELU__AVX_RR2_P6_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x48, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x48, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x48, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x48, xnn_init_f32_elu_avx_rr2_p6_params);
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X48, prescale) {
    TEST_REQUIRES_X86_AVX;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x48, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X48, alpha) {
    TEST_REQUIRES_X86_AVX;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x48, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX_RR2_P6_X48, beta) {
    TEST_REQUIRES_X86_AVX;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx_rr2_p6_x48, xnn_init_f32_elu_avx_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X8, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X8, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X8, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X16, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X16, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X16, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X24, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X24, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X24, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X32, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X32, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X32, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X40, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X40, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X40, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X48, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X48, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X48, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X56, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X56, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X56, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X64, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X64, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X64, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 73; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X72, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X72, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X72, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X80, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X80, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT4_P4_PERM_X80, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut4_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X8, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X8, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X8, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x8, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X16, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X16, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X16, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x16, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X24, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X24, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X24, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x24, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X32, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X32, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X32, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x32, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X40, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X40, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X40, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x40, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X48, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X48, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X48, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x48, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X56, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X56, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X56, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x56, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X64, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X64, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X64, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x64, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 73; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X72, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X72, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X72, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x72, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X80, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X80, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT8_P4_PERM_X80, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x80, xnn_init_f32_elu_avx2_rr1_lut8_p4_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x8, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x8, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x8, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x8, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x8, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X8, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x8, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X8, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x8, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X8, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x8, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x16, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x16, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x16, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x16, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x16, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X16, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x16, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X16, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x16, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X16, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x16, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x24, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x24, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x24, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x24, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x24, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X24, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x24, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X24, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x24, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X24, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x24, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x32, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x32, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x32, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x32, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x32, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X32, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x32, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X32, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x32, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X32, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x32, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x40, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x40, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x40, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x40, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x40, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X40, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x40, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X40, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x40, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X40, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x40, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x48, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x48, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x48, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x48, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x48, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X48, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x48, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X48, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x48, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X48, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x48, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x56, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x56, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x56, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x56, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x56, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X56, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x56, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X56, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x56, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X56, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x56, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x64, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x64, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x64, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x64, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x64, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X64, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x64, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X64, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x64, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X64, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x64, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x72, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x72, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x72, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 73; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x72, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x72, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X72, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x72, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X72, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x72, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X72, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x72, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x80, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x80, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x80, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x80, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x80, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X80, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x80, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X80, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x80, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_LUT16_P3_GATHER_X80, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x80, xnn_init_f32_elu_avx2_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_P6_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x8, xnn_init_f32_elu_avx2_rr1_p6_params);
  }

  TEST(F32_VELU__AVX2_RR1_P6_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x8, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x8, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x8, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x8, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X8, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x8, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X8, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x8, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X8, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x8, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_P6_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x16, xnn_init_f32_elu_avx2_rr1_p6_params);
  }

  TEST(F32_VELU__AVX2_RR1_P6_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x16, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x16, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x16, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x16, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X16, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x16, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X16, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x16, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X16, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x16, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_P6_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x24, xnn_init_f32_elu_avx2_rr1_p6_params);
  }

  TEST(F32_VELU__AVX2_RR1_P6_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x24, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x24, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x24, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x24, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X24, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x24, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X24, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x24, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X24, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x24, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_P6_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x32, xnn_init_f32_elu_avx2_rr1_p6_params);
  }

  TEST(F32_VELU__AVX2_RR1_P6_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x32, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x32, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x32, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x32, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X32, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x32, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X32, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x32, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X32, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x32, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_P6_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x40, xnn_init_f32_elu_avx2_rr1_p6_params);
  }

  TEST(F32_VELU__AVX2_RR1_P6_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x40, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x40, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x40, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x40, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X40, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x40, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X40, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x40, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X40, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x40, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_P6_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x48, xnn_init_f32_elu_avx2_rr1_p6_params);
  }

  TEST(F32_VELU__AVX2_RR1_P6_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x48, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x48, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x48, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x48, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X48, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x48, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X48, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x48, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X48, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x48, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_P6_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x56, xnn_init_f32_elu_avx2_rr1_p6_params);
  }

  TEST(F32_VELU__AVX2_RR1_P6_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x56, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x56, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x56, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x56, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X56, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x56, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X56, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x56, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X56, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x56, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_P6_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x64, xnn_init_f32_elu_avx2_rr1_p6_params);
  }

  TEST(F32_VELU__AVX2_RR1_P6_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x64, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x64, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x64, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x64, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X64, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x64, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X64, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x64, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X64, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x64, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_P6_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x72, xnn_init_f32_elu_avx2_rr1_p6_params);
  }

  TEST(F32_VELU__AVX2_RR1_P6_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x72, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x72, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 73; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x72, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x72, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X72, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x72, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X72, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x72, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X72, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x72, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX2_RR1_P6_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x80, xnn_init_f32_elu_avx2_rr1_p6_params);
  }

  TEST(F32_VELU__AVX2_RR1_P6_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x80, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x80, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x80, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x80, xnn_init_f32_elu_avx2_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X80, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x80, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X80, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x80, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX2_RR1_P6_X80, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx2_rr1_p6_x80, xnn_init_f32_elu_avx2_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x16, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x16, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x16, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x16, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x16, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X16, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x16, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X16, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x16, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X16, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x16, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x32, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x32, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x32, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x32, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x32, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X32, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x32, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X32, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x32, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X32, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x32, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x48, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x48, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x48, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x48, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x48, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X48, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x48, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X48, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x48, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X48, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x48, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x64, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x64, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x64, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x64, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x64, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X64, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x64, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X64, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x64, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X64, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x64, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x80, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x80, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x80, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x80, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x80, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X80, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x80, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X80, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x80, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X80, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x80, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x96, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x96, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x96, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 97; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x96, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x96, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X96, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x96, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X96, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x96, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X96, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x96, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x112, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x112, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x112, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 113; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x112, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x112, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X112, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x112, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X112, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x112, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X112, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x112, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x128, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x128, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x128, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 129; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x128, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x128, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X128, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x128, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X128, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x128, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_LUT16_P3_PERM_X128, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x128, xnn_init_f32_elu_avx512_rr1_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_P6_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x16, xnn_init_f32_elu_avx512_rr1_p6_params);
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x16, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x16, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x16, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x16, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X16, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x16, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X16, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x16, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X16, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x16, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_P6_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x32, xnn_init_f32_elu_avx512_rr1_p6_params);
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x32, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x32, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x32, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x32, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X32, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x32, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X32, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x32, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X32, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x32, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_P6_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x48, xnn_init_f32_elu_avx512_rr1_p6_params);
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x48, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x48, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x48, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x48, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X48, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x48, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X48, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x48, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X48, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x48, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_P6_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x64, xnn_init_f32_elu_avx512_rr1_p6_params);
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x64, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x64, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x64, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x64, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X64, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x64, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X64, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x64, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X64, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x64, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_P6_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x80, xnn_init_f32_elu_avx512_rr1_p6_params);
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x80, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x80, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x80, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X80, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x80, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X80, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x80, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X80, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x80, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X80, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x80, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_P6_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x96, xnn_init_f32_elu_avx512_rr1_p6_params);
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x96, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x96, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 97; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x96, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X96, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x96, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X96, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x96, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X96, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x96, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X96, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x96, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_P6_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x112, xnn_init_f32_elu_avx512_rr1_p6_params);
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x112, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x112, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 113; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x112, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X112, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x112, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X112, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x112, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X112, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x112, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X112, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x112, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VELU__AVX512F_RR1_P6_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x128, xnn_init_f32_elu_avx512_rr1_p6_params);
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x128, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x128, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 129; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x128, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X128, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x128, xnn_init_f32_elu_avx512_rr1_p6_params);
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X128, prescale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x128, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X128, alpha) {
    TEST_REQUIRES_X86_AVX512F;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x128, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }

  TEST(F32_VELU__AVX512F_RR1_P6_X128, beta) {
    TEST_REQUIRES_X86_AVX512F;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__avx512f_rr1_p6_x128, xnn_init_f32_elu_avx512_rr1_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X4, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X4, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X4, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X8, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X8, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X8, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X12, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X12, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X12, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X16, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X16, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X16, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X20, batch_gt_20) {
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X20, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X20, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X20, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X24, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X24, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_LUT16_P3_X24, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X4, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X4, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X4, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X8, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X8, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X8, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X12, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X12, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X12, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X16, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X16, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X16, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X20, batch_gt_20) {
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X20, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X20, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X20, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X24, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X24, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_ARM_RR2_P6_X24, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X4, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X4, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X4, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X8, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X8, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X8, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X12, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X12, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X12, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X16, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X16, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X16, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X20, batch_gt_20) {
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X20, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X20, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X20, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X24, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X24, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_LUT16_P3_X24, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X4, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X4, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X4, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X8, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X8, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X8, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X12, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X12, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X12, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X16, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X16, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X16, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X20, batch_gt_20) {
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X20, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X20, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X20, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X24, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X24, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMSIMD_X86_RR2_P6_X24, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X4, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X4, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X4, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X8, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X8, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X8, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X12, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X12, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X12, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X16, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X16, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X16, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X20, batch_gt_20) {
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X20, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X20, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X20, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X24, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X24, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_LUT16_P3_X24, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X4, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X4, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X4, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X8, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X8, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X8, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X12, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X12, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X12, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X16, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X16, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X16, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X20, batch_gt_20) {
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X20, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X20, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X20, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X24, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X24, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_FMA_RR2_P6_X24, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X4, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X4, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X4, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x4, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X8, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X8, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X8, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x8, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X12, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X12, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X12, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x12, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X16, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X16, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X16, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x16, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X20, batch_gt_20) {
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X20, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X20, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X20, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x20, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X24, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X24, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_LUT16_P3_X24, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x24, xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X4, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X4, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X4, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x4, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X8, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X8, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X8, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x8, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X12, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X12, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X12, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X12, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x12, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X16, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X16, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X16, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x16, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X20, batch_eq_20) {
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X20, batch_div_20) {
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X20, batch_lt_20) {
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X20, batch_gt_20) {
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X20, inplace) {
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X20, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X20, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X20, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x20, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X24, batch_eq_24) {
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X24, batch_div_24) {
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X24, batch_lt_24) {
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X24, batch_gt_24) {
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X24, inplace) {
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X24, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X24, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASMRELAXEDSIMD_RR2_P6_X24, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x24, xnn_init_f32_elu_wasmsimd_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASM_RR2_LUT16_P3_X1, batch_eq_1) {
    VUnaryMicrokernelTester()
      .batch_size(1)
      .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x1, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X1, batch_gt_1) {
    for (size_t batch_size = 2; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x1, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X1, inplace) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x1, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X1, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x1, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X1, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x1, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X1, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x1, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASM_RR2_LUT16_P3_X2, batch_eq_2) {
    VUnaryMicrokernelTester()
      .batch_size(2)
      .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X2, batch_div_2) {
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X2, batch_lt_2) {
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X2, batch_gt_2) {
    for (size_t batch_size = 3; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X2, inplace) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X2, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X2, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X2, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASM_RR2_LUT16_P3_X3, batch_eq_3) {
    VUnaryMicrokernelTester()
      .batch_size(3)
      .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X3, batch_div_3) {
    for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X3, batch_lt_3) {
    for (size_t batch_size = 1; batch_size < 3; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X3, batch_gt_3) {
    for (size_t batch_size = 4; batch_size < 6; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X3, inplace) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X3, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X3, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X3, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASM_RR2_LUT16_P3_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X4, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X4, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X4, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASM_RR2_LUT16_P3_X5, batch_eq_5) {
    VUnaryMicrokernelTester()
      .batch_size(5)
      .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X5, batch_div_5) {
    for (size_t batch_size = 10; batch_size < 50; batch_size += 5) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X5, batch_lt_5) {
    for (size_t batch_size = 1; batch_size < 5; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X5, batch_gt_5) {
    for (size_t batch_size = 6; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X5, inplace) {
    for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X5, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X5, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X5, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASM_RR2_LUT16_P3_X6, batch_eq_6) {
    VUnaryMicrokernelTester()
      .batch_size(6)
      .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X6, batch_div_6) {
    for (size_t batch_size = 12; batch_size < 60; batch_size += 6) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X6, batch_lt_6) {
    for (size_t batch_size = 1; batch_size < 6; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X6, batch_gt_6) {
    for (size_t batch_size = 7; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X6, inplace) {
    for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X6, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X6, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_LUT16_P3_X6, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASM_RR2_P6_X1, batch_eq_1) {
    VUnaryMicrokernelTester()
      .batch_size(1)
      .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x1, xnn_init_f32_elu_scalar_rr2_p6_params);
  }

  TEST(F32_VELU__WASM_RR2_P6_X1, batch_gt_1) {
    for (size_t batch_size = 2; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x1, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X1, inplace) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x1, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X1, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x1, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X1, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x1, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X1, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x1, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASM_RR2_P6_X2, batch_eq_2) {
    VUnaryMicrokernelTester()
      .batch_size(2)
      .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
  }

  TEST(F32_VELU__WASM_RR2_P6_X2, batch_div_2) {
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X2, batch_lt_2) {
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X2, batch_gt_2) {
    for (size_t batch_size = 3; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X2, inplace) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X2, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X2, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X2, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASM_RR2_P6_X3, batch_eq_3) {
    VUnaryMicrokernelTester()
      .batch_size(3)
      .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
  }

  TEST(F32_VELU__WASM_RR2_P6_X3, batch_div_3) {
    for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X3, batch_lt_3) {
    for (size_t batch_size = 1; batch_size < 3; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X3, batch_gt_3) {
    for (size_t batch_size = 4; batch_size < 6; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X3, inplace) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X3, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X3, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X3, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASM_RR2_P6_X4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
  }

  TEST(F32_VELU__WASM_RR2_P6_X4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X4, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X4, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X4, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASM_RR2_P6_X5, batch_eq_5) {
    VUnaryMicrokernelTester()
      .batch_size(5)
      .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
  }

  TEST(F32_VELU__WASM_RR2_P6_X5, batch_div_5) {
    for (size_t batch_size = 10; batch_size < 50; batch_size += 5) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X5, batch_lt_5) {
    for (size_t batch_size = 1; batch_size < 5; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X5, batch_gt_5) {
    for (size_t batch_size = 6; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X5, inplace) {
    for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X5, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X5, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X5, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VELU__WASM_RR2_P6_X6, batch_eq_6) {
    VUnaryMicrokernelTester()
      .batch_size(6)
      .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
  }

  TEST(F32_VELU__WASM_RR2_P6_X6, batch_div_6) {
    for (size_t batch_size = 12; batch_size < 60; batch_size += 6) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X6, batch_lt_6) {
    for (size_t batch_size = 1; batch_size < 6; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X6, batch_gt_6) {
    for (size_t batch_size = 7; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X6, inplace) {
    for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X6, prescale) {
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X6, alpha) {
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }

  TEST(F32_VELU__WASM_RR2_P6_X6, beta) {
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f32_velu_ukernel__wasm_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x1, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x1, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x1, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X1, prescale) {
  for (float prescale : std::vector<float>({0.1f, 10.0f})) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .prescale(prescale)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x1, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X1, alpha) {
  for (float alpha : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .alpha(alpha)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x1, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X1, beta) {
  for (float beta : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .beta(beta)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x1, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}


TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X2, prescale) {
  for (float prescale : std::vector<float>({0.1f, 10.0f})) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .prescale(prescale)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X2, alpha) {
  for (float alpha : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .alpha(alpha)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X2, beta) {
  for (float beta : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .beta(beta)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x2, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}


TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X3, batch_eq_3) {
  VUnaryMicrokernelTester()
    .batch_size(3)
    .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X3, inplace) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X3, prescale) {
  for (float prescale : std::vector<float>({0.1f, 10.0f})) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .prescale(prescale)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X3, alpha) {
  for (float alpha : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .alpha(alpha)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X3, beta) {
  for (float beta : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .beta(beta)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x3, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}


TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X4, prescale) {
  for (float prescale : std::vector<float>({0.1f, 10.0f})) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .prescale(prescale)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X4, alpha) {
  for (float alpha : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .alpha(alpha)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X4, beta) {
  for (float beta : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .beta(beta)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x4, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}


TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X5, batch_eq_5) {
  VUnaryMicrokernelTester()
    .batch_size(5)
    .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X5, batch_div_5) {
  for (size_t batch_size = 10; batch_size < 50; batch_size += 5) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X5, batch_lt_5) {
  for (size_t batch_size = 1; batch_size < 5; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X5, batch_gt_5) {
  for (size_t batch_size = 6; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X5, inplace) {
  for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X5, prescale) {
  for (float prescale : std::vector<float>({0.1f, 10.0f})) {
    for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .prescale(prescale)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X5, alpha) {
  for (float alpha : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .alpha(alpha)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X5, beta) {
  for (float beta : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .beta(beta)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x5, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}


TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X6, batch_eq_6) {
  VUnaryMicrokernelTester()
    .batch_size(6)
    .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X6, batch_div_6) {
  for (size_t batch_size = 12; batch_size < 60; batch_size += 6) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X6, batch_lt_6) {
  for (size_t batch_size = 1; batch_size < 6; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X6, batch_gt_6) {
  for (size_t batch_size = 7; batch_size < 12; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X6, inplace) {
  for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X6, prescale) {
  for (float prescale : std::vector<float>({0.1f, 10.0f})) {
    for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .prescale(prescale)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X6, alpha) {
  for (float alpha : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .alpha(alpha)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_LUT16_P3_X6, beta) {
  for (float beta : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .beta(beta)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x6, xnn_init_f32_elu_scalar_rr2_lut16_p3_params);
    }
  }
}


TEST(F32_VELU__SCALAR_RR2_P6_X1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x1, xnn_init_f32_elu_scalar_rr2_p6_params);
}

TEST(F32_VELU__SCALAR_RR2_P6_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x1, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x1, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X1, prescale) {
  for (float prescale : std::vector<float>({0.1f, 10.0f})) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .prescale(prescale)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x1, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X1, alpha) {
  for (float alpha : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .alpha(alpha)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x1, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X1, beta) {
  for (float beta : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .beta(beta)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x1, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}


TEST(F32_VELU__SCALAR_RR2_P6_X2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
}

TEST(F32_VELU__SCALAR_RR2_P6_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X2, prescale) {
  for (float prescale : std::vector<float>({0.1f, 10.0f})) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .prescale(prescale)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X2, alpha) {
  for (float alpha : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .alpha(alpha)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X2, beta) {
  for (float beta : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .beta(beta)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x2, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}


TEST(F32_VELU__SCALAR_RR2_P6_X3, batch_eq_3) {
  VUnaryMicrokernelTester()
    .batch_size(3)
    .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
}

TEST(F32_VELU__SCALAR_RR2_P6_X3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X3, inplace) {
  for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X3, prescale) {
  for (float prescale : std::vector<float>({0.1f, 10.0f})) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .prescale(prescale)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X3, alpha) {
  for (float alpha : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .alpha(alpha)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X3, beta) {
  for (float beta : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 15; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .beta(beta)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x3, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}


TEST(F32_VELU__SCALAR_RR2_P6_X4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
}

TEST(F32_VELU__SCALAR_RR2_P6_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X4, prescale) {
  for (float prescale : std::vector<float>({0.1f, 10.0f})) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .prescale(prescale)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X4, alpha) {
  for (float alpha : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .alpha(alpha)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X4, beta) {
  for (float beta : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .beta(beta)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x4, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}


TEST(F32_VELU__SCALAR_RR2_P6_X5, batch_eq_5) {
  VUnaryMicrokernelTester()
    .batch_size(5)
    .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
}

TEST(F32_VELU__SCALAR_RR2_P6_X5, batch_div_5) {
  for (size_t batch_size = 10; batch_size < 50; batch_size += 5) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X5, batch_lt_5) {
  for (size_t batch_size = 1; batch_size < 5; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X5, batch_gt_5) {
  for (size_t batch_size = 6; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X5, inplace) {
  for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X5, prescale) {
  for (float prescale : std::vector<float>({0.1f, 10.0f})) {
    for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .prescale(prescale)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X5, alpha) {
  for (float alpha : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .alpha(alpha)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X5, beta) {
  for (float beta : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 25; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .beta(beta)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x5, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}


TEST(F32_VELU__SCALAR_RR2_P6_X6, batch_eq_6) {
  VUnaryMicrokernelTester()
    .batch_size(6)
    .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
}

TEST(F32_VELU__SCALAR_RR2_P6_X6, batch_div_6) {
  for (size_t batch_size = 12; batch_size < 60; batch_size += 6) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X6, batch_lt_6) {
  for (size_t batch_size = 1; batch_size < 6; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X6, batch_gt_6) {
  for (size_t batch_size = 7; batch_size < 12; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X6, inplace) {
  for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X6, prescale) {
  for (float prescale : std::vector<float>({0.1f, 10.0f})) {
    for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .prescale(prescale)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X6, alpha) {
  for (float alpha : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .alpha(alpha)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}

TEST(F32_VELU__SCALAR_RR2_P6_X6, beta) {
  for (float beta : std::vector<float>({0.3f, 3.0f})) {
    for (size_t batch_size = 1; batch_size <= 30; batch_size += 5) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .beta(beta)
        .Test(xnn_f32_velu_ukernel__scalar_rr2_p6_x6, xnn_init_f32_elu_scalar_rr2_p6_params);
    }
  }
}
