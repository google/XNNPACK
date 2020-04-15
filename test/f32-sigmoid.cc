// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-sigmoid.yaml
//   Generator: tools/generate-vunary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vunary.h>
#include "vunary-microkernel-tester.h"


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_DIV_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2FMA_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR1RECPS1FMA_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_P5_NR2RECPS_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X20, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_P5_NR2RECPS_X24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_DIV_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2FMA_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR1RECPS1FMA_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT64_P2_NR2RECPS_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X20, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT64_P2_NR2RECPS_X24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_DIV_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2FMA_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR1RECPS1FMA_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X4, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X12, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X20, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_RR1_LUT2048_P1_NR2RECPS_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X20, batch_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X20, batch_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X20, batch_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X20, batch_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X20, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_RR2_LUT2048_P1_NR2RECPS_X24, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEON_FRAC_P9_P10_NR1RECPS_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neon_frac_p9_p10_nr1recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEON_FRAC_P9_P10_NR1RECPS_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_frac_p9_p10_nr1recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_FRAC_P9_P10_NR1RECPS_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_frac_p9_p10_nr1recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_FRAC_P9_P10_NR1RECPS_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neon_frac_p9_p10_nr1recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEON_FRAC_P9_P10_NR1RECPS_X16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neon_frac_p9_p10_nr1recps_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__SSE2_P5_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__SSE2_P5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__SSE2_P5_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__SSE2_P5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__SSE2_P5_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE2;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X20, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__SSE2_P5_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE2;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE2_P5_DIV_X24, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__sse2_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__SSE41_P5_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__SSE41_P5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__SSE41_P5_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__SSE41_P5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__SSE41_P5_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__SSE41_P5_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__SSE41_P5_DIV_X24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__sse41_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x32, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x32, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x32, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x32, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x32, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x40, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x40, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x40, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x40, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x40, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x48, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x48, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x48, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x48, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x48, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x56, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x56, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x56, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x56, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x56, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x64, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x64, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x64, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x64, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x64, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x72, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x72, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x72, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 73; batch_size < 144; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x72, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x72, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x80, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x80, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x80, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x80, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x80, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x32, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x32, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x32, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x32, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x32, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x40, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x40, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x40, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x40, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x40, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x48, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x48, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x48, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x48, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x48, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x56, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x56, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x56, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x56, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x56, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x64, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x64, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x64, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x64, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x64, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x72, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x72, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x72, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 73; batch_size < 144; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x72, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x72, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x80, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x80, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x80, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x80, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR1FMA_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x80, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x8, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x24, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x32, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x32, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x32, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x32, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x32, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x40, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x40, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x40, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x40, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x40, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x48, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x48, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x48, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x48, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x48, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x56, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x56, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x56, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x56, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x56, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x64, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x64, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x64, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x64, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x64, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x72, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x72, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x72, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 73; batch_size < 144; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x72, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x72, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnOpMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x80, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x80, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x80, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 81; batch_size < 160; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x80, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__AVX2_RR1_P5_NR2FMA_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x80, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_SIGMOID__PSIMD_P5_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_PSIMD;
    VUnOpMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X4, batch_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X4, inplace) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_SIGMOID__PSIMD_P5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_PSIMD;
    VUnOpMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X8, inplace) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x8, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_SIGMOID__PSIMD_P5_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_PSIMD;
    VUnOpMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X12, batch_div_12) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X12, inplace) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x12, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_SIGMOID__PSIMD_P5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_PSIMD;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X16, inplace) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x16, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_SIGMOID__PSIMD_P5_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_PSIMD;
    VUnOpMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X20, batch_div_20) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 21; batch_size < 40; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X20, inplace) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x20, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_SIGMOID__PSIMD_P5_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_PSIMD;
    VUnOpMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X24, batch_div_24) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_SIGMOID__PSIMD_P5_DIV_X24, inplace) {
    TEST_REQUIRES_PSIMD;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__psimd_p5_div_x24, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


TEST(F32_SIGMOID__SCALAR_LUT2048_P1_DIV_X1, batch_eq_1) {
  VUnOpMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x1, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
}

TEST(F32_SIGMOID__SCALAR_LUT2048_P1_DIV_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x1, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT2048_P1_DIV_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x1, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT2048_P1_DIV_X2, batch_eq_2) {
  VUnOpMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x2, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
}

TEST(F32_SIGMOID__SCALAR_LUT2048_P1_DIV_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x2, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT2048_P1_DIV_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x2, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT2048_P1_DIV_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x2, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT2048_P1_DIV_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x2, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT2048_P1_DIV_X4, batch_eq_4) {
  VUnOpMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
}

TEST(F32_SIGMOID__SCALAR_LUT2048_P1_DIV_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT2048_P1_DIV_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT2048_P1_DIV_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT2048_P1_DIV_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT64_P2_DIV_X1, batch_eq_1) {
  VUnOpMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x1, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
}

TEST(F32_SIGMOID__SCALAR_LUT64_P2_DIV_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x1, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT64_P2_DIV_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x1, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT64_P2_DIV_X2, batch_eq_2) {
  VUnOpMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x2, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
}

TEST(F32_SIGMOID__SCALAR_LUT64_P2_DIV_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x2, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT64_P2_DIV_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x2, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT64_P2_DIV_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x2, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT64_P2_DIV_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x2, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT64_P2_DIV_X4, batch_eq_4) {
  VUnOpMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
}

TEST(F32_SIGMOID__SCALAR_LUT64_P2_DIV_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT64_P2_DIV_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT64_P2_DIV_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_LUT64_P2_DIV_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_P5_DIV_X1, batch_eq_1) {
  VUnOpMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_sigmoid_ukernel__scalar_p5_div_x1, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
}

TEST(F32_SIGMOID__SCALAR_P5_DIV_X1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_p5_div_x1, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_P5_DIV_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_sigmoid_ukernel__scalar_p5_div_x1, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_P5_DIV_X2, batch_eq_2) {
  VUnOpMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_sigmoid_ukernel__scalar_p5_div_x2, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
}

TEST(F32_SIGMOID__SCALAR_P5_DIV_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_p5_div_x2, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_P5_DIV_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_p5_div_x2, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_P5_DIV_X2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_p5_div_x2, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_P5_DIV_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_sigmoid_ukernel__scalar_p5_div_x2, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_P5_DIV_X4, batch_eq_4) {
  VUnOpMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_sigmoid_ukernel__scalar_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
}

TEST(F32_SIGMOID__SCALAR_P5_DIV_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_P5_DIV_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_P5_DIV_X4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_sigmoid_ukernel__scalar_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_SIGMOID__SCALAR_P5_DIV_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnOpMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_sigmoid_ukernel__scalar_p5_div_x4, VUnOpMicrokernelTester::OpType::Sigmoid, VUnOpMicrokernelTester::Variant::Scalar);
  }
}