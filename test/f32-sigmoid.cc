// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-sigmoid.yaml
//   Generator: tools/generate-vunop-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vunop.h>
#include "vunop-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_SIGMOID__NEONFMA_P5_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    VUnOpMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_sigmoid_ukernel__neonfma_p5_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
  }

  TEST(F32_SIGMOID__NEONFMA_P5_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_p5_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_P5_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_p5_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_P5_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_p5_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }

  TEST(F32_SIGMOID__NEONFMA_P5_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnOpMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_sigmoid_ukernel__neonfma_p5_x16, VUnOpMicrokernelTester::OpType::Sigmoid);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
