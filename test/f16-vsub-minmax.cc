// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vsub-minmax.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/params-init.h>
#include <xnnpack/vbinary.h>
#include "vbinary-microkernel-tester.h"


#if XNN_ARCH_ARM64
  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x8, VBinaryMicrokernelTester::OpType::Sub);
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x8, VBinaryMicrokernelTester::OpType::Sub);
    }
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x8, VBinaryMicrokernelTester::OpType::Sub);
    }
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x8, VBinaryMicrokernelTester::OpType::Sub);
    }
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X8, inplace_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x8, VBinaryMicrokernelTester::OpType::Sub);
    }
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X8, inplace_b) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x8, VBinaryMicrokernelTester::OpType::Sub);
    }
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X8, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x8, VBinaryMicrokernelTester::OpType::Sub);
    }
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X8, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x8, VBinaryMicrokernelTester::OpType::Sub);
    }
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X8, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x8, VBinaryMicrokernelTester::OpType::Sub);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VBinaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x16, VBinaryMicrokernelTester::OpType::Sub);
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x16, VBinaryMicrokernelTester::OpType::Sub);
    }
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x16, VBinaryMicrokernelTester::OpType::Sub);
    }
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x16, VBinaryMicrokernelTester::OpType::Sub);
    }
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X16, inplace_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x16, VBinaryMicrokernelTester::OpType::Sub);
    }
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X16, inplace_b) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x16, VBinaryMicrokernelTester::OpType::Sub);
    }
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X16, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x16, VBinaryMicrokernelTester::OpType::Sub);
    }
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X16, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x16, VBinaryMicrokernelTester::OpType::Sub);
    }
  }

  TEST(F16_VSUB_MINMAX__NEONFP16ARITH_X16, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f16_vsub_minmax_ukernel__neonfp16arith_x16, VBinaryMicrokernelTester::OpType::Sub);
    }
  }
#endif  // XNN_ARCH_ARM64
