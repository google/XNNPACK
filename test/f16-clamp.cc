// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-clamp.yaml
//   Generator: tools/generate-clamp-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/clamp.h>
#include "clamp-microkernel-tester.h"


#if XNN_ARCH_ARM64
  TEST(F16_CLAMP__NEONFP16ARITH_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ClampMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_clamp_ukernel__neonfp16arith_x8);
  }

  TEST(F16_CLAMP__NEONFP16ARITH_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_clamp_ukernel__neonfp16arith_x8);
    }
  }

  TEST(F16_CLAMP__NEONFP16ARITH_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_clamp_ukernel__neonfp16arith_x8);
    }
  }

  TEST(F16_CLAMP__NEONFP16ARITH_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_clamp_ukernel__neonfp16arith_x8);
    }
  }

  TEST(F16_CLAMP__NEONFP16ARITH_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_clamp_ukernel__neonfp16arith_x8);
    }
  }

  TEST(F16_CLAMP__NEONFP16ARITH_X8, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (uint8_t qmin = 1; qmin < 255; qmin++) {
        ClampMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .qmax(255)
          .Test(xnn_f16_clamp_ukernel__neonfp16arith_x8);
      }
    }
  }

  TEST(F16_CLAMP__NEONFP16ARITH_X8, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (uint8_t qmax = 1; qmax < 255; qmax++) {
        ClampMicrokernelTester()
          .batch_size(batch_size)
          .qmin(0)
          .qmax(qmax)
          .Test(xnn_f16_clamp_ukernel__neonfp16arith_x8);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_CLAMP__NEONFP16ARITH_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ClampMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_clamp_ukernel__neonfp16arith_x16);
  }

  TEST(F16_CLAMP__NEONFP16ARITH_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_clamp_ukernel__neonfp16arith_x16);
    }
  }

  TEST(F16_CLAMP__NEONFP16ARITH_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_clamp_ukernel__neonfp16arith_x16);
    }
  }

  TEST(F16_CLAMP__NEONFP16ARITH_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_clamp_ukernel__neonfp16arith_x16);
    }
  }

  TEST(F16_CLAMP__NEONFP16ARITH_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      ClampMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_clamp_ukernel__neonfp16arith_x16);
    }
  }

  TEST(F16_CLAMP__NEONFP16ARITH_X16, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (uint8_t qmin = 1; qmin < 255; qmin++) {
        ClampMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .qmax(255)
          .Test(xnn_f16_clamp_ukernel__neonfp16arith_x16);
      }
    }
  }

  TEST(F16_CLAMP__NEONFP16ARITH_X16, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (uint8_t qmax = 1; qmax < 255; qmax++) {
        ClampMicrokernelTester()
          .batch_size(batch_size)
          .qmin(0)
          .qmax(qmax)
          .Test(xnn_f16_clamp_ukernel__neonfp16arith_x16);
      }
    }
  }
#endif  // XNN_ARCH_ARM64
