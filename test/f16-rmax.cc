// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/rmax.h>
#include "rmax-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RMAX__NEONFP16ARITH, n_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t n = 1; n < 32; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f16_rmax_ukernel__neonfp16arith);
    }
  }

  TEST(F16_RMAX__NEONFP16ARITH, n_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RMaxMicrokernelTester()
      .n(32)
      .Test(xnn_f16_rmax_ukernel__neonfp16arith);
  }

  TEST(F16_RMAX__NEONFP16ARITH, n_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t n = 64; n <= 128; n += 32) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f16_rmax_ukernel__neonfp16arith);
    }
  }

  TEST(F16_RMAX__NEONFP16ARITH, n_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t n = 32; n < 64; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f16_rmax_ukernel__neonfp16arith);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RMAX__F16C, n_lt_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t n = 1; n < 32; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f16_rmax_ukernel__f16c);
    }
  }

  TEST(F16_RMAX__F16C, n_eq_32) {
    TEST_REQUIRES_X86_F16C;
    RMaxMicrokernelTester()
      .n(32)
      .Test(xnn_f16_rmax_ukernel__f16c);
  }

  TEST(F16_RMAX__F16C, n_div_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t n = 64; n <= 128; n += 32) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f16_rmax_ukernel__f16c);
    }
  }

  TEST(F16_RMAX__F16C, n_gt_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t n = 32; n < 64; n++) {
      RMaxMicrokernelTester()
        .n(n)
        .Test(xnn_f16_rmax_ukernel__f16c);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
