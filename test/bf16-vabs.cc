// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/bf16-vabs.yaml
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


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_VABS__NEONBF16_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u8);
  }

  TEST(BF16_VABS__NEONBF16_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u8);
    }
  }

  TEST(BF16_VABS__NEONBF16_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u8);
    }
  }

  TEST(BF16_VABS__NEONBF16_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u8);
    }
  }

  TEST(BF16_VABS__NEONBF16_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_BF16;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u8);
    }
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_VABS__NEONBF16_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_BF16;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u16);
  }

  TEST(BF16_VABS__NEONBF16_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_BF16;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u16);
    }
  }

  TEST(BF16_VABS__NEONBF16_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_BF16;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u16);
    }
  }

  TEST(BF16_VABS__NEONBF16_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_BF16;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u16);
    }
  }

  TEST(BF16_VABS__NEONBF16_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_BF16;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u16);
    }
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_VABS__NEONBF16_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_BF16;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u24);
  }

  TEST(BF16_VABS__NEONBF16_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_BF16;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u24);
    }
  }

  TEST(BF16_VABS__NEONBF16_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_BF16;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u24);
    }
  }

  TEST(BF16_VABS__NEONBF16_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_BF16;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u24);
    }
  }

  TEST(BF16_VABS__NEONBF16_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_BF16;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u24);
    }
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
