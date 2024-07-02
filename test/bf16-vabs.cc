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
#include <cstddef>
#include <limits>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"
#include "vunary-microkernel-tester.h"


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_VABS__NEONBF16_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u8, xnn_init_bf16_abs_neon_params);
  }

  TEST(BF16_VABS__NEONBF16_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u8, xnn_init_bf16_abs_neon_params);
    }
  }

  TEST(BF16_VABS__NEONBF16_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u8, xnn_init_bf16_abs_neon_params);
    }
  }

  TEST(BF16_VABS__NEONBF16_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u8, xnn_init_bf16_abs_neon_params);
    }
  }

  TEST(BF16_VABS__NEONBF16_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u8, xnn_init_bf16_abs_neon_params);
    }
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_VABS__NEONBF16_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_BF16;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u16, xnn_init_bf16_abs_neon_params);
  }

  TEST(BF16_VABS__NEONBF16_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u16, xnn_init_bf16_abs_neon_params);
    }
  }

  TEST(BF16_VABS__NEONBF16_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u16, xnn_init_bf16_abs_neon_params);
    }
  }

  TEST(BF16_VABS__NEONBF16_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u16, xnn_init_bf16_abs_neon_params);
    }
  }

  TEST(BF16_VABS__NEONBF16_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u16, xnn_init_bf16_abs_neon_params);
    }
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(BF16_VABS__NEONBF16_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_BF16;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u24, xnn_init_bf16_abs_neon_params);
  }

  TEST(BF16_VABS__NEONBF16_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u24, xnn_init_bf16_abs_neon_params);
    }
  }

  TEST(BF16_VABS__NEONBF16_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u24, xnn_init_bf16_abs_neon_params);
    }
  }

  TEST(BF16_VABS__NEONBF16_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u24, xnn_init_bf16_abs_neon_params);
    }
  }

  TEST(BF16_VABS__NEONBF16_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_BF16;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestAbs(xnn_bf16_vabs_ukernel__neonbf16_u24, xnn_init_bf16_abs_neon_params);
    }
  }
#endif  // XNN_ENABLE_ARM_BF16 && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
