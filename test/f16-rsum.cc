// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-rsum.yaml
//   Generator: tools/generate-reduce-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/microparams-init.h>
#include <xnnpack/reduce.h>
#include "rsum-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RSUM__NEONFP16ARITH_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RSumMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_rsum_ukernel__neonfp16arith_u8, xnn_init_f16_scale_fp16arith_params);
  }

  TEST(F16_RSUM__NEONFP16ARITH_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u8, xnn_init_f16_scale_fp16arith_params);
    }
  }

  TEST(F16_RSUM__NEONFP16ARITH_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u8, xnn_init_f16_scale_fp16arith_params);
    }
  }

  TEST(F16_RSUM__NEONFP16ARITH_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u8, xnn_init_f16_scale_fp16arith_params);
    }
  }

  TEST(F16_RSUM__NEONFP16ARITH_U8, scale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(9)
        .scale(scale)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u8, xnn_init_f16_scale_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RSUM__NEONFP16ARITH_U16_ACC2, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RSumMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_rsum_ukernel__neonfp16arith_u16_acc2, xnn_init_f16_scale_fp16arith_params);
  }

  TEST(F16_RSUM__NEONFP16ARITH_U16_ACC2, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u16_acc2, xnn_init_f16_scale_fp16arith_params);
    }
  }

  TEST(F16_RSUM__NEONFP16ARITH_U16_ACC2, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u16_acc2, xnn_init_f16_scale_fp16arith_params);
    }
  }

  TEST(F16_RSUM__NEONFP16ARITH_U16_ACC2, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u16_acc2, xnn_init_f16_scale_fp16arith_params);
    }
  }

  TEST(F16_RSUM__NEONFP16ARITH_U16_ACC2, scale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(17)
        .scale(scale)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u16_acc2, xnn_init_f16_scale_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RSUM__NEONFP16ARITH_U24_ACC3, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RSumMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_rsum_ukernel__neonfp16arith_u24_acc3, xnn_init_f16_scale_fp16arith_params);
  }

  TEST(F16_RSUM__NEONFP16ARITH_U24_ACC3, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u24_acc3, xnn_init_f16_scale_fp16arith_params);
    }
  }

  TEST(F16_RSUM__NEONFP16ARITH_U24_ACC3, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u24_acc3, xnn_init_f16_scale_fp16arith_params);
    }
  }

  TEST(F16_RSUM__NEONFP16ARITH_U24_ACC3, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u24_acc3, xnn_init_f16_scale_fp16arith_params);
    }
  }

  TEST(F16_RSUM__NEONFP16ARITH_U24_ACC3, scale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(25)
        .scale(scale)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u24_acc3, xnn_init_f16_scale_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RSUM__NEONFP16ARITH_U32_ACC2, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RSumMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_rsum_ukernel__neonfp16arith_u32_acc2, xnn_init_f16_scale_fp16arith_params);
  }

  TEST(F16_RSUM__NEONFP16ARITH_U32_ACC2, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u32_acc2, xnn_init_f16_scale_fp16arith_params);
    }
  }

  TEST(F16_RSUM__NEONFP16ARITH_U32_ACC2, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u32_acc2, xnn_init_f16_scale_fp16arith_params);
    }
  }

  TEST(F16_RSUM__NEONFP16ARITH_U32_ACC2, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u32_acc2, xnn_init_f16_scale_fp16arith_params);
    }
  }

  TEST(F16_RSUM__NEONFP16ARITH_U32_ACC2, scale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(33)
        .scale(scale)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u32_acc2, xnn_init_f16_scale_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RSUM__NEONFP16ARITH_U32_ACC4, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RSumMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_rsum_ukernel__neonfp16arith_u32_acc4, xnn_init_f16_scale_fp16arith_params);
  }

  TEST(F16_RSUM__NEONFP16ARITH_U32_ACC4, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u32_acc4, xnn_init_f16_scale_fp16arith_params);
    }
  }

  TEST(F16_RSUM__NEONFP16ARITH_U32_ACC4, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u32_acc4, xnn_init_f16_scale_fp16arith_params);
    }
  }

  TEST(F16_RSUM__NEONFP16ARITH_U32_ACC4, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u32_acc4, xnn_init_f16_scale_fp16arith_params);
    }
  }

  TEST(F16_RSUM__NEONFP16ARITH_U32_ACC4, scale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(33)
        .scale(scale)
        .Test(xnn_f16_rsum_ukernel__neonfp16arith_u32_acc4, xnn_init_f16_scale_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
