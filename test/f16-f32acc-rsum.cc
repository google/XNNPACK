// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-f32acc-rsum.yaml
//   Generator: tools/generate-reduce-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/microparams-init.h>
#include <xnnpack/reduce.h>
#include "rsum-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32ACC_RSUM__NEONFP16_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON_FP16;
    RSumMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u4, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u4, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u4, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u4, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U4, scale) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(5)
        .scale(scale)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u4, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32ACC_RSUM__NEONFP16_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    RSumMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u8, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u8, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u8, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u8, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U8, scale) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(9)
        .scale(scale)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u8, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32ACC_RSUM__NEONFP16_U16_ACC2, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    RSumMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u16_acc2, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U16_ACC2, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u16_acc2, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U16_ACC2, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u16_acc2, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U16_ACC2, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u16_acc2, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U16_ACC2, scale) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(17)
        .scale(scale)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u16_acc2, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32ACC_RSUM__NEONFP16_U24_ACC3, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16;
    RSumMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u24_acc3, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U24_ACC3, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u24_acc3, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U24_ACC3, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u24_acc3, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U24_ACC3, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u24_acc3, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U24_ACC3, scale) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(25)
        .scale(scale)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u24_acc3, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32ACC_RSUM__NEONFP16_U32_ACC2, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16;
    RSumMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u32_acc2, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U32_ACC2, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u32_acc2, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U32_ACC2, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u32_acc2, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U32_ACC2, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u32_acc2, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U32_ACC2, scale) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(33)
        .scale(scale)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u32_acc2, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F16_F32ACC_RSUM__NEONFP16_U32_ACC4, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16;
    RSumMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u32_acc4, xnn_init_f16_f32acc_scale_scalar_params);
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U32_ACC4, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u32_acc4, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U32_ACC4, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u32_acc4, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U32_ACC4, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u32_acc4, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }

  TEST(F16_F32ACC_RSUM__NEONFP16_U32_ACC4, scale) {
    TEST_REQUIRES_ARM_NEON_FP16;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(33)
        .scale(scale)
        .Test(xnn_f16_f32acc_rsum_ukernel__neonfp16_u32_acc4, xnn_init_f16_f32acc_scale_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32ACC_RSUM__F16C_U8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    RSumMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u8, xnn_init_f16_f32acc_scale_avx_params);
  }

  TEST(F16_F32ACC_RSUM__F16C_U8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u8, xnn_init_f16_f32acc_scale_avx_params);
    }
  }

  TEST(F16_F32ACC_RSUM__F16C_U8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u8, xnn_init_f16_f32acc_scale_avx_params);
    }
  }

  TEST(F16_F32ACC_RSUM__F16C_U8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u8, xnn_init_f16_f32acc_scale_avx_params);
    }
  }

  TEST(F16_F32ACC_RSUM__F16C_U8, scale) {
    TEST_REQUIRES_X86_F16C;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(9)
        .scale(scale)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u8, xnn_init_f16_f32acc_scale_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32ACC_RSUM__F16C_U16_ACC2, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    RSumMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u16_acc2, xnn_init_f16_f32acc_scale_avx_params);
  }

  TEST(F16_F32ACC_RSUM__F16C_U16_ACC2, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u16_acc2, xnn_init_f16_f32acc_scale_avx_params);
    }
  }

  TEST(F16_F32ACC_RSUM__F16C_U16_ACC2, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u16_acc2, xnn_init_f16_f32acc_scale_avx_params);
    }
  }

  TEST(F16_F32ACC_RSUM__F16C_U16_ACC2, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u16_acc2, xnn_init_f16_f32acc_scale_avx_params);
    }
  }

  TEST(F16_F32ACC_RSUM__F16C_U16_ACC2, scale) {
    TEST_REQUIRES_X86_F16C;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(17)
        .scale(scale)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u16_acc2, xnn_init_f16_f32acc_scale_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32ACC_RSUM__F16C_U24_ACC3, batch_eq_24) {
    TEST_REQUIRES_X86_F16C;
    RSumMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u24_acc3, xnn_init_f16_f32acc_scale_avx_params);
  }

  TEST(F16_F32ACC_RSUM__F16C_U24_ACC3, batch_div_24) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u24_acc3, xnn_init_f16_f32acc_scale_avx_params);
    }
  }

  TEST(F16_F32ACC_RSUM__F16C_U24_ACC3, batch_lt_24) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u24_acc3, xnn_init_f16_f32acc_scale_avx_params);
    }
  }

  TEST(F16_F32ACC_RSUM__F16C_U24_ACC3, batch_gt_24) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u24_acc3, xnn_init_f16_f32acc_scale_avx_params);
    }
  }

  TEST(F16_F32ACC_RSUM__F16C_U24_ACC3, scale) {
    TEST_REQUIRES_X86_F16C;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(25)
        .scale(scale)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u24_acc3, xnn_init_f16_f32acc_scale_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32ACC_RSUM__F16C_U32_ACC2, batch_eq_32) {
    TEST_REQUIRES_X86_F16C;
    RSumMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc2, xnn_init_f16_f32acc_scale_avx_params);
  }

  TEST(F16_F32ACC_RSUM__F16C_U32_ACC2, batch_div_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc2, xnn_init_f16_f32acc_scale_avx_params);
    }
  }

  TEST(F16_F32ACC_RSUM__F16C_U32_ACC2, batch_lt_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc2, xnn_init_f16_f32acc_scale_avx_params);
    }
  }

  TEST(F16_F32ACC_RSUM__F16C_U32_ACC2, batch_gt_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc2, xnn_init_f16_f32acc_scale_avx_params);
    }
  }

  TEST(F16_F32ACC_RSUM__F16C_U32_ACC2, scale) {
    TEST_REQUIRES_X86_F16C;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(33)
        .scale(scale)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc2, xnn_init_f16_f32acc_scale_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_F32ACC_RSUM__F16C_U32_ACC4, batch_eq_32) {
    TEST_REQUIRES_X86_F16C;
    RSumMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc4, xnn_init_f16_f32acc_scale_avx_params);
  }

  TEST(F16_F32ACC_RSUM__F16C_U32_ACC4, batch_div_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc4, xnn_init_f16_f32acc_scale_avx_params);
    }
  }

  TEST(F16_F32ACC_RSUM__F16C_U32_ACC4, batch_lt_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc4, xnn_init_f16_f32acc_scale_avx_params);
    }
  }

  TEST(F16_F32ACC_RSUM__F16C_U32_ACC4, batch_gt_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc4, xnn_init_f16_f32acc_scale_avx_params);
    }
  }

  TEST(F16_F32ACC_RSUM__F16C_U32_ACC4, scale) {
    TEST_REQUIRES_X86_F16C;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(33)
        .scale(scale)
        .Test(xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc4, xnn_init_f16_f32acc_scale_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
