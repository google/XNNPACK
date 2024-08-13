// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-rsum.yaml
//   Generator: tools/generate-reduce-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/reduce.h"
#include "rsum-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RSUM__NEON_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_rsum_ukernel__neon_u4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__NEON_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__neon_u4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__neon_u4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__neon_u4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U4, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(5)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__neon_u4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U4, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(512)
      .Test(xnn_f32_rsum_ukernel__neon_u4, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RSUM__NEON_U8_ACC2, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_rsum_ukernel__neon_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__NEON_U8_ACC2, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__neon_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U8_ACC2, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__neon_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U8_ACC2, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__neon_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U8_ACC2, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(9)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__neon_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U8_ACC2, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(1024)
      .Test(xnn_f32_rsum_ukernel__neon_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RSUM__NEON_U12_ACC3, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_rsum_ukernel__neon_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__NEON_U12_ACC3, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__neon_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U12_ACC3, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__neon_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U12_ACC3, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__neon_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U12_ACC3, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(13)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__neon_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U12_ACC3, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(1536)
      .Test(xnn_f32_rsum_ukernel__neon_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RSUM__NEON_U16_ACC2, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rsum_ukernel__neon_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__NEON_U16_ACC2, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__neon_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U16_ACC2, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__neon_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U16_ACC2, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__neon_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U16_ACC2, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(17)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__neon_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U16_ACC2, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(2048)
      .Test(xnn_f32_rsum_ukernel__neon_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RSUM__NEON_U16_ACC4, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rsum_ukernel__neon_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__NEON_U16_ACC4, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__neon_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U16_ACC4, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__neon_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U16_ACC4, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__neon_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U16_ACC4, scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(17)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__neon_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__NEON_U16_ACC4, overflow_accumulator) {
    TEST_REQUIRES_ARM_NEON;
    RSumMicrokernelTester()
      .batch_size(2048)
      .Test(xnn_f32_rsum_ukernel__neon_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RSUM__SSE_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE;
    RSumMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_rsum_ukernel__sse_u4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__SSE_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__sse_u4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__sse_u4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__sse_u4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U4, scale) {
    TEST_REQUIRES_X86_SSE;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(5)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__sse_u4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U4, overflow_accumulator) {
    TEST_REQUIRES_X86_SSE;
    RSumMicrokernelTester()
      .batch_size(512)
      .Test(xnn_f32_rsum_ukernel__sse_u4, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RSUM__SSE_U8_ACC2, batch_eq_8) {
    TEST_REQUIRES_X86_SSE;
    RSumMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_rsum_ukernel__sse_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__SSE_U8_ACC2, batch_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__sse_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U8_ACC2, batch_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__sse_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U8_ACC2, batch_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__sse_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U8_ACC2, scale) {
    TEST_REQUIRES_X86_SSE;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(9)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__sse_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U8_ACC2, overflow_accumulator) {
    TEST_REQUIRES_X86_SSE;
    RSumMicrokernelTester()
      .batch_size(1024)
      .Test(xnn_f32_rsum_ukernel__sse_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RSUM__SSE_U12_ACC3, batch_eq_12) {
    TEST_REQUIRES_X86_SSE;
    RSumMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_rsum_ukernel__sse_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__SSE_U12_ACC3, batch_div_12) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__sse_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U12_ACC3, batch_lt_12) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__sse_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U12_ACC3, batch_gt_12) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__sse_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U12_ACC3, scale) {
    TEST_REQUIRES_X86_SSE;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(13)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__sse_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U12_ACC3, overflow_accumulator) {
    TEST_REQUIRES_X86_SSE;
    RSumMicrokernelTester()
      .batch_size(1536)
      .Test(xnn_f32_rsum_ukernel__sse_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RSUM__SSE_U16_ACC2, batch_eq_16) {
    TEST_REQUIRES_X86_SSE;
    RSumMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rsum_ukernel__sse_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__SSE_U16_ACC2, batch_div_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__sse_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U16_ACC2, batch_lt_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__sse_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U16_ACC2, batch_gt_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__sse_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U16_ACC2, scale) {
    TEST_REQUIRES_X86_SSE;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(17)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__sse_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U16_ACC2, overflow_accumulator) {
    TEST_REQUIRES_X86_SSE;
    RSumMicrokernelTester()
      .batch_size(2048)
      .Test(xnn_f32_rsum_ukernel__sse_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RSUM__SSE_U16_ACC4, batch_eq_16) {
    TEST_REQUIRES_X86_SSE;
    RSumMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rsum_ukernel__sse_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__SSE_U16_ACC4, batch_div_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__sse_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U16_ACC4, batch_lt_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__sse_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U16_ACC4, batch_gt_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__sse_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U16_ACC4, scale) {
    TEST_REQUIRES_X86_SSE;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(17)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__sse_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__SSE_U16_ACC4, overflow_accumulator) {
    TEST_REQUIRES_X86_SSE;
    RSumMicrokernelTester()
      .batch_size(2048)
      .Test(xnn_f32_rsum_ukernel__sse_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RSUM__AVX_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    RSumMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_rsum_ukernel__avx_u8, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__AVX_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx_u8, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx_u8, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx_u8, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U8, scale) {
    TEST_REQUIRES_X86_AVX;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(9)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__avx_u8, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U8, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX;
    RSumMicrokernelTester()
      .batch_size(1024)
      .Test(xnn_f32_rsum_ukernel__avx_u8, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RSUM__AVX_U16_ACC2, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    RSumMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rsum_ukernel__avx_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__AVX_U16_ACC2, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U16_ACC2, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U16_ACC2, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U16_ACC2, scale) {
    TEST_REQUIRES_X86_AVX;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(17)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__avx_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U16_ACC2, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX;
    RSumMicrokernelTester()
      .batch_size(2048)
      .Test(xnn_f32_rsum_ukernel__avx_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RSUM__AVX_U24_ACC3, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    RSumMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_rsum_ukernel__avx_u24_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__AVX_U24_ACC3, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx_u24_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U24_ACC3, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx_u24_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U24_ACC3, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx_u24_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U24_ACC3, scale) {
    TEST_REQUIRES_X86_AVX;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(25)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__avx_u24_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U24_ACC3, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX;
    RSumMicrokernelTester()
      .batch_size(3072)
      .Test(xnn_f32_rsum_ukernel__avx_u24_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RSUM__AVX_U32_ACC2, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    RSumMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_rsum_ukernel__avx_u32_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__AVX_U32_ACC2, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx_u32_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U32_ACC2, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx_u32_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U32_ACC2, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx_u32_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U32_ACC2, scale) {
    TEST_REQUIRES_X86_AVX;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(33)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__avx_u32_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U32_ACC2, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX;
    RSumMicrokernelTester()
      .batch_size(4096)
      .Test(xnn_f32_rsum_ukernel__avx_u32_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RSUM__AVX_U32_ACC4, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    RSumMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_rsum_ukernel__avx_u32_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__AVX_U32_ACC4, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx_u32_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U32_ACC4, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx_u32_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U32_ACC4, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx_u32_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U32_ACC4, scale) {
    TEST_REQUIRES_X86_AVX;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(33)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__avx_u32_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX_U32_ACC4, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX;
    RSumMicrokernelTester()
      .batch_size(4096)
      .Test(xnn_f32_rsum_ukernel__avx_u32_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RSUM__AVX512F_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    RSumMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rsum_ukernel__avx512f_u16, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__AVX512F_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx512f_u16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx512f_u16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx512f_u16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U16, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(17)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__avx512f_u16, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U16, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX512F;
    RSumMicrokernelTester()
      .batch_size(2048)
      .Test(xnn_f32_rsum_ukernel__avx512f_u16, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RSUM__AVX512F_U32_ACC2, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    RSumMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_rsum_ukernel__avx512f_u32_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__AVX512F_U32_ACC2, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx512f_u32_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U32_ACC2, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx512f_u32_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U32_ACC2, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx512f_u32_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U32_ACC2, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(33)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__avx512f_u32_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U32_ACC2, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX512F;
    RSumMicrokernelTester()
      .batch_size(4096)
      .Test(xnn_f32_rsum_ukernel__avx512f_u32_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RSUM__AVX512F_U48_ACC3, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    RSumMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_rsum_ukernel__avx512f_u48_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__AVX512F_U48_ACC3, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx512f_u48_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U48_ACC3, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx512f_u48_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U48_ACC3, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx512f_u48_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U48_ACC3, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(49)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__avx512f_u48_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U48_ACC3, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX512F;
    RSumMicrokernelTester()
      .batch_size(6144)
      .Test(xnn_f32_rsum_ukernel__avx512f_u48_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RSUM__AVX512F_U64_ACC2, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    RSumMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_rsum_ukernel__avx512f_u64_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__AVX512F_U64_ACC2, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx512f_u64_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U64_ACC2, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx512f_u64_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U64_ACC2, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx512f_u64_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U64_ACC2, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(65)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__avx512f_u64_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U64_ACC2, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX512F;
    RSumMicrokernelTester()
      .batch_size(8192)
      .Test(xnn_f32_rsum_ukernel__avx512f_u64_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RSUM__AVX512F_U64_ACC4, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    RSumMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_rsum_ukernel__avx512f_u64_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__AVX512F_U64_ACC4, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx512f_u64_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U64_ACC4, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx512f_u64_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U64_ACC4, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__avx512f_u64_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U64_ACC4, scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(65)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__avx512f_u64_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__AVX512F_U64_ACC4, overflow_accumulator) {
    TEST_REQUIRES_X86_AVX512F;
    RSumMicrokernelTester()
      .batch_size(8192)
      .Test(xnn_f32_rsum_ukernel__avx512f_u64_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_RSUM__HVX_U32, batch_eq_32) {
    TEST_REQUIRES_HVX;
    RSumMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_rsum_ukernel__hvx_u32, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__HVX_U32, batch_div_32) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__hvx_u32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U32, batch_lt_32) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__hvx_u32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U32, batch_gt_32) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__hvx_u32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U32, scale) {
    TEST_REQUIRES_HVX;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(33)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__hvx_u32, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U32, overflow_accumulator) {
    TEST_REQUIRES_HVX;
    RSumMicrokernelTester()
      .batch_size(4096)
      .Test(xnn_f32_rsum_ukernel__hvx_u32, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_RSUM__HVX_U64_ACC2, batch_eq_64) {
    TEST_REQUIRES_HVX;
    RSumMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_rsum_ukernel__hvx_u64_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__HVX_U64_ACC2, batch_div_64) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__hvx_u64_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U64_ACC2, batch_lt_64) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__hvx_u64_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U64_ACC2, batch_gt_64) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__hvx_u64_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U64_ACC2, scale) {
    TEST_REQUIRES_HVX;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(65)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__hvx_u64_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U64_ACC2, overflow_accumulator) {
    TEST_REQUIRES_HVX;
    RSumMicrokernelTester()
      .batch_size(8192)
      .Test(xnn_f32_rsum_ukernel__hvx_u64_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_RSUM__HVX_U96_ACC3, batch_eq_96) {
    TEST_REQUIRES_HVX;
    RSumMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_rsum_ukernel__hvx_u96_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__HVX_U96_ACC3, batch_div_96) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__hvx_u96_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U96_ACC3, batch_lt_96) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__hvx_u96_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U96_ACC3, batch_gt_96) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 97; batch_size < 192; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__hvx_u96_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U96_ACC3, scale) {
    TEST_REQUIRES_HVX;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(97)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__hvx_u96_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U96_ACC3, overflow_accumulator) {
    TEST_REQUIRES_HVX;
    RSumMicrokernelTester()
      .batch_size(12288)
      .Test(xnn_f32_rsum_ukernel__hvx_u96_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_RSUM__HVX_U128_ACC2, batch_eq_128) {
    TEST_REQUIRES_HVX;
    RSumMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_rsum_ukernel__hvx_u128_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__HVX_U128_ACC2, batch_div_128) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__hvx_u128_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U128_ACC2, batch_lt_128) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__hvx_u128_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U128_ACC2, batch_gt_128) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 129; batch_size < 256; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__hvx_u128_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U128_ACC2, scale) {
    TEST_REQUIRES_HVX;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(129)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__hvx_u128_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U128_ACC2, overflow_accumulator) {
    TEST_REQUIRES_HVX;
    RSumMicrokernelTester()
      .batch_size(16384)
      .Test(xnn_f32_rsum_ukernel__hvx_u128_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_RSUM__HVX_U128_ACC4, batch_eq_128) {
    TEST_REQUIRES_HVX;
    RSumMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_rsum_ukernel__hvx_u128_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__HVX_U128_ACC4, batch_div_128) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__hvx_u128_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U128_ACC4, batch_lt_128) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__hvx_u128_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U128_ACC4, batch_gt_128) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 129; batch_size < 256; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__hvx_u128_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U128_ACC4, scale) {
    TEST_REQUIRES_HVX;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(129)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__hvx_u128_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__HVX_U128_ACC4, overflow_accumulator) {
    TEST_REQUIRES_HVX;
    RSumMicrokernelTester()
      .batch_size(16384)
      .Test(xnn_f32_rsum_ukernel__hvx_u128_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RSUM__WASMSIMD_U4, batch_eq_4) {
    RSumMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_rsum_ukernel__wasmsimd_u4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__WASMSIMD_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U4, scale) {
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(5)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U4, overflow_accumulator) {
    RSumMicrokernelTester()
      .batch_size(512)
      .Test(xnn_f32_rsum_ukernel__wasmsimd_u4, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RSUM__WASMSIMD_U8_ACC2, batch_eq_8) {
    RSumMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_rsum_ukernel__wasmsimd_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__WASMSIMD_U8_ACC2, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U8_ACC2, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U8_ACC2, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U8_ACC2, scale) {
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(9)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U8_ACC2, overflow_accumulator) {
    RSumMicrokernelTester()
      .batch_size(1024)
      .Test(xnn_f32_rsum_ukernel__wasmsimd_u8_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RSUM__WASMSIMD_U12_ACC3, batch_eq_12) {
    RSumMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_rsum_ukernel__wasmsimd_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__WASMSIMD_U12_ACC3, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U12_ACC3, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U12_ACC3, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U12_ACC3, scale) {
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(13)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U12_ACC3, overflow_accumulator) {
    RSumMicrokernelTester()
      .batch_size(1536)
      .Test(xnn_f32_rsum_ukernel__wasmsimd_u12_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RSUM__WASMSIMD_U16_ACC2, batch_eq_16) {
    RSumMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rsum_ukernel__wasmsimd_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__WASMSIMD_U16_ACC2, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U16_ACC2, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U16_ACC2, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U16_ACC2, scale) {
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(17)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U16_ACC2, overflow_accumulator) {
    RSumMicrokernelTester()
      .batch_size(2048)
      .Test(xnn_f32_rsum_ukernel__wasmsimd_u16_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RSUM__WASMSIMD_U16_ACC4, batch_eq_16) {
    RSumMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rsum_ukernel__wasmsimd_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__WASMSIMD_U16_ACC4, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U16_ACC4, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U16_ACC4, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U16_ACC4, scale) {
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(17)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__wasmsimd_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__WASMSIMD_U16_ACC4, overflow_accumulator) {
    RSumMicrokernelTester()
      .batch_size(2048)
      .Test(xnn_f32_rsum_ukernel__wasmsimd_u16_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_RSUM__RVV_U1V, batch_eq_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    RSumMicrokernelTester()
      .batch_size(1 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_rsum_ukernel__rvv_u1v, xnn_init_f32_scaleminmax_scalar_params);
  }

  TEST(F32_RSUM__RVV_U1V, batch_div_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 10 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__rvv_u1v, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__RVV_U1V, batch_lt_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size < 1 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size++) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__rvv_u1v, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__RVV_U1V, batch_gt_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1;
                batch_size < 10 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 2) {
      RSumMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rsum_ukernel__rvv_u1v, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__RVV_U1V, scale) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
      RSumMicrokernelTester()
        .batch_size(1 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1)
        .scale(scale)
        .Test(xnn_f32_rsum_ukernel__rvv_u1v, xnn_init_f32_scaleminmax_scalar_params);
    }
  }

  TEST(F32_RSUM__RVV_U1V, overflow_accumulator) {
    TEST_REQUIRES_RISCV_VECTOR;
    RSumMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_rsum_ukernel__rvv_u1v, xnn_init_f32_scaleminmax_scalar_params);
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


TEST(F32_RSUM__SCALAR_U1, batch_eq_1) {
  RSumMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_rsum_ukernel__scalar_u1, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_RSUM__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rsum_ukernel__scalar_u1, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U1, scale) {
  for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
    RSumMicrokernelTester()
      .batch_size(2)
      .scale(scale)
      .Test(xnn_f32_rsum_ukernel__scalar_u1, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U1, overflow_accumulator) {
  RSumMicrokernelTester()
    .batch_size(128)
    .Test(xnn_f32_rsum_ukernel__scalar_u1, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_RSUM__SCALAR_U2_ACC2, batch_eq_2) {
  RSumMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_rsum_ukernel__scalar_u2_acc2, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_RSUM__SCALAR_U2_ACC2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rsum_ukernel__scalar_u2_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U2_ACC2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rsum_ukernel__scalar_u2_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U2_ACC2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rsum_ukernel__scalar_u2_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U2_ACC2, scale) {
  for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
    RSumMicrokernelTester()
      .batch_size(3)
      .scale(scale)
      .Test(xnn_f32_rsum_ukernel__scalar_u2_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U2_ACC2, overflow_accumulator) {
  RSumMicrokernelTester()
    .batch_size(256)
    .Test(xnn_f32_rsum_ukernel__scalar_u2_acc2, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_RSUM__SCALAR_U3_ACC3, batch_eq_3) {
  RSumMicrokernelTester()
    .batch_size(3)
    .Test(xnn_f32_rsum_ukernel__scalar_u3_acc3, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_RSUM__SCALAR_U3_ACC3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rsum_ukernel__scalar_u3_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U3_ACC3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rsum_ukernel__scalar_u3_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U3_ACC3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rsum_ukernel__scalar_u3_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U3_ACC3, scale) {
  for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
    RSumMicrokernelTester()
      .batch_size(4)
      .scale(scale)
      .Test(xnn_f32_rsum_ukernel__scalar_u3_acc3, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U3_ACC3, overflow_accumulator) {
  RSumMicrokernelTester()
    .batch_size(384)
    .Test(xnn_f32_rsum_ukernel__scalar_u3_acc3, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_RSUM__SCALAR_U4_ACC2, batch_eq_4) {
  RSumMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_rsum_ukernel__scalar_u4_acc2, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_RSUM__SCALAR_U4_ACC2, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rsum_ukernel__scalar_u4_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U4_ACC2, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rsum_ukernel__scalar_u4_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U4_ACC2, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rsum_ukernel__scalar_u4_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U4_ACC2, scale) {
  for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
    RSumMicrokernelTester()
      .batch_size(5)
      .scale(scale)
      .Test(xnn_f32_rsum_ukernel__scalar_u4_acc2, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U4_ACC2, overflow_accumulator) {
  RSumMicrokernelTester()
    .batch_size(512)
    .Test(xnn_f32_rsum_ukernel__scalar_u4_acc2, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_RSUM__SCALAR_U4_ACC4, batch_eq_4) {
  RSumMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_rsum_ukernel__scalar_u4_acc4, xnn_init_f32_scaleminmax_scalar_params);
}

TEST(F32_RSUM__SCALAR_U4_ACC4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rsum_ukernel__scalar_u4_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U4_ACC4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rsum_ukernel__scalar_u4_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U4_ACC4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    RSumMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rsum_ukernel__scalar_u4_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U4_ACC4, scale) {
  for (float scale = 0.3f; scale < 5.0f; scale *= 3.0f) {
    RSumMicrokernelTester()
      .batch_size(5)
      .scale(scale)
      .Test(xnn_f32_rsum_ukernel__scalar_u4_acc4, xnn_init_f32_scaleminmax_scalar_params);
  }
}

TEST(F32_RSUM__SCALAR_U4_ACC4, overflow_accumulator) {
  RSumMicrokernelTester()
    .batch_size(512)
    .Test(xnn_f32_rsum_ukernel__scalar_u4_acc4, xnn_init_f32_scaleminmax_scalar_params);
}