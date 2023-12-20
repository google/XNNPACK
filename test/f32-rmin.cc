// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-rmin.yaml
//   Generator: tools/generate-reduce-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/microparams-init.h>
#include <xnnpack/reduce.h>
#include "reduce-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RMIN__NEON_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_rmin_ukernel__neon_u4, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__NEON_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__neon_u4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__NEON_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__neon_u4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__NEON_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__neon_u4, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RMIN__NEON_U8_ACC2, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_rmin_ukernel__neon_u8_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__NEON_U8_ACC2, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__neon_u8_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__NEON_U8_ACC2, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__neon_u8_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__NEON_U8_ACC2, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__neon_u8_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RMIN__NEON_U12_ACC3, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_rmin_ukernel__neon_u12_acc3, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__NEON_U12_ACC3, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__neon_u12_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__NEON_U12_ACC3, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__neon_u12_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__NEON_U12_ACC3, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__neon_u12_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RMIN__NEON_U16_ACC2, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rmin_ukernel__neon_u16_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__NEON_U16_ACC2, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__neon_u16_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__NEON_U16_ACC2, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__neon_u16_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__NEON_U16_ACC2, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__neon_u16_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RMIN__NEON_U16_ACC4, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rmin_ukernel__neon_u16_acc4, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__NEON_U16_ACC4, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__neon_u16_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__NEON_U16_ACC4, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__neon_u16_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__NEON_U16_ACC4, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__neon_u16_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMIN__SSE_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE;
    ReduceMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_rmin_ukernel__sse_u4, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__SSE_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__sse_u4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__SSE_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__sse_u4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__SSE_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__sse_u4, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMIN__SSE_U8_ACC2, batch_eq_8) {
    TEST_REQUIRES_X86_SSE;
    ReduceMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_rmin_ukernel__sse_u8_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__SSE_U8_ACC2, batch_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__sse_u8_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__SSE_U8_ACC2, batch_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__sse_u8_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__SSE_U8_ACC2, batch_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__sse_u8_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMIN__SSE_U12_ACC3, batch_eq_12) {
    TEST_REQUIRES_X86_SSE;
    ReduceMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_rmin_ukernel__sse_u12_acc3, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__SSE_U12_ACC3, batch_div_12) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__sse_u12_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__SSE_U12_ACC3, batch_lt_12) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__sse_u12_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__SSE_U12_ACC3, batch_gt_12) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__sse_u12_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMIN__SSE_U16_ACC2, batch_eq_16) {
    TEST_REQUIRES_X86_SSE;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rmin_ukernel__sse_u16_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__SSE_U16_ACC2, batch_div_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__sse_u16_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__SSE_U16_ACC2, batch_lt_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__sse_u16_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__SSE_U16_ACC2, batch_gt_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__sse_u16_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMIN__SSE_U16_ACC4, batch_eq_16) {
    TEST_REQUIRES_X86_SSE;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rmin_ukernel__sse_u16_acc4, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__SSE_U16_ACC4, batch_div_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__sse_u16_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__SSE_U16_ACC4, batch_lt_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__sse_u16_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__SSE_U16_ACC4, batch_gt_16) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__sse_u16_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMIN__AVX_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    ReduceMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_rmin_ukernel__avx_u8, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
  }

  TEST(F32_RMIN__AVX_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx_u8, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
    }
  }

  TEST(F32_RMIN__AVX_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx_u8, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
    }
  }

  TEST(F32_RMIN__AVX_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx_u8, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMIN__AVX_U16_ACC2, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rmin_ukernel__avx_u16_acc2, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
  }

  TEST(F32_RMIN__AVX_U16_ACC2, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx_u16_acc2, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
    }
  }

  TEST(F32_RMIN__AVX_U16_ACC2, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx_u16_acc2, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
    }
  }

  TEST(F32_RMIN__AVX_U16_ACC2, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx_u16_acc2, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMIN__AVX_U24_ACC3, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    ReduceMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_rmin_ukernel__avx_u24_acc3, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
  }

  TEST(F32_RMIN__AVX_U24_ACC3, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx_u24_acc3, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
    }
  }

  TEST(F32_RMIN__AVX_U24_ACC3, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx_u24_acc3, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
    }
  }

  TEST(F32_RMIN__AVX_U24_ACC3, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx_u24_acc3, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMIN__AVX_U32_ACC2, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    ReduceMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_rmin_ukernel__avx_u32_acc2, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
  }

  TEST(F32_RMIN__AVX_U32_ACC2, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx_u32_acc2, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
    }
  }

  TEST(F32_RMIN__AVX_U32_ACC2, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx_u32_acc2, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
    }
  }

  TEST(F32_RMIN__AVX_U32_ACC2, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx_u32_acc2, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMIN__AVX_U32_ACC4, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    ReduceMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_rmin_ukernel__avx_u32_acc4, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
  }

  TEST(F32_RMIN__AVX_U32_ACC4, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx_u32_acc4, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
    }
  }

  TEST(F32_RMIN__AVX_U32_ACC4, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx_u32_acc4, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
    }
  }

  TEST(F32_RMIN__AVX_U32_ACC4, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx_u32_acc4, ReduceMicrokernelTester::OpType::Min, xnn_init_f32_default_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMIN__AVX512F_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rmin_ukernel__avx512f_u16, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__AVX512F_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx512f_u16, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__AVX512F_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx512f_u16, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__AVX512F_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx512f_u16, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMIN__AVX512F_U32_ACC2, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    ReduceMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_rmin_ukernel__avx512f_u32_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__AVX512F_U32_ACC2, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx512f_u32_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__AVX512F_U32_ACC2, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx512f_u32_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__AVX512F_U32_ACC2, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx512f_u32_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMIN__AVX512F_U48_ACC3, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    ReduceMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_rmin_ukernel__avx512f_u48_acc3, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__AVX512F_U48_ACC3, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx512f_u48_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__AVX512F_U48_ACC3, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx512f_u48_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__AVX512F_U48_ACC3, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx512f_u48_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMIN__AVX512F_U64_ACC2, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    ReduceMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_rmin_ukernel__avx512f_u64_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__AVX512F_U64_ACC2, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx512f_u64_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__AVX512F_U64_ACC2, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx512f_u64_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__AVX512F_U64_ACC2, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx512f_u64_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RMIN__AVX512F_U64_ACC4, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    ReduceMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_rmin_ukernel__avx512f_u64_acc4, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__AVX512F_U64_ACC4, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx512f_u64_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__AVX512F_U64_ACC4, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx512f_u64_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__AVX512F_U64_ACC4, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__avx512f_u64_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


TEST(F32_RMIN__SCALAR_U1, batch_eq_1) {
  ReduceMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_rmin_ukernel__scalar_u1, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RMIN__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rmin_ukernel__scalar_u1, ReduceMicrokernelTester::OpType::Min);
  }
}


TEST(F32_RMIN__SCALAR_U2_ACC2, batch_eq_2) {
  ReduceMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_rmin_ukernel__scalar_u2_acc2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RMIN__SCALAR_U2_ACC2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rmin_ukernel__scalar_u2_acc2, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F32_RMIN__SCALAR_U2_ACC2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rmin_ukernel__scalar_u2_acc2, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F32_RMIN__SCALAR_U2_ACC2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rmin_ukernel__scalar_u2_acc2, ReduceMicrokernelTester::OpType::Min);
  }
}


TEST(F32_RMIN__SCALAR_U3_ACC3, batch_eq_3) {
  ReduceMicrokernelTester()
    .batch_size(3)
    .Test(xnn_f32_rmin_ukernel__scalar_u3_acc3, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RMIN__SCALAR_U3_ACC3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rmin_ukernel__scalar_u3_acc3, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F32_RMIN__SCALAR_U3_ACC3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rmin_ukernel__scalar_u3_acc3, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F32_RMIN__SCALAR_U3_ACC3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rmin_ukernel__scalar_u3_acc3, ReduceMicrokernelTester::OpType::Min);
  }
}


TEST(F32_RMIN__SCALAR_U4_ACC2, batch_eq_4) {
  ReduceMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_rmin_ukernel__scalar_u4_acc2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RMIN__SCALAR_U4_ACC2, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rmin_ukernel__scalar_u4_acc2, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F32_RMIN__SCALAR_U4_ACC2, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rmin_ukernel__scalar_u4_acc2, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F32_RMIN__SCALAR_U4_ACC2, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rmin_ukernel__scalar_u4_acc2, ReduceMicrokernelTester::OpType::Min);
  }
}


TEST(F32_RMIN__SCALAR_U4_ACC4, batch_eq_4) {
  ReduceMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_rmin_ukernel__scalar_u4_acc4, ReduceMicrokernelTester::OpType::Min);
}

TEST(F32_RMIN__SCALAR_U4_ACC4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rmin_ukernel__scalar_u4_acc4, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F32_RMIN__SCALAR_U4_ACC4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rmin_ukernel__scalar_u4_acc4, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F32_RMIN__SCALAR_U4_ACC4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rmin_ukernel__scalar_u4_acc4, ReduceMicrokernelTester::OpType::Min);
  }
}


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RMIN__WASM_U1, batch_eq_1) {
    ReduceMicrokernelTester()
      .batch_size(1)
      .Test(xnn_f32_rmin_ukernel__wasm_u1, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__WASM_U1, batch_gt_1) {
    for (size_t batch_size = 2; batch_size < 10; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasm_u1, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RMIN__WASM_U2_ACC2, batch_eq_2) {
    ReduceMicrokernelTester()
      .batch_size(2)
      .Test(xnn_f32_rmin_ukernel__wasm_u2_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__WASM_U2_ACC2, batch_div_2) {
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasm_u2_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASM_U2_ACC2, batch_lt_2) {
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasm_u2_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASM_U2_ACC2, batch_gt_2) {
    for (size_t batch_size = 3; batch_size < 4; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasm_u2_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RMIN__WASM_U3_ACC3, batch_eq_3) {
    ReduceMicrokernelTester()
      .batch_size(3)
      .Test(xnn_f32_rmin_ukernel__wasm_u3_acc3, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__WASM_U3_ACC3, batch_div_3) {
    for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasm_u3_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASM_U3_ACC3, batch_lt_3) {
    for (size_t batch_size = 1; batch_size < 3; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasm_u3_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASM_U3_ACC3, batch_gt_3) {
    for (size_t batch_size = 4; batch_size < 6; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasm_u3_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RMIN__WASM_U4_ACC2, batch_eq_4) {
    ReduceMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_rmin_ukernel__wasm_u4_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__WASM_U4_ACC2, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasm_u4_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASM_U4_ACC2, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasm_u4_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASM_U4_ACC2, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasm_u4_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RMIN__WASM_U4_ACC4, batch_eq_4) {
    ReduceMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_rmin_ukernel__wasm_u4_acc4, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__WASM_U4_ACC4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasm_u4_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASM_U4_ACC4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasm_u4_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASM_U4_ACC4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasm_u4_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RMIN__WASMSIMD_MINMAX_U4, batch_eq_4) {
    ReduceMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u4, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__WASMSIMD_MINMAX_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_MINMAX_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_MINMAX_U4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u4, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RMIN__WASMSIMD_MINMAX_U8_ACC2, batch_eq_8) {
    ReduceMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u8_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__WASMSIMD_MINMAX_U8_ACC2, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u8_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_MINMAX_U8_ACC2, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u8_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_MINMAX_U8_ACC2, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u8_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RMIN__WASMSIMD_MINMAX_U12_ACC3, batch_eq_12) {
    ReduceMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u12_acc3, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__WASMSIMD_MINMAX_U12_ACC3, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u12_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_MINMAX_U12_ACC3, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u12_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_MINMAX_U12_ACC3, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u12_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RMIN__WASMSIMD_MINMAX_U16_ACC2, batch_eq_16) {
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u16_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__WASMSIMD_MINMAX_U16_ACC2, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u16_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_MINMAX_U16_ACC2, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u16_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_MINMAX_U16_ACC2, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u16_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RMIN__WASMSIMD_MINMAX_U16_ACC4, batch_eq_16) {
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u16_acc4, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__WASMSIMD_MINMAX_U16_ACC4, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u16_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_MINMAX_U16_ACC4, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u16_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_MINMAX_U16_ACC4, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_minmax_u16_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RMIN__WASMSIMD_PMINMAX_U4, batch_eq_4) {
    ReduceMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u4, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__WASMSIMD_PMINMAX_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_PMINMAX_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_PMINMAX_U4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u4, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RMIN__WASMSIMD_PMINMAX_U8_ACC2, batch_eq_8) {
    ReduceMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u8_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__WASMSIMD_PMINMAX_U8_ACC2, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u8_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_PMINMAX_U8_ACC2, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u8_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_PMINMAX_U8_ACC2, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u8_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RMIN__WASMSIMD_PMINMAX_U12_ACC3, batch_eq_12) {
    ReduceMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u12_acc3, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__WASMSIMD_PMINMAX_U12_ACC3, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u12_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_PMINMAX_U12_ACC3, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u12_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_PMINMAX_U12_ACC3, batch_gt_12) {
    for (size_t batch_size = 13; batch_size < 24; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u12_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RMIN__WASMSIMD_PMINMAX_U16_ACC2, batch_eq_16) {
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u16_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__WASMSIMD_PMINMAX_U16_ACC2, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u16_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_PMINMAX_U16_ACC2, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u16_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_PMINMAX_U16_ACC2, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u16_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RMIN__WASMSIMD_PMINMAX_U16_ACC4, batch_eq_16) {
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u16_acc4, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__WASMSIMD_PMINMAX_U16_ACC4, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u16_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_PMINMAX_U16_ACC4, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u16_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__WASMSIMD_PMINMAX_U16_ACC4, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__wasmsimd_pminmax_u16_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_RMIN__RVV_U1V, batch_eq_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    ReduceMicrokernelTester()
      .batch_size(1 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_rmin_ukernel__rvv_u1v, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__RVV_U1V, batch_gt_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1;
                batch_size < 10 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 2) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__rvv_u1v, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_RMIN__RVV_U2V, batch_eq_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    ReduceMicrokernelTester()
      .batch_size(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_rmin_ukernel__rvv_u2v, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__RVV_U2V, batch_div_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 20 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 2 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__rvv_u2v, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__RVV_U2V, batch_lt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size < 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__rvv_u2v, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__RVV_U2V, batch_gt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 2 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1;
                batch_size < 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 4) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__rvv_u2v, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_RMIN__RVV_U4V, batch_eq_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    ReduceMicrokernelTester()
      .batch_size(4 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_rmin_ukernel__rvv_u4v, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__RVV_U4V, batch_div_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 40 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 4 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__rvv_u4v, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__RVV_U4V, batch_lt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size < 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__rvv_u4v, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__RVV_U4V, batch_gt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 4 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1;
                batch_size < 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 8) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__rvv_u4v, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_RMIN__RVV_U8V, batch_eq_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    ReduceMicrokernelTester()
      .batch_size(8 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_rmin_ukernel__rvv_u8v, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F32_RMIN__RVV_U8V, batch_div_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 16 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 80 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 8 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__rvv_u8v, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__RVV_U8V, batch_lt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size < 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__rvv_u8v, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F32_RMIN__RVV_U8V, batch_gt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 8 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1;
                batch_size < 16 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_rmin_ukernel__rvv_u8v, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
