// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-velu.yaml
//   Generator: tools/generate-vunary-test.py


#include <vector>

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/vunary.h>

#include "vunary-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_fp16arith_rr1_p3_params);
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_fp16arith_rr1_p3_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_fp16arith_rr1_p3_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_fp16arith_rr1_p3_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_fp16arith_rr1_p3_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, prescale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_fp16arith_rr1_p3_params);
      }
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, alpha) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_fp16arith_rr1_p3_params);
      }
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, beta) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_fp16arith_rr1_p3_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_fp16arith_rr1_p3_params);
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_fp16arith_rr1_p3_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_fp16arith_rr1_p3_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_fp16arith_rr1_p3_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_fp16arith_rr1_p3_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, prescale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_fp16arith_rr1_p3_params);
      }
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, alpha) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_fp16arith_rr1_p3_params);
      }
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, beta) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_fp16arith_rr1_p3_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VELU__AVX2_RR1_P3_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_rr1_p3_params);
  }

  TEST(F16_VELU__AVX2_RR1_P3_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_rr1_p3_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_rr1_p3_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_rr1_p3_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_rr1_p3_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U8, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_rr1_p3_params);
      }
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U8, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_rr1_p3_params);
      }
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U8, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_rr1_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VELU__AVX2_RR1_P3_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_rr1_p3_params);
  }

  TEST(F16_VELU__AVX2_RR1_P3_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_rr1_p3_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_rr1_p3_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_rr1_p3_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_rr1_p3_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U16, prescale) {
    TEST_REQUIRES_X86_AVX2;
    for (float prescale : std::vector<float>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_rr1_p3_params);
      }
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U16, alpha) {
    TEST_REQUIRES_X86_AVX2;
    for (float alpha : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_rr1_p3_params);
      }
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U16, beta) {
    TEST_REQUIRES_X86_AVX2;
    for (float beta : std::vector<float>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_rr1_p3_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
