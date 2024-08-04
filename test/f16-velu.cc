// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-velu.yaml
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


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_scalar_params);
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_scalar_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_scalar_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_scalar_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_scalar_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, prescale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (float prescale : std::array<float, 2>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_scalar_params);
      }
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, alpha) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (float alpha : std::array<float, 2>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_scalar_params);
      }
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U8, beta) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (float beta : std::array<float, 2>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u8, xnn_init_f16_elu_scalar_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_scalar_params);
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_scalar_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_scalar_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_scalar_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_scalar_params);
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, prescale) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (float prescale : std::array<float, 2>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_scalar_params);
      }
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, alpha) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (float alpha : std::array<float, 2>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_scalar_params);
      }
    }
  }

  TEST(F16_VELU__NEONFP16ARITH_RR1_P3_U16, beta) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (float beta : std::array<float, 2>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_u16, xnn_init_f16_elu_scalar_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VELU__AVX2_RR1_P3_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_params);
  }

  TEST(F16_VELU__AVX2_RR1_P3_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U8, prescale) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (float prescale : std::array<float, 2>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_params);
      }
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U8, alpha) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (float alpha : std::array<float, 2>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_params);
      }
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U8, beta) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (float beta : std::array<float, 2>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u8, xnn_init_f16_elu_avx2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VELU__AVX2_RR1_P3_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_params);
  }

  TEST(F16_VELU__AVX2_RR1_P3_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_params);
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U16, prescale) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (float prescale : std::array<float, 2>({0.1f, 10.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .prescale(prescale)
          .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_params);
      }
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U16, alpha) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (float alpha : std::array<float, 2>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .alpha(alpha)
          .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_params);
      }
    }
  }

  TEST(F16_VELU__AVX2_RR1_P3_U16, beta) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (float beta : std::array<float, 2>({0.3f, 3.0f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .beta(beta)
          .Test(xnn_f16_velu_ukernel__avx2_rr1_p3_u16, xnn_init_f16_elu_avx2_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
