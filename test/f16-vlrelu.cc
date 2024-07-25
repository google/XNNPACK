// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vlrelu.yaml
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
  TEST(F16_VLRELU__NEONFP16ARITH_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_u8, xnn_init_f16_lrelu_fp16arith_params);
  }

  TEST(F16_VLRELU__NEONFP16ARITH_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_u8, xnn_init_f16_lrelu_fp16arith_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_u8, xnn_init_f16_lrelu_fp16arith_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_u8, xnn_init_f16_lrelu_fp16arith_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_u8, xnn_init_f16_lrelu_fp16arith_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_U8, slope) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_u8, xnn_init_f16_lrelu_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VLRELU__NEONFP16ARITH_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_u16, xnn_init_f16_lrelu_fp16arith_params);
  }

  TEST(F16_VLRELU__NEONFP16ARITH_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_u16, xnn_init_f16_lrelu_fp16arith_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_u16, xnn_init_f16_lrelu_fp16arith_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_u16, xnn_init_f16_lrelu_fp16arith_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_u16, xnn_init_f16_lrelu_fp16arith_params);
    }
  }

  TEST(F16_VLRELU__NEONFP16ARITH_U16, slope) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f16_vlrelu_ukernel__neonfp16arith_u16, xnn_init_f16_lrelu_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VLRELU__F16C_U8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vlrelu_ukernel__f16c_u8, xnn_init_f16_lrelu_avx_params);
  }

  TEST(F16_VLRELU__F16C_U8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__f16c_u8, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_U8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__f16c_u8, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_U8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__f16c_u8, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_U8, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vlrelu_ukernel__f16c_u8, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_U8, slope) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f16_vlrelu_ukernel__f16c_u8, xnn_init_f16_lrelu_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VLRELU__F16C_U16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vlrelu_ukernel__f16c_u16, xnn_init_f16_lrelu_avx_params);
  }

  TEST(F16_VLRELU__F16C_U16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__f16c_u16, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_U16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__f16c_u16, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_U16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vlrelu_ukernel__f16c_u16, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_U16, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vlrelu_ukernel__f16c_u16, xnn_init_f16_lrelu_avx_params);
    }
  }

  TEST(F16_VLRELU__F16C_U16, slope) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f16_vlrelu_ukernel__f16c_u16, xnn_init_f16_lrelu_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
