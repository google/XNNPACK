// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vclamp.yaml
//   Generator: tools/generate-vunary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vunary.h>
#include "vunary-microkernel-tester.h"


#if XNN_ARCH_ARM64
  TEST(F16_VCLAMP__NEONFP16ARITH_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vclamp_ukernel__neonfp16arith_x8, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_x8, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_x8, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_x8, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_x8, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_X8, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f16_vclamp_ukernel__neonfp16arith_x8, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_X8, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f16_vclamp_ukernel__neonfp16arith_x8, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_VCLAMP__NEONFP16ARITH_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vclamp_ukernel__neonfp16arith_x16, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_x16, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_x16, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_x16, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_x16, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_X16, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f16_vclamp_ukernel__neonfp16arith_x16, xnn_init_f16_minmax_neon_params);
      }
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_X16, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f16_vclamp_ukernel__neonfp16arith_x16, xnn_init_f16_minmax_neon_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VCLAMP__F16C_X8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vclamp_ukernel__f16c_x8, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_VCLAMP__F16C_X8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__f16c_x8, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_X8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__f16c_x8, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_X8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__f16c_x8, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_X8, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vclamp_ukernel__f16c_x8, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_X8, qmin) {
    TEST_REQUIRES_X86_F16C;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f16_vclamp_ukernel__f16c_x8, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VCLAMP__F16C_X8, qmax) {
    TEST_REQUIRES_X86_F16C;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f16_vclamp_ukernel__f16c_x8, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VCLAMP__F16C_X16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vclamp_ukernel__f16c_x16, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_VCLAMP__F16C_X16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__f16c_x16, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_X16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__f16c_x16, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_X16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__f16c_x16, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_X16, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vclamp_ukernel__f16c_x16, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_X16, qmin) {
    TEST_REQUIRES_X86_F16C;
    for (uint8_t qmin = 1; qmin < 255; qmin++) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f16_vclamp_ukernel__f16c_x16, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VCLAMP__F16C_X16, qmax) {
    TEST_REQUIRES_X86_F16C;
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f16_vclamp_ukernel__f16c_x16, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
