// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vdivc-minmax.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/params-init.h>
#include <xnnpack/vbinary.h>
#include "vbinaryc-microkernel-tester.h"


#if XNN_ARCH_ARM64
  TEST(F16_VDIVC_MINMAX__NEONFP16ARITH_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vdivc_minmax_ukernel__neonfp16arith_x8, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_VDIVC_MINMAX__NEONFP16ARITH_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vdivc_minmax_ukernel__neonfp16arith_x8, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__NEONFP16ARITH_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vdivc_minmax_ukernel__neonfp16arith_x8, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__NEONFP16ARITH_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vdivc_minmax_ukernel__neonfp16arith_x8, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__NEONFP16ARITH_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vdivc_minmax_ukernel__neonfp16arith_x8, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__NEONFP16ARITH_X8, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f16_vdivc_minmax_ukernel__neonfp16arith_x8, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__NEONFP16ARITH_X8, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f16_vdivc_minmax_ukernel__neonfp16arith_x8, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_neon_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F16_VDIVC_MINMAX__NEONFP16ARITH_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vdivc_minmax_ukernel__neonfp16arith_x16, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_neon_params);
  }

  TEST(F16_VDIVC_MINMAX__NEONFP16ARITH_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vdivc_minmax_ukernel__neonfp16arith_x16, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__NEONFP16ARITH_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vdivc_minmax_ukernel__neonfp16arith_x16, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__NEONFP16ARITH_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vdivc_minmax_ukernel__neonfp16arith_x16, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__NEONFP16ARITH_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vdivc_minmax_ukernel__neonfp16arith_x16, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__NEONFP16ARITH_X16, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f16_vdivc_minmax_ukernel__neonfp16arith_x16, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_neon_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__NEONFP16ARITH_X16, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f16_vdivc_minmax_ukernel__neonfp16arith_x16, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_neon_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VDIVC_MINMAX__F16C_X8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vdivc_minmax_ukernel__f16c_x8, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_VDIVC_MINMAX__F16C_X8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vdivc_minmax_ukernel__f16c_x8, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__F16C_X8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vdivc_minmax_ukernel__f16c_x8, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__F16C_X8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vdivc_minmax_ukernel__f16c_x8, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__F16C_X8, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vdivc_minmax_ukernel__f16c_x8, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__F16C_X8, qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f16_vdivc_minmax_ukernel__f16c_x8, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__F16C_X8, qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f16_vdivc_minmax_ukernel__f16c_x8, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VDIVC_MINMAX__F16C_X16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vdivc_minmax_ukernel__f16c_x16, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_VDIVC_MINMAX__F16C_X16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vdivc_minmax_ukernel__f16c_x16, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__F16C_X16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vdivc_minmax_ukernel__f16c_x16, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__F16C_X16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vdivc_minmax_ukernel__f16c_x16, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__F16C_X16, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vdivc_minmax_ukernel__f16c_x16, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__F16C_X16, qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f16_vdivc_minmax_ukernel__f16c_x16, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VDIVC_MINMAX__F16C_X16, qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f16_vdivc_minmax_ukernel__f16c_x16, VBinaryCMicrokernelTester::OpType::DivC, xnn_init_f16_minmax_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
