// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vmax.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/microparams-init.h>
#include <xnnpack/vbinary.h>
#include "vbinary-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VMAX__NEONFP16ARITH_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vmax_ukernel__neonfp16arith_u8, VBinaryMicrokernelTester::OpType::Max);
  }

  TEST(F16_VMAX__NEONFP16ARITH_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__neonfp16arith_u8, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__NEONFP16ARITH_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__neonfp16arith_u8, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__NEONFP16ARITH_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__neonfp16arith_u8, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__NEONFP16ARITH_U8, inplace_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f16_vmax_ukernel__neonfp16arith_u8, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__NEONFP16ARITH_U8, inplace_b) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f16_vmax_ukernel__neonfp16arith_u8, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__NEONFP16ARITH_U8, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f16_vmax_ukernel__neonfp16arith_u8, VBinaryMicrokernelTester::OpType::Max);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VMAX__NEONFP16ARITH_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VBinaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vmax_ukernel__neonfp16arith_u16, VBinaryMicrokernelTester::OpType::Max);
  }

  TEST(F16_VMAX__NEONFP16ARITH_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__neonfp16arith_u16, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__NEONFP16ARITH_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__neonfp16arith_u16, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__NEONFP16ARITH_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__neonfp16arith_u16, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__NEONFP16ARITH_U16, inplace_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f16_vmax_ukernel__neonfp16arith_u16, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__NEONFP16ARITH_U16, inplace_b) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f16_vmax_ukernel__neonfp16arith_u16, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__NEONFP16ARITH_U16, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f16_vmax_ukernel__neonfp16arith_u16, VBinaryMicrokernelTester::OpType::Max);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VMAX__FP16ARITH_U1, batch_eq_1) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    VBinaryMicrokernelTester()
      .batch_size(1)
      .Test(xnn_f16_vmax_ukernel__fp16arith_u1, VBinaryMicrokernelTester::OpType::Max);
  }

  TEST(F16_VMAX__FP16ARITH_U1, batch_gt_1) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 2; batch_size < 10; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u1, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__FP16ARITH_U1, inplace_a) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u1, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__FP16ARITH_U1, inplace_b) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u1, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__FP16ARITH_U1, inplace_a_and_b) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u1, VBinaryMicrokernelTester::OpType::Max);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VMAX__FP16ARITH_U2, batch_eq_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    VBinaryMicrokernelTester()
      .batch_size(2)
      .Test(xnn_f16_vmax_ukernel__fp16arith_u2, VBinaryMicrokernelTester::OpType::Max);
  }

  TEST(F16_VMAX__FP16ARITH_U2, batch_div_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u2, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__FP16ARITH_U2, batch_lt_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u2, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__FP16ARITH_U2, batch_gt_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 3; batch_size < 4; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u2, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__FP16ARITH_U2, inplace_a) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u2, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__FP16ARITH_U2, inplace_b) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u2, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__FP16ARITH_U2, inplace_a_and_b) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u2, VBinaryMicrokernelTester::OpType::Max);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VMAX__FP16ARITH_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    VBinaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f16_vmax_ukernel__fp16arith_u4, VBinaryMicrokernelTester::OpType::Max);
  }

  TEST(F16_VMAX__FP16ARITH_U4, batch_div_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u4, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__FP16ARITH_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u4, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__FP16ARITH_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u4, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__FP16ARITH_U4, inplace_a) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u4, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__FP16ARITH_U4, inplace_b) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u4, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__FP16ARITH_U4, inplace_a_and_b) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f16_vmax_ukernel__fp16arith_u4, VBinaryMicrokernelTester::OpType::Max);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VMAX__F16C_U8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vmax_ukernel__f16c_u8, VBinaryMicrokernelTester::OpType::Max);
  }

  TEST(F16_VMAX__F16C_U8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__f16c_u8, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__F16C_U8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__f16c_u8, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__F16C_U8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__f16c_u8, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__F16C_U8, inplace_a) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f16_vmax_ukernel__f16c_u8, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__F16C_U8, inplace_b) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f16_vmax_ukernel__f16c_u8, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__F16C_U8, inplace_a_and_b) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f16_vmax_ukernel__f16c_u8, VBinaryMicrokernelTester::OpType::Max);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VMAX__F16C_U16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VBinaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vmax_ukernel__f16c_u16, VBinaryMicrokernelTester::OpType::Max);
  }

  TEST(F16_VMAX__F16C_U16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__f16c_u16, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__F16C_U16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__f16c_u16, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__F16C_U16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vmax_ukernel__f16c_u16, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__F16C_U16, inplace_a) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f16_vmax_ukernel__f16c_u16, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__F16C_U16, inplace_b) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f16_vmax_ukernel__f16c_u16, VBinaryMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_VMAX__F16C_U16, inplace_a_and_b) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f16_vmax_ukernel__f16c_u16, VBinaryMicrokernelTester::OpType::Max);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
