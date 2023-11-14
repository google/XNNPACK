// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vcmul.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/microparams-init.h>
#include <xnnpack/vbinary.h>
#include "vcmul-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VCMUL__NEONFP16ARITH_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VCMulMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u8);
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u8);
    }
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u8);
    }
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u8);
    }
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U8, inplace_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u8);
    }
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U8, inplace_b) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u8);
    }
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U8, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u8);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VCMUL__NEONFP16ARITH_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VCMulMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u16);
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u16);
    }
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u16);
    }
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u16);
    }
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U16, inplace_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u16);
    }
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U16, inplace_b) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u16);
    }
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U16, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u16);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VCMUL__NEONFP16ARITH_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VCMulMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u32);
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u32);
    }
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u32);
    }
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u32);
    }
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U32, inplace_a) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u32);
    }
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U32, inplace_b) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u32);
    }
  }

  TEST(F16_VCMUL__NEONFP16ARITH_U32, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VCMulMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f16_vcmul_ukernel__neonfp16arith_u32);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
