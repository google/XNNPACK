// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vsqrt.yaml
//   Generator: tools/generate-vunary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vunary.h>
#include "vunary-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_x8);
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_x8);
    }
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_x8);
    }
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_x8);
    }
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_x8);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_x16);
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_x16);
    }
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_x16);
    }
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_x16);
    }
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_x16);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x8);
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x8);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x8);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x8);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x8);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x16);
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x16);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x16);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x16);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x16);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x24);
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x24);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x24);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x24);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x24);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x32);
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x32);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x32);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x32);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_X32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x32);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSQRT__FP16ARITH_SQRT_X1, batch_eq_1) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(1)
      .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x1);
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_X1, batch_gt_1) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 2; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x1);
    }
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_X1, inplace) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x1);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSQRT__FP16ARITH_SQRT_X2, batch_eq_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(2)
      .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x2);
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_X2, batch_div_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x2);
    }
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_X2, batch_lt_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x2);
    }
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_X2, batch_gt_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 3; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x2);
    }
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_X2, inplace) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x2);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSQRT__FP16ARITH_SQRT_X4, batch_eq_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x4);
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_X4, batch_div_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x4);
    }
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_X4, batch_lt_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x4);
    }
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_X4, batch_gt_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x4);
    }
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_X4, inplace) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x4);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSQRT__F16C_SQRT_X8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_x8);
  }

  TEST(F16_VSQRT__F16C_SQRT_X8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_x8);
    }
  }

  TEST(F16_VSQRT__F16C_SQRT_X8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_x8);
    }
  }

  TEST(F16_VSQRT__F16C_SQRT_X8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_x8);
    }
  }

  TEST(F16_VSQRT__F16C_SQRT_X8, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_x8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSQRT__F16C_SQRT_X16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_x16);
  }

  TEST(F16_VSQRT__F16C_SQRT_X16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_x16);
    }
  }

  TEST(F16_VSQRT__F16C_SQRT_X16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_x16);
    }
  }

  TEST(F16_VSQRT__F16C_SQRT_X16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_x16);
    }
  }

  TEST(F16_VSQRT__F16C_SQRT_X16, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
