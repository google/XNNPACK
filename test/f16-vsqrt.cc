// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vsqrt.yaml
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


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u8);
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u8);
    }
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u8);
    }
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u8);
    }
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u8);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u16);
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u16);
    }
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u16);
    }
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u16);
    }
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u16);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u32);
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u32);
    }
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u32);
    }
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u32);
    }
  }

  TEST(F16_VSQRT__AARCH64_NEONFP16ARITH_SQRT_U32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u32);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u8);
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u8);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u8);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u8);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u8);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u16);
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u16);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u16);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u16);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u16);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u32);
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u32);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u32);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u32);
    }
  }

  TEST(F16_VSQRT__NEONFP16ARITH_NR1FMA1ADJ_U32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_u32);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSQRT__FP16ARITH_SQRT_U1, batch_eq_1) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(1)
      .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u1);
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_U1, batch_gt_1) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    const size_t batch_step = 1;
    for (size_t batch_size = batch_step + 1; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u1);
    }
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_U1, inplace) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    const size_t batch_step = 1;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u1);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSQRT__FP16ARITH_SQRT_U2, batch_eq_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(2)
      .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u2);
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_U2, batch_div_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    const size_t batch_step = 2;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u2);
    }
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_U2, batch_lt_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    const size_t batch_step = 2;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u2);
    }
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_U2, batch_gt_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    const size_t batch_step = 2;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u2);
    }
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_U2, inplace) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    const size_t batch_step = 2;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u2);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSQRT__FP16ARITH_SQRT_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u4);
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_U4, batch_div_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u4);
    }
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u4);
    }
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u4);
    }
  }

  TEST(F16_VSQRT__FP16ARITH_SQRT_U4, inplace) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_u4);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F16_VSQRT__AVX512FP16_SQRT_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512FP16;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u32);
  }

  TEST(F16_VSQRT__AVX512FP16_SQRT_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512FP16;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u32);
    }
  }

  TEST(F16_VSQRT__AVX512FP16_SQRT_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512FP16;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u32);
    }
  }

  TEST(F16_VSQRT__AVX512FP16_SQRT_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512FP16;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u32);
    }
  }

  TEST(F16_VSQRT__AVX512FP16_SQRT_U32, inplace) {
    TEST_REQUIRES_X86_AVX512FP16;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u32);
    }
  }
#endif  // XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F16_VSQRT__AVX512FP16_SQRT_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512FP16;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u64);
  }

  TEST(F16_VSQRT__AVX512FP16_SQRT_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512FP16;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u64);
    }
  }

  TEST(F16_VSQRT__AVX512FP16_SQRT_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512FP16;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u64);
    }
  }

  TEST(F16_VSQRT__AVX512FP16_SQRT_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512FP16;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u64);
    }
  }

  TEST(F16_VSQRT__AVX512FP16_SQRT_U64, inplace) {
    TEST_REQUIRES_X86_AVX512FP16;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u64);
    }
  }
#endif  // XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F16_VSQRT__AVX512FP16_SQRT_U128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512FP16;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u128);
  }

  TEST(F16_VSQRT__AVX512FP16_SQRT_U128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512FP16;
    const size_t batch_step = 128;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u128);
    }
  }

  TEST(F16_VSQRT__AVX512FP16_SQRT_U128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512FP16;
    const size_t batch_step = 128;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u128);
    }
  }

  TEST(F16_VSQRT__AVX512FP16_SQRT_U128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512FP16;
    const size_t batch_step = 128;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u128);
    }
  }

  TEST(F16_VSQRT__AVX512FP16_SQRT_U128, inplace) {
    TEST_REQUIRES_X86_AVX512FP16;
    const size_t batch_step = 128;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__avx512fp16_sqrt_u128);
    }
  }
#endif  // XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSQRT__AVX512SKX_SQRT_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u16);
  }

  TEST(F16_VSQRT__AVX512SKX_SQRT_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u16);
    }
  }

  TEST(F16_VSQRT__AVX512SKX_SQRT_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u16);
    }
  }

  TEST(F16_VSQRT__AVX512SKX_SQRT_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u16);
    }
  }

  TEST(F16_VSQRT__AVX512SKX_SQRT_U16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSQRT__AVX512SKX_SQRT_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u32);
  }

  TEST(F16_VSQRT__AVX512SKX_SQRT_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u32);
    }
  }

  TEST(F16_VSQRT__AVX512SKX_SQRT_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u32);
    }
  }

  TEST(F16_VSQRT__AVX512SKX_SQRT_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u32);
    }
  }

  TEST(F16_VSQRT__AVX512SKX_SQRT_U32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSQRT__AVX512SKX_SQRT_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u64);
  }

  TEST(F16_VSQRT__AVX512SKX_SQRT_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u64);
    }
  }

  TEST(F16_VSQRT__AVX512SKX_SQRT_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u64);
    }
  }

  TEST(F16_VSQRT__AVX512SKX_SQRT_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u64);
    }
  }

  TEST(F16_VSQRT__AVX512SKX_SQRT_U64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__avx512skx_sqrt_u64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSQRT__F16C_RSQRT_U8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsqrt_ukernel__f16c_rsqrt_u8);
  }

  TEST(F16_VSQRT__F16C_RSQRT_U8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_rsqrt_u8);
    }
  }

  TEST(F16_VSQRT__F16C_RSQRT_U8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_rsqrt_u8);
    }
  }

  TEST(F16_VSQRT__F16C_RSQRT_U8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_rsqrt_u8);
    }
  }

  TEST(F16_VSQRT__F16C_RSQRT_U8, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__f16c_rsqrt_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSQRT__F16C_RSQRT_U16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsqrt_ukernel__f16c_rsqrt_u16);
  }

  TEST(F16_VSQRT__F16C_RSQRT_U16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_rsqrt_u16);
    }
  }

  TEST(F16_VSQRT__F16C_RSQRT_U16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_rsqrt_u16);
    }
  }

  TEST(F16_VSQRT__F16C_RSQRT_U16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_rsqrt_u16);
    }
  }

  TEST(F16_VSQRT__F16C_RSQRT_U16, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__f16c_rsqrt_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSQRT__F16C_RSQRT_U32, batch_eq_32) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsqrt_ukernel__f16c_rsqrt_u32);
  }

  TEST(F16_VSQRT__F16C_RSQRT_U32, batch_div_32) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_rsqrt_u32);
    }
  }

  TEST(F16_VSQRT__F16C_RSQRT_U32, batch_lt_32) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_rsqrt_u32);
    }
  }

  TEST(F16_VSQRT__F16C_RSQRT_U32, batch_gt_32) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_rsqrt_u32);
    }
  }

  TEST(F16_VSQRT__F16C_RSQRT_U32, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__f16c_rsqrt_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSQRT__F16C_SQRT_U8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_u8);
  }

  TEST(F16_VSQRT__F16C_SQRT_U8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_u8);
    }
  }

  TEST(F16_VSQRT__F16C_SQRT_U8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_u8);
    }
  }

  TEST(F16_VSQRT__F16C_SQRT_U8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_u8);
    }
  }

  TEST(F16_VSQRT__F16C_SQRT_U8, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSQRT__F16C_SQRT_U16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_u16);
  }

  TEST(F16_VSQRT__F16C_SQRT_U16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_u16);
    }
  }

  TEST(F16_VSQRT__F16C_SQRT_U16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_u16);
    }
  }

  TEST(F16_VSQRT__F16C_SQRT_U16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_u16);
    }
  }

  TEST(F16_VSQRT__F16C_SQRT_U16, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSQRT__F16C_SQRT_U32, batch_eq_32) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_u32);
  }

  TEST(F16_VSQRT__F16C_SQRT_U32, batch_div_32) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_u32);
    }
  }

  TEST(F16_VSQRT__F16C_SQRT_U32, batch_lt_32) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_u32);
    }
  }

  TEST(F16_VSQRT__F16C_SQRT_U32, batch_gt_32) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_u32);
    }
  }

  TEST(F16_VSQRT__F16C_SQRT_U32, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsqrt_ukernel__f16c_sqrt_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
