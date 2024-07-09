// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vrdivc-minmax.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vbinary.h"
#include "vbinaryc-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VRDIVC_MINMAX__AARCH64_NEONFP16ARITH_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vrdivc_minmax_ukernel__aarch64_neonfp16arith_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_VRDIVC_MINMAX__AARCH64_NEONFP16ARITH_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__aarch64_neonfp16arith_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AARCH64_NEONFP16ARITH_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__aarch64_neonfp16arith_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AARCH64_NEONFP16ARITH_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__aarch64_neonfp16arith_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AARCH64_NEONFP16ARITH_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vrdivc_minmax_ukernel__aarch64_neonfp16arith_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AARCH64_NEONFP16ARITH_U8, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__aarch64_neonfp16arith_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AARCH64_NEONFP16ARITH_U8, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__aarch64_neonfp16arith_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VRDIVC_MINMAX__AARCH64_NEONFP16ARITH_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vrdivc_minmax_ukernel__aarch64_neonfp16arith_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_VRDIVC_MINMAX__AARCH64_NEONFP16ARITH_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__aarch64_neonfp16arith_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AARCH64_NEONFP16ARITH_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__aarch64_neonfp16arith_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AARCH64_NEONFP16ARITH_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__aarch64_neonfp16arith_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AARCH64_NEONFP16ARITH_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vrdivc_minmax_ukernel__aarch64_neonfp16arith_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AARCH64_NEONFP16ARITH_U16, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__aarch64_neonfp16arith_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AARCH64_NEONFP16ARITH_U16, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__aarch64_neonfp16arith_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U1, batch_eq_1) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    VBinaryCMicrokernelTester()
      .batch_size(1)
      .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u1, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U1, batch_gt_1) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 2; batch_size < 10; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u1, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U1, inplace) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u1, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U1, qmin) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u1, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U1, qmax) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u1, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U2, batch_eq_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    VBinaryCMicrokernelTester()
      .batch_size(2)
      .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U2, batch_div_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U2, batch_lt_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U2, batch_gt_2) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 3; batch_size < 4; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U2, inplace) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U2, qmin) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U2, qmax) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    VBinaryCMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U4, batch_div_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U4, inplace) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U4, qmin) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__FP16ARITH_U4, qmax) {
    TEST_REQUIRES_ARM_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__fp16arith_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_SCALAR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F16_VRDIVC_MINMAX__AVX512FP16_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512FP16;
    VBinaryCMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u32, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_VRDIVC_MINMAX__AVX512FP16_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512FP16;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u32, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AVX512FP16_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512FP16;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u32, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AVX512FP16_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512FP16;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u32, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AVX512FP16_U32, inplace) {
    TEST_REQUIRES_X86_AVX512FP16;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u32, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AVX512FP16_U32, qmin) {
    TEST_REQUIRES_X86_AVX512FP16;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u32, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AVX512FP16_U32, qmax) {
    TEST_REQUIRES_X86_AVX512FP16;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u32, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F16_VRDIVC_MINMAX__AVX512FP16_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512FP16;
    VBinaryCMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u64, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_VRDIVC_MINMAX__AVX512FP16_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512FP16;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u64, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AVX512FP16_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512FP16;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u64, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AVX512FP16_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512FP16;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u64, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AVX512FP16_U64, inplace) {
    TEST_REQUIRES_X86_AVX512FP16;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u64, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AVX512FP16_U64, qmin) {
    TEST_REQUIRES_X86_AVX512FP16;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u64, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__AVX512FP16_U64, qmax) {
    TEST_REQUIRES_X86_AVX512FP16;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u64, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_fp16arith_params);
    }
  }
#endif  // XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VRDIVC_MINMAX__F16C_U8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vrdivc_minmax_ukernel__f16c_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_VRDIVC_MINMAX__F16C_U8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__f16c_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__F16C_U8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__f16c_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__F16C_U8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__f16c_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__F16C_U8, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vrdivc_minmax_ukernel__f16c_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__F16C_U8, qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__f16c_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__F16C_U8, qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__f16c_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VRDIVC_MINMAX__F16C_U16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vrdivc_minmax_ukernel__f16c_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_VRDIVC_MINMAX__F16C_U16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__f16c_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__F16C_U16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__f16c_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__F16C_U16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vrdivc_minmax_ukernel__f16c_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__F16C_U16, inplace) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vrdivc_minmax_ukernel__f16c_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__F16C_U16, qmin) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__f16c_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VRDIVC_MINMAX__F16C_U16, qmax) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f16_vrdivc_minmax_ukernel__f16c_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f16_minmax_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
