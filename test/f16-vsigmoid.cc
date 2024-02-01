// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vsigmoid.yaml
//   Generator: tools/generate-vunary-test.py


#include <vector>

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/vunary.h>

#include "vunary-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U40, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U48, batch_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U48, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U56, batch_div_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U56, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_U64, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U40, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U48, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U48, batch_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U48, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U48, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U48, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U56, batch_eq_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U56, batch_div_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U56, batch_lt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U56, batch_gt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U56, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_U64, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_u64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U24, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U40, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U48, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U48, batch_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U48, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U48, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U48, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U56, batch_eq_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U56, batch_div_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U56, batch_lt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U56, batch_gt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U56, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_U64, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_u64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_U64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_u64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
