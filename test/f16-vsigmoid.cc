// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vsigmoid.yaml
//   Generator: tools/generate-vunary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vunary.h>
#include "vunary-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X40, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X48, batch_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X48, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X56, batch_div_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X56, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64
  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AARCH64_NEONFP16ARITH_RR2_P2_DIV_X64, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X40, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X48, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X48, batch_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X48, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X48, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X48, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X56, batch_eq_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X56, batch_div_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X56, batch_lt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X56, batch_gt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X56, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1FMA_X64, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x8, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x16, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X24, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x24, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X32, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x32, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X40, batch_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X40, batch_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X40, batch_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X40, batch_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X40, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x40, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X48, batch_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X48, batch_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X48, batch_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X48, batch_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X48, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x48, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X56, batch_eq_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X56, batch_div_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X56, batch_lt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X56, batch_gt_56) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X56, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x56, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }

  TEST(F16_VSIGMOID__NEONFP16ARITH_RR2_P2_NR1RECPS_X64, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x64, xnn_init_f16_sigmoid_fp16arith_rr2_p2_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x8, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x16, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x24, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x32, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 41; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x40, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 49; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x48, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 57; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x56, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }

  TEST(F16_VSIGMOID__AVX2_RR1_P2_RCP_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x64, xnn_init_f16_sigmoid_avx2_rr1_p2_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
