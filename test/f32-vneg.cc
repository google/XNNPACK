// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vneg.yaml
//   Generator: tools/generate-vunary-test.py


#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"
#include "vunary-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VNEG__NEON_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .TestNeg(xnn_f32_vneg_ukernel__neon_u4);
  }

  TEST(F32_VNEG__NEON_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__neon_u4);
    }
  }

  TEST(F32_VNEG__NEON_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__neon_u4);
    }
  }

  TEST(F32_VNEG__NEON_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__neon_u4);
    }
  }

  TEST(F32_VNEG__NEON_U4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__neon_u4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VNEG__NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .TestNeg(xnn_f32_vneg_ukernel__neon_u8);
  }

  TEST(F32_VNEG__NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__neon_u8);
    }
  }

  TEST(F32_VNEG__NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__neon_u8);
    }
  }

  TEST(F32_VNEG__NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__neon_u8);
    }
  }

  TEST(F32_VNEG__NEON_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__neon_u8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VNEG__NEON_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .TestNeg(xnn_f32_vneg_ukernel__neon_u12);
  }

  TEST(F32_VNEG__NEON_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__neon_u12);
    }
  }

  TEST(F32_VNEG__NEON_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__neon_u12);
    }
  }

  TEST(F32_VNEG__NEON_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__neon_u12);
    }
  }

  TEST(F32_VNEG__NEON_U12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__neon_u12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VNEG__RVV_U1V, batch_eq_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(1 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .TestNeg(xnn_f32_vneg_ukernel__rvv_u1v);
  }

  TEST(F32_VNEG__RVV_U1V, batch_div_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 2 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size < 10 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 1 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u1v);
    }
  }

  TEST(F32_VNEG__RVV_U1V, batch_lt_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size < 1 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u1v);
    }
  }

  TEST(F32_VNEG__RVV_U1V, batch_gt_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1; batch_size < 10 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u1v);
    }
  }

  TEST(F32_VNEG__RVV_U1V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size <= 5 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u1v);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VNEG__RVV_U2V, batch_eq_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .TestNeg(xnn_f32_vneg_ukernel__rvv_u2v);
  }

  TEST(F32_VNEG__RVV_U2V, batch_div_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 4 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size < 20 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 2 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u2v);
    }
  }

  TEST(F32_VNEG__RVV_U2V, batch_lt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size < 2 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u2v);
    }
  }

  TEST(F32_VNEG__RVV_U2V, batch_gt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 2 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1; batch_size < 4 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u2v);
    }
  }

  TEST(F32_VNEG__RVV_U2V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size <= 10 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u2v);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VNEG__RVV_U4V, batch_eq_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(4 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .TestNeg(xnn_f32_vneg_ukernel__rvv_u4v);
  }

  TEST(F32_VNEG__RVV_U4V, batch_div_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 8 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size < 40 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 4 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u4v);
    }
  }

  TEST(F32_VNEG__RVV_U4V, batch_lt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size < 4 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u4v);
    }
  }

  TEST(F32_VNEG__RVV_U4V, batch_gt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 4 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1; batch_size < 8 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u4v);
    }
  }

  TEST(F32_VNEG__RVV_U4V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size <= 20 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u4v);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VNEG__RVV_U8V, batch_eq_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(8 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .TestNeg(xnn_f32_vneg_ukernel__rvv_u8v);
  }

  TEST(F32_VNEG__RVV_U8V, batch_div_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 16 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size < 80 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 8 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u8v);
    }
  }

  TEST(F32_VNEG__RVV_U8V, batch_lt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size < 8 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u8v);
    }
  }

  TEST(F32_VNEG__RVV_U8V, batch_gt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 8 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1; batch_size < 16 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u8v);
    }
  }

  TEST(F32_VNEG__RVV_U8V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1; batch_size <= 40 * xnn_init_hardware_config()->vlenb / sizeof(float); batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__rvv_u8v);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VNEG__SSE2_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .TestNeg(xnn_f32_vneg_ukernel__sse2_u4);
  }

  TEST(F32_VNEG__SSE2_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__sse2_u4);
    }
  }

  TEST(F32_VNEG__SSE2_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__sse2_u4);
    }
  }

  TEST(F32_VNEG__SSE2_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__sse2_u4);
    }
  }

  TEST(F32_VNEG__SSE2_U4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__sse2_u4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VNEG__SSE2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .TestNeg(xnn_f32_vneg_ukernel__sse2_u8);
  }

  TEST(F32_VNEG__SSE2_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__sse2_u8);
    }
  }

  TEST(F32_VNEG__SSE2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__sse2_u8);
    }
  }

  TEST(F32_VNEG__SSE2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__sse2_u8);
    }
  }

  TEST(F32_VNEG__SSE2_U8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__sse2_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VNEG__SSE2_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .TestNeg(xnn_f32_vneg_ukernel__sse2_u12);
  }

  TEST(F32_VNEG__SSE2_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__sse2_u12);
    }
  }

  TEST(F32_VNEG__SSE2_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__sse2_u12);
    }
  }

  TEST(F32_VNEG__SSE2_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__sse2_u12);
    }
  }

  TEST(F32_VNEG__SSE2_U12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__sse2_u12);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VNEG__AVX_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .TestNeg(xnn_f32_vneg_ukernel__avx_u8);
  }

  TEST(F32_VNEG__AVX_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx_u8);
    }
  }

  TEST(F32_VNEG__AVX_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx_u8);
    }
  }

  TEST(F32_VNEG__AVX_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx_u8);
    }
  }

  TEST(F32_VNEG__AVX_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__avx_u8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VNEG__AVX_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .TestNeg(xnn_f32_vneg_ukernel__avx_u16);
  }

  TEST(F32_VNEG__AVX_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx_u16);
    }
  }

  TEST(F32_VNEG__AVX_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx_u16);
    }
  }

  TEST(F32_VNEG__AVX_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx_u16);
    }
  }

  TEST(F32_VNEG__AVX_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__avx_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VNEG__AVX_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .TestNeg(xnn_f32_vneg_ukernel__avx_u24);
  }

  TEST(F32_VNEG__AVX_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx_u24);
    }
  }

  TEST(F32_VNEG__AVX_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx_u24);
    }
  }

  TEST(F32_VNEG__AVX_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx_u24);
    }
  }

  TEST(F32_VNEG__AVX_U24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__avx_u24);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VNEG__AVX512F_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .TestNeg(xnn_f32_vneg_ukernel__avx512f_u16);
  }

  TEST(F32_VNEG__AVX512F_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx512f_u16);
    }
  }

  TEST(F32_VNEG__AVX512F_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx512f_u16);
    }
  }

  TEST(F32_VNEG__AVX512F_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx512f_u16);
    }
  }

  TEST(F32_VNEG__AVX512F_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__avx512f_u16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VNEG__AVX512F_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .TestNeg(xnn_f32_vneg_ukernel__avx512f_u32);
  }

  TEST(F32_VNEG__AVX512F_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx512f_u32);
    }
  }

  TEST(F32_VNEG__AVX512F_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx512f_u32);
    }
  }

  TEST(F32_VNEG__AVX512F_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx512f_u32);
    }
  }

  TEST(F32_VNEG__AVX512F_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__avx512f_u32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VNEG__AVX512F_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .TestNeg(xnn_f32_vneg_ukernel__avx512f_u48);
  }

  TEST(F32_VNEG__AVX512F_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx512f_u48);
    }
  }

  TEST(F32_VNEG__AVX512F_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx512f_u48);
    }
  }

  TEST(F32_VNEG__AVX512F_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__avx512f_u48);
    }
  }

  TEST(F32_VNEG__AVX512F_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__avx512f_u48);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_VNEG__HVX_U32, batch_eq_32) {
    TEST_REQUIRES_HVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .TestNeg(xnn_f32_vneg_ukernel__hvx_u32);
  }

  TEST(F32_VNEG__HVX_U32, batch_div_32) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__hvx_u32);
    }
  }

  TEST(F32_VNEG__HVX_U32, batch_lt_32) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__hvx_u32);
    }
  }

  TEST(F32_VNEG__HVX_U32, batch_gt_32) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__hvx_u32);
    }
  }

  TEST(F32_VNEG__HVX_U32, inplace) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__hvx_u32);
    }
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_VNEG__HVX_U64, batch_eq_64) {
    TEST_REQUIRES_HVX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .TestNeg(xnn_f32_vneg_ukernel__hvx_u64);
  }

  TEST(F32_VNEG__HVX_U64, batch_div_64) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__hvx_u64);
    }
  }

  TEST(F32_VNEG__HVX_U64, batch_lt_64) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__hvx_u64);
    }
  }

  TEST(F32_VNEG__HVX_U64, batch_gt_64) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__hvx_u64);
    }
  }

  TEST(F32_VNEG__HVX_U64, inplace) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__hvx_u64);
    }
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_VNEG__HVX_U128, batch_eq_128) {
    TEST_REQUIRES_HVX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .TestNeg(xnn_f32_vneg_ukernel__hvx_u128);
  }

  TEST(F32_VNEG__HVX_U128, batch_div_128) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__hvx_u128);
    }
  }

  TEST(F32_VNEG__HVX_U128, batch_lt_128) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__hvx_u128);
    }
  }

  TEST(F32_VNEG__HVX_U128, batch_gt_128) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__hvx_u128);
    }
  }

  TEST(F32_VNEG__HVX_U128, inplace) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__hvx_u128);
    }
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VNEG__WASMSIMD_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .TestNeg(xnn_f32_vneg_ukernel__wasmsimd_u4);
  }

  TEST(F32_VNEG__WASMSIMD_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__wasmsimd_u4);
    }
  }

  TEST(F32_VNEG__WASMSIMD_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__wasmsimd_u4);
    }
  }

  TEST(F32_VNEG__WASMSIMD_U4, batch_gt_4) {
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__wasmsimd_u4);
    }
  }

  TEST(F32_VNEG__WASMSIMD_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__wasmsimd_u4);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VNEG__WASMSIMD_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .TestNeg(xnn_f32_vneg_ukernel__wasmsimd_u8);
  }

  TEST(F32_VNEG__WASMSIMD_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_VNEG__WASMSIMD_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_VNEG__WASMSIMD_U8, batch_gt_8) {
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__wasmsimd_u8);
    }
  }

  TEST(F32_VNEG__WASMSIMD_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__wasmsimd_u8);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VNEG__WASMSIMD_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .TestNeg(xnn_f32_vneg_ukernel__wasmsimd_u12);
  }

  TEST(F32_VNEG__WASMSIMD_U12, batch_div_12) {
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__wasmsimd_u12);
    }
  }

  TEST(F32_VNEG__WASMSIMD_U12, batch_lt_12) {
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__wasmsimd_u12);
    }
  }

  TEST(F32_VNEG__WASMSIMD_U12, batch_gt_12) {
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestNeg(xnn_f32_vneg_ukernel__wasmsimd_u12);
    }
  }

  TEST(F32_VNEG__WASMSIMD_U12, inplace) {
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestNeg(xnn_f32_vneg_ukernel__wasmsimd_u12);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_VNEG__SCALAR_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .TestNeg(xnn_f32_vneg_ukernel__scalar_u1);
}

TEST(F32_VNEG__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestNeg(xnn_f32_vneg_ukernel__scalar_u1);
  }
}

TEST(F32_VNEG__SCALAR_U1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestNeg(xnn_f32_vneg_ukernel__scalar_u1);
  }
}


TEST(F32_VNEG__SCALAR_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .TestNeg(xnn_f32_vneg_ukernel__scalar_u2);
}

TEST(F32_VNEG__SCALAR_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestNeg(xnn_f32_vneg_ukernel__scalar_u2);
  }
}

TEST(F32_VNEG__SCALAR_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestNeg(xnn_f32_vneg_ukernel__scalar_u2);
  }
}

TEST(F32_VNEG__SCALAR_U2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestNeg(xnn_f32_vneg_ukernel__scalar_u2);
  }
}

TEST(F32_VNEG__SCALAR_U2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestNeg(xnn_f32_vneg_ukernel__scalar_u2);
  }
}


TEST(F32_VNEG__SCALAR_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .TestNeg(xnn_f32_vneg_ukernel__scalar_u4);
}

TEST(F32_VNEG__SCALAR_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestNeg(xnn_f32_vneg_ukernel__scalar_u4);
  }
}

TEST(F32_VNEG__SCALAR_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestNeg(xnn_f32_vneg_ukernel__scalar_u4);
  }
}

TEST(F32_VNEG__SCALAR_U4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestNeg(xnn_f32_vneg_ukernel__scalar_u4);
  }
}

TEST(F32_VNEG__SCALAR_U4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestNeg(xnn_f32_vneg_ukernel__scalar_u4);
  }
}
