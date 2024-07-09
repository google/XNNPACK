// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vsub-minmax.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vbinary.h"
#include "vbinary-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSUB_MINMAX__NEON_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VBinaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsub_minmax_ukernel__neon_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VSUB_MINMAX__NEON_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__NEON_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__NEON_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__NEON_U4, inplace_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__NEON_U4, inplace_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__NEON_U4, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__NEON_U4, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__NEON_U4, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VSUB_MINMAX__NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsub_minmax_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VSUB_MINMAX__NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__NEON_U8, inplace_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__NEON_U8, inplace_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__NEON_U8, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__NEON_U8, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__NEON_U8, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VSUB_MINMAX__RVV_U4V, batch_eq_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VBinaryMicrokernelTester()
      .batch_size(4 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vsub_minmax_ukernel__rvv_u4v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VSUB_MINMAX__RVV_U4V, batch_div_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 40 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 4 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u4v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__RVV_U4V, batch_lt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u4v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__RVV_U4V, batch_gt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 5 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u4v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__RVV_U4V, inplace_a) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 20 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 3 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u4v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__RVV_U4V, inplace_b) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 20 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 3 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u4v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__RVV_U4V, inplace_a_and_b) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 20 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 3 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u4v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__RVV_U4V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 20 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 3 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u4v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__RVV_U4V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 20 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 3 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u4v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VSUB_MINMAX__RVV_U8V, batch_eq_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VBinaryMicrokernelTester()
      .batch_size(8 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vsub_minmax_ukernel__rvv_u8v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VSUB_MINMAX__RVV_U8V, batch_div_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 16 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 80 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 8 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u8v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__RVV_U8V, batch_lt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u8v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__RVV_U8V, batch_gt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 9 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 16 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u8v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__RVV_U8V, inplace_a) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 40 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 7 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u8v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__RVV_U8V, inplace_b) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 40 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 7 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u8v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__RVV_U8V, inplace_a_and_b) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 40 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 7 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u8v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__RVV_U8V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 40 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 7 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u8v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__RVV_U8V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 40 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 7 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__rvv_u8v, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSUB_MINMAX__SSE_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE;
    VBinaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsub_minmax_ukernel__sse_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_VSUB_MINMAX__SSE_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VSUB_MINMAX__SSE_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VSUB_MINMAX__SSE_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VSUB_MINMAX__SSE_U4, inplace_a) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VSUB_MINMAX__SSE_U4, inplace_b) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VSUB_MINMAX__SSE_U4, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VSUB_MINMAX__SSE_U4, qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VSUB_MINMAX__SSE_U4, qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSUB_MINMAX__SSE_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE;
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsub_minmax_ukernel__sse_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_VSUB_MINMAX__SSE_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VSUB_MINMAX__SSE_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VSUB_MINMAX__SSE_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VSUB_MINMAX__SSE_U8, inplace_a) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VSUB_MINMAX__SSE_U8, inplace_b) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VSUB_MINMAX__SSE_U8, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VSUB_MINMAX__SSE_U8, qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VSUB_MINMAX__SSE_U8, qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__sse_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_sse_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSUB_MINMAX__AVX_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsub_minmax_ukernel__avx_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_VSUB_MINMAX__AVX_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX_U8, inplace_a) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX_U8, inplace_b) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX_U8, inplace_a_and_b) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX_U8, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX_U8, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSUB_MINMAX__AVX_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VBinaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsub_minmax_ukernel__avx_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_VSUB_MINMAX__AVX_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX_U16, inplace_a) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX_U16, inplace_b) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX_U16, inplace_a_and_b) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX_U16, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX_U16, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__avx_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSUB_MINMAX__AVX512F_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VBinaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U16, inplace_a) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U16, inplace_b) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U16, inplace_a_and_b) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U16, qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U16, qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSUB_MINMAX__AVX512F_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VBinaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U32, inplace_a) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U32, inplace_b) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U32, inplace_a_and_b) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U32, qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__AVX512F_U32, qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__avx512f_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U4, batch_eq_4) {
    VBinaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U4, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U4, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U4, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U4, qmin) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U4, qmax) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U8, batch_eq_8) {
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U8, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U8, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U8, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U8, qmin) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U8, qmax) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U16, batch_eq_16) {
    VBinaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U16, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U16, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U16, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U16, qmin) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_ARM_U16, qmax) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U4, batch_eq_4) {
    VBinaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U4, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U4, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U4, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U4, qmin) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U4, qmax) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U8, batch_eq_8) {
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U8, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U8, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U8, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U8, qmin) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U8, qmax) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U16, batch_eq_16) {
    VBinaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U16, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U16, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U16, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U16, qmin) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASMSIMD_X86_U16, qmax) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasmsimd_x86_u16, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_wasmsimd_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSUB_MINMAX__WASM_U1, batch_eq_1) {
    VBinaryMicrokernelTester()
      .batch_size(1)
      .Test(xnn_f32_vsub_minmax_ukernel__wasm_u1, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VSUB_MINMAX__WASM_U1, batch_gt_1) {
    for (size_t batch_size = 2; batch_size < 10; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u1, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U1, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u1, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U1, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u1, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U1, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u1, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U1, qmin) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u1, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U1, qmax) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u1, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSUB_MINMAX__WASM_U2, batch_eq_2) {
    VBinaryMicrokernelTester()
      .batch_size(2)
      .Test(xnn_f32_vsub_minmax_ukernel__wasm_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VSUB_MINMAX__WASM_U2, batch_div_2) {
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U2, batch_lt_2) {
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U2, batch_gt_2) {
    for (size_t batch_size = 3; batch_size < 4; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U2, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U2, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U2, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U2, qmin) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U2, qmax) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSUB_MINMAX__WASM_U4, batch_eq_4) {
    VBinaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsub_minmax_ukernel__wasm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VSUB_MINMAX__WASM_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U4, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U4, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U4, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U4, qmin) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U4, qmax) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSUB_MINMAX__WASM_U8, batch_eq_8) {
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsub_minmax_ukernel__wasm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VSUB_MINMAX__WASM_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U8, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U8, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U8, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U8, qmin) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__WASM_U8, qmax) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__wasm_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_VSUB_MINMAX__HVX_U32, batch_eq_32) {
    TEST_REQUIRES_HVX;
    VBinaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsub_minmax_ukernel__hvx_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VSUB_MINMAX__HVX_U32, batch_div_32) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U32, batch_lt_32) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U32, batch_gt_32) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U32, inplace_a) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U32, inplace_b) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U32, inplace_a_and_b) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U32, qmin) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U32, qmax) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u32, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_VSUB_MINMAX__HVX_U64, batch_eq_64) {
    TEST_REQUIRES_HVX;
    VBinaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vsub_minmax_ukernel__hvx_u64, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VSUB_MINMAX__HVX_U64, batch_div_64) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u64, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U64, batch_lt_64) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u64, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U64, batch_gt_64) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u64, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U64, inplace_a) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u64, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U64, inplace_b) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u64, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U64, inplace_a_and_b) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u64, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U64, qmin) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u64, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U64, qmax) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u64, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_VSUB_MINMAX__HVX_U128, batch_eq_128) {
    TEST_REQUIRES_HVX;
    VBinaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vsub_minmax_ukernel__hvx_u128, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VSUB_MINMAX__HVX_U128, batch_div_128) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u128, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U128, batch_lt_128) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u128, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U128, batch_gt_128) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 129; batch_size < 256; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u128, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U128, inplace_a) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u128, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U128, inplace_b) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u128, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U128, inplace_a_and_b) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u128, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U128, qmin) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u128, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VSUB_MINMAX__HVX_U128, qmax) {
    TEST_REQUIRES_HVX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vsub_minmax_ukernel__hvx_u128, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


TEST(F32_VSUB_MINMAX__SCALAR_U1, batch_eq_1) {
  VBinaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vsub_minmax_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_VSUB_MINMAX__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U1, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U1, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U1, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U1, qmin) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U1, qmax) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U2, batch_eq_2) {
  VBinaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vsub_minmax_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_VSUB_MINMAX__SCALAR_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U2, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U2, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U2, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U2, qmin) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U2, qmax) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U4, batch_eq_4) {
  VBinaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vsub_minmax_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_VSUB_MINMAX__SCALAR_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U4, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U4, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U4, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U4, qmin) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U4, qmax) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U8, batch_eq_8) {
  VBinaryMicrokernelTester()
    .batch_size(8)
    .Test(xnn_f32_vsub_minmax_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_VSUB_MINMAX__SCALAR_U8, batch_div_8) {
  for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U8, batch_lt_8) {
  for (size_t batch_size = 1; batch_size < 8; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U8, batch_gt_8) {
  for (size_t batch_size = 9; batch_size < 16; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U8, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U8, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U8, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U8, qmin) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VSUB_MINMAX__SCALAR_U8, qmax) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_f32_vsub_minmax_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Sub, xnn_init_f32_minmax_scalar_params);
  }
}