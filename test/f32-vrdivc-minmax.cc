// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vrdivc-minmax.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vbinary.h"
#include "vbinaryc-microkernel-tester.h"


#if XNN_ARCH_ARM64
  TEST(F32_VRDIVC_MINMAX__AARCH64_NEON_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VBinaryCMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VRDIVC_MINMAX__AARCH64_NEON_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AARCH64_NEON_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AARCH64_NEON_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AARCH64_NEON_U4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AARCH64_NEON_U4, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AARCH64_NEON_U4, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VRDIVC_MINMAX__AARCH64_NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VRDIVC_MINMAX__AARCH64_NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AARCH64_NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AARCH64_NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AARCH64_NEON_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AARCH64_NEON_U8, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AARCH64_NEON_U8, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__aarch64_neon_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_ARM64


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VRDIVC_MINMAX__RVV_U4V, batch_eq_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VBinaryCMicrokernelTester()
      .batch_size(4 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vrdivc_minmax_ukernel__rvv_u4v, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VRDIVC_MINMAX__RVV_U4V, batch_div_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 40 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 4 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__rvv_u4v, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__RVV_U4V, batch_lt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__rvv_u4v, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__RVV_U4V, batch_gt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 5 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__rvv_u4v, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__RVV_U4V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 20 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 3 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__rvv_u4v, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__RVV_U4V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 20 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 3 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__rvv_u4v, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__RVV_U4V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 20 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 3 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__rvv_u4v, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VRDIVC_MINMAX__RVV_U8V, batch_eq_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VBinaryCMicrokernelTester()
      .batch_size(8 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vrdivc_minmax_ukernel__rvv_u8v, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VRDIVC_MINMAX__RVV_U8V, batch_div_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 16 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 80 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 8 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__rvv_u8v, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__RVV_U8V, batch_lt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__rvv_u8v, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__RVV_U8V, batch_gt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 9 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size < 16 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__rvv_u8v, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__RVV_U8V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 40 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 7 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__rvv_u8v, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__RVV_U8V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 40 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 7 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__rvv_u8v, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__RVV_U8V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t batch_size = 1;
                batch_size <= 40 * xnn_init_hardware_config()->vlenb / sizeof(float);
                batch_size += 7 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__rvv_u8v, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VRDIVC_MINMAX__SSE_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE;
    VBinaryCMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vrdivc_minmax_ukernel__sse_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_VRDIVC_MINMAX__SSE_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__sse_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__SSE_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__sse_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__SSE_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__sse_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__SSE_U4, inplace) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__sse_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__SSE_U4, qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__sse_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__SSE_U4, qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__sse_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_sse_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VRDIVC_MINMAX__SSE_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE;
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vrdivc_minmax_ukernel__sse_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_sse_params);
  }

  TEST(F32_VRDIVC_MINMAX__SSE_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__sse_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__SSE_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__sse_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__SSE_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__sse_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__SSE_U8, inplace) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__sse_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__SSE_U8, qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__sse_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_sse_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__SSE_U8, qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__sse_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_sse_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VRDIVC_MINMAX__AVX_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vrdivc_minmax_ukernel__avx_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_VRDIVC_MINMAX__AVX_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX_U8, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX_U8, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VRDIVC_MINMAX__AVX_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vrdivc_minmax_ukernel__avx_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_avx_params);
  }

  TEST(F32_VRDIVC_MINMAX__AVX_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX_U16, qmin) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_avx_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX_U16, qmax) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_avx_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VRDIVC_MINMAX__AVX512F_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vrdivc_minmax_ukernel__avx512f_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VRDIVC_MINMAX__AVX512F_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx512f_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX512F_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx512f_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX512F_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx512f_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX512F_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx512f_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX512F_U16, qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx512f_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX512F_U16, qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx512f_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VRDIVC_MINMAX__AVX512F_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VBinaryCMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vrdivc_minmax_ukernel__avx512f_u32, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VRDIVC_MINMAX__AVX512F_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx512f_u32, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX512F_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx512f_u32, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX512F_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx512f_u32, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX512F_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx512f_u32, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX512F_U32, qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx512f_u32, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__AVX512F_U32, qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__avx512f_u32, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U4, batch_eq_4) {
    VBinaryCMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U4, qmin) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U4, qmax) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U8, batch_eq_8) {
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U8, qmin) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U8, qmax) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U16, batch_eq_16) {
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U16, qmin) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_ARM_U16, qmax) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_arm_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U4, batch_eq_4) {
    VBinaryCMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U4, qmin) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U4, qmax) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U8, batch_eq_8) {
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U8, qmin) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U8, qmax) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U16, batch_eq_16) {
    VBinaryCMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U16, inplace) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U16, qmin) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASMSIMD_X86_U16, qmax) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasmsimd_x86_u16, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_wasmsimd_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRDIVC_MINMAX__WASM_U1, batch_eq_1) {
    VBinaryCMicrokernelTester()
      .batch_size(1)
      .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u1, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U1, batch_gt_1) {
    for (size_t batch_size = 2; batch_size < 10; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u1, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U1, inplace) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u1, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U1, qmin) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u1, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U1, qmax) {
    for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u1, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRDIVC_MINMAX__WASM_U2, batch_eq_2) {
    VBinaryCMicrokernelTester()
      .batch_size(2)
      .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U2, batch_div_2) {
    for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U2, batch_lt_2) {
    for (size_t batch_size = 1; batch_size < 2; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U2, batch_gt_2) {
    for (size_t batch_size = 3; batch_size < 4; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U2, inplace) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U2, qmin) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U2, qmax) {
    for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRDIVC_MINMAX__WASM_U4, batch_eq_4) {
    VBinaryCMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U4, batch_div_4) {
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U4, batch_lt_4) {
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U4, batch_gt_4) {
    for (size_t batch_size = 5; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U4, inplace) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U4, qmin) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U4, qmax) {
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VRDIVC_MINMAX__WASM_U8, batch_eq_8) {
    VBinaryCMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U8, inplace) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U8, qmin) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }

  TEST(F32_VRDIVC_MINMAX__WASM_U8, qmax) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryCMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_f32_vrdivc_minmax_ukernel__wasm_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_VRDIVC_MINMAX__SCALAR_U1, batch_eq_1) {
  VBinaryCMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u1, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u1, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u1, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U1, qmin) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u1, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U1, qmax) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u1, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U2, batch_eq_2) {
  VBinaryCMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U2, qmin) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U2, qmax) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u2, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U4, batch_eq_4) {
  VBinaryCMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U4, qmin) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U4, qmax) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u4, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U8, batch_eq_8) {
  VBinaryCMicrokernelTester()
    .batch_size(8)
    .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U8, batch_div_8) {
  for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U8, batch_lt_8) {
  for (size_t batch_size = 1; batch_size < 8; batch_size++) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U8, batch_gt_8) {
  for (size_t batch_size = 9; batch_size < 16; batch_size++) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U8, inplace) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U8, qmin) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}

TEST(F32_VRDIVC_MINMAX__SCALAR_U8, qmax) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryCMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_f32_vrdivc_minmax_ukernel__scalar_u8, VBinaryCMicrokernelTester::OpType::RDivC, xnn_init_f32_minmax_scalar_params);
  }
}