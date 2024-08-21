// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vlrelu.yaml
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


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VLRELU__NEON_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vlrelu_ukernel__neon_u4, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__NEON_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__neon_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__NEON_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__neon_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__NEON_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__neon_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__NEON_U4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__neon_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__NEON_U4, slope) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__neon_u4, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VLRELU__NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vlrelu_ukernel__neon_u8, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__neon_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__neon_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__neon_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__NEON_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__neon_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__NEON_U8, slope) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__neon_u8, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VLRELU__RVV_U1V, batch_eq_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(1 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vlrelu_ukernel__rvv_u1v, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__RVV_U1V, batch_div_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 1 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u1v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U1V, batch_lt_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 1 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u1v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U1V, batch_gt_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 1 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = batch_step + 1; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u1v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U1V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 1 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u1v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U1V, slope) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 1 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__rvv_u1v, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VLRELU__RVV_U2V, batch_eq_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vlrelu_ukernel__rvv_u2v, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__RVV_U2V, batch_div_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u2v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U2V, batch_lt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u2v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U2V, batch_gt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u2v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U2V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u2v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U2V, slope) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__rvv_u2v, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VLRELU__RVV_U4V, batch_eq_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(4 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vlrelu_ukernel__rvv_u4v, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__RVV_U4V, batch_div_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u4v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U4V, batch_lt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u4v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U4V, batch_gt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u4v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U4V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u4v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U4V, slope) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__rvv_u4v, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VLRELU__RVV_U8V, batch_eq_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(8 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vlrelu_ukernel__rvv_u8v, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__RVV_U8V, batch_div_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u8v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U8V, batch_lt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u8v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U8V, batch_gt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u8v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U8V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__rvv_u8v, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__RVV_U8V, slope) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__rvv_u8v, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLRELU__SSE_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vlrelu_ukernel__sse_u4, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__SSE_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE_U4, inplace) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__sse_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE_U4, slope) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 4;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__sse_u4, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLRELU__SSE_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vlrelu_ukernel__sse_u8, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__SSE_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE_U8, inplace) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__sse_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE_U8, slope) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 8;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__sse_u8, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLRELU__SSE2_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vlrelu_ukernel__sse2_u4, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__SSE2_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse2_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE2_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse2_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE2_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse2_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE2_U4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__sse2_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE2_U4, slope) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__sse2_u4, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLRELU__SSE2_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vlrelu_ukernel__sse2_u8, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__SSE2_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse2_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE2_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse2_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE2_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse2_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE2_U8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__sse2_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE2_U8, slope) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__sse2_u8, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLRELU__SSE41_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vlrelu_ukernel__sse41_u4, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__SSE41_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse41_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE41_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse41_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE41_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse41_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE41_U4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__sse41_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE41_U4, slope) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 4;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__sse41_u4, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLRELU__SSE41_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vlrelu_ukernel__sse41_u8, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__SSE41_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse41_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE41_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse41_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE41_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__sse41_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE41_U8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__sse41_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__SSE41_U8, slope) {
    TEST_REQUIRES_X86_SSE41;
    const size_t batch_step = 8;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__sse41_u8, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLRELU__AVX_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vlrelu_ukernel__avx_u8, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__AVX_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__avx_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__avx_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__avx_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__avx_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX_U8, slope) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__avx_u8, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLRELU__AVX_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vlrelu_ukernel__avx_u16, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__AVX_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__avx_u16, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__avx_u16, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__avx_u16, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__avx_u16, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX_U16, slope) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__avx_u16, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLRELU__AVX512F_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vlrelu_ukernel__avx512f_u16, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__AVX512F_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__avx512f_u16, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX512F_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__avx512f_u16, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX512F_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__avx512f_u16, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX512F_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__avx512f_u16, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX512F_U16, slope) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__avx512f_u16, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLRELU__AVX512F_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vlrelu_ukernel__avx512f_u32, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__AVX512F_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__avx512f_u32, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX512F_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__avx512f_u32, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX512F_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__avx512f_u32, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX512F_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__avx512f_u32, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__AVX512F_U32, slope) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 31) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__avx512f_u32, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VLRELU__WASMSIMD_IMINMAX_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_u4, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__WASMSIMD_IMINMAX_U4, batch_div_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_IMINMAX_U4, batch_lt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_IMINMAX_U4, batch_gt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_IMINMAX_U4, inplace) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_IMINMAX_U4, slope) {
    const size_t batch_step = 4;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_u4, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VLRELU__WASMSIMD_IMINMAX_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_u8, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__WASMSIMD_IMINMAX_U8, batch_div_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_IMINMAX_U8, batch_lt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_IMINMAX_U8, batch_gt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_IMINMAX_U8, inplace) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_IMINMAX_U8, slope) {
    const size_t batch_step = 8;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_u8, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VLRELU__WASMSIMD_LANESELECT_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_u4, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__WASMSIMD_LANESELECT_U4, batch_div_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_LANESELECT_U4, batch_lt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_LANESELECT_U4, batch_gt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_LANESELECT_U4, inplace) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_LANESELECT_U4, slope) {
    const size_t batch_step = 4;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_u4, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VLRELU__WASMSIMD_LANESELECT_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_u8, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__WASMSIMD_LANESELECT_U8, batch_div_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_LANESELECT_U8, batch_lt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_LANESELECT_U8, batch_gt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_LANESELECT_U8, inplace) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMSIMD_LANESELECT_U8, slope) {
    const size_t batch_step = 8;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_u8, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VLRELU__WASMRELAXEDSIMD_IMINMAX_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_u4, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_IMINMAX_U4, batch_div_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_IMINMAX_U4, batch_lt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_IMINMAX_U4, batch_gt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_IMINMAX_U4, inplace) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_IMINMAX_U4, slope) {
    const size_t batch_step = 4;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_u4, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VLRELU__WASMRELAXEDSIMD_IMINMAX_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_u8, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_IMINMAX_U8, batch_div_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_IMINMAX_U8, batch_lt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_IMINMAX_U8, batch_gt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_IMINMAX_U8, inplace) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_IMINMAX_U8, slope) {
    const size_t batch_step = 8;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_u8, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VLRELU__WASMRELAXEDSIMD_LANESELECT_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_u4, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_LANESELECT_U4, batch_div_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_LANESELECT_U4, batch_lt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_LANESELECT_U4, batch_gt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_LANESELECT_U4, inplace) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_LANESELECT_U4, slope) {
    const size_t batch_step = 4;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_u4, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VLRELU__WASMRELAXEDSIMD_LANESELECT_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_u8, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_LANESELECT_U8, batch_div_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_LANESELECT_U8, batch_lt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_LANESELECT_U8, batch_gt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_LANESELECT_U8, inplace) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_u8, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASMRELAXEDSIMD_LANESELECT_U8, slope) {
    const size_t batch_step = 8;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_u8, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VLRELU__WASM_U1, batch_eq_1) {
    VUnaryMicrokernelTester()
      .batch_size(1)
      .Test(xnn_f32_vlrelu_ukernel__wasm_u1, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__WASM_U1, batch_gt_1) {
    const size_t batch_step = 1;
    for (size_t batch_size = batch_step + 1; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasm_u1, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASM_U1, inplace) {
    const size_t batch_step = 1;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__wasm_u1, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASM_U1, slope) {
    const size_t batch_step = 1;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__wasm_u1, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VLRELU__WASM_U2, batch_eq_2) {
    VUnaryMicrokernelTester()
      .batch_size(2)
      .Test(xnn_f32_vlrelu_ukernel__wasm_u2, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__WASM_U2, batch_div_2) {
    const size_t batch_step = 2;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasm_u2, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASM_U2, batch_lt_2) {
    const size_t batch_step = 2;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasm_u2, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASM_U2, batch_gt_2) {
    const size_t batch_step = 2;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasm_u2, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASM_U2, inplace) {
    const size_t batch_step = 2;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__wasm_u2, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASM_U2, slope) {
    const size_t batch_step = 2;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__wasm_u2, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VLRELU__WASM_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vlrelu_ukernel__wasm_u4, xnn_init_f32_lrelu_scalar_params);
  }

  TEST(F32_VLRELU__WASM_U4, batch_div_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasm_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASM_U4, batch_lt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasm_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASM_U4, batch_gt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vlrelu_ukernel__wasm_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASM_U4, inplace) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vlrelu_ukernel__wasm_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }

  TEST(F32_VLRELU__WASM_U4, slope) {
    const size_t batch_step = 4;
    for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 3) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .slope(slope)
          .Test(xnn_f32_vlrelu_ukernel__wasm_u4, xnn_init_f32_lrelu_scalar_params);
      }
    }
  }
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_VLRELU__SCALAR_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vlrelu_ukernel__scalar_u1, xnn_init_f32_lrelu_scalar_params);
}

TEST(F32_VLRELU__SCALAR_U1, batch_gt_1) {
  const size_t batch_step = 1;
  for (size_t batch_size = batch_step + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vlrelu_ukernel__scalar_u1, xnn_init_f32_lrelu_scalar_params);
  }
}

TEST(F32_VLRELU__SCALAR_U1, inplace) {
  const size_t batch_step = 1;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vlrelu_ukernel__scalar_u1, xnn_init_f32_lrelu_scalar_params);
  }
}

TEST(F32_VLRELU__SCALAR_U1, slope) {
  const size_t batch_step = 1;
  for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
    for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .slope(slope)
        .Test(xnn_f32_vlrelu_ukernel__scalar_u1, xnn_init_f32_lrelu_scalar_params);
    }
  }
}


TEST(F32_VLRELU__SCALAR_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vlrelu_ukernel__scalar_u2, xnn_init_f32_lrelu_scalar_params);
}

TEST(F32_VLRELU__SCALAR_U2, batch_div_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vlrelu_ukernel__scalar_u2, xnn_init_f32_lrelu_scalar_params);
  }
}

TEST(F32_VLRELU__SCALAR_U2, batch_lt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vlrelu_ukernel__scalar_u2, xnn_init_f32_lrelu_scalar_params);
  }
}

TEST(F32_VLRELU__SCALAR_U2, batch_gt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vlrelu_ukernel__scalar_u2, xnn_init_f32_lrelu_scalar_params);
  }
}

TEST(F32_VLRELU__SCALAR_U2, inplace) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vlrelu_ukernel__scalar_u2, xnn_init_f32_lrelu_scalar_params);
  }
}

TEST(F32_VLRELU__SCALAR_U2, slope) {
  const size_t batch_step = 2;
  for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
    for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .slope(slope)
        .Test(xnn_f32_vlrelu_ukernel__scalar_u2, xnn_init_f32_lrelu_scalar_params);
    }
  }
}


TEST(F32_VLRELU__SCALAR_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vlrelu_ukernel__scalar_u4, xnn_init_f32_lrelu_scalar_params);
}

TEST(F32_VLRELU__SCALAR_U4, batch_div_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vlrelu_ukernel__scalar_u4, xnn_init_f32_lrelu_scalar_params);
  }
}

TEST(F32_VLRELU__SCALAR_U4, batch_lt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vlrelu_ukernel__scalar_u4, xnn_init_f32_lrelu_scalar_params);
  }
}

TEST(F32_VLRELU__SCALAR_U4, batch_gt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vlrelu_ukernel__scalar_u4, xnn_init_f32_lrelu_scalar_params);
  }
}

TEST(F32_VLRELU__SCALAR_U4, inplace) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vlrelu_ukernel__scalar_u4, xnn_init_f32_lrelu_scalar_params);
  }
}

TEST(F32_VLRELU__SCALAR_U4, slope) {
  const size_t batch_step = 4;
  for (float slope : std::array<float, 3>({-0.7f, 0.3f, 1.3f})) {
    for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .slope(slope)
        .Test(xnn_f32_vlrelu_ukernel__scalar_u4, xnn_init_f32_lrelu_scalar_params);
    }
  }
}
