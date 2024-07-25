// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-vclamp.yaml
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


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VCLAMP__NEONFP16ARITH_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vclamp_ukernel__neonfp16arith_u8, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_u8, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_u8, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_u8, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_U8, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_u8, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_U8, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t qmin = 1; qmin < 255; qmin = xnnpack::NextPrime(qmin)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f16_vclamp_ukernel__neonfp16arith_u8, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_U8, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 8;
    for (size_t qmax = 1; qmax < 255; qmax = xnnpack::NextPrime(qmax)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f16_vclamp_ukernel__neonfp16arith_u8, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_VCLAMP__NEONFP16ARITH_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vclamp_ukernel__neonfp16arith_u16, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_u16, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_u16, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_u16, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_U16, inplace) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vclamp_ukernel__neonfp16arith_u16, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_U16, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t qmin = 1; qmin < 255; qmin = xnnpack::NextPrime(qmin)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f16_vclamp_ukernel__neonfp16arith_u16, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VCLAMP__NEONFP16ARITH_U16, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t batch_step = 16;
    for (size_t qmax = 1; qmax < 255; qmax = xnnpack::NextPrime(qmax)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f16_vclamp_ukernel__neonfp16arith_u16, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_RISCV_FP16_VECTOR && XNN_ARCH_RISCV
  TEST(F16_VCLAMP__RVVFP16ARITH_U1V, batch_eq_1v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(1 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t))
      .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u1v, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U1V, batch_div_1v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 1 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u1v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U1V, batch_lt_1v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 1 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u1v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U1V, batch_gt_1v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 1 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = batch_step + 1; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u1v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U1V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 1 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u1v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U1V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 1 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t qmin = 1; qmin < 255; qmin = xnnpack::NextPrime(qmin)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += batch_step - 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u1v, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U1V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 1 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t qmax = 1; qmax < 255; qmax = xnnpack::NextPrime(qmax)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += batch_step - 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u1v, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_FP16_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_FP16_VECTOR && XNN_ARCH_RISCV
  TEST(F16_VCLAMP__RVVFP16ARITH_U2V, batch_eq_2v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(2 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t))
      .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u2v, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U2V, batch_div_2v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 2 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u2v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U2V, batch_lt_2v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 2 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u2v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U2V, batch_gt_2v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 2 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u2v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U2V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 2 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u2v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U2V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 2 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t qmin = 1; qmin < 255; qmin = xnnpack::NextPrime(qmin)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += batch_step - 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u2v, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U2V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 2 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t qmax = 1; qmax < 255; qmax = xnnpack::NextPrime(qmax)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += batch_step - 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u2v, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_FP16_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_FP16_VECTOR && XNN_ARCH_RISCV
  TEST(F16_VCLAMP__RVVFP16ARITH_U4V, batch_eq_4v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(4 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t))
      .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u4v, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U4V, batch_div_4v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 4 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u4v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U4V, batch_lt_4v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 4 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u4v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U4V, batch_gt_4v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 4 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u4v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U4V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 4 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u4v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U4V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 4 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t qmin = 1; qmin < 255; qmin = xnnpack::NextPrime(qmin)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += batch_step - 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u4v, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U4V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 4 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t qmax = 1; qmax < 255; qmax = xnnpack::NextPrime(qmax)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += batch_step - 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u4v, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_FP16_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_FP16_VECTOR && XNN_ARCH_RISCV
  TEST(F16_VCLAMP__RVVFP16ARITH_U8V, batch_eq_8v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    VUnaryMicrokernelTester()
      .batch_size(8 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t))
      .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u8v, xnn_init_f16_minmax_fp16arith_params);
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U8V, batch_div_8v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 8 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u8v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U8V, batch_lt_8v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 8 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u8v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U8V, batch_gt_8v) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 8 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u8v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U8V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 8 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u8v, xnn_init_f16_minmax_fp16arith_params);
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U8V, qmin) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 8 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t qmin = 1; qmin < 255; qmin = xnnpack::NextPrime(qmin)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += batch_step - 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u8v, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }

  TEST(F16_VCLAMP__RVVFP16ARITH_U8V, qmax) {
    TEST_REQUIRES_RISCV_VECTOR_FP16_ARITH;
    const size_t batch_step = 8 * xnn_init_hardware_config()->vlenb / sizeof(uint16_t);
    for (size_t qmax = 1; qmax < 255; qmax = xnnpack::NextPrime(qmax)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += batch_step - 1) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f16_vclamp_ukernel__rvvfp16arith_u8v, xnn_init_f16_minmax_fp16arith_params);
      }
    }
  }
#endif  // XNN_ENABLE_RISCV_FP16_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VCLAMP__F16C_U8, batch_eq_8) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_vclamp_ukernel__f16c_u8, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_VCLAMP__F16C_U8, batch_div_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__f16c_u8, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_U8, batch_lt_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__f16c_u8, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_U8, batch_gt_8) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__f16c_u8, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_U8, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vclamp_ukernel__f16c_u8, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_U8, qmin) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t qmin = 1; qmin < 255; qmin = xnnpack::NextPrime(qmin)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f16_vclamp_ukernel__f16c_u8, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VCLAMP__F16C_U8, qmax) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 8;
    for (size_t qmax = 1; qmax < 255; qmax = xnnpack::NextPrime(qmax)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 7) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f16_vclamp_ukernel__f16c_u8, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_VCLAMP__F16C_U16, batch_eq_16) {
    TEST_REQUIRES_X86_F16C;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_vclamp_ukernel__f16c_u16, xnn_init_f16_minmax_avx_params);
  }

  TEST(F16_VCLAMP__F16C_U16, batch_div_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__f16c_u16, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_U16, batch_lt_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__f16c_u16, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_U16, batch_gt_16) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_vclamp_ukernel__f16c_u16, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_U16, inplace) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f16_vclamp_ukernel__f16c_u16, xnn_init_f16_minmax_avx_params);
    }
  }

  TEST(F16_VCLAMP__F16C_U16, qmin) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t qmin = 1; qmin < 255; qmin = xnnpack::NextPrime(qmin)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmin(qmin)
          .Test(xnn_f16_vclamp_ukernel__f16c_u16, xnn_init_f16_minmax_avx_params);
      }
    }
  }

  TEST(F16_VCLAMP__F16C_U16, qmax) {
    TEST_REQUIRES_X86_F16C;
    const size_t batch_step = 16;
    for (size_t qmax = 1; qmax < 255; qmax = xnnpack::NextPrime(qmax)) {
      for (size_t batch_size = 1; batch_size <= 5 * batch_step; batch_size += 15) {
        VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .qmax(qmax)
          .Test(xnn_f16_vclamp_ukernel__f16c_u16, xnn_init_f16_minmax_avx_params);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
