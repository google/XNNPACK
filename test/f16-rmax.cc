// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-rmax.yaml
//   Generator: tools/generate-reduce-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/microparams-init.h>
#include <xnnpack/reduce.h>
#include "reduce-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RMAX__NEONFP16ARITH_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_rmax_ukernel__neonfp16arith_u32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RMAX__NEONFP16ARITH_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmax_ukernel__neonfp16arith_u32, ReduceMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_RMAX__NEONFP16ARITH_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmax_ukernel__neonfp16arith_u32, ReduceMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_RMAX__NEONFP16ARITH_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmax_ukernel__neonfp16arith_u32, ReduceMicrokernelTester::OpType::Max);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RMAX__F16C_U32, batch_eq_32) {
    TEST_REQUIRES_X86_F16C;
    ReduceMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_rmax_ukernel__f16c_u32, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(F16_RMAX__F16C_U32, batch_div_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmax_ukernel__f16c_u32, ReduceMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_RMAX__F16C_U32, batch_lt_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmax_ukernel__f16c_u32, ReduceMicrokernelTester::OpType::Max);
    }
  }

  TEST(F16_RMAX__F16C_U32, batch_gt_32) {
    TEST_REQUIRES_X86_F16C;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmax_ukernel__f16c_u32, ReduceMicrokernelTester::OpType::Max);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
