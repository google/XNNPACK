// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/bf16-rminmax.yaml
//   Generator: tools/generate-reduce-test.py


#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/reduce.h"
#include "test/reduce-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(BF16_RMINMAX__NEON_U16_ACC4, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_bf16_rminmax_ukernel__neon_u16_acc4, ReduceMicrokernelTester::OpType::MinMax);
  }

  TEST(BF16_RMINMAX__NEON_U16_ACC4, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_bf16_rminmax_ukernel__neon_u16_acc4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(BF16_RMINMAX__NEON_U16_ACC4, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_bf16_rminmax_ukernel__neon_u16_acc4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }

  TEST(BF16_RMINMAX__NEON_U16_ACC4, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_bf16_rminmax_ukernel__neon_u16_acc4, ReduceMicrokernelTester::OpType::MinMax);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
