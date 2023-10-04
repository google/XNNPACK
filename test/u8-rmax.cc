// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/u8-rmax.yaml
//   Generator: tools/generate-reduce-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/microparams-init.h>
#include <xnnpack/reduce.h>
#include "reduce-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(U8_RMAX__NEON_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_u8_rmax_ukernel__neon_u16, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(U8_RMAX__NEON_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rmax_ukernel__neon_u16, ReduceMicrokernelTester::OpType::Max);
    }
  }

  TEST(U8_RMAX__NEON_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rmax_ukernel__neon_u16, ReduceMicrokernelTester::OpType::Max);
    }
  }

  TEST(U8_RMAX__NEON_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rmax_ukernel__neon_u16, ReduceMicrokernelTester::OpType::Max);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(U8_RMAX__SSE2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_u8_rmax_ukernel__sse2_u16, ReduceMicrokernelTester::OpType::Max);
  }

  TEST(U8_RMAX__SSE2_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rmax_ukernel__sse2_u16, ReduceMicrokernelTester::OpType::Max);
    }
  }

  TEST(U8_RMAX__SSE2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rmax_ukernel__sse2_u16, ReduceMicrokernelTester::OpType::Max);
    }
  }

  TEST(U8_RMAX__SSE2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_u8_rmax_ukernel__sse2_u16, ReduceMicrokernelTester::OpType::Max);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


TEST(U8_RMAX__SCALAR_U2, batch_eq_2) {
  ReduceMicrokernelTester()
    .batch_size(2)
    .Test(xnn_u8_rmax_ukernel__scalar_u2, ReduceMicrokernelTester::OpType::Max);
}

TEST(U8_RMAX__SCALAR_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rmax_ukernel__scalar_u2, ReduceMicrokernelTester::OpType::Max);
  }
}

TEST(U8_RMAX__SCALAR_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rmax_ukernel__scalar_u2, ReduceMicrokernelTester::OpType::Max);
  }
}

TEST(U8_RMAX__SCALAR_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_u8_rmax_ukernel__scalar_u2, ReduceMicrokernelTester::OpType::Max);
  }
}
