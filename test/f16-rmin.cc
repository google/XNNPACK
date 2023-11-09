// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-rmin.yaml
//   Generator: tools/generate-reduce-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/microparams-init.h>
#include <xnnpack/reduce.h>
#include "reduce-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RMIN__NEONFP16ARITH_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f16_rmin_ukernel__neonfp16arith_u8, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F16_RMIN__NEONFP16ARITH_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u8, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u8, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u8, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RMIN__NEONFP16ARITH_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_rmin_ukernel__neonfp16arith_u16, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F16_RMIN__NEONFP16ARITH_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u16, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u16, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u16, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RMIN__NEONFP16ARITH_U16_ACC2, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f16_rmin_ukernel__neonfp16arith_u16_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F16_RMIN__NEONFP16ARITH_U16_ACC2, batch_div_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u16_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U16_ACC2, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u16_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U16_ACC2, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u16_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RMIN__NEONFP16ARITH_U24, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_rmin_ukernel__neonfp16arith_u24, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F16_RMIN__NEONFP16ARITH_U24, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u24, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U24, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u24, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U24, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u24, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RMIN__NEONFP16ARITH_U24_ACC2, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_rmin_ukernel__neonfp16arith_u24_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F16_RMIN__NEONFP16ARITH_U24_ACC2, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u24_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U24_ACC2, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u24_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U24_ACC2, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u24_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RMIN__NEONFP16ARITH_U24_ACC3, batch_eq_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f16_rmin_ukernel__neonfp16arith_u24_acc3, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F16_RMIN__NEONFP16ARITH_U24_ACC3, batch_div_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u24_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U24_ACC3, batch_lt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u24_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U24_ACC3, batch_gt_24) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 25; batch_size < 48; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u24_acc3, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RMIN__NEONFP16ARITH_U32, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_rmin_ukernel__neonfp16arith_u32, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F16_RMIN__NEONFP16ARITH_U32, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u32, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U32, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u32, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U32, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u32, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RMIN__NEONFP16ARITH_U32_ACC2, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_rmin_ukernel__neonfp16arith_u32_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F16_RMIN__NEONFP16ARITH_U32_ACC2, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u32_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U32_ACC2, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u32_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U32_ACC2, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u32_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RMIN__NEONFP16ARITH_U32_ACC4, batch_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f16_rmin_ukernel__neonfp16arith_u32_acc4, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F16_RMIN__NEONFP16ARITH_U32_ACC4, batch_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u32_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U32_ACC4, batch_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u32_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U32_ACC4, batch_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u32_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RMIN__NEONFP16ARITH_U64, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_rmin_ukernel__neonfp16arith_u64, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F16_RMIN__NEONFP16ARITH_U64, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u64, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U64, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u64, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U64, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u64, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RMIN__NEONFP16ARITH_U64_ACC2, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_rmin_ukernel__neonfp16arith_u64_acc2, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F16_RMIN__NEONFP16ARITH_U64_ACC2, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u64_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U64_ACC2, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u64_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U64_ACC2, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u64_acc2, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RMIN__NEONFP16ARITH_U64_ACC4, batch_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    ReduceMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f16_rmin_ukernel__neonfp16arith_u64_acc4, ReduceMicrokernelTester::OpType::Min);
  }

  TEST(F16_RMIN__NEONFP16ARITH_U64_ACC4, batch_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u64_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U64_ACC4, batch_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u64_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }

  TEST(F16_RMIN__NEONFP16ARITH_U64_ACC4, batch_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      ReduceMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f16_rmin_ukernel__neonfp16arith_u64_acc4, ReduceMicrokernelTester::OpType::Min);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


TEST(F16_RMIN__SCALAR_U1, batch_eq_1) {
  ReduceMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f16_rmin_ukernel__scalar_u1, ReduceMicrokernelTester::OpType::Min);
}

TEST(F16_RMIN__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_rmin_ukernel__scalar_u1, ReduceMicrokernelTester::OpType::Min);
  }
}


TEST(F16_RMIN__SCALAR_U2_ACC2, batch_eq_2) {
  ReduceMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f16_rmin_ukernel__scalar_u2_acc2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F16_RMIN__SCALAR_U2_ACC2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_rmin_ukernel__scalar_u2_acc2, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F16_RMIN__SCALAR_U2_ACC2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_rmin_ukernel__scalar_u2_acc2, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F16_RMIN__SCALAR_U2_ACC2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_rmin_ukernel__scalar_u2_acc2, ReduceMicrokernelTester::OpType::Min);
  }
}


TEST(F16_RMIN__SCALAR_U3_ACC3, batch_eq_3) {
  ReduceMicrokernelTester()
    .batch_size(3)
    .Test(xnn_f16_rmin_ukernel__scalar_u3_acc3, ReduceMicrokernelTester::OpType::Min);
}

TEST(F16_RMIN__SCALAR_U3_ACC3, batch_div_3) {
  for (size_t batch_size = 6; batch_size < 30; batch_size += 3) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_rmin_ukernel__scalar_u3_acc3, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F16_RMIN__SCALAR_U3_ACC3, batch_lt_3) {
  for (size_t batch_size = 1; batch_size < 3; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_rmin_ukernel__scalar_u3_acc3, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F16_RMIN__SCALAR_U3_ACC3, batch_gt_3) {
  for (size_t batch_size = 4; batch_size < 6; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_rmin_ukernel__scalar_u3_acc3, ReduceMicrokernelTester::OpType::Min);
  }
}


TEST(F16_RMIN__SCALAR_U4_ACC2, batch_eq_4) {
  ReduceMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f16_rmin_ukernel__scalar_u4_acc2, ReduceMicrokernelTester::OpType::Min);
}

TEST(F16_RMIN__SCALAR_U4_ACC2, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_rmin_ukernel__scalar_u4_acc2, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F16_RMIN__SCALAR_U4_ACC2, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_rmin_ukernel__scalar_u4_acc2, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F16_RMIN__SCALAR_U4_ACC2, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_rmin_ukernel__scalar_u4_acc2, ReduceMicrokernelTester::OpType::Min);
  }
}


TEST(F16_RMIN__SCALAR_U4_ACC4, batch_eq_4) {
  ReduceMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f16_rmin_ukernel__scalar_u4_acc4, ReduceMicrokernelTester::OpType::Min);
}

TEST(F16_RMIN__SCALAR_U4_ACC4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_rmin_ukernel__scalar_u4_acc4, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F16_RMIN__SCALAR_U4_ACC4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_rmin_ukernel__scalar_u4_acc4, ReduceMicrokernelTester::OpType::Min);
  }
}

TEST(F16_RMIN__SCALAR_U4_ACC4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    ReduceMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f16_rmin_ukernel__scalar_u4_acc4, ReduceMicrokernelTester::OpType::Min);
  }
}
