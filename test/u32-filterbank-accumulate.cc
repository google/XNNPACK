// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/u32-filterbank-accumulate.yaml
//   Generator: tools/generate-filterbank-accumulate-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/filterbank.h>
#include "filterbank-accumulate-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(U32_FILTERBANK_ACCUMULATE__NEON_X1, batch_eq_1) {
    TEST_REQUIRES_ARM_NEON;
    FilterbankAccumulateMicrokernelTester()
      .batch(1)
      .Test(xnn_u32_filterbank_accumulate_ukernel__neon_x1);
  }

  TEST(U32_FILTERBANK_ACCUMULATE__NEON_X1, batch_gt_1) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 2; batch < 10; batch++) {
      FilterbankAccumulateMicrokernelTester()
        .batch(batch)
        .Test(xnn_u32_filterbank_accumulate_ukernel__neon_x1);
    }
  }

  TEST(U32_FILTERBANK_ACCUMULATE__NEON_X1, rows_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    FilterbankAccumulateMicrokernelTester()
      .rows(2)
      .batch(1)
      .Test(xnn_u32_filterbank_accumulate_ukernel__neon_x1);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(U32_FILTERBANK_ACCUMULATE__NEON_X2, batch_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    FilterbankAccumulateMicrokernelTester()
      .batch(2)
      .Test(xnn_u32_filterbank_accumulate_ukernel__neon_x2);
  }

  TEST(U32_FILTERBANK_ACCUMULATE__NEON_X2, batch_div_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 4; batch < 20; batch += 2) {
      FilterbankAccumulateMicrokernelTester()
        .batch(batch)
        .Test(xnn_u32_filterbank_accumulate_ukernel__neon_x2);
    }
  }

  TEST(U32_FILTERBANK_ACCUMULATE__NEON_X2, batch_lt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 1; batch < 2; batch++) {
      FilterbankAccumulateMicrokernelTester()
        .batch(batch)
        .Test(xnn_u32_filterbank_accumulate_ukernel__neon_x2);
    }
  }

  TEST(U32_FILTERBANK_ACCUMULATE__NEON_X2, batch_gt_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch = 3; batch < 4; batch++) {
      FilterbankAccumulateMicrokernelTester()
        .batch(batch)
        .Test(xnn_u32_filterbank_accumulate_ukernel__neon_x2);
    }
  }

  TEST(U32_FILTERBANK_ACCUMULATE__NEON_X2, rows_eq_2) {
    TEST_REQUIRES_ARM_NEON;
    FilterbankAccumulateMicrokernelTester()
      .rows(2)
      .batch(2)
      .Test(xnn_u32_filterbank_accumulate_ukernel__neon_x2);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


TEST(U32_FILTERBANK_ACCUMULATE__SCALAR_X1, batch_eq_1) {
  FilterbankAccumulateMicrokernelTester()
    .batch(1)
    .Test(xnn_u32_filterbank_accumulate_ukernel__scalar_x1);
}

TEST(U32_FILTERBANK_ACCUMULATE__SCALAR_X1, batch_gt_1) {
  for (size_t batch = 2; batch < 10; batch++) {
    FilterbankAccumulateMicrokernelTester()
      .batch(batch)
      .Test(xnn_u32_filterbank_accumulate_ukernel__scalar_x1);
  }
}

TEST(U32_FILTERBANK_ACCUMULATE__SCALAR_X1, rows_eq_2) {
  FilterbankAccumulateMicrokernelTester()
    .rows(2)
    .batch(1)
    .Test(xnn_u32_filterbank_accumulate_ukernel__scalar_x1);
}
