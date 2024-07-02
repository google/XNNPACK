// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/u32-filterbank-subtract.yaml
//   Generator: tools/generate-filterbank-subtract-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/filterbank.h"
#include "xnnpack/isa-checks.h"
#include "filterbank-subtract-microkernel-tester.h"


TEST(U32_FILTERBANK_SUBTRACT__SCALAR_X2, batch_eq_2) {
  FilterbankSubtractMicrokernelTester()
    .batch(2)
    .Test(xnn_u32_filterbank_subtract_ukernel__scalar_x2);
}

TEST(U32_FILTERBANK_SUBTRACT__SCALAR_X2, batch_div_2) {
  for (size_t batch = 4; batch < 20; batch += 2) {
    FilterbankSubtractMicrokernelTester()
      .batch(batch)
      .Test(xnn_u32_filterbank_subtract_ukernel__scalar_x2);
  }
}

TEST(U32_FILTERBANK_SUBTRACT__SCALAR_X2, batch_lt_2) {
  for (size_t batch = 2; batch < 2; batch += 2) {
    FilterbankSubtractMicrokernelTester()
      .batch(batch)
      .Test(xnn_u32_filterbank_subtract_ukernel__scalar_x2);
  }
}

TEST(U32_FILTERBANK_SUBTRACT__SCALAR_X2, batch_gt_2) {
  for (size_t batch = 4; batch < 4; batch += 2) {
    FilterbankSubtractMicrokernelTester()
      .batch(batch)
      .Test(xnn_u32_filterbank_subtract_ukernel__scalar_x2);
  }
}

TEST(U32_FILTERBANK_SUBTRACT__SCALAR_X2, inplace) {
  for (size_t batch = 4; batch < 4; batch += 2) {
    FilterbankSubtractMicrokernelTester()
      .batch(batch)
      .inplace(true)
      .Test(xnn_u32_filterbank_subtract_ukernel__scalar_x2);
  }
}