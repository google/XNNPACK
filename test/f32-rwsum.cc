// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-rwsum.yaml
//   Generator: tools/generate-rw-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/reduce.h"
#include "reducewindow-microkernel-tester.h"


TEST(F32_RWSUM__SCALAR_U1, batch_eq_1) {
  ReduceWindowMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_rwsum_ukernel__scalar_u1, ReduceWindowMicrokernelTester::OpType::Sum);
}

TEST(F32_RWSUM__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    ReduceWindowMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_rwsum_ukernel__scalar_u1, ReduceWindowMicrokernelTester::OpType::Sum);
  }
}
